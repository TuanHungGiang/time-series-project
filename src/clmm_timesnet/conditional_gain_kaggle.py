"""One-seed comparison of conditional and marginal reconstruction gain on Kaggle."""

import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display


METHODS = ("reconstruction", "marginal_gain", "conditional_gain")
TAILS = ("upper", "lower", "two_sided")
RIDGE = 0.10
COV_REGULARIZATION = 0.10
GAIN_WEIGHT = 0.25
HEALTHY_QUANTILE = 0.90


def robust_scale(reference, floor):
    center = np.median(reference)
    mad = 1.4826 * np.median(np.abs(reference - center))
    return center, max(mad, 0.25 * reference.std(), floor)


def gain_features(gain):
    return np.column_stack((gain.mean(axis=1), gain.std(axis=1)))


class ConditionalGain:
    """Fit all score references from healthy windows exposed to the test mask."""

    def __init__(self, reconstruction, gain=None):
        x = np.log1p(reconstruction)
        self.rec_center, self.rec_scale = robust_scale(x, 1e-6)
        self.slope = None
        if gain is None:
            return
        if len(x) < 6 or len(x) != len(gain):
            raise ValueError("Expected at least six paired healthy reference windows")
        u = gain_features(gain)
        self.gain_center = u.mean(axis=0)
        self.gain_scale = np.maximum(u.std(axis=0), 1e-6)
        u = (u - self.gain_center) / self.gain_scale
        self.x_center = x.mean()
        self.x_scale = max(x.std(), 1e-6)
        z = (x - self.x_center) / self.x_scale
        self.slope = (z @ u / len(z)) / (np.mean(z ** 2) + RIDGE)
        residual = u - z[:, None] * self.slope
        self.covariances = tuple(np.cov(v, rowvar=False) + COV_REGULARIZATION * np.eye(2)
                                 for v in (u, residual))
        self.distance_references = [robust_scale(np.log1p(self.distance(v, c)), 0.05)
                                    for v, c in zip((u, residual), self.covariances, strict=True)]

    @staticmethod
    def distance(values, covariance):
        whitened = np.linalg.solve(np.linalg.cholesky(covariance), values.T)
        return np.square(whitened).sum(axis=0)

    def scores(self, reconstruction, gain=None):
        x = np.log1p(reconstruction)
        base = (x - self.rec_center) / self.rec_scale
        if self.slope is None:
            zero = np.zeros_like(base)
            return base, zero, zero, {}
        raw_u = gain_features(gain)
        u = (raw_u - self.gain_center) / self.gain_scale
        residual = u - ((x - self.x_center) / self.x_scale)[:, None] * self.slope
        terms = []
        for v, covariance, (center, scale) in zip(
            (u, residual), self.covariances, self.distance_references, strict=True,
        ):
            terms.append((np.log1p(self.distance(v, covariance)) - center) / scale)
        return base, *terms, dict(gain_mean=raw_u[:, 0], gain_sd=raw_u[:, 1],
                                  residual_mean=residual[:, 0], residual_sd=residual[:, 1])


def head_fit_split(dataset, group, n_val):
    if dataset == "Z24":
        return group.reference, np.ones(n_val, dtype=bool)
    val = group.metadata.iloc[:n_val]
    if (n_val != 20 or val.label.ne(0).any() or not all(c == 1 for c in group.counts[:n_val])
            or not np.array_equal(group.reference, group.values[:n_val])):
        raise ValueError("Unexpected HellBridge healthy development split")
    # The 20 healthy development windows are five pairs in each of two records.
    # Allocate whole pairs so threshold windows are excluded from head fitting.
    threshold = np.zeros(n_val, dtype=bool)
    for _, record in val.groupby("record_id", sort=False):
        windows = record.sort_values("start").index.to_numpy()
        if len(windows) != 10:
            raise ValueError("Expected ten development windows per healthy recording")
        threshold[windows.reshape(5, 2)[[1, 3]].ravel()] = True
    if int(threshold.sum()) != 8:
        raise ValueError("Expected eight windows for threshold calibration")
    return group.values[:n_val][~threshold], threshold


def development_signals(forward, values, reference, threshold):
    fit = forward(reference)
    if threshold.all():
        return fit, forward(values)
    # This branch holds entire, one-window HellBridge samples apart in both passes.
    held_out = forward(values[threshold])
    combined = []
    for fitted, calibration in zip(fit, held_out, strict=True):
        output = np.empty((len(values), *fitted.shape[1:]), dtype=fitted.dtype)
        output[~threshold], output[threshold] = fitted, calibration
        combined.append(output)
    return fit, combined


def evaluate_group(dataset, group, order, index, steps, args, functions, device, out, priors):
    from clmm_timesnet.laplacian_kaggle import RATES, read_gain
    from clmm_timesnet.old_gain_diagnostics import aggregate_windows
    from clmm_timesnet.targeted_baselines import atomic_csv
    from clmm_timesnet.targeted_gain_head import fit_or_restore_head, score_columns

    cache = out / "groups" / f"{group.name}.csv.gz"
    fit_file = out / "groups" / f"{group.name}_fit.csv"
    if cache.is_file() and fit_file.is_file():
        print(f"{dataset}/{group.name} scores restored", flush=True)
        return pd.read_csv(cache), pd.read_csv(fit_file)

    _, aggregate, _, _, source = functions
    weights = args.root / "weights" / dataset / str(args.seed)
    weights.mkdir(parents=True, exist_ok=True)
    args.previous_clmm_weights = next((root for root in priors if all(
        (root / dataset / str(args.seed) / name).is_file()
        for name in (f"{group.name}.pt", f"{group.name}_adapter.pt")
    )), None)
    scorer, head, norm = fit_or_restore_head(dataset, group, index, args.seed, args, functions, device, weights)
    metadata = group.metadata.drop(columns="path", errors="ignore").copy()
    n_val = int(metadata.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    head_reference, threshold = head_fit_split(dataset, group, n_val)
    threshold_rows = np.r_[threshold, np.zeros(len(metadata) - n_val, dtype=bool)]
    metadata["threshold_healthy"] = threshold_rows & metadata.label.eq(0).to_numpy()
    val, test = group.values[:offset], group.values[offset:]
    # Keep validation and test forward passes separate because TimesNet selects periods per batch.
    (raw_ref_clean,), (raw_val_clean,) = development_signals(
        lambda v: (scorer(v),), val, head_reference, threshold,
    )
    clean_tail = score_columns(dataset, raw_val_clean, group.counts[:n_val], norm, aggregate)
    results, fits = [], []

    for rate in RATES:
        step, actual = steps[rate]
        kept = np.ones(len(group.sensors), dtype=bool)
        kept[order[:min(step, len(kept) - 1)]] = False
        if kept.all():
            fitted = ConditionalGain(raw_ref_clean)
            reconstruction = np.concatenate((raw_val_clean, scorer(test)))
            base, marginal, conditional, _ = fitted.scores(reconstruction)
            components = {name: np.zeros_like(base) for name in ("gain_mean", "gain_sd", "residual_mean", "residual_sd")}
        else:
            (ref_rec, ref_gain), (val_rec, val_gain) = development_signals(
                lambda v: read_gain(head, source, v, kept), val, head_reference, threshold,
            )
            fitted = ConditionalGain(ref_rec, ref_gain)
            test_rec, test_gain = read_gain(head, source, test, kept)
            base, marginal, conditional, components = fitted.scores(
                np.concatenate((val_rec, test_rec)), np.concatenate((val_gain, test_gain)),
            )
        sensor_names = " ".join(np.asarray(group.sensors)[~kept])
        fit = dict(dataset=dataset, group=group.name, requested_loss=rate, actual_loss=actual,
                   removed_sensors=sensor_names, head_reference_windows=len(head_reference),
                   threshold_healthy_samples=int(metadata.threshold_healthy.sum()),
                   reconstruction_center=fitted.rec_center, reconstruction_scale=fitted.rec_scale)
        if fitted.slope is not None:
            fit.update(slope_gain_mean=fitted.slope[0], slope_gain_sd=fitted.slope[1])
        fits.append(fit)
        scores = {}
        for tail, x in zip(TAILS, (base, -base, np.abs(base)), strict=True):
            for method, gain_term in zip(METHODS, (0.0, marginal, conditional), strict=True):
                scores[f"{method}_{tail}"] = aggregate_windows(x + GAIN_WEIGHT * gain_term, group.counts)
        for name, values in dict(**components, marginal_term=marginal, conditional_term=conditional).items():
            scores[name] = aggregate_windows(values, group.counts)
        native_validation = {tail: np.r_[values, np.full(len(metadata) - n_val, np.nan)]
                             for tail, values in clean_tail.items()} if rate == 0 else {}
        results.append(metadata.assign(dataset=dataset, group=group.name, requested_loss=rate,
                                       actual_loss=actual, **scores, **native_validation))
        print(f"  loss={rate}% actual={actual:.2f}% | healthy reference={len(head_reference)} windows | "
              f"retained={kept.sum()}", flush=True)

    scores, fits = pd.concat(results, ignore_index=True), pd.DataFrame(fits)
    atomic_csv(fits, fit_file)
    atomic_csv(scores, cache)
    return scores, fits


def summarize(dataset, scores, out, select_threshold):
    from clmm_timesnet.laplacian_kaggle import RATES, SEED
    from clmm_timesnet.metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
    from clmm_timesnet.targeted_baselines import atomic_csv, clean_calibration

    validation = scores.loc[scores.split.eq("val") & scores.requested_loss.eq(0)]
    tail, _ = clean_calibration(dataset, validation, select_threshold)
    threshold_rows = scores.loc[scores.threshold_healthy]
    test = scores.loc[scores.split.eq("test")].copy()
    clean_ids = test.loc[test.requested_loss.eq(0)].set_index("sample_id").label.sort_index()
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    expected_calibration = 6 if dataset == "Z24" else 8
    rows = []
    if set(test.requested_loss) != set(RATES):
        raise ValueError(f"Missing test conditions: {dataset}")
    for rate, part in test.groupby("requested_loss"):
        if (part.label.value_counts().to_dict() != expected or part.sample_id.duplicated().any()
                or not part.set_index("sample_id").label.sort_index().equals(clean_ids)):
            raise ValueError(f"Incorrect test cohort: {dataset}/{rate}")
        calibration = threshold_rows.loc[threshold_rows.requested_loss.eq(rate)]
        clean = threshold_rows.loc[threshold_rows.requested_loss.eq(0)]
        if (len(calibration) != expected_calibration or calibration.sample_id.duplicated().any()
                or set(calibration.sample_id) != set(clean.sample_id)):
            raise ValueError(f"Incorrect healthy calibration cohort: {dataset}/{rate}")
        y = part.label.to_numpy()
        for method in METHODS:
            column = f"{method}_{tail}"
            values = part[column].to_numpy()
            if not all(np.isfinite(table[column]).all() for table in (calibration, clean, part)):
                raise ValueError(f"Nonfinite scores: {dataset}/{rate}/{method}")
            matched_threshold = threshold_from_healthy(calibration[column], HEALTHY_QUANTILE)
            clean_threshold = threshold_from_healthy(clean[column], HEALTHY_QUANTILE)
            matched = publication_metrics(y, values, matched_threshold)
            frozen = publication_metrics(y, values, clean_threshold)
            rows.append(dict(
                dataset=dataset, seed=SEED, requested_loss=rate, actual_loss=part.actual_loss.iloc[0],
                method=method, tail=tail, auc=matched["auc"], ap=matched["ap"],
                recall_at_fpr_05=recall_at_fpr(y, values, .05), recall_at_fpr_10=recall_at_fpr(y, values, .10),
                matched_threshold=matched_threshold, clean_threshold=clean_threshold,
                **{f"matched_q90_{k}": matched[k] for k in ("ba", "recall", "fpr", "precision", "f1")},
                **{f"clean_q90_{k}": frozen[k] for k in ("ba", "recall", "fpr")},
            ))
    report = pd.DataFrame(rows)
    metrics = ["auc", "ap", "recall_at_fpr_05", "recall_at_fpr_10", "matched_q90_recall", "matched_q90_fpr"]
    indexed = report.set_index(["requested_loss", "method"])
    proposed = indexed.xs("conditional_gain", level="method")
    comparisons = []
    for method in METHODS[:2]:
        paired = proposed[metrics] - indexed.xs(method, level="method")[metrics]
        comparisons.append(paired.add_prefix("delta_").assign(dataset=dataset, against=method).reset_index())
    deltas = pd.concat(comparisons, ignore_index=True)
    for name, frame in (("metrics", report), ("paired_deltas", deltas), ("test_scores", test),
                        ("validation_scores", scores.loc[scores.split.eq("val")])):
        atomic_csv(frame, out / f"{name}.csv")
    return report, deltas


def main():
    from clmm_timesnet.laplacian_kaggle import BASELINE_COMMIT, DATASETS, HELL, RATES, SEED, WORK, Z24, checkout
    from clmm_timesnet.old_gain_diagnostics import loss_steps
    from clmm_timesnet.targeted_baselines import archive_functions, atomic_csv, load_hellbridge, load_z24
    from clmm_timesnet.targeted_gain_head import STRENGTH, sensor_orders, targeted_protocol

    for path in (Z24 / "metadata/records_manifest.csv", HELL):
        if not path.is_file():
            raise FileNotFoundError(f"Attach the required Kaggle dataset: {path}")
    os.chdir(WORK)
    root = WORK / f"clmm_conditional_gain_v1_{SEED}"
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    archive = root / "source_baselines"
    checkout(archive, "codex/mdr-shm-kaggle", BASELINE_COMMIT)
    priors = (WORK / f"clmm_laplacian_v1_{SEED}/weights", WORK / f"clmm_old_gain_diagnostics_{SEED}/weights")
    args = SimpleNamespace(root=root, seed=SEED, epochs=30, adapter_epochs=8, batch_size=16,
                           smoke_window_limit=0, strength=STRENGTH, gain_weight=GAIN_WEIGHT, center_gain=False,
                           baseline_runs=None, previous_clmm_weights=None)
    args.protocol = targeted_protocol(args)
    protocol = dict(seed=SEED, datasets=list(DATASETS), rates=list(RATES), z24=str(Z24), hellbridge=str(HELL),
                    baseline_commit=BASELINE_COMMIT, checkpoint_protocol=args.protocol,
                    missingness="cumulative healthy PCA sensor-sensitivity ordering",
                    gain_features="mean and SD of signed log-error gain across available channels",
                    ridge=RIDGE, covariance_regularization=COV_REGULARIZATION, gain_weight=GAIN_WEIGHT,
                    healthy_quantile=HEALTHY_QUANTILE,
                    z24_calibration="internal healthy training reference for head fit; six healthy validation records for Q90",
                    hellbridge_calibration="12 development windows for head fit and eight separate windows for Q90; "
                                            "healthy development and test share source recordings",
                    test_cohort="unchanged targeted benchmark; six healthy records and 126 damaged records on Z24; "
                                "80 healthy and 640 damaged windows on HellBridge")
    manifest = out / "protocol.json"
    if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Protocol changed. Use a fresh output path: {out}")
    manifest.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    print(f"device={device} | seed={SEED}", flush=True)
    functions = archive_functions(archive / "research/mdr_shm_kaggle")
    reports, deltas = [], []
    for dataset in DATASETS:
        dataset_out = out / dataset
        (dataset_out / "groups").mkdir(parents=True, exist_ok=True)
        groups = list(load_z24(Z24, 0)) if dataset == "Z24" else [load_hellbridge(HELL, dataset_out, 0)]
        saved_order = dataset_out / "sensor_removal_order.csv"
        cached_order = args.root / "weights"
        prior_order = next((p for p in priors if (p.parent / "reports" / dataset / saved_order.name).is_file()), None)
        args.previous_clmm_weights = cached_order if saved_order.is_file() else prior_order
        orders = sensor_orders(dataset, groups, args, dataset_out, functions[0])
        steps = loss_steps(groups)
        tables = [evaluate_group(dataset, g, orders[g.name], i, steps, args, functions, device, dataset_out, priors)
                  for i, g in enumerate(groups)]
        scores, fits = (pd.concat(parts, ignore_index=True) for parts in zip(*tables, strict=True))
        atomic_csv(fits, dataset_out / "head_fit_summary.csv")
        report, paired = summarize(dataset, scores, dataset_out, functions[3])
        reports.append(report)
        deltas.append(paired)
        print(f"\n=== {dataset} ===", flush=True)
        display(report[["requested_loss", "actual_loss", "method", "auc", "ap", "recall_at_fpr_05",
                        "recall_at_fpr_10", "matched_q90_recall", "matched_q90_fpr"]].round(4))
        del groups, tables, scores

    atomic_csv(pd.concat(reports, ignore_index=True), out / "head_comparison.csv")
    atomic_csv(pd.concat(deltas, ignore_index=True), out / "paired_deltas.csv")
    bundle = shutil.make_archive(str(root / "clmm_conditional_gain_reports"), "zip", root_dir=out)
    display(FileLink(Path(bundle).relative_to(WORK).as_posix()))
    print(f"CLMM_CONDITIONAL_GAIN_PASS reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
