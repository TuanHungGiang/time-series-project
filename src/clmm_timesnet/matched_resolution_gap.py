"""Compare CLMM whitening and the manuscript baselines under matched sensor loss."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from . import hellbridge_spectral_kaggle as spectral
from . import mask_pair_block as windows
from . import post_block_audit
from .baseline_validation import validation_callback
from .data.hellbridge import Normalizer
from .data.z24 import load_manifest
from .matched_baselines_kaggle import BaselineBlock, MODELS
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .whitening_two_bridges_kaggle import CANDIDATE, MASKS, METHODS, Z24WhiteningAudit, orient_z24


NATIVE, ADAPTER, GAIN, DIAGONAL, CLMM = METHODS
BASELINES = tuple(model for model in MODELS if model != NATIVE)
REPORT_MODELS = (*MODELS, ADAPTER, GAIN, DIAGONAL, CANDIDATE)
SEEDS = (20270905, 20270913, 20270929)
CONDITIONS = {"clean": (0,), **{f"sensor_{rate:.2f}": tuple(MASKS) for rate in (.10, .30, .50)}}
COHORTS = {"Z24": (6, 126, 6), "HellBridge": (80, 640, 20)}
POLICIES = ("clean_frozen", "mask_matched")
METRICS = (
    "auc", "ap", "ba", "recall", "fpr", "precision", "f1", "diagnostic_recall",
    "recall_at_fpr_05", "recall_at_fpr_10", "auc_within_group",
)
IDENTIFIERS = ["dataset", "points", "seed"]
KEYS = ["condition", "replicate"]


def with_resolution(loader, points):
    def read(*args):
        raw = loader(*args)
        if raw.ndim != 3 or raw.shape[-1] != 1024:
            raise ValueError(f"Expected [windows, sensors, 1024], got {raw.shape}")
        if points == 1024:
            return raw
        return resample_poly(raw, up=1, down=1024 // points, axis=-1).astype(np.float32)
    return read


class MatchedModels:
    def __init__(self, fit, reference, seed, config, device, group, source):
        expected_fit = 8 if config.dataset == "Z24" else 50
        expected_ref = 8 if config.dataset == "Z24" else 10
        if fit.shape != (expected_fit, reference.shape[1], config.points) or len(reference) != expected_ref:
            raise ValueError(f"{config.dataset}/{group}: unexpected shapes {fit.shape}, {reference.shape}")
        print(f"{config.dataset}/{group}: healthy fit={fit.shape}, reference={reference.shape}", flush=True)
        detector_type = post_block_audit.Detector
        normalized_ref = Normalizer.fit(fit).transform(reference)
        callback = validation_callback(
            config.epoch_losses, config.dataset, group, config.seed, NATIVE,
            normalized_ref, config.batch_size, device, source,
        )

        def train_backbone(*args):
            return detector_type(*args, epoch_callback=callback)

        detector = Z24WhiteningAudit if config.dataset == "Z24" else spectral.SpectralGainAudit
        with patch.object(post_block_audit, "Detector", train_backbone):
            self.clmm = detector(fit, reference, seed, config, device, group)
        self.baselines = BaselineBlock(fit, reference, seed, config, device, group, source)
        self.clmm_columns = self.clmm.columns if config.dataset == "Z24" else METHODS
        self.columns = (*self.clmm_columns, *self.baselines.columns)
        self.clmm_indices = [self.clmm.columns.index(name) for name in self.clmm_columns]

    def prepare(self, raw):
        return self.clmm.prepare(raw)

    def scores(self, data, kept):
        candidate = self.clmm.scores(data, kept)[:, self.clmm_indices]
        competitors = self.baselines.scores(data, kept)
        return np.column_stack((candidate, competitors))


def capture_window_scores(save_cases, window_rows):
    def capture(block, raw, metadata, mask_key, args, rows, masks, group, record_level=False, clean_only=False):
        if not record_level:
            return save_cases(block, raw, metadata, mask_key, args, rows, masks, group,
                              record_level=record_level, clean_only=clean_only)

        original = block.scores
        record = metadata.iloc[0]
        cases = iter(windows.cases(raw.shape[1], mask_key, () if clean_only else args.replicates))

        def traced(data, kept):
            condition, replicate, expected = next(cases)
            np.testing.assert_array_equal(kept, expected)
            values = original(data, kept)
            frame = pd.DataFrame(values, columns=block.columns)
            for name in ("record_id", "sample_id", "split", "state", "label"):
                frame[name] = record[name]
            frame["window_index"] = np.arange(len(frame))
            frame["group"], frame["condition"], frame["replicate"] = group, condition, replicate
            window_rows.append(frame)
            return values

        with patch.object(block, "scores", traced):
            save_cases(block, raw, metadata, mask_key, args, rows, masks, group,
                       record_level=record_level, clean_only=clean_only)
    return capture


def orient_scores(rows, window_rows, dataset):
    raw = pd.DataFrame(rows)
    clean = raw.loc[raw.split.eq("val") & raw.condition.eq("clean")]
    if dataset == "Z24":
        calibration_auc = {m: roc_auc_score(clean.label, clean[m]) for m in REPORT_MODELS}
        native_sign = orient_z24(rows)
        signs = {m: native_sign if m in METHODS else windows.native_score_direction(dataset, raw, m)
                 for m in REPORT_MODELS}
    else:
        calibration_auc = dict.fromkeys(REPORT_MODELS, None)
        signs = dict.fromkeys(REPORT_MODELS, 1)
    scores = pd.DataFrame(rows)
    scores[list(REPORT_MODELS)] = scores[list(REPORT_MODELS)].mul(pd.Series(signs), axis=1)
    polarity = pd.DataFrame({"method": REPORT_MODELS, "sign": [signs[m] for m in REPORT_MODELS],
                             "clean_val_raw_auc": [calibration_auc[m] for m in REPORT_MODELS]})
    if window_rows:
        frame = pd.concat(window_rows, ignore_index=True)
        for name in METHODS[2:]:
            alternate = frame.pop(f"{name}_lower_raw")
            if signs[name] == -1:
                frame[name] = alternate
        frame[list(REPORT_MODELS)] = frame[list(REPORT_MODELS)].mul(pd.Series(signs), axis=1)
    else:
        frame = None
    return scores, frame, polarity


def pair_auc(pos, neg):
    distance = pos[:, None] - neg[None, :]
    return float(np.mean((distance > 0) + 0.5 * (distance == 0)))


def measure(scores, dataset):
    n_healthy, n_damage, n_calibration = COHORTS[dataset]
    calibration = scores.loc[scores.split.eq("val") & scores.label.eq(0)]
    test = scores.loc[scores.split.eq("test")]
    clean_cal = calibration.loc[calibration.condition.eq("clean")].set_index("sample_id", verify_integrity=True)
    clean_test = test.loc[test.condition.eq("clean")].set_index("sample_id", verify_integrity=True)
    limits = 0.0 if dataset == "Z24" else .05
    metrics, groups, states = [], [], []
    expected = {(c, r) for c, reps in CONDITIONS.items() for r in reps}
    if set(test[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"{dataset}: unexpected mask conditions")
    if not np.isfinite(scores[list(REPORT_MODELS)]).all().all():
        raise ValueError(f"{dataset}: non-finite scores")
    for (condition, replicate), block in test.groupby(KEYS, sort=False):
        trial = block.set_index("sample_id", verify_integrity=True)
        cal = calibration.loc[calibration.condition.eq(condition) & calibration.replicate.eq(replicate)]
        cal = cal.set_index("sample_id", verify_integrity=True)
        if (set(trial.index) != set(clean_test.index) or set(cal.index) != set(clean_cal.index)
                or len(cal) != n_calibration or trial.label.value_counts().to_dict() != {0: n_healthy, 1: n_damage}):
            raise ValueError(f"{dataset}: different test or calibration cohort at {condition}/{replicate}")
        y = trial.label.to_numpy()
        healthy = y == 0
        strata = {k: np.flatnonzero(trial.group.eq(k)) for k in trial.group.unique()}
        states_by_id = {k: np.flatnonzero(trial.state.eq(k) & trial.label.eq(1))
                        for k in trial.loc[trial.label.eq(1), "state"].unique()}
        for model in REPORT_MODELS:
            values = trial[model].to_numpy()
            clean = clean_test.loc[trial.index, model].to_numpy()
            ranked_clean = clean[~healthy, None] > clean[healthy][None, :]
            reversals = ranked_clean & (values[~healthy, None] <= values[healthy][None, :])
            overlap_pairs = 0
            within_auc_sum = 0.0
            local = {}
            for key, ids in strata.items():
                mask = healthy[ids]
                pairs = int(mask.sum() * (~mask).sum())
                auc = pair_auc(values[ids][~mask], values[ids][mask]) if pairs else np.nan
                if pairs:
                    overlap_pairs += pairs
                    within_auc_sum += auc * pairs
                local[key] = (ids, auc, pairs)
            if not overlap_pairs:
                raise ValueError(f"{dataset}: no within-group healthy–damage pairs")
            diagnostic = recall_at_fpr(y, values, limits)
            r05, r10 = (recall_at_fpr(y, values, limit) for limit in (.05, .10))
            for policy, ref in zip(POLICIES, (clean_cal, cal), strict=True):
                reference = ref[model].to_numpy()
                threshold = (threshold_from_healthy(reference, .90) if dataset == "Z24"
                             else spectral.calibration_cutoff(reference, .05))
                prediction = values >= threshold
                meta = dict(condition=condition, replicate=replicate, method=model, policy=policy)
                metrics.append(dict(
                    **meta, threshold=threshold, healthy=n_healthy, damage=n_damage,
                    healthy_calibration=len(ref), tp=int((prediction & ~healthy).sum()),
                    fp=int((prediction & healthy).sum()), tn=int((~prediction & healthy).sum()),
                    fn=int((~prediction & ~healthy).sum()), rank_reversal=reversals.sum() / max(1, ranked_clean.sum()),
                    within_group_pairs=overlap_pairs, all_pairs=n_healthy * n_damage,
                    auc_within_group=within_auc_sum / overlap_pairs, diagnostic_recall=diagnostic,
                    recall_at_fpr_05=r05, recall_at_fpr_10=r10,
                    **publication_metrics(y, values, threshold),
                ))
                for key, (ids, auc, pairs) in local.items():
                    mask = healthy[ids]
                    alarms = prediction[ids]
                    groups.append(dict(**meta, group=key, auc=auc, pairs=pairs,
                                       healthy=int(mask.sum()), damage=int((~mask).sum()),
                                       healthy_calibration=int(ref.group.eq(key).sum()),
                                       fpr=alarms[mask].mean() if mask.any() else np.nan,
                                       recall=alarms[~mask].mean() if (~mask).any() else np.nan))
                for state, ids in states_by_id.items():
                    states.append(dict(**meta, state=state, damage=len(ids), tp=int(prediction[ids].sum()),
                                       recall=prediction[ids].mean()))
    return pd.DataFrame(metrics), pd.DataFrame(groups), pd.DataFrame(states)


def run_bridge(config, device, source, out):
    out.mkdir(parents=True, exist_ok=True)
    config.epoch_losses, config.train_log, config.reference_log = [], [], []
    window_rows = []

    def make_model(fit, reference, seed, args, accelerator, group):
        return MatchedModels(fit, reference, seed, args, accelerator, group, source)

    factor = 1024 // config.points
    with patch.multiple(
        windows, load_z24=with_resolution(windows.load_z24, config.points),
        load_hellbridge=with_resolution(windows.load_hellbridge, config.points),
        save_cases=capture_window_scores(windows.save_cases, window_rows),
    ), patch.multiple(
        spectral, SEGMENT=spectral.SEGMENT // factor, STRIDE=spectral.STRIDE // factor,
        SAMPLE_HZ=100 / factor,
    ):
        if config.dataset == "Z24":
            rows, masks, fits = windows.run_z24(config, device, make_block=make_model)
        else:
            rows, masks, fits = windows.run_hellbridge(config, device, out, make_block=make_model)
    scores, window_scores, polarity = orient_scores(rows, window_rows, config.dataset)
    metrics, groups, states = measure(scores, config.dataset)
    mask_table = pd.DataFrame(masks)
    sensor_map = {f["group"]: f["channels"] for f in fits}
    mask_table["total_sensors"] = mask_table.group.map(lambda key: len(sensor_map[key]))
    mask_table["actual_loss"] = 1 - mask_table.channels / mask_table.total_sensors
    mask_table["missing_sensors"] = [
        " ".join(label for index, label in enumerate(sensor_map[row.group])
                 if index not in set(map(int, row.kept_indices.split())))
        for row in mask_table.itertuples()
    ]
    adapter = pd.DataFrame(config.train_log).assign(model=ADAPTER, objective="observed-channel MSE")
    loss = pd.DataFrame(config.epoch_losses)
    tables = {
        "scores.csv.gz": scores, "metrics_by_mask.csv": metrics, "group_metrics.csv.gz": groups,
        "damage_state_recall.csv.gz": states, "masks.csv": mask_table,
        "score_orientation.csv": polarity, "epoch_losses.csv": loss, "adapter_train_loss.csv": adapter,
    }
    if window_scores is not None:
        tables["z24_window_scores.csv.gz"] = window_scores
    for name, table in tables.items():
        table.assign(dataset=config.dataset, points=config.points, seed=config.seed).to_csv(out / name, index=False)
    (out / "fit_groups.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    (out / "complete.json").write_text(json.dumps({"metrics_rows": len(metrics)}), encoding="utf-8")
    print(f"Complete {config.dataset} | {config.points} points | seed {config.seed}", flush=True)


def paired_changes(per_mask, out):
    conditions = [*IDENTIFIERS, *KEYS, "policy"]
    masked = per_mask.loc[per_mask.condition.ne("clean")]
    reference = per_mask.loc[per_mask.condition.eq("clean"), [*IDENTIFIERS, "policy", "method", *METRICS]]
    clean_pairs = masked.merge(reference, on=[*IDENTIFIERS, "policy", "method"],
                               suffixes=("", "_ref"), validate="many_to_one")
    paired = [clean_pairs.assign(comparison="sensor_loss_vs_clean", reference=clean_pairs.method)]
    contrasts = [(CLMM, model, "clmm_vs_baseline") for model in MODELS]
    contrasts += [(a, b, "ablation") for a, b in ((ADAPTER, NATIVE), (GAIN, ADAPTER), (DIAGONAL, GAIN), (CLMM, DIAGONAL))]
    for candidate, comparator, kind in contrasts:
        left = per_mask.loc[per_mask.method.eq(candidate)]
        right = per_mask.loc[per_mask.method.eq(comparator), [*conditions, *METRICS]]
        pair = left.merge(right, on=conditions, suffixes=("", "_ref"), validate="one_to_one")
        if len(pair) != len(left):
            raise ValueError(f"Incomplete comparison {candidate} vs {comparator}")
        paired.append(pair.assign(comparison=kind, reference=comparator))
    all_pairs = pd.concat(paired, ignore_index=True)
    pair_keys = [*conditions, "comparison", "method", "reference"]
    deltas = pd.DataFrame({metric: all_pairs[metric] - all_pairs[f"{metric}_ref"] for metric in METRICS})
    deltas = pd.concat([all_pairs[pair_keys], deltas], axis=1)
    deltas.to_csv(out / "paired_by_mask.csv", index=False)
    seed_keys = [key for key in pair_keys if key != "replicate"]
    per_seed = deltas.groupby(seed_keys, as_index=False)[list(METRICS)].mean()
    per_seed.to_csv(out / "paired_by_seed.csv", index=False)
    brief = per_seed.melt(id_vars=seed_keys, var_name="metric", value_name="delta")
    group_keys = [*[key for key in seed_keys if key != "seed"], "metric"]
    summary = brief.groupby(group_keys, as_index=False).delta.agg(
        mean="mean", sd="std", min="min", max="max", seeds="size",
        seeds_positive=lambda x: (x > 0).sum(), seeds_negative=lambda x: (x < 0).sum(),
    )
    summary.to_csv(out / "paired_summary.csv", index=False)
    return summary


def plot_metrics(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = summary.loc[summary.policy.eq("mask_matched")]
    names = {
        "pca_spe": "PCA-SPE", "lstm_ae": "LSTM-AE", NATIVE: "TimesNet", "tranad": "TranAD",
        "moderntcn": "ModernTCN", "scatterad": "ScatterAD", "cnn_ae_1d": "CNN-AE 1D", CLMM: "CLMM-Net",
    }
    for (dataset, points), block in selected.groupby(["dataset", "points"], sort=False):
        fig, plots = plt.subplots(2, 3, figsize=(12.4, 7.5))
        panels = (("auc", "ROC-AUC"), ("ap", "Average precision"), ("ba", "Balanced accuracy"),
                  ("diagnostic_recall", "Diagnostic recall"), ("recall", "Alarm recall"), ("fpr", "Alarm FPR"))
        for plot, (metric, ylabel) in zip(plots.ravel(), panels, strict=True):
            for model, label in names.items():
                rows = block.loc[block.method.eq(model)].set_index("condition").loc[list(CONDITIONS)]
                plot.errorbar([0, 10, 30, 50], rows[f"{metric}_mean"],
                              yerr=rows[f"{metric}_std"], label=label, marker="o", capsize=2,
                              linewidth=2.1 if model == CLMM else 1.0, markersize=4)
            plot.set(xlabel="Nominal sensor loss (%)", ylabel=ylabel, xticks=[0, 10, 30, 50])
            plot.grid(alpha=.2)
        fig.suptitle(f"{dataset} | {points} points | mean ± SD over three training seeds")
        handles, labels = plots.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, .11, 1, .94))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{dataset}_{points}_sensor_loss.{ext}", dpi=165)
        plt.close(fig)


def summarize(shards, out):
    metrics = pd.concat([pd.read_csv(path / "metrics_by_mask.csv") for path in shards], ignore_index=True)
    keys = [*IDENTIFIERS, "condition", "method", "policy"]
    if metrics.duplicated([*keys, "replicate"]).any():
        raise ValueError("Duplicate seed or mask rows")
    expected = len(REPORT_MODELS) * len(POLICIES) * sum(map(len, CONDITIONS.values()))
    if not metrics.groupby(IDENTIFIERS).size().eq(expected).all():
        raise ValueError("Incomplete seed results")
    metrics.to_csv(out / "metrics_by_mask.csv", index=False)
    group = metrics.groupby(keys)
    per_seed = group[list(METRICS)].mean().reset_index()
    per_seed.to_csv(out / "metrics_by_seed.csv", index=False)
    spread = group[list(METRICS)].agg(["std", "min", "max", "size"])
    spread.columns = [f"{metric}_{stat}" for metric, stat in spread.columns]
    spread.reset_index().to_csv(out / "mask_variability.csv", index=False)
    summary_keys = [key for key in keys if key != "seed"]
    summary = per_seed.groupby(summary_keys)[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["seed_runs"] = len(SEEDS)
    summary["effective_seed_runs"] = np.where(summary.method.eq("pca_spe"), 1, len(SEEDS))
    summary.to_csv(out / "summary_mean_sd.csv", index=False)
    paper = summary[[*summary_keys, "seed_runs", "effective_seed_runs"]].copy()
    for metric in METRICS:
        paper[metric] = [
            f"{avg:.4f}" if model == "pca_spe" else f"{avg:.4f} ± {sd:.4f}"
            for avg, sd, model in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"], summary.method, strict=True)
        ]
    paper.to_csv(out / "paper_table.csv", index=False, encoding="utf-8-sig")
    paired_changes(metrics, out)
    for name in ("epoch_losses.csv", "adapter_train_loss.csv", "score_orientation.csv"):
        pd.concat([pd.read_csv(path / name) for path in shards], ignore_index=True).to_csv(out / name, index=False)
    plot_metrics(summary, out)


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", nargs="+", type=int, choices=(128, 1024), default=[128])
    args = parser.parse_args()
    for path in (args.hell_h5, args.z24_root / "metadata/records_manifest.csv",
                 args.baseline_dir / "run_hellbridge_source_baselines.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator.")
    if len(args.points) != len(set(args.points)):
        parser.error("Pass each input resolution only once.")
    os.environ["CLMM_LOG_EACH_EPOCH"] = "1"
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    source = importlib.import_module("run_hellbridge_source_baselines")
    from torch_geometric.nn import GATConv  # noqa: F401

    manifest = load_manifest(args.z24_root)
    damage = manifest.loc[manifest.label.eq(1)]
    if manifest.record_id.duplicated().any() or set(damage.loc[damage.split.eq("val"), "state"]) & set(
        damage.loc[damage.split.eq("test"), "state"]
    ):
        raise ValueError("Z24 has duplicate records or shared validation and test damage states")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    config = argparse.Namespace(
        z24_root=args.z24_root, hell_h5=args.hell_h5, epochs=5, adapter_epochs=8, batch_size=8,
        z24_windows=8, strength=.50, gain_weight=.25, replicates=MASKS, layer_probe=False,
        segment_consensus=False, residual_whitening=True, active_models=BASELINES,
    )
    protocol = dict(
        repo_commit=git_head(Path(__file__).parent), source_commit=git_head(args.baseline_dir),
        z24_manifest_sha256=sha256((args.z24_root / "metadata/records_manifest.csv").read_bytes()).hexdigest(),
        z24_root=str(args.z24_root.resolve()), hell_h5=str(args.hell_h5.resolve()),
        hell_file_bytes=args.hell_h5.stat().st_size, hell_modified_ns=args.hell_h5.stat().st_mtime_ns,
        seeds=SEEDS, models=REPORT_MODELS, masks=MASKS, resolutions=args.points,
        raw_hz=100, raw_points=1024, downsampling="detrend then polyphase antialias",
        epochs=config.epochs, adapter_epochs=config.adapter_epochs, batch_size=config.batch_size,
        no_overlap=True, weighting="mean over masks within seed, then sample SD over seeds",
        thresholds={"Z24": "healthy validation Q90", "HellBridge": "healthy validation rank at alpha=0.05"},
        orientation="Z24 clean labeled validation: shared TimesNet sign for CLMM, model-specific for other baselines",
        test_units={"Z24": "median of eight windows per MAT record", "HellBridge": "windows from held-out damage records and two shared healthy source files"},
        torch=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
    )
    resolved = json.dumps(protocol, indent=2)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text(encoding="utf-8") != resolved:
        raise ValueError("Output belongs to a different protocol. Use a different output directory.")
    protocol_path.write_text(resolved, encoding="utf-8")
    print(resolved, flush=True)
    manifest[["record_id", "setup", "excitation", "split", "state", "label"]].to_csv(
        out / "z24_split_inventory.csv", index=False,
    )
    shards = []
    for points in args.points:
        config.points = points
        for seed in SEEDS:
            config.seed = seed
            for dataset in COHORTS:
                config.dataset = dataset
                shard = out / f"points_{points}" / f"seed_{seed}" / dataset
                shards.append(shard)
                if (shard / "complete.json").is_file():
                    print(f"Resume: {shard}", flush=True)
                else:
                    run_bridge(config, device, source, shard)
    summarize(shards, out)
    print(f"CLMM_MATCHED_GAP_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
