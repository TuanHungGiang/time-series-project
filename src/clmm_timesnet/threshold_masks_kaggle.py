"""Audit two healthy Q90 threshold policies on targeted and random sensor masks."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


MASK_REPLICATES = 5
WATCH_RECORD = "z24_s08_fvt_setup01"
GROUP_KEYS = ["design", "mask_id", "requested_loss"]
SCORING_HEADS = ("pooled_mse", "per_sensor")


def mask_plan(order, random_orders, steps):
    yield "clean", 0, 0, ()
    for design, mask_id, permutation in [("targeted", 0, order), *(
        ("random", index, permutation) for index, permutation in enumerate(random_orders, 1)
    )]:
        for rate, step in steps.items():
            if rate:
                yield design, mask_id, rate, tuple(permutation[:min(step, len(permutation) - 1)])


@torch.inference_mode()
def adapted_reconstruction(head, data, kept, *, include_channels=False, enabled=True):
    device = head.detector.device
    mask = torch.as_tensor(kept, device=device)
    data = torch.as_tensor(data, device=device)
    errors, channel_errors = [], []
    for chunk in data.split(head.detector.batch_size):
        residual, _ = head.residual(chunk, mask, enabled=enabled)
        errors.append(residual.mean((1, 2)).cpu().numpy().astype(np.float64))
        if include_channels:
            channel_errors.append(residual.mean(-1).cpu().numpy().astype(np.float64))
    pooled = np.concatenate(errors)
    return (pooled, np.concatenate(channel_errors)) if include_channels else pooled


def normalized_channel_score(channel_mse, reference_mse):
    """Fit and apply a separate masked healthy reference for each observed sensor."""
    reference = np.log1p(reference_mse)
    center = np.median(reference, axis=0)
    mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
    scale = np.maximum(np.maximum(mad, .25 * reference.std(axis=0)), 1e-6)
    z = (np.log1p(channel_mse) - center) / scale
    return z.mean(axis=1), center, scale


def score_group(dataset, group, index, order, steps, args, functions, device, out):
    from clmm_timesnet.conditional_gain_kaggle import ConditionalGain, TAILS, development_signals, head_fit_split
    from clmm_timesnet.data.masking import stable_rng
    from clmm_timesnet.old_gain_diagnostics import aggregate_windows
    from clmm_timesnet.targeted_baselines import atomic_csv
    from clmm_timesnet.targeted_gain_head import fit_or_restore_head, score_columns

    path = out / "groups" / f"{group.name}.csv.gz"
    channel_path = out / "groups" / f"{group.name}_sensor_reference.csv"
    if path.is_file() and (not args.per_sensor or channel_path.is_file()):
        print(f"{dataset}/{group.name} scores restored", flush=True)
        return pd.read_csv(path, float_precision="round_trip")
    weights = args.root / "weights" / dataset / str(args.seed)
    weights.mkdir(parents=True, exist_ok=True)
    names = (f"{group.name}.pt", f"{group.name}_adapter.pt")
    args.previous_clmm_weights = next((p for p in args.priors if all(
        (p / dataset / str(args.seed) / name).is_file() for name in names
    )), None)
    scorer, head, norm = fit_or_restore_head(dataset, group, index, args.seed, args, functions, device, weights)
    meta = group.metadata.drop(columns="path", errors="ignore").copy()
    n_val = int(meta.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    reference, threshold = head_fit_split(dataset, group, n_val)
    chosen = np.r_[threshold, np.zeros(len(meta) - n_val, dtype=bool)]
    meta["threshold_healthy"] = chosen & meta.label.eq(0).to_numpy()
    val, test = group.values[:offset], group.values[offset:]
    channels = len(group.sensors)
    random_orders = [stable_rng(f"{dataset}/{group.name}", f"threshold-q90-{args.seed}", mask_id).permutation(channels)
                     for mask_id in range(1, MASK_REPLICATES + 1)]
    rows, channel_references = [], []
    for design, mask_id, rate, removed in mask_plan(order, random_orders, steps):
        kept = np.ones(channels, dtype=bool)
        kept[list(removed)] = False

        def forward(values):
            if args.per_sensor:
                return adapted_reconstruction(head, values, kept, include_channels=True, enabled=bool(removed))
            return (adapted_reconstruction(head, values, kept) if removed else scorer(values),)

        ref_signals, val_signals = development_signals(forward, val, reference, threshold)
        test_signals = forward(test)
        ref_raw, val_raw, test_raw = ref_signals[0], val_signals[0], test_signals[0]
        raw = np.r_[val_raw, test_raw]
        fitted = ConditionalGain(ref_raw)
        base, _, _, _ = fitted.scores(raw)
        by_tail = {f"score_{name}": aggregate_windows(x, group.counts)
                   for name, x in zip(TAILS, (base, -base, np.abs(base)), strict=True)}
        if args.per_sensor:
            channel_mse = np.concatenate((val_signals[1], test_signals[1]))
            channel_score, center, scale = normalized_channel_score(channel_mse, ref_signals[1])
            by_tail.update({f"channel_score_{name}": aggregate_windows(x, group.counts)
                            for name, x in zip(TAILS, (channel_score, -channel_score, np.abs(channel_score)), strict=True)})
            for sensor, loc, spread in zip(np.asarray(group.sensors)[kept], center, scale, strict=True):
                channel_references.append(dict(group=group.name, design=design, mask_id=mask_id,
                                               requested_loss=rate, sensor=sensor, log_center=loc, log_scale=spread))
        native = {}
        if design == "clean":
            clean_tail = score_columns(dataset, val_raw, group.counts[:n_val], norm, functions[1])
            native = {key: np.r_[x, np.full(len(meta) - n_val, np.nan)] for key, x in clean_tail.items()}
        rows.append(meta.assign(dataset=dataset, group=group.name, design=design, mask_id=mask_id,
                                requested_loss=rate, removed_count=len(removed), total_channels=channels,
                                removed_sensors=" ".join(np.asarray(group.sensors)[~kept]),
                                reference_windows=len(ref_raw), ref_log_center=fitted.rec_center,
                                ref_log_scale=fitted.rec_scale,
                                raw_mse_median=aggregate_windows(raw, group.counts),
                                log_mse_median=aggregate_windows(np.log1p(raw), group.counts),
                                **by_tail, **native))
    table = pd.concat(rows, ignore_index=True)
    if args.per_sensor:
        atomic_csv(pd.DataFrame(channel_references), channel_path)
    atomic_csv(table, path)
    print(f"{dataset}/{group.name} scored {len(rows)} mask conditions", flush=True)
    return table


def summarize(dataset, scores, out, select_threshold):
    from clmm_timesnet.conditional_gain_kaggle import HEALTHY_QUANTILE
    from clmm_timesnet.metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
    from clmm_timesnet.targeted_baselines import atomic_csv, clean_calibration

    validation = scores.loc[scores.split.eq("val")]
    n_calibration = 6 if dataset == "Z24" else 8
    expected_labels = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    tail, _ = clean_calibration(dataset, validation.loc[validation.design.eq("clean")], select_threshold)
    score_column = f"score_{tail}"
    healthy = validation.loc[validation.threshold_healthy].copy()
    test = scores.loc[scores.split.eq("test")]
    clean_healthy = healthy.loc[healthy.design.eq("clean")]
    clean_test = test.loc[test.design.eq("clean")]
    clean_labels = clean_test.set_index("sample_id").label.sort_index()
    clean_threshold = threshold_from_healthy(clean_healthy[score_column], HEALTHY_QUANTILE)
    rows, thresholds = [], []
    for (design, mask_id, rate), part in test.groupby(GROUP_KEYS):
        calibration = healthy.loc[healthy.design.eq(design) & healthy.mask_id.eq(mask_id)
                                  & healthy.requested_loss.eq(rate)]
        if (len(calibration) != n_calibration or not calibration.sample_id.is_unique
                or set(calibration.sample_id) != set(clean_healthy.sample_id)
                or part.label.value_counts().to_dict() != expected_labels or not part.sample_id.is_unique
                or not part.set_index("sample_id").label.sort_index().equals(clean_labels)):
            raise ValueError(f"Cohort mismatch: {dataset}/{design}/{mask_id}/{rate}")
        labels, values = part.label.to_numpy(), part[score_column].to_numpy()
        if not np.isfinite(values).all() or not np.isfinite(calibration[score_column]).all():
            raise ValueError(f"Non-finite reconstruction score: {dataset}/{design}/{mask_id}/{rate}")
        threshold = threshold_from_healthy(calibration[score_column], HEALTHY_QUANTILE)
        key = dict(dataset=dataset, design=design, mask_id=mask_id, requested_loss=rate,
                   actual_loss=100 * part.removed_count.sum() / part.total_channels.sum(), tail=tail)
        for policy, cutoff in (("clean_q90", clean_threshold), ("mask_q90", threshold)):
            metrics = publication_metrics(labels, values, cutoff)
            rows.append(dict(**key, policy=policy, threshold=cutoff, n_healthy=int((labels == 0).sum()),
                             n_damage=int((labels == 1).sum()), recall_at_fpr_05=recall_at_fpr(labels, values, .05),
                             recall_at_fpr_10=recall_at_fpr(labels, values, .10), **metrics))
        strongest = calibration.nlargest(2, score_column)
        thresholds.append(dict(**key, top_healthy=strongest.sample_id.iloc[0],
                               highest_healthy_score=strongest[score_column].iloc[0],
                               second_healthy_score=strongest[score_column].iloc[1],
                               top_gap=strongest[score_column].iloc[0] - strongest[score_column].iloc[1]))

    report = pd.DataFrame(rows)
    if len(report) != 2 * (1 + 2 * (1 + MASK_REPLICATES)):
        raise ValueError(f"Unexpected condition count for {dataset}")
    q90_audit = pd.DataFrame(thresholds)
    development = healthy.assign(score=healthy[score_column]).merge(q90_audit[GROUP_KEYS + ["top_healthy", "top_gap"]],
                                                                     on=GROUP_KEYS, validate="many_to_one")
    ordered = development.sort_values([*GROUP_KEYS, "score"], ascending=[True, True, True, False])
    ordered["rank"] = ordered.groupby(GROUP_KEYS).cumcount() + 1
    atomic_csv(ordered, out / "healthy_validation_scores.csv")
    atomic_csv(q90_audit, out / "threshold_audit.csv")
    atomic_csv(scores[[*GROUP_KEYS, "group", "removed_count", "total_channels", "removed_sensors"]].drop_duplicates(),
               out / "mask_manifest.csv")
    atomic_csv(report, out / "metrics_by_mask.csv")
    if dataset == "Z24":
        watch = ordered.loc[ordered.sample_id.eq(WATCH_RECORD)].copy()
        if len(watch) != len(q90_audit):
            raise ValueError(f"Missing tracked validation record {WATCH_RECORD}")
        atomic_csv(watch, out / "z24_01_fvt_watch.csv")
    return report


def compare_scoring_heads(dataset, scores, out, select_threshold):
    from clmm_timesnet.conditional_gain_kaggle import TAILS
    from clmm_timesnet.targeted_baselines import atomic_csv

    pooled_path, channel_path = (out / name for name in SCORING_HEADS)
    pooled_path.mkdir(exist_ok=True)
    channel_path.mkdir(exist_ok=True)
    pooled = summarize(dataset, scores, pooled_path, select_threshold).assign(head=SCORING_HEADS[0])
    aliases = {f"channel_score_{tail}": f"score_{tail}" for tail in TAILS}
    channel_scores = scores.drop(columns=list(aliases.values())).rename(columns=aliases)
    channel = summarize(dataset, channel_scores, channel_path, select_threshold).assign(head=SCORING_HEADS[1])
    report = pd.concat((pooled, channel), ignore_index=True)
    atomic_csv(report, out / "metrics_by_mask.csv")
    if dataset == "Z24":
        watches = []
        for folder in (pooled_path, channel_path):
            watch = pd.read_csv(folder / "z24_01_fvt_watch.csv", float_precision="round_trip")
            policy = report.loc[report["head"].eq(folder.name) & report.policy.eq("mask_q90")]
            watch = watch.merge(policy[[*GROUP_KEYS, "threshold"]], on=GROUP_KEYS, validate="one_to_one")
            watch["margin_to_mask_q90"] = watch.score - watch.threshold
            watches.append(watch.assign(head=folder.name))
        atomic_csv(pd.concat(watches, ignore_index=True), out / "z24_01_fvt_watch.csv")
    return report


def main():
    from clmm_timesnet.laplacian_kaggle import BASELINE_COMMIT, DATASETS, HELL, RATES, SEED, WORK, Z24, checkout
    from clmm_timesnet.old_gain_diagnostics import loss_steps
    from clmm_timesnet.targeted_baselines import archive_functions, atomic_csv, load_hellbridge, load_z24
    from clmm_timesnet.targeted_gain_head import GAIN_WEIGHT, STRENGTH, sensor_orders, targeted_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sensor", action="store_true", help="Compare normalize-before-pooling to the pooled MSE control")
    per_sensor = parser.parse_args().per_sensor
    for path in (Z24 / "metadata/records_manifest.csv", HELL):
        if not path.is_file():
            raise FileNotFoundError(f"Attach the Kaggle dataset: {path}")
    os.chdir(WORK)
    experiment = "clmm_per_sensor_score_v1" if per_sensor else "clmm_threshold_masks_v1"
    root = WORK / f"{experiment}_{SEED}"
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    prior_names = ["clmm_conditional_gain_v1", "clmm_laplacian_v1", "clmm_old_gain_diagnostics"]
    if per_sensor:
        prior_names.insert(0, "clmm_threshold_masks_v1")
    prior_runs = [WORK / f"{name}_{SEED}" for name in prior_names]
    archive = next((p / "source_baselines" for p in prior_runs if (p / "source_baselines/.git").is_dir()),
                   root / "source_baselines")
    checkout(archive, "codex/mdr-shm-kaggle", BASELINE_COMMIT)
    prior_weights = [p / "weights" for p in prior_runs]
    args = SimpleNamespace(root=root, priors=prior_weights, seed=SEED, epochs=30, adapter_epochs=8,
                           batch_size=16, smoke_window_limit=0, strength=STRENGTH, gain_weight=GAIN_WEIGHT,
                           center_gain=False, baseline_runs=None, previous_clmm_weights=None, per_sensor=per_sensor)
    args.protocol = targeted_protocol(args)
    protocol = dict(seed=SEED, datasets=list(DATASETS), rates=list(RATES), random_mask_replicates=MASK_REPLICATES,
                    baseline_commit=BASELINE_COMMIT, z24=str(Z24), hellbridge=str(HELL),
                    validation="same split and score normalization as conditional gain experiment",
                    targeted="healthy PCA-sensitivity order", random="nested group-specific permutations with matched counts",
                    policies="Q90 clean frozen and exact-mask Q90, both on same mask-normalized reconstruction score",
                    checkpoint_protocol=args.protocol)
    if per_sensor:
        protocol["scoring_heads"] = list(SCORING_HEADS)
        protocol["sensor_scaling"] = "log1p MSE, healthy exact-mask sensor median and robust scale, observed-sensor mean"
    path = out / "protocol.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Protocol changed. Use a fresh output path: {out}")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | seed={SEED} | random masks={MASK_REPLICATES}", flush=True)
    functions = archive_functions(archive / "research/mdr_shm_kaggle")
    reports = []
    for dataset in DATASETS:
        dataset_out = out / dataset
        (dataset_out / "groups").mkdir(parents=True, exist_ok=True)
        groups = list(load_z24(Z24, 0)) if dataset == "Z24" else [load_hellbridge(HELL, dataset_out, 0)]
        filename = "sensor_removal_order.csv"
        args.previous_clmm_weights = next((p for p in [root / "weights", *prior_weights]
                                          if (p.parent / "reports" / dataset / filename).is_file()), None)
        orders = sensor_orders(dataset, groups, args, dataset_out, functions[0])
        steps = {rate: step for rate, (step, _) in loss_steps(groups).items() if rate in RATES}
        scores = pd.concat([score_group(dataset, g, i, orders[g.name], steps, args, functions, device, dataset_out)
                            for i, g in enumerate(groups)], ignore_index=True)
        scoring_summary = compare_scoring_heads if per_sensor else summarize
        reports.append(scoring_summary(dataset, scores, dataset_out, functions[3]))
        del groups, scores
    metrics = pd.concat(reports, ignore_index=True)
    keys = ["dataset", "requested_loss", "policy"]
    measures = ["auc", "recall", "fpr", "ba", "recall_at_fpr_05", "recall_at_fpr_10"]
    if per_sensor:
        keys.append("head")
        index = ["dataset", *GROUP_KEYS, "policy"]
        heads = metrics.set_index([*index, "head"])
        paired = heads.xs(SCORING_HEADS[1], level="head")[measures] - heads.xs(SCORING_HEADS[0], level="head")[measures]
        atomic_csv(paired.add_prefix("delta_").reset_index(), out / "paired_deltas.csv")
    random = metrics.loc[metrics.design.eq("random")].groupby(keys)[measures].agg(["mean", "std"])
    random.columns = [f"{metric}_{stat}" for metric, stat in random.columns]
    atomic_csv(metrics, out / "metrics_by_mask.csv")
    atomic_csv(random.reset_index(), out / "random_mask_summary.csv")
    bundle_name = "clmm_per_sensor_score_reports" if per_sensor else "clmm_threshold_masks_reports"
    bundle = shutil.make_archive(str(root / bundle_name), "zip", root_dir=out)
    marker = "CLMM_PER_SENSOR_SCORE_PASS" if per_sensor else "CLMM_THRESHOLD_MASKS_PASS"
    print(f"{marker} reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
