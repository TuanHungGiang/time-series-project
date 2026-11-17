from __future__ import annotations

"""CLMM V19 screening on Z24 under severe sensor loss.

CLMM extends the frozen source-USAD control with two deliberately isolated
components:

1. a cross-sensor adapter trained only on healthy windows with a worst-case
   50% mask curriculum and full-vs-masked latent/score consistency; and
2. mask-conditional split-conformal calibration fitted only on healthy
   validation records.

The detector branch (imputed USAD) and score orientation (lower tail) are
fixed before evaluation.  Damage validation labels and test labels are never
used for branch, tail, or threshold selection.  A clean-frozen calibrator is
reported as a stress test, not used to select the primary protocol.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_care_usad_v18 as care  # noqa: E402
import run_z24_missingness_baselines as missing  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax,
    Normalizer,
    fit_usad,
    score_usad,
    seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
)


CONDITIONS = care.CONDITIONS
SCENARIOS = care.SCENARIOS
PRIMARY_PROTOCOL = "mask_conditional_conformal"
STRESS_PROTOCOL = "clean_frozen_conformal"


def usad_tensor(model: nn.Module, values: torch.Tensor):
    flat = values.flatten(1)
    latent = model.encoder(flat)
    first = model.decoder1(latent)
    cycle = model.decoder2(model.encoder(first))
    score = (
        0.5 * (flat - first).pow(2)
        + 0.5 * (flat - cycle).pow(2)
    ).mean(1)
    return latent, score


def clmm_loss_per_sample(
    adapter: nn.Module,
    source: nn.Module,
    clean: torch.Tensor,
    available: torch.Tensor,
    score_scale: float,
    latent_weight: float,
    score_weight: float,
):
    masked = clean * available.unsqueeze(-1)
    reconstruction = adapter(masked, available)
    missing_mask = 1.0 - available
    completed = (
        available.unsqueeze(-1) * clean
        + missing_mask.unsqueeze(-1) * reconstruction
    )

    missing_denominator = (
        missing_mask.sum(1) * clean.shape[2]
    ).clamp_min(1.0)
    missing_error = (
        (reconstruction - clean).pow(2) * missing_mask.unsqueeze(-1)
    ).sum((1, 2)) / missing_denominator

    observed_denominator = (
        available.sum(1) * clean.shape[2]
    ).clamp_min(1.0)
    observed_error = (
        (reconstruction - clean).pow(2) * available.unsqueeze(-1)
    ).sum((1, 2)) / observed_denominator

    with torch.no_grad():
        full_latent, full_score = usad_tensor(source, clean)
    masked_latent, masked_score = usad_tensor(source, completed)

    latent_error = 1.0 - F.cosine_similarity(
        masked_latent, full_latent.detach(), dim=1, eps=1e-8,
    )
    score_gap = (
        torch.log1p(masked_score)
        - torch.log1p(full_score.detach())
    ) / max(float(score_scale), 1e-8)
    score_error = F.smooth_l1_loss(
        score_gap, torch.zeros_like(score_gap), reduction="none",
    )

    total = (
        missing_error
        + 0.10 * observed_error
        + latent_weight * latent_error
        + score_weight * score_error
    )
    parts = {
        "reconstruction": missing_error + 0.10 * observed_error,
        "latent": latent_error,
        "score": score_error,
    }
    return total, parts


def curriculum_masks(batch: int, channels: int, importance: np.ndarray, rng, index: int):
    # Always include pure random and deterministic criticality masks.  The
    # remaining candidates mix both regimes, after which the current model
    # selects the candidate causing the largest healthy consistency loss.
    if index == 0:
        probability = 0.0
    elif index == 1:
        probability = 1.0
    else:
        probability = 0.5
    return care.masks_50(
        batch, channels, importance, rng, targeted_probability=probability,
    )


def hardest_mask(
    adapter,
    source,
    clean,
    importance,
    rng,
    candidates,
    score_scale,
    latent_weight,
    score_weight,
):
    options, losses = [], []
    with torch.no_grad():
        for index in range(max(2, candidates)):
            current = torch.as_tensor(
                curriculum_masks(
                    len(clean), clean.shape[1], importance, rng, index,
                ),
                device=clean.device,
            )
            options.append(current)
            loss, _ = clmm_loss_per_sample(
                adapter, source, clean, current, score_scale,
                latent_weight, score_weight,
            )
            losses.append(loss)
    selected = torch.stack(losses).argmax(0)
    stacked = torch.stack(options)
    indices = torch.arange(len(clean), device=clean.device)
    return stacked[selected, indices]


def fit_adapter(
    fit_scaled,
    importance,
    source,
    score_reference,
    epochs,
    batch_size,
    embedding,
    heads,
    candidates,
    latent_weight,
    score_weight,
    device,
    seed,
):
    _, channels, length = fit_scaled.shape
    adapter = care.CrossSensorAdapter(
        channels, length, embedding, heads,
    ).to(device)
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_scaled)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=5e-4, weight_decay=1e-5,
    )
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_state = None
    score_scale = float(score_reference[1])

    for epoch in range(1, epochs + 1):
        adapter.train()
        epoch_rows = []
        for (clean,) in loader:
            clean = clean.to(device)
            available = hardest_mask(
                adapter, source, clean, importance, rng, candidates,
                score_scale, latent_weight, score_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses, parts = clmm_loss_per_sample(
                adapter, source, clean, available, score_scale,
                latent_weight, score_weight,
            )
            loss = losses.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            epoch_rows.append([
                float(loss.detach().cpu()),
                float(parts["reconstruction"].mean().detach().cpu()),
                float(parts["latent"].mean().detach().cpu()),
                float(parts["score"].mean().detach().cpu()),
            ])
        means = np.mean(epoch_rows, axis=0)
        if means[0] < best:
            best = float(means[0])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  CLMM adapter epoch={epoch}/{epochs} "
                f"total={means[0]:.6f} recon={means[1]:.6f} "
                f"latent={means[2]:.6f} score={means[3]:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("CLMM adapter did not produce a checkpoint")
    adapter.load_state_dict(best_state)
    return adapter.eval()


def run_group(args, group, labels, group_index, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit)
    internal_scaled = scale.transform(internal)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed = args.seed + group_index * 1009
    seed_everything(seed)
    source = fit_usad(
        fit_scaled, args.epochs, args.batch_size, device,
    )
    source_reference = robust_score_reference(
        score_usad(source, internal_scaled, args.batch_size, device),
    )
    seed_everything(seed + 97)
    adapter = fit_adapter(
        fit_scaled,
        importance,
        source,
        source_reference,
        args.adapter_epochs,
        args.batch_size,
        args.embedding,
        args.heads,
        args.hard_candidates,
        args.latent_weight,
        args.score_weight,
        device,
        seed + 97,
    )
    conditional_usad, conditional_consistency = care.conditional_reference(
        adapter,
        source,
        internal,
        scale,
        importance,
        args.batch_size,
        device,
        seed + 193,
    )
    source_parameters = care.count_parameters(source)
    proposed_parameters = source_parameters + care.count_parameters(adapter)

    frames = []
    for records in (validation, test):
        for condition, kind, rate, replicate in SCENARIOS:
            values, available, counts, rows = care.build_cohort(
                records,
                cache,
                condition,
                kind,
                rate,
                replicate,
                importance,
            )
            scaled_values = scale.transform(values)
            baseline_scores = score_usad(
                source, scaled_values, args.batch_size, device,
            )
            frames.append(care.source_frame(
                rows,
                baseline_scores,
                counts,
                source_reference,
                source_parameters,
            ))
            completed, consistency = care.complete(
                adapter,
                scaled_values,
                available,
                args.batch_size,
                device,
            )
            imputed_scores = score_usad(
                source, completed, args.batch_size, device,
            )
            reference_u = (
                source_reference if condition == "clean"
                else conditional_usad
            )
            reference_c = (
                robust_score_reference(np.zeros(max(2, len(consistency))))
                if condition == "clean" else conditional_consistency
            )
            frames.extend(care.proposed_frames(
                rows,
                imputed_scores,
                consistency,
                counts,
                reference_u,
                reference_c,
                proposed_parameters,
                args.fusion_weight,
            ))

    result = pd.concat(frames, ignore_index=True)
    result["policy_parameters"] = proposed_parameters
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del source, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def conformal_pvalues(calibration_scores, scores):
    calibration = np.asarray(calibration_scores, dtype=float)
    values = np.asarray(scores, dtype=float)
    if len(calibration) < 2:
        raise RuntimeError(
            f"Need at least two healthy calibration records, got {len(calibration)}"
        )
    return (
        1.0
        + (calibration[None, :] >= values[:, None]).sum(axis=1)
    ) / (len(calibration) + 1.0)


def metrics_from_predictions(labels, scores, prediction):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = np.asarray(prediction, dtype=bool)
    healthy = labels == 0
    damage = labels == 1
    recall = float(prediction[damage].mean())
    specificity = float((~prediction[healthy]).mean())
    fpr = 1.0 - specificity
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "ba": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def method_for(output_method: str, condition: str):
    if output_method == "source_usad" or condition == "clean":
        return "source_usad"
    return "imputed_usad"


def calibration_scores(validation, output_method, condition, protocol):
    calibration_condition = (
        "clean" if protocol == STRESS_PROTOCOL else condition
    )
    source_method = method_for(
        output_method, calibration_condition,
    )
    current = validation[
        (validation.label == 0)
        & (validation.condition == calibration_condition)
        & (validation.method == source_method)
    ]
    values = current.score_lower.to_numpy(dtype=float)
    if len(values) < 2:
        raise RuntimeError(
            f"Insufficient healthy calibration: protocol={protocol} "
            f"method={output_method} condition={condition} n={len(values)}"
        )
    return values, source_method


def evaluate_protocol(frame, protocol: str, alpha: float):
    validation = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    metric_rows, score_rows, calibration_rows = [], [], []

    for condition in CONDITIONS:
        for output_method in ("source_usad", "clmm_v19"):
            selected_method = method_for(output_method, condition)
            calibration, calibration_method = calibration_scores(
                validation, output_method, condition, protocol,
            )
            calibration_rows.append({
                "protocol": protocol,
                "output_method": output_method,
                "condition": condition,
                "selected_method": selected_method,
                "calibration_method": calibration_method,
                "score_tail": "lower",
                "alpha": alpha,
                "healthy_calibration_scores": len(calibration),
                "calibration_score_min": float(np.min(calibration)),
                "calibration_score_median": float(np.median(calibration)),
                "calibration_score_max": float(np.max(calibration)),
            })
            selected = test[
                (test.condition == condition)
                & (test.method == selected_method)
            ].copy()
            for replicate, cohort in selected.groupby("replicate"):
                pvalue = conformal_pvalues(
                    calibration, cohort.score_lower.to_numpy(),
                )
                prediction = pvalue <= alpha
                metric = metrics_from_predictions(
                    cohort.label,
                    cohort.score_lower,
                    prediction,
                )
                metric_rows.append({
                    "protocol": protocol,
                    "method": output_method,
                    "selected_method": selected_method,
                    "condition": condition,
                    "replicate": int(replicate),
                    "score_tail": "lower",
                    "alpha": alpha,
                    "healthy_calibration_scores": len(calibration),
                    **metric,
                    "records": len(cohort),
                    "healthy_records": int((cohort.label == 0).sum()),
                    "damage_records": int((cohort.label == 1).sum()),
                    "coverage_mean": float(cohort.coverage.mean()),
                    "parameters": int(
                        cohort.policy_parameters.iloc[0]
                        if output_method == "clmm_v19"
                        else cohort.parameters.iloc[0]
                    ),
                })
                scored = cohort.copy()
                scored["protocol"] = protocol
                scored["output_method"] = output_method
                scored["score"] = scored.score_lower
                scored["conformal_pvalue"] = pvalue
                scored["prediction"] = prediction.astype(int)
                score_rows.append(scored)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(calibration_rows),
    )


def summarize(metrics):
    return metrics.groupby(
        ["protocol", "method", "condition"], as_index=False,
    ).agg(
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"),
        replicates=("replicate", "count"),
        parameters=("parameters", "mean"),
    )


def paired_effects(metrics):
    keys = ["protocol", "condition", "replicate"]
    wide = metrics.pivot_table(
        index=keys,
        columns="method",
        values=["auc", "ba", "recall", "fpr", "f1"],
    )
    rows = []
    for index, values in wide.iterrows():
        row = dict(zip(keys, index))
        for metric in ("auc", "ba", "recall", "fpr", "f1"):
            row[f"delta_{metric}"] = float(
                values[(metric, "clmm_v19")]
                - values[(metric, "source_usad")]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def finish(args, frame):
    all_metrics, all_scores, all_calibration = [], [], []
    for protocol in (PRIMARY_PROTOCOL, STRESS_PROTOCOL):
        metrics, scores, calibration = evaluate_protocol(
            frame, protocol, args.conformal_alpha,
        )
        all_metrics.append(metrics)
        all_scores.append(scores)
        all_calibration.append(calibration)

    metrics = pd.concat(all_metrics, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    summary = summarize(metrics)
    effects = paired_effects(metrics)
    effect_summary = effects.groupby(
        ["protocol", "condition"], as_index=False,
    ).agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_median=("delta_auc", "median"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"),
        positive_auc_pairs=("delta_auc", lambda x: int((x > 0).sum())),
        positive_ba_pairs=("delta_ba", lambda x: int((x > 0).sum())),
        pairs=("delta_auc", "count"),
    )

    primary = effect_summary[
        (effect_summary.protocol == PRIMARY_PROTOCOL)
        & (effect_summary.condition == "sensor_0.50")
    ].iloc[0]
    clean = effect_summary[
        (effect_summary.protocol == PRIMARY_PROTOCOL)
        & (effect_summary.condition == "clean")
    ].iloc[0]
    decision = {
        "promote_to_three_seed": bool(
            primary.delta_auc_mean > 0.0
            and primary.delta_ba_mean > 0.0
            and primary.delta_recall_mean > 0.0
            and primary.positive_auc_pairs >= 3
            and abs(clean.delta_auc_mean) < 1e-12
            and abs(clean.delta_ba_mean) < 1e-12
        ),
        "primary_protocol": PRIMARY_PROTOCOL,
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "calibration_labels": "healthy-only",
        "damage_validation_used_for_selection": False,
        "test_used_for_selection": False,
        "screening_note": "One seed is diagnostic, not publication evidence.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    calibration.to_csv(
        args.output_dir / "healthy_calibration_audit.csv", index=False,
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    effect_summary.to_csv(
        args.output_dir / "effect_summary.csv", index=False,
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    primary_summary = summary[summary.protocol == PRIMARY_PROTOCOL]
    pivot = primary_summary.pivot(
        index="condition", columns="method", values="ba_mean",
    )
    axis = pivot.plot.bar(
        figsize=(10, 5), ylim=(0, 1.02), rot=15, grid=True,
    )
    axis.set_ylabel("Record-level balanced accuracy")
    axis.set_title("CLMM V19 — healthy-only mask-conditional calibration")
    axis.figure.tight_layout()
    axis.figure.savefig(
        args.output_dir / "clmm_v19_screening.png", dpi=220,
    )
    plt.close(axis.figure)

    print("\n=== CLMM V19 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== CLMM V19 PAIRED EFFECTS ===")
    print(effects.to_string(index=False))
    print("\n=== CLMM V19 EFFECT SUMMARY ===")
    print(effect_summary.to_string(index=False))
    print("\n=== HEALTHY-ONLY CALIBRATION AUDIT ===")
    print(calibration.to_string(index=False))
    print("\n=== SCREENING DECISION ===")
    print(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270701)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=4)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--latent-weight", type=float, default=0.20)
    parser.add_argument("--score-weight", type=float, default=0.10)
    parser.add_argument("--conformal-alpha", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.0 < args.conformal_alpha < 1.0:
        raise ValueError("conformal-alpha must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} seed={args.seed} "
        f"source_epochs={args.epochs} adapter_epochs={args.adapter_epochs} "
        f"alpha={args.conformal_alpha}",
        flush=True,
    )

    for index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = care.group_file(
            args.output_dir, key[0], key[1],
        )
        if path.exists():
            print(f"SKIP cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[
            (used.setup == key[0])
            & (used.excitation == key[1])
        ]
        run_group(args, group, labels, index, device, path)

    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(
            f"Expected 18 group files, got {len(paths)}"
        )
    frame = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True,
    )
    decision = finish(args, frame)
    protocol = {
        "dataset": "Z24",
        "method": "CLMM V19",
        "full_name": "Counteracting Loss from Missing Measurements",
        "baseline": "vendored official EURECOM USAD",
        "conditions": list(CONDITIONS),
        "adapter_fit": (
            "healthy-only random/critical/worst-candidate 50% mask curriculum"
        ),
        "consistency": "full-vs-masked USAD latent and anomaly score",
        "source_usad_frozen_during_adapter_fit": True,
        "primary_branch": "imputed_usad fixed before evaluation",
        "score_tail": "lower fixed before evaluation",
        "calibration": (
            "split-conformal empirical p-values from healthy validation only"
        ),
        "conformal_alpha": args.conformal_alpha,
        "clean_branch": "forced untouched source USAD",
        "damage_validation_used_for_selection": False,
        "test_used_for_selection": False,
        "seed": args.seed,
        "decision": decision,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"CLMM_V19_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

