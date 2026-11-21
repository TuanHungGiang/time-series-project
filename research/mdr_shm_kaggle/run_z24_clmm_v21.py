from __future__ import annotations

"""CLMM V21: one-class training with validation-calibrated thresholds.

The scientific target is robustness to random 50% sensor loss.  USAD and the
CLMM adapter are fitted using healthy training records only.  Detector branch
(imputed USAD) and score orientation (lower tail) are fixed in advance.  The
labelled validation split is used only to calibrate one operating threshold
per method/condition under a predeclared FPR cap.  Test labels are evaluation
only.  Targeted sensor loss is retained as a stress test and is not a
promotion criterion.

V21 removes the USAD latent-cosine term from V19/V20 because ReLU collapse
made that term either approximately zero or a constant one in several cells.
The adapter objective contains only masked reconstruction and frozen-USAD
score consistency.
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_care_usad_v18 as care  # noqa: E402
import run_z24_clmm_v19 as v19  # noqa: E402
from run_hellbridge_source_baselines import publication_metrics  # noqa: E402
from run_z24_mdr_v16 import threshold_under_fpr_cap  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    load_manifest,
    stable_sensor_map,
)


CONDITIONS = care.CONDITIONS


def reconstruction_and_score_loss(
    adapter,
    source,
    clean,
    available,
    score_scale,
    score_weight,
):
    masked = clean * available.unsqueeze(-1)
    reconstruction = adapter(masked, available)
    missing = 1.0 - available
    completed = (
        available.unsqueeze(-1) * clean
        + missing.unsqueeze(-1) * reconstruction
    )

    missing_denominator = (
        missing.sum(1) * clean.shape[2]
    ).clamp_min(1.0)
    missing_error = (
        (reconstruction - clean).pow(2) * missing.unsqueeze(-1)
    ).sum((1, 2)) / missing_denominator

    observed_denominator = (
        available.sum(1) * clean.shape[2]
    ).clamp_min(1.0)
    observed_error = (
        (reconstruction - clean).pow(2) * available.unsqueeze(-1)
    ).sum((1, 2)) / observed_denominator
    reconstruction_error = missing_error + 0.10 * observed_error

    with torch.no_grad():
        _, full_score = v19.usad_tensor(source, clean)
    _, completed_score = v19.usad_tensor(source, completed)
    score_gap = (
        torch.log1p(completed_score)
        - torch.log1p(full_score.detach())
    ) / max(float(score_scale), 1e-8)
    score_error = F.smooth_l1_loss(
        score_gap,
        torch.zeros_like(score_gap),
        reduction="none",
    )
    total = reconstruction_error + score_weight * score_error
    return total, reconstruction_error, score_error


def hardest_mask(
    adapter,
    source,
    clean,
    importance,
    rng,
    candidates,
    score_scale,
    score_weight,
):
    options, losses = [], []
    with torch.no_grad():
        for index in range(max(2, candidates)):
            current = torch.as_tensor(
                v19.curriculum_masks(
                    len(clean), clean.shape[1], importance, rng, index,
                ),
                device=clean.device,
            )
            options.append(current)
            loss, _, _ = reconstruction_and_score_loss(
                adapter,
                source,
                clean,
                current,
                score_scale,
                score_weight,
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
    del latent_weight
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
    score_scale = max(float(score_reference[1]), 0.02)
    best = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        adapter.train()
        rows = []
        for (clean,) in loader:
            clean = clean.to(device)
            available = hardest_mask(
                adapter,
                source,
                clean,
                importance,
                rng,
                candidates,
                score_scale,
                score_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses, reconstruction, score = reconstruction_and_score_loss(
                adapter,
                source,
                clean,
                available,
                score_scale,
                score_weight,
            )
            loss = losses.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            rows.append([
                float(loss.detach().cpu()),
                float(reconstruction.mean().detach().cpu()),
                float(score.mean().detach().cpu()),
            ])

        means = np.mean(rows, axis=0)
        if means[0] < best:
            best = float(means[0])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  CLMM-V21 adapter epoch={epoch}/{epochs} "
                f"total={means[0]:.6f} recon={means[1]:.6f} "
                f"score={means[2]:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("CLMM-V21 adapter did not produce a checkpoint")
    adapter.load_state_dict(best_state)
    return adapter.eval()


def selected_method(output_method: str, condition: str):
    if output_method == "source_usad" or condition == "clean":
        return "source_usad"
    return "imputed_usad"


def evaluate(frame, args):
    validation = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    policy_rows, metric_rows, score_rows = [], [], []

    for condition in CONDITIONS:
        for output_method in ("source_usad", "clmm_v21"):
            method = selected_method(output_method, condition)
            calibration = validation[
                (validation.condition == condition)
                & (validation.method == method)
            ].copy()
            threshold, validation_metric = threshold_under_fpr_cap(
                calibration.label.to_numpy(),
                calibration.score_lower.to_numpy(),
                args.fpr_cap,
            )
            policy_rows.append({
                "output_method": output_method,
                "condition": condition,
                "selected_method": method,
                "score_tail": "lower",
                "threshold": threshold,
                "fpr_cap": args.fpr_cap,
                "validation_records": len(calibration),
                "validation_healthy_records": int(
                    (calibration.label == 0).sum()
                ),
                "validation_damage_records": int(
                    (calibration.label == 1).sum()
                ),
                **{
                    f"validation_{key}": value
                    for key, value in validation_metric.items()
                },
            })

            selected = test[
                (test.condition == condition)
                & (test.method == method)
            ].copy()
            for replicate, cohort in selected.groupby("replicate"):
                metric = publication_metrics(
                    cohort.label,
                    cohort.score_lower,
                    threshold,
                )
                metric_rows.append({
                    "method": output_method,
                    "selected_method": method,
                    "condition": condition,
                    "replicate": int(replicate),
                    "score_tail": "lower",
                    "threshold": threshold,
                    "fpr_cap": args.fpr_cap,
                    **metric,
                    "records": len(cohort),
                    "healthy_records": int((cohort.label == 0).sum()),
                    "damage_records": int((cohort.label == 1).sum()),
                    "coverage_mean": float(cohort.coverage.mean()),
                    "parameters": int(
                        cohort.policy_parameters.iloc[0]
                        if output_method == "clmm_v21"
                        else cohort.parameters.iloc[0]
                    ),
                })
                scored = cohort.copy()
                scored["output_method"] = output_method
                scored["score"] = scored.score_lower
                scored["threshold"] = threshold
                scored["prediction"] = (
                    scored.score >= threshold
                ).astype(int)
                score_rows.append(scored)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(policy_rows),
    )


def summarize(metrics):
    return metrics.groupby(
        ["method", "condition"], as_index=False,
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
    keys = ["condition", "replicate"]
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
                values[(metric, "clmm_v21")]
                - values[(metric, "source_usad")]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def finish(args, frame):
    metrics, scores, policy = evaluate(frame, args)
    summary = summarize(metrics)
    effects = paired_effects(metrics)
    effect_summary = effects.groupby(
        "condition", as_index=False,
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

    random_effect = effect_summary[
        effect_summary.condition == "sensor_0.50"
    ].iloc[0]
    clean_effect = effect_summary[
        effect_summary.condition == "clean"
    ].iloc[0]
    criteria = {
        "clean_auc_preserved": bool(abs(clean_effect.delta_auc_mean) < 1e-12),
        "clean_ba_preserved": bool(abs(clean_effect.delta_ba_mean) < 1e-12),
        "random50_delta_auc_at_least_0_03": bool(
            random_effect.delta_auc_mean >= 0.03
        ),
        "random50_delta_ba_at_least_0_05": bool(
            random_effect.delta_ba_mean >= 0.05
        ),
        "random50_delta_recall_at_least_0_10": bool(
            random_effect.delta_recall_mean >= 0.10
        ),
        "random50_positive_auc_masks_at_least_3_of_5": bool(
            random_effect.positive_auc_pairs >= 3
        ),
    }
    decision = {
        "promote_to_three_seed": bool(all(criteria.values())),
        "criteria": criteria,
        "training_labels": "healthy-only",
        "threshold_calibration": "labelled validation under fixed FPR cap",
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "damage_validation_used_for_architecture_selection": False,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
    }

    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    policy.to_csv(args.output_dir / "validation_policy.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    effect_summary.to_csv(
        args.output_dir / "effect_summary.csv", index=False,
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = summary.pivot(
        index="condition", columns="method", values="ba_mean",
    )
    axis = pivot.plot.bar(
        figsize=(10, 5), ylim=(0, 1.02), rot=15, grid=True,
    )
    axis.set_ylabel("Record-level balanced accuracy")
    axis.set_title("CLMM V21 — validation-calibrated operating point")
    axis.figure.tight_layout()
    axis.figure.savefig(
        args.output_dir / "clmm_v21_screening.png", dpi=220,
    )
    plt.close(axis.figure)

    print("\n=== CLMM V21 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== CLMM V21 PAIRED EFFECTS ===")
    print(effects.to_string(index=False))
    print("\n=== CLMM V21 EFFECT SUMMARY ===")
    print(effect_summary.to_string(index=False))
    print("\n=== FIXED-BRANCH VALIDATION POLICY ===")
    print(policy.to_string(index=False))
    print("\n=== CLMM V21 SCREENING DECISION ===")
    print(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270803)
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
    parser.add_argument("--latent-weight", type=float, default=0.0)
    parser.add_argument("--score-weight", type=float, default=0.10)
    parser.add_argument("--fpr-cap", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.0 <= args.fpr_cap < 1.0:
        raise ValueError("fpr-cap must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} seed={args.seed} "
        f"source_epochs={args.epochs} adapter_epochs={args.adapter_epochs} "
        f"fpr_cap={args.fpr_cap}",
        flush=True,
    )

    # Reuse V19's audited data path while replacing only the adapter fitter.
    v19.fit_adapter = fit_adapter
    for index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = care.group_file(args.output_dir, key[0], key[1])
        if path.exists():
            print(f"SKIP cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[
            (used.setup == key[0])
            & (used.excitation == key[1])
        ]
        v19.run_group(args, group, labels, index, device, path)

    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files, got {len(paths)}")
    frame = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True,
    )
    decision = finish(args, frame)
    protocol = {
        "dataset": "Z24",
        "method": "CLMM V21",
        "full_name": "Counteracting Loss from Missing Measurements",
        "baseline": "vendored official EURECOM USAD",
        "training_labels": "healthy-only",
        "adapter_objective": "reconstruction plus frozen-USAD score consistency",
        "latent_cosine_removed": True,
        "fixed_branch": "imputed_usad",
        "fixed_tail": "lower",
        "threshold_calibration": "labelled validation under FPR cap",
        "fpr_cap": args.fpr_cap,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
        "seed": args.seed,
        "decision": decision,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    gc.collect()
    print(f"CLMM_V21_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

