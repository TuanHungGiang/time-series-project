"""Final validation selection and one-shot test for modal-centroid preprocessing."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_z24_deterministic_preprocessing_selection as benchmark
import run_z24_modal_peak_sensitivity as sensitivity
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal


CONFIGURATIONS = (
    sensitivity.Configuration("centroid_locked"),
    sensitivity.Configuration("centroid_modes4", modes=4),
    sensitivity.Configuration("centroid_window4096", window=4096, stride=2048),
    sensitivity.Configuration("centroid_separation070", separation=.70),
    sensitivity.Configuration("centroid_modes4_window4096", modes=4,
                              window=4096, stride=2048),
    sensitivity.Configuration("centroid_modes4_window4096_separation070",
                              modes=4, separation=.70,
                              window=4096, stride=2048),
)
BASELINE = "centroid_locked"


def centroid_cache(rows: pd.DataFrame, train: pd.DataFrame, spectra,
                   frequency, config):
    anchors = modal.learn_anchors(
        [spectra[key] for key in train.record_id], frequency,
        config.modes, config.separation,
    )
    cache = {
        row.record_id: modal.track_modes_centroid(
            frequency, spectra[row.record_id], anchors, config.radius
        ) for row in rows.itertuples()
    }
    return cache, anchors


def select_configuration(metrics: pd.DataFrame, models: list[str]):
    ranked = metrics.copy()
    ranked["rank"] = ranked.groupby(["model", "seed"])["auroc"].rank(
        method="average", ascending=False
    )
    aggregate = ranked.groupby("configuration", as_index=False).agg(
        mean_rank=("rank", "mean"), mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_recall=("recall", "mean"),
        mean_specificity=("specificity", "mean"),
    )
    model_means = metrics.groupby(
        ["configuration", "model"], as_index=False
    ).auroc.mean()
    base = model_means[model_means.configuration == BASELINE].set_index("model")
    deltas = []
    for row in model_means.itertuples():
        delta = float(row.auroc - base.loc[row.model, "auroc"])
        deltas.append({"configuration": row.configuration,
                       "model": row.model, "auroc_mean": row.auroc,
                       "delta_vs_centroid_locked": delta})
    deltas = pd.DataFrame(deltas)
    robustness = deltas.groupby("configuration", as_index=False).agg(
        improved_detectors=("delta_vs_centroid_locked",
                            lambda x: int((x > 0).sum())),
        noninferior_detectors=("delta_vs_centroid_locked",
                              lambda x: int((x >= -.02).sum())),
        worst_detector_delta=("delta_vs_centroid_locked", "min"),
        mean_delta=("delta_vs_centroid_locked", "mean"),
    )
    aggregate = aggregate.merge(robustness, on="configuration")
    required = max(1, int(np.ceil(.75 * len(models))))
    aggregate["eligible"] = (
        (aggregate.improved_detectors >= required) &
        (aggregate.noninferior_detectors == len(models))
    )
    aggregate.loc[aggregate.configuration == BASELINE, "eligible"] = True
    candidates = aggregate[aggregate.eligible].sort_values(
        ["mean_rank", "mean_auroc", "mean_balanced_accuracy"],
        ascending=[True, False, False],
    )
    selected = str(candidates.iloc[0].configuration)
    aggregate["selected"] = aggregate.configuration == selected
    aggregate = aggregate.sort_values(
        ["selected", "eligible", "mean_rank", "mean_auroc"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return selected, aggregate, deltas


def markdown_table(frame: pd.DataFrame, columns: list[str], digits=4):
    values = frame[columns].copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(
                lambda x: "" if pd.isna(x) else f"{x:.{digits}f}"
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |"
            for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(path: Path, selected: str, selection: pd.DataFrame,
                 validation_by_model: pd.DataFrame,
                 test_summary: pd.DataFrame, consensus_metrics: dict,
                 anchor_summary: pd.DataFrame, edge_rate: float,
                 passed: bool, config) -> None:
    selected_row = selection[selection.selected].iloc[0]
    report = f"""# Z24 final preprocessing lock report

## Decision

- **Status:** {'PASS — preprocessing can be locked' if passed else 'REVIEW — do not lock yet'}
- **Selected preprocessing:** `{selected}`
- **Representation:** continuous modal centroid
- **Parameters:** modes={config.modes}, window={config.window}, stride={config.stride}, separation={config.separation:.2f} Hz, radius={config.radius:.2f} Hz
- **Selection source:** validation only
- **Damage used for feature/detector/threshold fitting:** no
- **Locked test used for configuration selection:** no

## Validation selection

{markdown_table(selection, ['configuration', 'mean_rank', 'mean_auroc', 'std_auroc', 'improved_detectors', 'noninferior_detectors', 'worst_detector_delta', 'eligible', 'selected'])}

Selected mean validation AUROC: **{selected_row.mean_auroc:.4f}**.

## Selected validation performance by detector

{markdown_table(validation_by_model, ['model', 'seeds', 'auroc_mean', 'auroc_std', 'balanced_accuracy_mean', 'recall_mean', 'specificity_mean'])}

## One-shot locked-test performance

{markdown_table(test_summary, ['model', 'seeds', 'auroc_mean', 'auroc_std', 'balanced_accuracy_mean', 'recall_mean', 'specificity_mean', 'f1_mean'])}

Detector/seed consensus: AUROC **{consensus_metrics['auroc']:.4f}**, balanced accuracy **{consensus_metrics['balanced_accuracy']:.4f}**, recall **{consensus_metrics['recall']:.4f}**, specificity **{consensus_metrics['specificity']:.4f}**, F1 **{consensus_metrics['f1']:.4f}**.

## Structural audit

{markdown_table(anchor_summary, ['mode_index', 'anchor_mean_hz', 'anchor_std_hz', 'anchor_range_hz'])}

- Mean healthy-train tracking edge-hit rate: **{edge_rate:.4%}**.
- Anchor criterion: maximum fold range must not exceed tracking radius ({config.radius:.2f} Hz).
- Edge criterion: mean edge-hit rate must be below 5%.

## Lock rule

PASS requires: validation eligibility (improves at least 3/4 detectors and no detector loses more than 0.02 AUROC versus the locked centroid control), stable grouped-OOF anchors, and edge-hit below 5%. Test metrics are reported but do not determine PASS.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_centroid_final"))
    parser.add_argument("--models", default="ocsvm,pca,iforest,ae")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--configurations", default=",".join(
        config.name for config in CONFIGURATIONS
    ))
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--fs", type=float, default=50.)
    parser.add_argument("--fmin", type=float, default=3.)
    parser.add_argument("--modal-fmax", type=float, default=20.)
    parser.add_argument("--record-quantile", type=float, default=.95)
    parser.add_argument("--ocsvm-nu", type=float, default=.05)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--ae-latent", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = benchmark.parse_csv(args.models)
    seeds = benchmark.parse_seeds(args.seeds)
    requested = benchmark.parse_csv(args.configurations)
    lookup = {config.name: config for config in CONFIGURATIONS}
    unknown = set(requested) - set(lookup)
    if unknown or BASELINE not in requested:
        raise ValueError(f"Invalid configurations: unknown={unknown}, baseline_required={BASELINE}")

    frame = common.resolve_manifest(args.manifest, args.data_root)
    train = frame[(frame.split == "train") &
                  (frame.label.astype(int) == 0)].copy()
    healthy_val = frame[(frame.split == "val") &
                        (frame.label.astype(int) == 0)].copy()
    val = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    if tuple(map(len, (train, healthy_val, val, test))) != (24, 6, 132, 132):
        raise ValueError("Unexpected manifest split")
    parts = (train, healthy_val, val, test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis_rows = pd.concat([train, val], ignore_index=True)
    print(f"[1/4] Loading validation-stage spectra for "
          f"{len(analysis_rows)} records...", flush=True)
    started = time.time()

    spectra_cache, caches, anchors_by_config, frequency_by_config = {}, {}, {}, {}
    validation_rows = []
    for number, name in enumerate(requested, 1):
        config = lookup[name]
        key = (config.window, config.stride)
        if key not in spectra_cache:
            spectra_cache[key] = sensitivity.spectra_for_configuration(
                analysis_rows, config, args
            )
        spectra, frequency = spectra_cache[key]
        cache, anchors = centroid_cache(
            analysis_rows, train, spectra, frequency, config
        )
        caches[name], anchors_by_config[name] = cache, anchors
        frequency_by_config[name] = frequency
        for model in models:
            for seed in seeds:
                result, _ = benchmark.evaluate_validation(
                    name, cache, model, seed, parts, args, device
                )
                result["configuration"] = result.pop("preprocessing")
                validation_rows.append(result)
        print(f"[2/4] Validation {number}/{len(requested)} complete: {name}", flush=True)

    validation = pd.DataFrame(validation_rows)
    selected, selection, deltas = select_configuration(validation, models)
    selected_config = lookup[selected]
    selected_cache = caches[selected]
    print(f"[3/4] Selected from validation: {selected}", flush=True)

    # The locked test is first opened only after validation has selected the
    # complete preprocessing configuration. Anchors remain train-only.
    test_spectra, test_frequency = sensitivity.spectra_for_configuration(
        test, selected_config, args
    )
    selected_frequency = frequency_by_config[selected]
    if not np.array_equal(test_frequency, selected_frequency):
        raise ValueError("Selected test frequency grid differs")
    selected_anchors = anchors_by_config[selected]
    for row in test.itertuples():
        selected_cache[row.record_id] = modal.track_modes_centroid(
            test_frequency, test_spectra[row.record_id], selected_anchors,
            selected_config.radius,
        )

    # Suppress per-seed test lines; retain them in a diagnostic text file.
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        test_metrics, test_scores = benchmark.evaluate_test(
            selected, selected_cache, models, seeds, parts, args, device
        )
    (args.output / "detailed_test_log.txt").write_text(
        capture.getvalue(), encoding="utf-8"
    )
    consensus = test_scores.groupby(
        ["record_id", "state", "scenario", "excitation", "setup", "label"],
        as_index=False,
    ).agg(normalized_score=("normalized_score", "mean"))
    consensus["alarm"] = consensus.normalized_score > 1.
    consensus_metrics = common.metrics(
        consensus.label, consensus.normalized_score, 1.
    )

    spectra, frequency = spectra_cache[(selected_config.window,
                                        selected_config.stride)]
    fold_anchors = sensitivity.anchor_fold_audit(
        train, spectra, frequency, selected_config, args.oof_folds
    )
    anchor_summary = fold_anchors.groupby("mode_index", as_index=False).agg(
        anchor_mean_hz=("anchor_hz", "mean"),
        anchor_std_hz=("anchor_hz", "std"),
        anchor_range_hz=("anchor_hz", lambda x: float(x.max() - x.min())),
    )
    edge = sensitivity.edge_hit_audit(
        train, selected_cache, anchors_by_config[selected], frequency,
        selected_config,
    )
    edge_rate = float(edge.edge_hit_fraction.mean())
    selected_row = selection[selection.selected].iloc[0]
    anchor_pass = bool(anchor_summary.anchor_range_hz.max() <= selected_config.radius)
    edge_pass = edge_rate < .05
    passed = bool(selected_row.eligible and anchor_pass and edge_pass)

    validation_by_model = validation[
        validation.configuration == selected
    ].groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
    )
    test_summary = test_metrics.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), f1_mean=("f1", "mean"),
    )

    validation.to_csv(args.output / "validation_metrics_by_seed.csv", index=False)
    selection.to_csv(args.output / "configuration_selection.csv", index=False)
    deltas.to_csv(args.output / "configuration_deltas.csv", index=False)
    test_metrics.to_csv(args.output / "selected_test_metrics.csv", index=False)
    test_scores.to_csv(args.output / "selected_test_record_scores.csv", index=False)
    consensus.to_csv(args.output / "selected_test_consensus_scores.csv", index=False)
    fold_anchors.to_csv(args.output / "selected_anchor_folds.csv", index=False)
    anchor_summary.to_csv(args.output / "selected_anchor_stability.csv", index=False)
    edge.to_csv(args.output / "selected_tracking_edge_hits.csv", index=False)

    write_report(
        args.output / "FINAL_PREPROCESSING_REPORT.md", selected, selection,
        validation_by_model, test_summary, consensus_metrics,
        anchor_summary, edge_rate, passed, selected_config,
    )
    decision = {
        "status": "PASS" if passed else "REVIEW",
        "selected_preprocessing": selected,
        "representation": "modal_centroid",
        "parameters": asdict(selected_config),
        "selection_used_validation_only": True,
        "test_used_for_selection": False,
        "damage_used_for_fit": False,
        "eligibility_pass": bool(selected_row.eligible),
        "anchor_stability_pass": anchor_pass,
        "edge_hit_pass": edge_pass,
        "edge_hit_rate": edge_rate,
        "consensus_test_metrics": consensus_metrics,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "final_preprocessing_decision.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    plot = selection.sort_values("mean_rank")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(plot.configuration, plot.mean_auroc)
    axes[0].set_ylim(0, 1); axes[0].set_title("Validation mean AUROC")
    axes[0].tick_params(axis="x", rotation=30)
    test_summary.plot.bar(x="model", y="auroc_mean", ylim=(0, 1),
                          legend=False, ax=axes[1], title="Selected test AUROC")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(args.output / "final_preprocessing_summary.png", dpi=190)
    plt.close(figure)

    print(f"[4/4] Final audit complete. Status={decision['status']}", flush=True)
    print(f"REPORT={args.output / 'FINAL_PREPROCESSING_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

