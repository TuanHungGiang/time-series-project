"""Validation-only robustness audit for the locked Z24 modal-peak transform.

The nominal modal configuration is never re-selected from the sensitivity
results.  One factor is changed at a time around the predeclared configuration
(6 modes, 0.50-Hz separation, 0.45-Hz tracking radius, 8192/4096
window/stride).  Damage validation is used only to measure robustness; the
locked test is never loaded into a detector or scored by this script.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import run_z24_deterministic_preprocessing_selection as benchmark
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal


@dataclass(frozen=True)
class Configuration:
    name: str
    modes: int = 6
    separation: float = .50
    radius: float = .45
    window: int = 8192
    stride: int = 4096


CONFIGURATIONS = (
    Configuration("locked"),
    Configuration("modes_4", modes=4),
    Configuration("modes_8", modes=8),
    Configuration("radius_030", radius=.30),
    Configuration("radius_060", radius=.60),
    Configuration("window_4096", window=4096, stride=2048),
    Configuration("separation_030", separation=.30),
    Configuration("separation_070", separation=.70),
)


def spectra_for_configuration(rows: pd.DataFrame, config: Configuration,
                              args):
    spectra, frequency = {}, None
    for index, row in enumerate(rows.itertuples(), 1):
        current_frequency, current = modal.record_spectra(
            row.path, args.fs, config.window, config.stride,
            args.fmin, args.modal_fmax,
        )
        if frequency is None:
            frequency = current_frequency
        elif not np.array_equal(frequency, current_frequency):
            raise ValueError("Modal frequency grids differ")
        spectra[row.record_id] = current
        if index % 24 == 0 or index == len(rows):
            print(f"{config.name} spectra={index}/{len(rows)}", flush=True)
    return spectra, frequency


def modal_cache(rows: pd.DataFrame, train: pd.DataFrame,
                spectra: dict[str, np.ndarray], frequency: np.ndarray,
                config: Configuration):
    anchors = modal.learn_anchors(
        [spectra[key] for key in train.record_id], frequency,
        config.modes, config.separation,
    )
    cache = {
        row.record_id: modal.track_modes(
            frequency, spectra[row.record_id], anchors, config.radius
        ) for row in rows.itertuples()
    }
    return cache, anchors


def anchor_fold_audit(train: pd.DataFrame, spectra, frequency,
                      config: Configuration, folds: int):
    splitter = GroupKFold(n_splits=folds)
    rows = []
    for fold, (fit_index, held_index) in enumerate(splitter.split(
            train, groups=train.setup.astype(int)), 1):
        fit = train.iloc[fit_index]
        held = train.iloc[held_index]
        anchors = modal.learn_anchors(
            [spectra[key] for key in fit.record_id], frequency,
            config.modes, config.separation,
        )
        for mode_index, value in enumerate(anchors, 1):
            rows.append({
                "fold": fold, "mode_index": mode_index,
                "anchor_hz": float(value),
                "fit_records": len(fit),
                "heldout_setups": ",".join(map(str, sorted(held.setup.unique()))),
            })
    return pd.DataFrame(rows)


def edge_hit_audit(train: pd.DataFrame, cache, anchors, frequency,
                   config: Configuration):
    step = float(np.median(np.diff(frequency)))
    boundary = max(config.radius - step, 0.)
    rows = []
    for row in train.itertuples():
        values = cache[row.record_id]
        for mode_index, anchor in enumerate(anchors):
            distances = np.abs(values[:, mode_index] - anchor)
            rows.append({
                "record_id": row.record_id,
                "excitation": str(row.excitation).upper(),
                "setup": int(row.setup), "mode_index": mode_index + 1,
                "anchor_hz": float(anchor),
                "edge_hit_fraction": float(np.mean(distances >= boundary)),
                "mean_distance_hz": float(np.mean(distances)),
            })
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame):
    locked = metrics[metrics.configuration == "locked"].groupby(
        "model"
    ).auroc.mean()
    rows = []
    for (configuration, model), group in metrics.groupby(
            ["configuration", "model"]):
        mean = float(group.auroc.mean())
        rows.append({
            "configuration": configuration, "model": model,
            "seeds": group.seed.nunique(), "auroc_mean": mean,
            "auroc_std": float(group.auroc.std()),
            "balanced_accuracy_mean": float(group.balanced_accuracy.mean()),
            "recall_mean": float(group.recall.mean()),
            "specificity_mean": float(group.specificity.mean()),
            "delta_auroc_vs_locked": mean - float(locked.loc[model]),
        })
    by_model = pd.DataFrame(rows)
    summary = by_model.groupby("configuration", as_index=False).agg(
        detectors=("model", "nunique"),
        auroc_mean=("auroc_mean", "mean"),
        auroc_worst_detector=("auroc_mean", "min"),
        mean_delta_vs_locked=("delta_auroc_vs_locked", "mean"),
        worst_delta_vs_locked=("delta_auroc_vs_locked", "min"),
    )
    summary["predeclared_locked"] = summary.configuration == "locked"
    return by_model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_modal_sensitivity"))
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
    model_names = benchmark.parse_csv(args.models)
    seeds = benchmark.parse_seeds(args.seeds)
    requested = benchmark.parse_csv(args.configurations)
    config_lookup = {config.name: config for config in CONFIGURATIONS}
    unknown = set(requested) - set(config_lookup)
    if unknown:
        raise ValueError(f"Unknown configurations: {sorted(unknown)}")
    if "locked" not in requested:
        raise ValueError("The predeclared locked configuration is mandatory")

    frame = common.resolve_manifest(args.manifest, args.data_root)
    train = frame[(frame.split == "train") &
                  (frame.label.astype(int) == 0)].copy()
    healthy_val = frame[(frame.split == "val") &
                        (frame.label.astype(int) == 0)].copy()
    val = frame[frame.split == "val"].copy()
    if (len(train), len(healthy_val), len(val)) != (24, 6, 132):
        raise ValueError(
            f"Unexpected train/healthy-val/val counts "
            f"{(len(train), len(healthy_val), len(val))}"
        )
    # Test rows are deliberately excluded before any feature extraction.
    analysis_rows = pd.concat([train, val], ignore_index=True)
    parts = (train, healthy_val, val, pd.DataFrame())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train={len(train)} validation={len(val)} "
          f"models={model_names} seeds={seeds}", flush=True)
    print("locked_test_loaded=False damage_used_for_fit=False "
          "configuration_selected_from_sensitivity=False", flush=True)

    started = time.time()
    metrics, score_frames, anchor_rows, edge_rows = [], [], [], []
    spectra_cache = {}
    for name in requested:
        config = config_lookup[name]
        spectrum_key = (config.window, config.stride)
        if spectrum_key not in spectra_cache:
            spectra_cache[spectrum_key] = spectra_for_configuration(
                analysis_rows, config, args
            )
        spectra, frequency = spectra_cache[spectrum_key]
        cache, anchors = modal_cache(
            analysis_rows, train, spectra, frequency, config
        )
        for mode_index, anchor in enumerate(anchors, 1):
            anchor_rows.append({
                "configuration": name, "source": "all_train",
                "fold": 0, "mode_index": mode_index,
                "anchor_hz": float(anchor), "fit_records": len(train),
                "heldout_setups": "",
            })
        if name == "locked":
            folded = anchor_fold_audit(
                train, spectra, frequency, config, args.oof_folds
            )
            folded.insert(0, "configuration", name)
            folded.insert(1, "source", "grouped_oof")
            anchor_rows.extend(folded.to_dict("records"))
        edge = edge_hit_audit(train, cache, anchors, frequency, config)
        edge.insert(0, "configuration", name)
        edge_rows.append(edge)

        for model_name in model_names:
            for seed in seeds:
                summary, scores = benchmark.evaluate_validation(
                    name, cache, model_name, seed, parts, args, device
                )
                summary["configuration"] = summary.pop("preprocessing")
                scores["configuration"] = name
                metrics.append(summary); score_frames.append(scores)
                print(f"config={name} model={model_name} seed={seed} "
                      f"AUROC={summary['auroc']:.4f} "
                      f"BA={summary['balanced_accuracy']:.4f}", flush=True)

    metrics = pd.DataFrame(metrics)
    scores = pd.concat(score_frames, ignore_index=True)
    anchors = pd.DataFrame(anchor_rows)
    edge_hits = pd.concat(edge_rows, ignore_index=True)
    by_model, summary = aggregate_metrics(metrics)
    fold_anchor_summary = anchors[anchors.source == "grouped_oof"].groupby(
        "mode_index", as_index=False
    ).agg(anchor_mean_hz=("anchor_hz", "mean"),
          anchor_std_hz=("anchor_hz", "std"),
          anchor_range_hz=("anchor_hz", lambda x: float(x.max() - x.min())))

    metrics.to_csv(args.output / "sensitivity_metrics_by_seed.csv", index=False)
    scores.to_csv(args.output / "sensitivity_validation_scores.csv", index=False)
    by_model.to_csv(args.output / "sensitivity_by_detector.csv", index=False)
    summary.to_csv(args.output / "sensitivity_summary.csv", index=False)
    anchors.to_csv(args.output / "modal_anchor_audit.csv", index=False)
    fold_anchor_summary.to_csv(
        args.output / "locked_anchor_stability.csv", index=False
    )
    edge_hits.to_csv(args.output / "tracking_edge_hits.csv", index=False)

    plot = by_model.pivot(index="configuration", columns="model",
                          values="auroc_mean").reindex(requested)
    figure, axis = plt.subplots(figsize=(12, 6))
    plot.plot.bar(ax=axis, ylim=(0, 1))
    axis.set_title("Validation-only modal-peak sensitivity")
    axis.set_ylabel("Mean AUROC across seeds")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(args.output / "modal_peak_sensitivity.png", dpi=190)
    plt.close(figure)

    metadata = {
        "purpose": "validation-only lock robustness audit",
        "predeclared_locked_configuration": asdict(config_lookup["locked"]),
        "one_factor_at_a_time": True,
        "configuration_reselected_from_results": False,
        "damage_used_for_feature_or_detector_fit": False,
        "damage_validation_used_for_robustness_measurement": True,
        "locked_test_loaded_or_scored": False,
        "models": model_names, "seeds": seeds,
        "pca_components_requested": args.pca_components,
        "elapsed_seconds": time.time() - started,
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nSENSITIVITY SUMMARY\n" + summary.to_string(index=False), flush=True)
    print("\nLOCKED ANCHOR STABILITY\n" +
          fold_anchor_summary.to_string(index=False), flush=True)
    print("\nTRACKING EDGE-HIT SUMMARY\n" + edge_hits.groupby(
        "configuration", as_index=False
    ).edge_hit_fraction.mean().to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

