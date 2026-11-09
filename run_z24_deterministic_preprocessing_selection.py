"""Select a model-agnostic deterministic preprocessing for Z24.

This benchmark separates signal preprocessing from learned representation.
All candidate transforms are deterministic after fitting healthy-train-only
statistics.  Four fixed one-class detectors evaluate every transform.  Damage
validation labels select at most one preprocessing; the locked test is scored
only after that selection is complete.

Selection is conservative: a non-baseline transform must improve validation
AUROC for at least three of four detectors and may not reduce any detector by
more than 0.02 AUROC.  Otherwise energy-normalised logPSD remains selected.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import run_z24_cm_hrr_benchmark as detectors
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal
from run_z24_strict_acf_pca import acf_features


PREPROCESSING = (
    "logpsd",
    "domain_cmhrr",
    "acf",
    "modal_peak",
    "modal_centroid",
)
MODELS = ("ocsvm", "pca", "iforest", "ae")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv(value)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    return seeds


@dataclass
class Reference:
    center: np.ndarray
    scale: np.ndarray
    records: int
    windows: int


class DomainCMHRR:
    """Healthy-train-only references respecting pre/post-lifting domains."""

    def __init__(self, clip: float):
        self.clip = clip
        self.references: dict[tuple, Reference] = {}

    def fit(self, rows: pd.DataFrame,
            raw: dict[str, np.ndarray]) -> "DomainCMHRR":
        if len(rows) == 0 or set(rows.label.astype(int)) != {0}:
            raise ValueError("Domain CM-HRR requires healthy training rows only")
        groups: dict[tuple, list[tuple[str, np.ndarray]]] = {}
        for row in rows.itertuples():
            domain = modal.domain_for_state(row.state)
            excitation = str(row.excitation).upper()
            keys = [
                ("domain", domain),
                ("excitation", domain, excitation),
                ("condition", domain, excitation, int(row.setup)),
            ]
            for key in keys:
                groups.setdefault(key, []).append(
                    (row.record_id, raw[row.record_id])
                )
        for key, parts in groups.items():
            values = np.concatenate([value for _, value in parts])
            center, scale = detectors.robust_location_scale(values)
            self.references[key] = Reference(
                center=center, scale=scale,
                records=len({record for record, _ in parts}),
                windows=len(values),
            )
        return self

    def key_for(self, state: str, excitation: str, setup: int) -> tuple:
        domain = modal.domain_for_state(state)
        excitation = str(excitation).upper()
        candidates = (
            ("condition", domain, excitation, int(setup)),
            ("excitation", domain, excitation),
            ("domain", domain),
        )
        for key in candidates:
            if key in self.references:
                return key
        raise ValueError(f"No healthy reference for domain={domain}")

    def transform(self, value: np.ndarray, state: str, excitation: str,
                  setup: int) -> tuple[np.ndarray, tuple]:
        key = self.key_for(state, excitation, setup)
        reference = self.references[key]
        result = np.clip(
            (value - reference.center) / reference.scale,
            -self.clip, self.clip,
        )
        return result.astype(np.float32), key

    def audit(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"key": "|".join(map(str, key)), "records": value.records,
             "windows": value.windows}
            for key, value in sorted(
                self.references.items(), key=lambda item: str(item[0])
            )
        ])


def build_feature_caches(frame: pd.DataFrame, train: pd.DataFrame, args):
    logpsd: dict[str, np.ndarray] = {}
    acf: dict[str, np.ndarray] = {}
    spectra: dict[str, np.ndarray] = {}
    frequency = None
    for index, row in enumerate(frame.itertuples(), 1):
        logpsd[row.record_id] = common.psd_vectors(
            row.path, args.fs, args.window, args.stride, args.psd_bins,
            args.fmin, args.fmax,
        )
        acf[row.record_id] = acf_features(
            row.path, args.acf_window, args.acf_stride
        )
        current_frequency, current_spectrum = modal.record_spectra(
            row.path, args.fs, args.modal_window, args.modal_stride,
            args.fmin, args.modal_fmax,
        )
        if frequency is None:
            frequency = current_frequency
        elif not np.array_equal(frequency, current_frequency):
            raise ValueError("Modal frequency grids differ between records")
        spectra[row.record_id] = current_spectrum
        if index % 24 == 0 or index == len(frame):
            print(f"deterministic features={index}/{len(frame)}", flush=True)

    residualizer = DomainCMHRR(args.cmhrr_clip).fit(train, logpsd)
    domain_cmhrr = {}
    assignments = []
    for row in frame.itertuples():
        value, key = residualizer.transform(
            logpsd[row.record_id], row.state, row.excitation, int(row.setup)
        )
        domain_cmhrr[row.record_id] = value
        assignments.append({
            "record_id": row.record_id, "split": row.split,
            "state": row.state, "reference_key": "|".join(map(str, key)),
        })

    anchors = modal.learn_anchors(
        [spectra[key] for key in train.record_id], frequency,
        args.modes, args.mode_separation,
    )
    modal_peak = {
        key: modal.track_modes(
            frequency, value, anchors, args.tracking_radius
        ) for key, value in spectra.items()
    }
    modal_centroid = {
        key: modal.track_modes_centroid(
            frequency, value, anchors, args.tracking_radius
        ) for key, value in spectra.items()
    }
    caches = {
        "logpsd": logpsd,
        "domain_cmhrr": domain_cmhrr,
        "acf": acf,
        "modal_peak": modal_peak,
        "modal_centroid": modal_centroid,
    }
    audit = {
        "reference_audit": residualizer.audit(),
        "reference_assignments": pd.DataFrame(assignments),
        "modal_anchors": anchors,
    }
    return caches, audit


def build_detector(model_name: str, args, seed: int,
                   device: torch.device):
    detector_args = copy.copy(args)
    detector_args.seed = seed
    seed_all(seed)
    return detectors.build_detector(model_name, detector_args, device)


def record_score(detector, value: np.ndarray, quantile: float) -> float:
    return float(np.quantile(detector.score(value), quantile))


def fit_detector(model_name: str, cache: dict[str, np.ndarray],
                 train: pd.DataFrame, healthy_val: pd.DataFrame,
                 args, seed: int, device: torch.device):
    train_x = np.concatenate([cache[key] for key in train.record_id])
    healthy_val_x = np.concatenate(
        [cache[key] for key in healthy_val.record_id]
    )
    detector = build_detector(model_name, args, seed, device)
    detector.fit(train_x, healthy_val_x)
    calibration = {
        row.record_id: record_score(
            detector, cache[row.record_id], args.record_quantile
        ) for row in healthy_val.itertuples()
    }
    threshold = max(max(calibration.values()), np.finfo(float).tiny)
    return detector, calibration, threshold


def score_rows(rows: pd.DataFrame, cache: dict[str, np.ndarray], detector,
               calibration: dict[str, float], threshold: float,
               args, split: str) -> pd.DataFrame:
    output = []
    for row in rows.itertuples():
        current_threshold = threshold
        if split == "validation" and int(row.label) == 0 and \
                row.record_id in calibration:
            other = [value for key, value in calibration.items()
                     if key != row.record_id]
            current_threshold = max(max(other), np.finfo(float).tiny)
        score = record_score(
            detector, cache[row.record_id], args.record_quantile
        )
        output.append({
            "record_id": row.record_id, "state": row.state,
            "scenario": row.scenario, "excitation": row.excitation,
            "setup": int(row.setup), "label": int(row.label),
            "split": split, "score": score,
            "threshold": current_threshold,
            "normalized_score": score / current_threshold,
            "alarm": score > current_threshold,
        })
    return pd.DataFrame(output)


def evaluate_validation(preprocessing, cache, model_name, seed, parts,
                        args, device):
    train, healthy_val, val_rows, _ = parts
    detector, calibration, threshold = fit_detector(
        model_name, cache, train, healthy_val, args, seed, device
    )
    scores = score_rows(
        val_rows, cache, detector, calibration, threshold,
        args, "validation",
    )
    metrics = common.metrics(scores.label, scores.normalized_score, 1.)
    summary = {
        "preprocessing": preprocessing, "model": model_name,
        "seed": seed, "train_records": len(train),
        "healthy_calibration_records": len(healthy_val), **metrics,
    }
    scores.insert(0, "preprocessing", preprocessing)
    scores.insert(1, "model", model_name)
    scores.insert(2, "seed", seed)
    return summary, scores


def preprocessing_selection(validation_metrics: pd.DataFrame,
                            model_names: list[str]):
    means = validation_metrics.groupby(
        ["preprocessing", "model"], as_index=False
    ).agg(auroc_mean=("auroc", "mean"),
          balanced_accuracy_mean=("balanced_accuracy", "mean"))
    baseline = means[means.preprocessing == "logpsd"].set_index("model")
    delta_rows = []
    for row in means.itertuples():
        before = baseline.loc[row.model]
        delta_rows.append({
            "preprocessing": row.preprocessing, "model": row.model,
            "auroc_mean": row.auroc_mean,
            "delta_auroc_vs_logpsd": row.auroc_mean - before.auroc_mean,
            "balanced_accuracy_mean": row.balanced_accuracy_mean,
        })
    deltas = pd.DataFrame(delta_rows)

    ranked = validation_metrics.copy()
    ranked["rank"] = ranked.groupby(["model", "seed"])["auroc"].rank(
        method="average", ascending=False
    )
    aggregate = ranked.groupby("preprocessing", as_index=False).agg(
        mean_rank=("rank", "mean"), mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_recall=("recall", "mean"),
        mean_specificity=("specificity", "mean"),
    )
    robustness = deltas.groupby("preprocessing", as_index=False).agg(
        improved_models=("delta_auroc_vs_logpsd", lambda x: int((x > 0).sum())),
        noninferior_models=("delta_auroc_vs_logpsd", lambda x: int((x >= -.02).sum())),
        worst_model_delta=("delta_auroc_vs_logpsd", "min"),
        mean_delta_vs_logpsd=("delta_auroc_vs_logpsd", "mean"),
    )
    aggregate = aggregate.merge(robustness, on="preprocessing")
    required_improvements = max(1, int(np.ceil(.75 * len(model_names))))
    aggregate["eligible"] = (
        (aggregate.improved_models >= required_improvements) &
        (aggregate.noninferior_models == len(model_names))
    )
    aggregate.loc[aggregate.preprocessing == "logpsd", "eligible"] = True
    candidates = aggregate[aggregate.eligible].sort_values(
        ["mean_rank", "mean_auroc", "mean_balanced_accuracy"],
        ascending=[True, False, False],
    )
    selected = str(candidates.iloc[0].preprocessing)
    aggregate["selected"] = aggregate.preprocessing == selected
    aggregate = aggregate.sort_values(
        ["selected", "eligible", "mean_rank", "mean_auroc"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return selected, aggregate, deltas


def evaluate_test(selected, cache, model_names, seeds, parts, args, device):
    train, healthy_val, _, test_rows = parts
    summaries, details = [], []
    for model_name in model_names:
        for seed in seeds:
            detector, calibration, threshold = fit_detector(
                model_name, cache, train, healthy_val, args, seed, device
            )
            scores = score_rows(
                test_rows, cache, detector, calibration, threshold,
                args, "locked_test",
            )
            metrics = common.metrics(scores.label, scores.normalized_score, 1.)
            summaries.append({
                "preprocessing": selected, "model": model_name,
                "seed": seed, **metrics,
            })
            scores.insert(0, "preprocessing", selected)
            scores.insert(1, "model", model_name)
            scores.insert(2, "seed", seed)
            details.append(scores)
            print(f"test selected={selected} model={model_name} seed={seed} "
                  f"AUROC={metrics['auroc']:.4f} "
                  f"BA={metrics['balanced_accuracy']:.4f}", flush=True)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def save_figures(validation_metrics, selection, test_metrics,
                 consensus, output):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    validation_pivot = validation_metrics.groupby(
        ["preprocessing", "model"]
    ).auroc.mean().unstack("model")
    validation_pivot.plot.bar(ax=axes[0], ylim=(0, 1),
                              title="Validation AUROC by detector")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(selection.preprocessing, selection.mean_rank)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(title="Mean validation rank (lower is better)",
                ylabel="Rank")
    test_metrics.groupby("model").auroc.mean().plot.bar(
        ax=axes[2], ylim=(0, 1), title="Selected-only locked-test AUROC"
    )
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "deterministic_preprocessing_selection.png",
                dpi=190, bbox_inches="tight")
    plt.close(fig)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(consensus.label, consensus.alarm, labels=[0, 1]),
        display_labels=["Healthy", "Damage"],
    ).plot(ax=axis, cmap="Greens", colorbar=False)
    axis.set_title("Selected preprocessing: detector/seed consensus")
    figure.tight_layout()
    figure.savefig(output / "selected_consensus_confusion.png", dpi=190,
                   bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_deterministic_preprocessing"))
    parser.add_argument("--preprocessing", default=",".join(PREPROCESSING))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--fs", type=float, default=50.)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--psd-bins", type=int, default=96)
    parser.add_argument("--fmin", type=float, default=3.)
    parser.add_argument("--fmax", type=float, default=30.)
    parser.add_argument("--acf-window", type=int, default=2048)
    parser.add_argument("--acf-stride", type=int, default=1024)
    parser.add_argument("--modal-window", type=int, default=8192)
    parser.add_argument("--modal-stride", type=int, default=4096)
    parser.add_argument("--modal-fmax", type=float, default=20.)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--mode-separation", type=float, default=.5)
    parser.add_argument("--tracking-radius", type=float, default=.45)
    parser.add_argument("--cmhrr-clip", type=float, default=10.)
    parser.add_argument("--record-quantile", type=float, default=.95)
    parser.add_argument("--ocsvm-nu", type=float, default=.05)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--ae-latent", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11,
                        help="Compatibility field; --seeds controls runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preprocessing_names = parse_csv(args.preprocessing)
    model_names = parse_csv(args.models)
    seeds = parse_seeds(args.seeds)
    unknown_preprocessing = set(preprocessing_names) - set(PREPROCESSING)
    unknown_models = set(model_names) - set(MODELS)
    if unknown_preprocessing or unknown_models:
        raise ValueError(
            f"Unknown preprocessing={unknown_preprocessing}, models={unknown_models}"
        )
    if "logpsd" not in preprocessing_names:
        raise ValueError("logpsd control is mandatory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    train = frame[(frame.split == "train") & (frame.label.astype(int) == 0)].copy()
    healthy_val = frame[(frame.split == "val") &
                        (frame.label.astype(int) == 0)].copy()
    val_rows = frame[frame.split == "val"].copy()
    test_rows = frame[frame.split == "test"].copy()
    parts = (train, healthy_val, val_rows, test_rows)
    counts = tuple(map(len, parts))
    if counts != (24, 6, 132, 132):
        raise ValueError(f"Unexpected manifest split {counts}")
    if set(train.state) != {"01", "08"}:
        raise ValueError(f"Healthy train must contain S01/S08, got {set(train.state)}")
    print(f"device={device} split={counts} seeds={seeds} "
          f"preprocessing={preprocessing_names} models={model_names}", flush=True)
    print("damage_used_for_fit=False damage_val_used_for_selection=True "
          "test_used_for_selection=False", flush=True)

    started = time.time()
    caches, audit = build_feature_caches(frame, train, args)
    audit["reference_audit"].to_csv(
        args.output / "healthy_reference_audit.csv", index=False
    )
    audit["reference_assignments"].to_csv(
        args.output / "reference_assignments.csv", index=False
    )
    pd.DataFrame({"modal_anchor_hz": audit["modal_anchors"]}).to_csv(
        args.output / "modal_anchors.csv", index=False
    )

    summaries, score_frames = [], []
    for preprocessing in preprocessing_names:
        for model_name in model_names:
            for seed in seeds:
                summary, scores = evaluate_validation(
                    preprocessing, caches[preprocessing], model_name, seed,
                    parts, args, device,
                )
                summaries.append(summary)
                score_frames.append(scores)
                print(f"validation preprocessing={preprocessing} "
                      f"model={model_name} seed={seed} "
                      f"AUROC={summary['auroc']:.4f} "
                      f"BA={summary['balanced_accuracy']:.4f}", flush=True)
    validation_metrics = pd.DataFrame(summaries)
    validation_scores = pd.concat(score_frames, ignore_index=True)
    selected, selection, deltas = preprocessing_selection(
        validation_metrics, model_names
    )
    print(f"\nSELECTED FROM VALIDATION ONLY: {selected}", flush=True)

    test_metrics, test_scores = evaluate_test(
        selected, caches[selected], model_names, seeds, parts, args, device
    )
    consensus = test_scores.groupby(
        ["record_id", "state", "scenario", "excitation", "setup", "label"],
        as_index=False,
    ).agg(normalized_score=("normalized_score", "mean"))
    consensus["alarm"] = consensus.normalized_score > 1.
    consensus_metrics = common.metrics(
        consensus.label, consensus.normalized_score, 1.
    )
    state_scores = consensus.groupby(
        ["state", "scenario", "label"], as_index=False
    ).agg(records=("record_id", "size"),
          score_median=("normalized_score", "median"),
          alarm_rate=("alarm", "mean"))

    validation_metrics.to_csv(
        args.output / "validation_metrics_by_model_seed.csv", index=False
    )
    validation_scores.to_csv(
        args.output / "validation_record_scores.csv", index=False
    )
    selection.to_csv(args.output / "preprocessing_selection.csv", index=False)
    deltas.to_csv(args.output / "validation_deltas_vs_logpsd.csv", index=False)
    test_metrics.to_csv(
        args.output / "selected_test_metrics_by_model_seed.csv", index=False
    )
    test_scores.to_csv(
        args.output / "selected_test_record_scores.csv", index=False
    )
    consensus.to_csv(
        args.output / "selected_test_consensus_scores.csv", index=False
    )
    pd.DataFrame([{"preprocessing": selected, **consensus_metrics}]).to_csv(
        args.output / "selected_test_consensus_metrics.csv", index=False
    )
    state_scores.to_csv(
        args.output / "selected_test_consensus_state_scores.csv", index=False
    )
    save_figures(validation_metrics, selection, test_metrics,
                 consensus, args.output)

    metadata = {
        "step": "deterministic model-agnostic preprocessing selection",
        "selected_preprocessing": selected,
        "selection_rule": (
            "lowest mean rank among candidates improving >=75% detectors, "
            "with no detector AUROC loss below -0.02; logpsd is fallback"
        ),
        "damage_used_for_fit_scaling_threshold": False,
        "damage_validation_used_for_selection": True,
        "locked_test_used_for_selection": False,
        "healthy_train_states": sorted(train.state.unique().tolist()),
        "preprocessing": preprocessing_names,
        "models": model_names,
        "seeds": seeds,
        "modal_anchors_hz": audit["modal_anchors"].tolist(),
        "elapsed_seconds": time.time() - started,
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nPREPROCESSING SELECTION\n" + selection.to_string(index=False),
          flush=True)
    print("\nVALIDATION DELTAS VS LOGPSD\n" + deltas.to_string(index=False),
          flush=True)
    print("\nSELECTED TEST METRICS BY MODEL/SEED\n" +
          test_metrics.to_string(index=False), flush=True)
    print("\nSELECTED TEST CONSENSUS\n" +
          pd.DataFrame([{"preprocessing": selected,
                         **consensus_metrics}]).to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

