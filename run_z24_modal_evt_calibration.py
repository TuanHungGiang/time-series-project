"""Leak-free EVT alarm calibration for the locked Z24 modal-peak pipeline.

This runner implements the alarm-setting part of Toshkova et al. (MSSP,
2020, doi:10.1016/j.ymssp.2019.106417) without changing the selected modal
preprocessing or the one-class detectors:

* modal anchors and detectors are cross-fitted on healthy train records;
* setup is the grouped OOF unit, so paired records never cross a fold;
* AVT/FVT is the known operational regime (the paper used process-variable
  clustering followed by a classifier);
* each held-out record supplies one block extreme (record score);
* GEV parameters are fitted by maximum likelihood and the alarm is the
  inverse CDF at 1-alpha.

Damage validation and locked test records are evaluation-only.  The script
also reports max-healthy and empirical-quantile controls.  With the default
record quantile of .95 this is an explicitly labelled robust block-extreme
adaptation; set --record-quantile 1.0 for literal block maxima.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import genextreme, kstest
from sklearn.model_selection import GroupKFold

import run_z24_cm_hrr_benchmark as detector_lib
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal


MODELS = ("ocsvm", "pca", "iforest", "ae")
SCOPES = ("pooled", "excitation")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str) -> list[int]:
    result = [int(value) for value in parse_csv(value)]
    if not result or len(result) != len(set(result)):
        raise ValueError("--seeds must contain unique integers")
    return result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_detector(name: str, args, seed: int, device: torch.device):
    current = copy.copy(args)
    current.seed = seed
    seed_all(seed)
    return detector_lib.build_detector(name, current, device)


def record_score(detector, value: np.ndarray, quantile: float) -> float:
    return float(np.quantile(detector.score(value), quantile))


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
    return center, max(scale, 1e-8)


def learn_modal_cache(rows: pd.DataFrame, fit_rows: pd.DataFrame,
                      spectra: dict[str, np.ndarray], frequency: np.ndarray,
                      args) -> tuple[dict[str, np.ndarray], np.ndarray]:
    anchors = modal.learn_anchors(
        [spectra[key] for key in fit_rows.record_id], frequency,
        args.modes, args.mode_separation,
    )
    cache = {
        row.record_id: modal.track_modes(
            frequency, spectra[row.record_id], anchors,
            args.tracking_radius,
        )
        for row in rows.itertuples()
    }
    return cache, anchors


def fit_score_reference(detector, cache: dict[str, np.ndarray],
                        rows: pd.DataFrame, quantile: float) -> dict[str, tuple[float, float]]:
    table = rows[["record_id", "excitation"]].copy()
    table["raw_score"] = [
        record_score(detector, cache[key], quantile)
        for key in table.record_id
    ]
    references = {}
    for excitation, group in table.groupby("excitation"):
        references[str(excitation).upper()] = robust_location_scale(
            group.raw_score.to_numpy()
        )
    references["__POOLED__"] = robust_location_scale(
        table.raw_score.to_numpy()
    )
    return references


def standardized_score(detector, value: np.ndarray, excitation: str,
                       references: dict[str, tuple[float, float]],
                       quantile: float) -> tuple[float, float]:
    raw = record_score(detector, value, quantile)
    key = str(excitation).upper()
    center, scale = references.get(key, references["__POOLED__"])
    return raw, (raw - center) / scale


def cross_fitted_healthy_scores(train: pd.DataFrame,
                                spectra: dict[str, np.ndarray],
                                frequency: np.ndarray, model_name: str,
                                seed: int, args, device: torch.device):
    unique_groups = np.sort(train.setup.astype(int).unique())
    if len(unique_groups) < args.oof_folds:
        raise ValueError(
            f"Need at least {args.oof_folds} setup groups, got {unique_groups}"
        )
    splitter = GroupKFold(n_splits=args.oof_folds)
    output, anchor_audit = [], []
    for fold, (fit_index, held_index) in enumerate(splitter.split(
            train, groups=train.setup.astype(int)), 1):
        fit_rows = train.iloc[fit_index].copy()
        held_rows = train.iloc[held_index].copy()
        cache, anchors = learn_modal_cache(
            pd.concat([fit_rows, held_rows]), fit_rows,
            spectra, frequency, args,
        )
        detector = build_detector(model_name, args, seed + fold, device)
        train_x = np.concatenate([cache[key] for key in fit_rows.record_id])
        # No OOF record is used for AE early stopping.
        detector.fit(train_x, None)
        references = fit_score_reference(
            detector, cache, fit_rows, args.record_quantile
        )
        for row in held_rows.itertuples():
            raw, score = standardized_score(
                detector, cache[row.record_id], row.excitation,
                references, args.record_quantile,
            )
            output.append({
                "record_id": row.record_id, "state": row.state,
                "excitation": str(row.excitation).upper(),
                "setup": int(row.setup), "fold": fold,
                "raw_score": raw, "score": score,
            })
        anchor_audit.append({
            "model": model_name, "seed": seed, "fold": fold,
            "fit_records": len(fit_rows),
            "heldout_records": len(held_rows),
            "heldout_setups": ",".join(map(str, sorted(held_rows.setup.unique()))),
            "anchors_hz": json.dumps(anchors.tolist()),
        })
    result = pd.DataFrame(output)
    if len(result) != len(train) or result.record_id.nunique() != len(train):
        raise RuntimeError("OOF calibration did not score each healthy record once")
    return result, pd.DataFrame(anchor_audit)


def fit_gev(values: np.ndarray, alpha: float, min_samples: int) -> dict:
    values = np.asarray(values, dtype=float)
    result = {
        "n": len(values), "threshold": np.nan, "location": np.nan,
        "scale": np.nan, "shape_xi": np.nan, "ks_statistic": np.nan,
        "ks_pvalue": np.nan, "fit_status": "not_fitted",
    }
    if len(values) < min_samples:
        result["fit_status"] = f"insufficient_n<{min_samples}"
        return result
    if not np.isfinite(values).all() or np.ptp(values) < 1e-10:
        result["fit_status"] = "degenerate_values"
        return result
    try:
        # scipy uses c=-xi relative to the paper's GEV shape convention.
        c, location, scale = genextreme.fit(values)
        threshold = float(genextreme.ppf(1. - alpha, c, loc=location, scale=scale))
        ks = kstest(values, "genextreme", args=(c, location, scale))
        if not np.isfinite([c, location, scale, threshold]).all() or scale <= 0:
            raise ValueError("invalid GEV parameters")
        result.update({
            "threshold": threshold, "location": float(location),
            "scale": float(scale), "shape_xi": float(-c),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue), "fit_status": "ok",
        })
    except Exception as error:  # keep failed fits visible; never silently fallback
        result["fit_status"] = f"failed:{type(error).__name__}"
    return result


def bootstrap_gev_ci(values: np.ndarray, alpha: float, samples: int,
                     seed: int, min_samples: int) -> tuple[float, float, int]:
    if samples <= 0 or len(values) < min_samples:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sample = rng.choice(values, size=len(values), replace=True)
        fit = fit_gev(sample, alpha, min_samples)
        if fit["fit_status"] == "ok":
            estimates.append(fit["threshold"])
    if len(estimates) < max(20, samples // 5):
        return np.nan, np.nan, len(estimates)
    return (*np.quantile(estimates, [.025, .975]).tolist(), len(estimates))


def fit_threshold_table(oof: pd.DataFrame, model_name: str, seed: int,
                        args) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        groups = [("__POOLED__", oof)] if scope == "pooled" else list(
            oof.groupby("excitation")
        )
        for regime, group in groups:
            values = group.score.to_numpy(float)
            controls = {
                "max_healthy": float(np.max(values)),
                "empirical": float(np.quantile(
                    values, 1. - args.alpha, method="higher"
                )),
            }
            for method, threshold in controls.items():
                rows.append({
                    "model": model_name, "seed": seed, "scope": scope,
                    "regime": regime, "method": method,
                    "alpha": args.alpha, "n": len(values),
                    "threshold": threshold, "location": np.nan,
                    "scale": np.nan, "shape_xi": np.nan,
                    "ks_statistic": np.nan, "ks_pvalue": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan,
                    "bootstrap_success": 0, "fit_status": "ok",
                })
            gev = fit_gev(values, args.alpha, args.gev_min_samples)
            low, high, successes = bootstrap_gev_ci(
                values, args.alpha, args.bootstrap_samples,
                seed + (0 if scope == "pooled" else 1000),
                args.gev_min_samples,
            )
            rows.append({
                "model": model_name, "seed": seed, "scope": scope,
                "regime": regime, "method": "gev",
                "alpha": args.alpha, **gev,
                "ci_low": low, "ci_high": high,
                "bootstrap_success": successes,
            })
    return pd.DataFrame(rows)


def threshold_lookup(table: pd.DataFrame, method: str, scope: str,
                     excitation: str) -> tuple[float, str]:
    regime = "__POOLED__" if scope == "pooled" else str(excitation).upper()
    selected = table[(table.method == method) &
                     (table.scope == scope) &
                     (table.regime == regime)]
    if len(selected) != 1:
        raise ValueError(f"Missing threshold {method}/{scope}/{regime}")
    row = selected.iloc[0]
    return float(row.threshold), str(row.fit_status)


def score_evaluation(rows: pd.DataFrame, cache: dict[str, np.ndarray],
                     detector, references, thresholds: pd.DataFrame,
                     model_name: str, seed: int, args, split: str):
    base = []
    for row in rows.itertuples():
        raw, score = standardized_score(
            detector, cache[row.record_id], row.excitation,
            references, args.record_quantile,
        )
        base.append({
            "record_id": row.record_id, "state": row.state,
            "scenario": row.scenario,
            "excitation": str(row.excitation).upper(),
            "setup": int(row.setup), "label": int(row.label),
            "split": split, "raw_score": raw, "score": score,
        })
    base = pd.DataFrame(base)
    details, summaries = [], []
    for method in ("max_healthy", "empirical", "gev"):
        for scope in SCOPES:
            current = base.copy()
            values = [threshold_lookup(
                thresholds, method, scope, excitation
            ) for excitation in current.excitation]
            current["threshold"] = [value[0] for value in values]
            current["fit_status"] = [value[1] for value in values]
            current["decision_margin"] = current.score - current.threshold
            current["alarm"] = current.decision_margin > 0.
            current.insert(0, "method", method)
            current.insert(1, "scope", scope)
            current.insert(2, "model", model_name)
            current.insert(3, "seed", seed)
            details.append(current)
            if current.threshold.isna().any():
                metrics = {key: np.nan for key in (
                    "tn", "fp", "fn", "tp", "accuracy",
                    "balanced_accuracy", "precision", "recall", "f1",
                    "specificity", "auroc", "auprc", "threshold",
                )}
            else:
                metrics = common.metrics(
                    current.label, current.decision_margin, 0.
                )
            summaries.append({
                "method": method, "scope": scope,
                "model": model_name, "seed": seed,
                "evaluation": split, **metrics,
            })
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def save_diagnostics(oof: pd.DataFrame, thresholds: pd.DataFrame,
                     metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for excitation, group in oof.groupby("excitation"):
        axes[0].hist(group.score, bins=max(4, len(group) // 2), alpha=.55,
                     label=excitation)
    axes[0].set(title="Grouped-OOF healthy block extremes",
                xlabel="Standardized anomaly score", ylabel="Records")
    axes[0].legend()
    valid = thresholds[(thresholds.method == "gev") &
                       (thresholds.fit_status == "ok")]
    axes[1].scatter(valid.shape_xi, valid.ks_pvalue,
                    c=(valid.scope == "excitation").astype(int), alpha=.8)
    axes[1].axhline(.05, color="red", linestyle="--", linewidth=1)
    axes[1].set(title="GEV fit audit", xlabel="Shape xi",
                ylabel="KS p-value")
    figure.tight_layout()
    figure.savefig(output / "evt_fit_diagnostics.png", dpi=190)
    plt.close(figure)

    test = metrics[metrics.evaluation == "locked_test"]
    plot = test.groupby(["method", "scope"], as_index=False).agg(
        recall=("recall", "mean"), specificity=("specificity", "mean")
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = plot.method + "_" + plot.scope
    x = np.arange(len(plot))
    axis.bar(x - .18, plot.recall, .36, label="Recall")
    axis.bar(x + .18, plot.specificity, .36, label="Specificity")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_title("Locked-test calibration trade-off")
    axis.legend(); axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(output / "evt_test_tradeoff.png", dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_modal_evt"))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--gev-min-samples", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--fs", type=float, default=50.)
    parser.add_argument("--modal-window", type=int, default=8192)
    parser.add_argument("--modal-stride", type=int, default=4096)
    parser.add_argument("--fmin", type=float, default=3.)
    parser.add_argument("--modal-fmax", type=float, default=20.)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--mode-separation", type=float, default=.5)
    parser.add_argument("--tracking-radius", type=float, default=.45)
    parser.add_argument("--record-quantile", type=float, default=.95)
    parser.add_argument("--ocsvm-nu", type=float, default=.05)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--ae-latent", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11,
                        help="Compatibility field; use --seeds")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be in (0,1)")
    if not 0 < args.record_quantile <= 1:
        raise ValueError("--record-quantile must be in (0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    models = parse_csv(args.models)
    seeds = parse_seeds(args.seeds)
    if set(models) - set(MODELS):
        raise ValueError(f"Unknown models: {set(models) - set(MODELS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    train = frame[(frame.split == "train") &
                  (frame.label.astype(int) == 0)].copy()
    val = frame[frame.split == "val"].copy()
    test = frame[frame.split == "test"].copy()
    if (len(train), len(val), len(test)) != (24, 132, 132):
        raise ValueError(
            f"Unexpected split {(len(train), len(val), len(test))}; "
            "expected (24,132,132)"
        )
    if set(train.state) != {"01", "08"} or set(train.label.astype(int)) != {0}:
        raise ValueError("Training must contain healthy S01/S08 only")
    if set(train.record_id) & (set(val.record_id) | set(test.record_id)):
        raise ValueError("Record leakage across splits")
    print(f"device={device} train={len(train)} val={len(val)} test={len(test)} "
          f"models={models} seeds={seeds}", flush=True)
    print("preprocessing=locked_modal_peak damage_used_for_fit=False "
          "damage_used_for_threshold=False test_used_for_selection=False",
          flush=True)

    started = time.time()
    spectra, frequency = {}, None
    for index, row in enumerate(frame.itertuples(), 1):
        current_frequency, current_spectrum = modal.record_spectra(
            row.path, args.fs, args.modal_window, args.modal_stride,
            args.fmin, args.modal_fmax,
        )
        if frequency is None:
            frequency = current_frequency
        elif not np.array_equal(frequency, current_frequency):
            raise ValueError("Modal frequency grids differ")
        spectra[row.record_id] = current_spectrum
        if index % 24 == 0 or index == len(frame):
            print(f"modal spectra={index}/{len(frame)}", flush=True)

    final_cache, final_anchors = learn_modal_cache(
        frame, train, spectra, frequency, args
    )
    all_oof, all_thresholds, all_metrics, all_details, audits = [], [], [], [], []
    for model_name in models:
        for seed in seeds:
            print(f"\n=== model={model_name} seed={seed} ===", flush=True)
            oof, audit = cross_fitted_healthy_scores(
                train, spectra, frequency, model_name, seed, args, device
            )
            oof.insert(0, "model", model_name)
            oof.insert(1, "seed", seed)
            thresholds = fit_threshold_table(oof, model_name, seed, args)

            detector = build_detector(model_name, args, seed, device)
            train_x = np.concatenate([
                final_cache[key] for key in train.record_id
            ])
            detector.fit(train_x, None)
            references = fit_score_reference(
                detector, final_cache, train, args.record_quantile
            )
            for rows, split in ((val, "validation"), (test, "locked_test")):
                metrics, details = score_evaluation(
                    rows, final_cache, detector, references, thresholds,
                    model_name, seed, args, split,
                )
                all_metrics.append(metrics); all_details.append(details)
                primary = metrics[(metrics.method == "gev") &
                                  (metrics.scope == "excitation")].iloc[0]
                print(f"{split} GEV/excitation AUROC={primary.auroc:.4f} "
                      f"recall={primary.recall:.4f} "
                      f"specificity={primary.specificity:.4f}", flush=True)
            all_oof.append(oof); all_thresholds.append(thresholds)
            audits.append(audit)

    oof = pd.concat(all_oof, ignore_index=True)
    thresholds = pd.concat(all_thresholds, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    details = pd.concat(all_details, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)

    oof.to_csv(args.output / "healthy_grouped_oof_scores.csv", index=False)
    thresholds.to_csv(args.output / "calibration_thresholds.csv", index=False)
    metrics.to_csv(args.output / "calibration_metrics_by_seed.csv", index=False)
    details.to_csv(args.output / "calibration_record_scores.csv", index=False)
    audit.to_csv(args.output / "oof_anchor_audit.csv", index=False)
    pd.DataFrame({"modal_anchor_hz": final_anchors}).to_csv(
        args.output / "final_modal_anchors.csv", index=False
    )
    summary = metrics.groupby(
        ["evaluation", "method", "scope", "model"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"), auroc_mean=("auroc", "mean"),
        auprc_mean=("auprc", "mean"),
    )
    summary.to_csv(args.output / "calibration_summary.csv", index=False)
    save_diagnostics(oof, thresholds, metrics, args.output)

    metadata = {
        "paper": "Toshkova et al., MSSP 2020, DOI 10.1016/j.ymssp.2019.106417",
        "preprocessing": "locked modal_peak",
        "operational_regime": "manifest AVT/FVT excitation",
        "oof_group": "setup",
        "block_extreme": f"record quantile {args.record_quantile}",
        "literal_paper_block_maxima": args.record_quantile == 1.,
        "damage_used_for_fit_or_threshold": False,
        "validation_damage_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "primary_predeclared_method": "gev/excitation",
        "final_modal_anchors_hz": final_anchors.tolist(),
        "elapsed_seconds": time.time() - started,
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nCALIBRATION SUMMARY\n" + summary.to_string(index=False), flush=True)
    print("\nGEV FIT AUDIT\n" + thresholds[
        thresholds.method == "gev"
    ].to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

