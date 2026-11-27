from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from run_hellbridge_b2_external import extract_window_spectra, fit_fold, parse_name
from run_hellbridge_record_ablation import add_cross_regime_conformal
from run_z24_b2 import features


METHODS = ("median_score", "exceedance_fraction", "bdc_surprisal", "ewma_q90")


def normalized_window_scores(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    matrix = np.stack([features(item, frequencies, fold["anchors"]) for item in spectra])
    raw = np.asarray([fold["b2"].raw_score(item) for item in matrix])
    return fold["b2_calibration"].transform(raw)


def temporal_statistics(scores: np.ndarray, calibration_windows: int, ewma_lambda: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    exceedances = scores > 1.0
    count = int(exceedances.sum())
    total = len(scores)

    # With a max-calibration threshold, zero calibration exceedances are expected.
    # 1/(m+1) is the finite-sample upper-tail probability implied by conformal ranks.
    null_probability = 1.0 / (calibration_windows + 1.0)
    tail_probability = float(binom.sf(count - 1, total, null_probability)) if count else 1.0

    ewma = np.empty_like(scores)
    ewma[0] = scores[0]
    for index in range(1, len(scores)):
        ewma[index] = ewma_lambda * scores[index] + (1.0 - ewma_lambda) * ewma[index - 1]

    return {
        "median_score": float(np.median(scores)),
        "exceedance_fraction": float(count / total),
        "bdc_surprisal": float(-np.log10(max(tail_probability, np.finfo(float).tiny))),
        "ewma_q90": float(np.quantile(ewma, 0.90)),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=False):
        labels = group.label.to_numpy(dtype=int)
        predictions = group.alarm.to_numpy(dtype=bool)
        rows.append({
            "method": method,
            "records": len(group),
            "healthy_records": int((labels == 0).sum()),
            "damage_records": int((labels == 1).sum()),
            "auroc": float(roc_auc_score(labels, group.score)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "alarm_rate": float(predictions.mean()),
        })
    return pd.DataFrame(rows).set_index("method").loc[list(METHODS)].reset_index()


def severity_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    states = sorted((x for x in frame.state.unique() if x != "UDS"), key=lambda x: int(x[2:]))
    for method, group in frame.groupby("method", sort=False):
        healthy = group[group.state == "UDS"]
        for state in states:
            damage = group[group.state == state]
            subset = pd.concat([healthy, damage])
            rows.append({
                "method": method,
                "state": state,
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
                "damage_alarm_rate": float(damage.alarm.mean()),
                "median_statistic": float(damage.score.median()),
            })
    return pd.DataFrame(rows)


def regime_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, regime), group in frame.groupby(["method", "regime"], sort=False):
        labels = group.label.to_numpy(dtype=int)
        predictions = group.alarm.to_numpy(dtype=bool)
        rows.append({
            "method": method,
            "regime": regime,
            "auroc": float(roc_auc_score(labels, group.score)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    wide_score = frame.pivot(index="record_id", columns="method", values="score")
    wide_alarm = frame.pivot(index="record_id", columns="method", values="alarm")
    metadata = frame.drop_duplicates("record_id").set_index("record_id").loc[wide_score.index]
    labels = metadata.label.to_numpy(dtype=int)
    healthy = np.flatnonzero(labels == 0)
    damage = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    baseline_score = wide_score.median_score.to_numpy(dtype=float)
    baseline_alarm = wide_alarm.median_score.to_numpy(dtype=bool)
    rows = []
    for method in METHODS[1:]:
        candidate_score = wide_score[method].to_numpy(dtype=float)
        candidate_alarm = wide_alarm[method].to_numpy(dtype=bool)

        def difference(indices: np.ndarray) -> tuple[float, float]:
            y = labels[indices]
            return (
                float(roc_auc_score(y, candidate_score[indices]) - roc_auc_score(y, baseline_score[indices])),
                float(balanced_accuracy_score(y, candidate_alarm[indices]) - balanced_accuracy_score(y, baseline_alarm[indices])),
            )

        estimate = difference(np.arange(len(labels)))
        draws = np.empty((samples, 2), dtype=float)
        for index in range(samples):
            sampled = np.concatenate([
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ])
            draws[index] = difference(sampled)
        for column, metric in enumerate(("auroc", "balanced_accuracy")):
            rows.append({
                "candidate": method,
                "baseline": "median_score",
                "metric": metric,
                "estimate_candidate_minus_baseline": estimate[column],
                "ci_lower": float(np.quantile(draws[:, column], 0.025)),
                "ci_upper": float(np.quantile(draws[:, column], 0.975)),
                "probability_candidate_better": float(np.mean(draws[:, column] > 0)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    labels = ["Median", "Exceedance rate", "BDC surprisal", "EWMA q90"]
    colors = ["#5B6C8F", "#4C956C", "#F2B134", "#D95D39"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, metrics.auroc, color=colors)
    axes[1].bar(labels, metrics.balanced_accuracy, color=colors)
    for axis, ylabel in zip(axes, ("AUROC", "Balanced accuracy")):
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Temporal evidence ranking")
    axes[1].set_title("Healthy-only conformal alarms")
    figure.suptitle("HellBridge temporal alarm ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hellbridge_temporal_alarm"))
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--ewma-lambda", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    metadata = {}
    with h5py.File(args.h5, "r") as source:
        all_names = sorted(source.keys())
        for name in all_names:
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1"):
                continue
            metadata[name] = item
            cache[name] = extract_window_spectra(source[name], item["axis"], args.window_seconds)
            print(f"loaded {len(cache)}/{len(all_names)} {name}", flush=True)

    statistics = defaultdict(lambda: defaultdict(list))
    window_rows = []
    fold_rows = []
    regimes = sorted({item["regime"] for item in metadata.values()})
    for regime in regimes:
        healthy = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] == "UDS")
        damage = sorted(name for name, item in metadata.items() if item["regime"] == regime and item["state"] != "UDS")
        if len(healthy) != 2:
            raise ValueError(f"Expected two UDS recordings in {regime}, found {len(healthy)}")
        for fold_index, heldout in enumerate(healthy):
            train_name = healthy[1 - fold_index]
            frequencies, train_spectra = cache[train_name]
            fold = fit_fold(train_spectra, frequencies)
            for name in [heldout] + damage:
                record_frequencies, record_spectra = cache[name]
                scores = normalized_window_scores(fold, record_spectra, record_frequencies)
                values = temporal_statistics(scores, fold["calibration_windows"], args.ewma_lambda)
                for method, value in values.items():
                    statistics[name][method].append(value)
                for window, score in enumerate(scores):
                    window_rows.append({
                        "record_id": name,
                        "regime": regime,
                        "fold": fold_index,
                        "window": window,
                        "normalized_score": float(score),
                        "exceeds_window_threshold": bool(score > 1.0),
                    })
            fold_rows.append({
                "regime": regime,
                "fold": fold_index,
                "train_uds": train_name,
                "heldout_uds": heldout,
                "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
                "null_exceedance_probability": 1.0 / (fold["calibration_windows"] + 1.0),
            })
        print(f"scored regime={regime}", flush=True)

    rows = []
    for name, item in sorted(metadata.items()):
        for method in METHODS:
            rows.append({
                **item,
                "model": "b2",
                "aggregation": method,
                "method": method,
                "score": float(np.mean(statistics[name][method])),
                "fold_scores": len(statistics[name][method]),
            })
    records = add_cross_regime_conformal(pd.DataFrame(rows), args.alpha).drop(columns=["model", "aggregation"])
    metrics = summarize(records)
    severity = severity_table(records)
    regimes_frame = regime_table(records)
    bootstrap = paired_bootstrap(records, args.bootstrap_samples, 20261112)

    records.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "temporal_alarm_metrics.csv", index=False)
    severity.to_csv(args.output_dir / "severity_metrics.csv", index=False)
    regimes_frame.to_csv(args.output_dir / "regime_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(window_rows).to_csv(args.output_dir / "window_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    make_figure(metrics, args.output_dir / "fig_temporal_alarm.png")

    protocol = {
        "experiment": "Step 3 - temporal persistence and alarm stabilization",
        "scientific_basis": [
            "Binomial Distribution Classifier for SHM false-alarm reduction",
            "EWMA filtering and control-chart thresholding for bridge damage assessment",
        ],
        "frozen_components": [
            "global median sensor PSD",
            "full B2 13D representation",
            "Z24-compatible spectral preprocessing",
            "leave-target-regime-out record-level conformal calibration",
            f"alpha={args.alpha}",
        ],
        "changed_component": "temporal record statistic only",
        "methods": list(METHODS),
        "predeclared_candidate": "bdc_surprisal",
        "ewma_lambda": args.ewma_lambda,
        "damage_labels_used_for_fit_model_selection_or_calibration": False,
        "status": "exploratory HellBridge development; retain all variants and paired comparisons",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(f"HELLBRIDGE_TEMPORAL_ALARM_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

