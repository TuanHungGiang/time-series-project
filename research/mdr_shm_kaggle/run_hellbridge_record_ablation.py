from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from run_hellbridge_b2_external import extract_window_spectra, fit_fold, parse_name
from run_z24_b2 import features


AGGREGATIONS = ("median", "q75", "q90", "top20_mean")
MODELS = ("b2", "ocsvm")


def aggregate(values: np.ndarray, method: str) -> float:
    values = np.asarray(values, dtype=float)
    if method == "median":
        return float(np.median(values))
    if method == "q75":
        return float(np.quantile(values, 0.75))
    if method == "q90":
        return float(np.quantile(values, 0.90))
    if method == "top20_mean":
        count = max(1, int(np.ceil(0.20 * len(values))))
        return float(np.mean(np.sort(values)[-count:]))
    raise ValueError(f"Unknown aggregation: {method}")


def score_windows(fold: dict, spectra: np.ndarray, frequencies: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.stack([features(item, frequencies, fold["anchors"]) for item in spectra])
    b2_raw = np.asarray([fold["b2"].raw_score(item) for item in matrix])
    ocsvm_raw = -fold["ocsvm"].decision_function(fold["scaler"].transform(matrix))
    return {
        "b2": fold["b2_calibration"].transform(b2_raw),
        "ocsvm": fold["ocsvm_calibration"].transform(ocsvm_raw),
    }


def add_cross_regime_conformal(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    output = frame.copy()
    pvalues = []
    calibration_sizes = []
    thresholds = []
    for row in output.itertuples(index=False):
        calibration = output.loc[
            (output.model == row.model)
            & (output.aggregation == row.aggregation)
            & (output.state == "UDS")
            & (output.regime != row.regime),
            "score",
        ].to_numpy(dtype=float)
        if len(calibration) == 0:
            raise RuntimeError(f"No external healthy calibration records for {row.regime}")
        # Conservative upper-tail inductive conformal p-value. Ties count against an alarm.
        pvalue = (1.0 + float(np.sum(calibration >= row.score))) / (len(calibration) + 1.0)
        pvalues.append(pvalue)
        calibration_sizes.append(len(calibration))
        thresholds.append(float(np.max(calibration)))
    output["conformal_p"] = pvalues
    output["calibration_records"] = calibration_sizes
    output["external_healthy_max"] = thresholds
    output["alarm"] = output.conformal_p <= alpha
    return output


def metric_dict(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(recall_score(1 - labels, ~predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "alarm_rate": float(np.mean(predictions)),
    }


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, aggregation), group in frame.groupby(["model", "aggregation"], sort=False):
        rows.append({
            "model": model,
            "aggregation": aggregation,
            "records": len(group),
            "healthy_records": int((group.label == 0).sum()),
            "damage_records": int((group.label == 1).sum()),
            **metric_dict(
                group.label.to_numpy(dtype=int),
                group.score.to_numpy(dtype=float),
                group.alarm.to_numpy(dtype=bool),
            ),
        })
    return pd.DataFrame(rows).sort_values(["model", "aggregation"]).reset_index(drop=True)


def alpha_sensitivity(frame: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        for (model, aggregation), group in frame.groupby(["model", "aggregation"], sort=False):
            prediction = group.conformal_p.to_numpy(dtype=float) <= alpha
            rows.append({
                "alpha": alpha,
                "model": model,
                "aggregation": aggregation,
                **metric_dict(
                    group.label.to_numpy(dtype=int),
                    group.score.to_numpy(dtype=float),
                    prediction,
                ),
            })
    return pd.DataFrame(rows)


def severity_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    states = sorted((value for value in frame.state.unique() if value != "UDS"), key=lambda x: int(x[2:]))
    for (model, aggregation), group in frame.groupby(["model", "aggregation"], sort=False):
        healthy = group[group.state == "UDS"]
        for state in states:
            damage = group[group.state == state]
            subset = pd.concat([healthy, damage], ignore_index=True)
            rows.append({
                "model": model,
                "aggregation": aggregation,
                "state": state,
                "damage_records": len(damage),
                "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
                "damage_alarm_rate": float(damage.alarm.mean()),
                "median_damage_score": float(damage.score.median()),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(frame: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for aggregation in AGGREGATIONS:
        subset = frame[frame.aggregation == aggregation]
        wide_score = subset.pivot(index="record_id", columns="model", values="score")
        wide_alarm = subset.pivot(index="record_id", columns="model", values="alarm")
        metadata = subset.drop_duplicates("record_id").set_index("record_id")
        record_ids = wide_score.index.to_numpy()
        labels = metadata.loc[record_ids, "label"].to_numpy(dtype=int)
        healthy = np.flatnonzero(labels == 0)
        damage = np.flatnonzero(labels == 1)
        b2_score = wide_score.loc[record_ids, "b2"].to_numpy(dtype=float)
        ocsvm_score = wide_score.loc[record_ids, "ocsvm"].to_numpy(dtype=float)
        b2_alarm = wide_alarm.loc[record_ids, "b2"].to_numpy(dtype=bool)
        ocsvm_alarm = wide_alarm.loc[record_ids, "ocsvm"].to_numpy(dtype=bool)

        def differences(indices: np.ndarray) -> tuple[float, float]:
            y = labels[indices]
            auc = roc_auc_score(y, b2_score[indices]) - roc_auc_score(y, ocsvm_score[indices])
            ba = balanced_accuracy_score(y, b2_alarm[indices]) - balanced_accuracy_score(y, ocsvm_alarm[indices])
            return float(auc), float(ba)

        estimates = differences(np.arange(len(labels)))
        draws = np.empty((samples, 2), dtype=float)
        for index in range(samples):
            sampled = np.concatenate([
                rng.choice(healthy, len(healthy), replace=True),
                rng.choice(damage, len(damage), replace=True),
            ])
            draws[index] = differences(sampled)
        for column, metric in enumerate(("auroc", "balanced_accuracy")):
            rows.append({
                "aggregation": aggregation,
                "metric": metric,
                "estimate_b2_minus_ocsvm": estimates[column],
                "ci_lower": float(np.quantile(draws[:, column], 0.025)),
                "ci_upper": float(np.quantile(draws[:, column], 0.975)),
                "probability_b2_better": float(np.mean(draws[:, column] > 0)),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(rows)


def make_figure(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(AGGREGATIONS))
    width = 0.36
    for offset, model in ((-width / 2, "b2"), (width / 2, "ocsvm")):
        ordered = metrics[metrics.model == model].set_index("aggregation").loc[list(AGGREGATIONS)]
        axes[0].bar(x + offset, ordered.auroc, width, label=model.upper())
        axes[1].bar(x + offset, ordered.balanced_accuracy, width, label=model.upper())
    for axis, title in zip(axes, ("Ranking performance", "Conformal alarm performance")):
        axis.set_xticks(x, AGGREGATIONS, rotation=25, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("Balanced accuracy")
    figure.suptitle("HellBridge temporal aggregation ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/hellbridge_record_ablation"),
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectra_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metadata: dict[str, dict[str, str]] = {}
    with h5py.File(args.h5, "r") as source:
        all_names = sorted(source.keys())
        selected_names = []
        for name in all_names:
            item = parse_name(name)
            if args.mode == "quick" and item["state"] not in ("UDS", "DS1"):
                continue
            selected_names.append(name)
            metadata[name] = item
            spectra_cache[name] = extract_window_spectra(source[name], item["axis"], args.window_seconds)
            print(f"loaded {len(selected_names)}/{len(all_names)} {name}", flush=True)

    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
            frequencies, train_spectra = spectra_cache[train_name]
            fold = fit_fold(train_spectra, frequencies)
            for name in [heldout] + damage:
                record_frequencies, record_spectra = spectra_cache[name]
                values = score_windows(fold, record_spectra, record_frequencies)
                for model, vector in values.items():
                    for method in AGGREGATIONS:
                        aggregated[name][model][method].append(aggregate(vector, method))
                    for window_index, score in enumerate(vector):
                        window_rows.append({
                            "record_id": name,
                            "regime": regime,
                            "fold": fold_index,
                            "model": model,
                            "window": window_index,
                            "normalized_window_score": float(score),
                        })
            fold_rows.append({
                "regime": regime,
                "fold": fold_index,
                "train_uds": train_name,
                "heldout_uds": heldout,
                "fit_windows": fold["fit_windows"],
                "calibration_windows": fold["calibration_windows"],
            })
        print(f"scored regime={regime}", flush=True)

    rows = []
    for name, item in sorted(metadata.items()):
        for model in MODELS:
            for method in AGGREGATIONS:
                fold_values = aggregated[name][model][method]
                rows.append({
                    **item,
                    "model": model,
                    "aggregation": method,
                    "score": float(np.mean(fold_values)),
                    "fold_scores": len(fold_values),
                })
    records = add_cross_regime_conformal(pd.DataFrame(rows), args.alpha)
    metrics = summarize_metrics(records)
    sensitivity = alpha_sensitivity(records, [0.12, 0.20, 0.25, 0.34, 0.50])
    severity = severity_metrics(records)
    bootstrap = paired_bootstrap(records, args.bootstrap_samples, 20261112)

    records.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "ablation_metrics.csv", index=False)
    sensitivity.to_csv(args.output_dir / "alpha_sensitivity.csv", index=False)
    severity.to_csv(args.output_dir / "severity_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(window_rows).to_csv(args.output_dir / "window_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "folds.csv", index=False)
    make_figure(metrics, args.output_dir / "fig_hellbridge_ablation.png")

    protocol = {
        "dataset": "Hell Bridge Test Arena",
        "mode": args.mode,
        "protocol": "regime-matched two-fold leave-one-UDS-record-out",
        "signal_preprocessing": "shared Z24 detrend + Welch PSD + 0.5-30 Hz sensor normalization",
        "window_seconds": args.window_seconds,
        "window_overlap": 0,
        "aggregations": list(AGGREGATIONS),
        "primary_aggregation": "q75",
        "primary_alpha": args.alpha,
        "record_calibration": "leave-target-regime-out healthy-only inductive conformal",
        "damage_labels_used_for_fit_model_selection_or_calibration": False,
        "minimum_conformal_p": "1/(8+1)=0.111... in full mode",
        "interpretation": "Ablation is exploratory because prior median HellBridge results were already observed.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print("\nPrimary rows (q75):")
    print(metrics[metrics.aggregation == "q75"].to_string(index=False))
    print(f"HELLBRIDGE_RECORD_ABLATION_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

