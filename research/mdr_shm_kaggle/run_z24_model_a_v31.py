from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from run_z24_b2 import (
    features,
    load_vertical,
    normalized_sensor_spectra,
    stable_rng,
)
from run_z24_model_a_v3 import (
    b2_score,
    calibrate_reliability,
    fit_b2,
    normalized_reliability,
    prepare_frame,
    reliability_features,
)


METRIC_NAMES = (
    "auroc",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
)
STANDARD_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)


def corrupt_v31(
    signals: np.ndarray,
    condition: str,
    record_id: str,
    replicate: int,
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Apply a deterministic-per-seed corruption with auditable sensor IDs.

    Targeted dropout is stochastic within the highest-evidence sensor pool.
    This keeps the attack evidence-targeted while making replicate IDs actual
    independent perturbation realizations instead of duplicate rows.
    """
    output = signals.copy()
    sensors, samples = output.shape
    availability = np.ones(sensors, dtype=bool)
    quality = np.ones(sensors, dtype=np.float32)
    if condition == "clean":
        return output, availability, quality, ""

    rate = float(condition.split("_")[-1])
    rng = stable_rng(record_id, f"selective_b2_v31_{condition}", replicate)
    affected_count = max(1, int(round(0.30 * sensors)))
    affected = np.sort(rng.choice(sensors, affected_count, replace=False))
    corrupted = affected

    if condition.startswith("missing"):
        length = int(round(rate * samples))
        coordinates = np.arange(samples)
        for sensor in affected:
            start = int(rng.integers(0, samples - length + 1))
            keep = np.ones(samples, dtype=bool)
            keep[start : start + length] = False
            output[sensor, ~keep] = np.interp(
                coordinates[~keep], coordinates[keep], output[sensor, keep]
            )
            quality[sensor] = 1.0 - rate
    elif condition.startswith("noise"):
        for sensor in affected:
            centered = output[sensor] - output[sensor].mean()
            power = float(np.mean(centered**2))
            sigma = np.sqrt(max(power, 1e-30) / 10 ** (rate / 10))
            output[sensor] += rng.normal(0.0, sigma, samples).astype(np.float32)
    elif condition.startswith("dropout"):
        count = min(sensors - 1, int(round(rate * sensors)))
        corrupted = np.sort(rng.choice(sensors, count, replace=False))
        availability[corrupted] = False
    elif condition.startswith("targeted"):
        frequencies, psd = normalized_sensor_spectra(output)
        evidence = np.asarray(
            [
                model.raw_score(features(item, frequencies, model.anchors))
                for item in psd
            ]
        )
        count = min(sensors - 1, int(round(rate * sensors)))
        pool_count = min(sensors, max(count + 1, int(np.ceil(1.5 * count))))
        pool = np.argsort(evidence)[-pool_count:]
        pool_evidence = evidence[pool]
        weights = pool_evidence - pool_evidence.min() + 1e-12
        weights = weights / weights.sum()
        corrupted = np.sort(
            rng.choice(pool, count, replace=False, p=weights)
        )
        availability[corrupted] = False
    else:
        raise ValueError(condition)

    signature = ";".join(str(int(item)) for item in corrupted)
    return output, availability, quality, signature


def safe_metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {name: float("nan") for name in METRIC_NAMES}
    predictions = scores > 1.0
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    false_positive = int(np.sum(predictions & negative))
    false_negative = int(np.sum(~predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    return {
        "auroc": binary_auroc(labels, scores),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(f1),
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with half credit for ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    differences = positive[:, None] - negative[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def bootstrap_intervals(
    group: pd.DataFrame,
    subset: str,
    samples: int,
    seed: int,
) -> list[dict]:
    labels = group.label.to_numpy(dtype=int)
    scores = group.score.to_numpy(dtype=float)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    if min(map(len, class_indices)) < 2 or samples <= 0:
        return []

    rng = np.random.default_rng(seed)
    boot = {name: [] for name in METRIC_NAMES}
    for _ in range(samples):
        draw = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        values = safe_metric_dict(labels[draw], scores[draw])
        for name in METRIC_NAMES:
            boot[name].append(values[name])

    estimates = safe_metric_dict(labels, scores)
    predictions = scores > 1.0
    positive = labels == 1
    negative = labels == 0
    exact_counts = {
        "recall": (int(np.sum(predictions & positive)), int(np.sum(positive))),
        "specificity": (
            int(np.sum(~predictions & negative)),
            int(np.sum(negative)),
        ),
    }
    rows = []
    for name in METRIC_NAMES:
        if name in exact_counts:
            successes, trials = exact_counts[name]
            lower, upper = clopper_pearson(successes, trials)
            method = "clopper_pearson_exact"
        else:
            lower, upper = np.quantile(boot[name], [0.025, 0.975])
            successes, trials = None, None
            method = "stratified_percentile_bootstrap"
        rows.append(
            {
                "subset": subset,
                "metric": name,
                "estimate": estimates[name],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "interval_method": method,
                "successes": successes,
                "trials": trials,
                "bootstrap_samples": samples,
                "healthy_records": int(len(class_indices[0])),
                "damage_records": int(len(class_indices[1])),
            }
        )
    return rows


def clopper_pearson(
    successes: int, trials: int, alpha: float = 0.05
) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def reliability_error_auroc(group: pd.DataFrame) -> float:
    prediction = group.score.to_numpy() > 1.0
    error = prediction != group.label.to_numpy(dtype=bool)
    if np.unique(error).size < 2:
        return float("nan")
    return binary_auroc(error.astype(int), group.reliability.to_numpy())


def risk_coverage_rows(group: pd.DataFrame) -> list[dict]:
    ordered = group.sort_values("reliability", kind="stable")
    output = []
    for requested in STANDARD_COVERAGES:
        count = max(2, int(np.ceil(len(ordered) * requested)))
        kept = ordered.iloc[:count]
        class_counts = kept.label.value_counts()
        valid = kept.label.nunique() == 2
        values = safe_metric_dict(kept.label.to_numpy(), kept.score.to_numpy())
        output.append(
            {
                "coverage": requested,
                "realized_coverage": float(len(kept) / len(ordered)),
                "records": int(len(kept)),
                "healthy_records": int(class_counts.get(0, 0)),
                "damage_records": int(class_counts.get(1, 0)),
                "valid_two_class_cohort": bool(valid),
                "balanced_risk": float(1.0 - values["balanced_accuracy"])
                if valid
                else float("nan"),
                **values,
            }
        )
    return output


def balanced_aurc(group: pd.DataFrame) -> float:
    ordered = group.sort_values("reliability", kind="stable")
    coverage = []
    risk = []
    for count in range(2, len(ordered) + 1):
        kept = ordered.iloc[:count]
        if kept.label.nunique() < 2:
            continue
        values = safe_metric_dict(kept.label.to_numpy(), kept.score.to_numpy())
        coverage.append(count / len(ordered))
        risk.append(1.0 - values["balanced_accuracy"])
    if len(coverage) < 2:
        return float("nan")
    span = coverage[-1] - coverage[0]
    if span <= 0:
        return float("nan")
    return float(np.trapz(risk, coverage) / span)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/model_a_v31"),
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261110)
    args = parser.parse_args()

    frame, manifest = prepare_frame(args.input_root)
    models = {}
    reliability_calibrations = {}
    for excitation_name in ("avt", "fvt"):
        rows = frame[frame.excitation_resolved == excitation_name]
        model = fit_b2(rows)
        models[excitation_name] = model
        reliability_calibrations[excitation_name] = calibrate_reliability(
            model, rows
        )
        print(
            f"{excitation_name}: score_threshold={model.threshold:.6f} "
            "reliability_threshold="
            f"{reliability_calibrations[excitation_name].threshold:.6f}",
            flush=True,
        )

    conditions = (
        ["clean", "missing_0.30", "noise_5", "dropout_0.30", "targeted_0.30"]
        if args.mode == "quick"
        else [
            "clean",
            "missing_0.10",
            "missing_0.30",
            "missing_0.50",
            "noise_20",
            "noise_10",
            "noise_5",
            "dropout_0.10",
            "dropout_0.30",
            "dropout_0.50",
            "targeted_0.10",
            "targeted_0.30",
            "targeted_0.50",
        ]
    )

    output = []
    test = frame[frame.role == "test"]
    for index, (_, row) in enumerate(test.iterrows(), 1):
        signals = load_vertical(Path(row.resolved_path))
        model = models[row.excitation_resolved]
        reliability_calibration = reliability_calibrations[row.excitation_resolved]
        for condition in conditions:
            if condition == "clean":
                repeats = 1
            elif args.mode == "quick" and condition.startswith("targeted"):
                # The quick gate must verify that the repaired targeted attack
                # really produces distinct seeded realizations.
                repeats = 3
            else:
                repeats = 1 if args.mode == "quick" else 3
            for replicate in range(repeats):
                changed, availability, quality, signature = corrupt_v31(
                    signals, condition, str(row.record_id), replicate, model
                )
                frequencies, psd = normalized_sensor_spectra(changed)
                score = b2_score(model, frequencies, psd, availability)
                reliability_vector, details = reliability_features(
                    model, frequencies, psd, availability, quality
                )
                reliability = normalized_reliability(
                    reliability_vector, reliability_calibration
                )
                output.append(
                    {
                        "record_id": row.record_id,
                        "label": int(row.label),
                        "excitation": row.excitation_resolved,
                        "condition": condition,
                        "replicate": replicate,
                        "corrupted_sensors": signature,
                        "score": score,
                        "reliability": reliability,
                        "accepted": reliability <= 1.0,
                        **details,
                    }
                )
        if index % 36 == 0:
            print(f"test={index}/{len(test)}", flush=True)

    scores = pd.DataFrame(output)
    metrics = []
    risk_coverage = {}
    confidence_intervals = []
    for group_index, ((condition, replicate), group) in enumerate(
        scores.groupby(["condition", "replicate"], sort=True)
    ):
        accepted = group[group.accepted]
        healthy = group.label == 0
        damage = group.label == 1
        accepted_counts = accepted.label.value_counts()
        accepted_valid = accepted.label.nunique() == 2
        entry = {
            "condition": condition,
            "replicate": int(replicate),
            **safe_metric_dict(
                group.label.to_numpy(), group.score.to_numpy()
            ),
            "records": int(len(group)),
            "healthy_records": int(healthy.sum()),
            "damage_records": int(damage.sum()),
            "accepted_records": int(len(accepted)),
            "accepted_healthy_records": int(accepted_counts.get(0, 0)),
            "accepted_damage_records": int(accepted_counts.get(1, 0)),
            "acceptance_rate": float(group.accepted.mean()),
            "healthy_acceptance_rate": float(group.loc[healthy, "accepted"].mean()),
            "damage_acceptance_rate": float(group.loc[damage, "accepted"].mean()),
            "accepted_valid_two_class_cohort": bool(accepted_valid),
            "accepted_status": "valid" if accepted_valid else "invalid_missing_class",
            "accepted_low_class_support": bool(
                accepted_valid and int(accepted_counts.min()) < 5
            ),
            "reliability_error_auroc": reliability_error_auroc(group),
            "balanced_aurc": balanced_aurc(group),
        }
        entry["false_alarm_rate"] = 1.0 - entry["specificity"]
        accepted_metrics = safe_metric_dict(
            accepted.label.to_numpy(), accepted.score.to_numpy()
        )
        entry.update(
            {f"accepted_{key}": value for key, value in accepted_metrics.items()}
        )
        entry["accepted_false_alarm_rate"] = (
            1.0 - accepted_metrics["specificity"]
            if accepted_valid
            else float("nan")
        )
        metrics.append(entry)

        curve = risk_coverage_rows(group)
        risk_coverage[f"{condition}_rep{replicate}"] = curve
        seed = args.bootstrap_seed + group_index * 1009
        for subset_name, subset in (("all", group), ("accepted", accepted)):
            rows = bootstrap_intervals(
                subset, subset_name, args.bootstrap_samples, seed
            )
            for item in rows:
                item.update(
                    {"condition": condition, "replicate": int(replicate)}
                )
            confidence_intervals.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "record_scores.csv", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(confidence_intervals).to_csv(
        args.output_dir / "bootstrap_ci.csv", index=False
    )
    risk_rows = []
    for key, rows in risk_coverage.items():
        condition, replicate = key.rsplit("_rep", 1)
        for row in rows:
            risk_rows.append(
                {"condition": condition, "replicate": int(replicate), **row}
            )
    pd.DataFrame(risk_rows).to_csv(
        args.output_dir / "risk_coverage.csv", index=False
    )
    (args.output_dir / "risk_coverage.json").write_text(
        json.dumps(risk_coverage, indent=2)
    )

    artifact = {
        "protocol": "model-a-v3.1-selective-b2",
        "manifest": str(manifest),
        "mode": args.mode,
        "selection_uses_damage": False,
        "evaluation_uses_damage_labels": True,
        "targeted_attack": "weighted sample from top-1.5x evidence pool",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "models": {key: value.as_dict() for key, value in models.items()},
        "reliability_calibrations": {
            key: {
                "center": value.center.tolist(),
                "scale": value.scale.tolist(),
                "threshold": value.threshold,
            }
            for key, value in reliability_calibrations.items()
        },
    }
    (args.output_dir / "model.json").write_text(json.dumps(artifact, indent=2))
    print(metrics_frame.to_string(index=False))
    print(f"MODEL_A_V31_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

