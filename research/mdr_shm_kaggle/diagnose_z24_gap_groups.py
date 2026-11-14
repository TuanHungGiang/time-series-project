from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


RECORD_PATTERN = re.compile(
    r"z24_s(?P<state>\d+)_?(?P<excitation>avt|fvt)?_?setup(?P<setup>\d+)",
    flags=re.IGNORECASE,
)


def as_bool(value: object, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def safe_mean(values):
    values = finite(values)
    return mean(values) if values else math.nan


def safe_median(values):
    values = finite(values)
    return median(values) if values else math.nan


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        stop = index + 1
        while stop < len(pairs) and pairs[stop][0] == pairs[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:stop])
        index = stop
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_thresholds(model_json: Path) -> dict[str, float]:
    model = json.loads(model_json.read_text(encoding="utf-8"))
    models = model.get("models", model)
    thresholds = {}
    for excitation in ("avt", "fvt"):
        entry = models.get(excitation, {})
        if "threshold" not in entry:
            raise KeyError(f"No score threshold for {excitation} in {model_json}")
        thresholds[excitation] = float(entry["threshold"])
    return thresholds


def enrich(row, thresholds, clean_scores, scores_normalized):
    match = RECORD_PATTERN.search(row["record_id"])
    if match:
        row["state"] = int(match.group("state"))
        row["setup"] = int(match.group("setup"))
        row["excitation"] = (row.get("excitation") or match.group("excitation")).lower()
    else:
        row["state"] = -1
        row["setup"] = -1
        row["excitation"] = row.get("excitation", "unknown").lower()
    row["label"] = int(row["label"])
    row["replicate"] = int(row.get("replicate", 0))
    row["score"] = float(row["score"])
    row["raw_threshold"] = thresholds[row["excitation"]]
    row["threshold"] = 1.0 if scores_normalized else row["raw_threshold"]
    row["prediction"] = int(row["score"] > row["threshold"])
    row["correct"] = int(row["prediction"] == row["label"])
    row["accepted"] = as_bool(row.get("accepted"), default=True)
    reliability = row.get("reliability")
    row["reliability"] = float(reliability) if reliability not in (None, "") else math.nan
    row["margin_ratio"] = row["score"] / max(row["threshold"], 1e-12)
    clean = clean_scores.get(row["record_id"], row["score"])
    row["clean_score"] = clean
    row["score_retention"] = row["score"] / max(clean, 1e-12)
    row["score_delta"] = row["score"] - clean
    return row


def summarize(rows):
    labels = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    accepted = [row for row in rows if row["accepted"]]
    damage = [row for row in rows if row["label"] == 1]
    healthy = [row for row in rows if row["label"] == 0]
    tp = sum(row["prediction"] == 1 for row in damage)
    tn = sum(row["prediction"] == 0 for row in healthy)
    accepted_damage = [row for row in accepted if row["label"] == 1]
    accepted_healthy = [row for row in accepted if row["label"] == 0]
    return {
        "records": len(rows),
        "healthy_records": len(healthy),
        "damage_records": len(damage),
        "auroc": auroc(labels, [row["score"] for row in rows]),
        "recall": tp / len(damage) if damage else math.nan,
        "specificity": tn / len(healthy) if healthy else math.nan,
        "balanced_accuracy": 0.5 * (tp / len(damage) + tn / len(healthy)) if damage and healthy else math.nan,
        "acceptance_rate": len(accepted) / len(rows) if rows else math.nan,
        "damage_acceptance_rate": len(accepted_damage) / len(damage) if damage else math.nan,
        "healthy_acceptance_rate": len(accepted_healthy) / len(healthy) if healthy else math.nan,
        "accepted_recall": safe_mean(row["prediction"] == 1 for row in accepted_damage) if accepted_damage else math.nan,
        "accepted_specificity": safe_mean(row["prediction"] == 0 for row in accepted_healthy) if accepted_healthy else math.nan,
        "mean_margin_ratio": safe_mean(row["margin_ratio"] for row in rows),
        "median_score_retention": safe_median(row["score_retention"] for row in rows),
        "mean_score_delta": safe_mean(row["score_delta"] for row in rows),
        "errors": sum(label != prediction for label, prediction in zip(labels, predictions)),
        "false_negatives": len(damage) - tp,
        "false_positives": len(healthy) - tn,
    }


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [
        {**dict(zip(keys, group_key)), **summarize(group_rows)}
        for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))
    ]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-scores", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-scores", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.model_json)
    with args.record_scores.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    clean_scores = {
        row["record_id"]: float(row["score"])
        for row in raw
        if row.get("condition") == "clean" and int(row.get("replicate", 0)) == 0
    }
    rows = [enrich(dict(row), thresholds, clean_scores, not args.raw_scores) for row in raw]

    condition = grouped(rows, ["condition", "replicate"])
    condition_class = grouped(rows, ["condition", "replicate", "label"])
    excitation = grouped(rows, ["condition", "replicate", "excitation", "label"])
    setup = grouped(rows, ["condition", "replicate", "setup", "label"])
    damage_state = grouped(
        [row for row in rows if row["label"] == 1],
        ["condition", "replicate", "state", "excitation"],
    )
    failures = [
        {key: row[key] for key in (
            "record_id", "condition", "replicate", "state", "setup", "excitation",
            "label", "score", "threshold", "margin_ratio", "prediction", "accepted",
            "reliability", "score_retention", "score_delta",
        )}
        for row in rows if not row["correct"] or not row["accepted"]
    ]

    write_csv(args.output_dir / "condition_metrics.csv", condition)
    write_csv(args.output_dir / "condition_class_metrics.csv", condition_class)
    write_csv(args.output_dir / "excitation_class_metrics.csv", excitation)
    write_csv(args.output_dir / "setup_class_metrics.csv", setup)
    write_csv(args.output_dir / "damage_state_metrics.csv", damage_state)
    write_csv(args.output_dir / "failure_records.csv", failures)

    weakest_states = sorted(
        damage_state,
        key=lambda row: (
            row["recall"] if math.isfinite(row["recall"]) else 2.0,
            row["median_score_retention"] if math.isfinite(row["median_score_retention"]) else 2.0,
        ),
    )[:15]
    weakest_setups = sorted(
        [row for row in setup if row["label"] == 1],
        key=lambda row: row["recall"] if math.isfinite(row["recall"]) else 2.0,
    )[:15]

    lines = [
        "Z24 GAP DIAGNOSTIC",
        (
            f"score_threshold={'raw model thresholds' if args.raw_scores else '1.0 (normalized score)'}; "
            f"raw_thresholds avt={thresholds['avt']:.6f} fvt={thresholds['fvt']:.6f}"
        ),
        "", "[CONDITION SUMMARY]",
    ]
    for row in condition:
        lines.append(" ".join(
            f"{key}={fmt(row[key])}" for key in (
                "condition", "replicate", "records", "auroc", "recall", "specificity",
                "balanced_accuracy", "acceptance_rate", "accepted_recall",
                "median_score_retention", "false_negatives", "false_positives",
            )
        ))
    lines.extend(["", "[WEAKEST DAMAGE STATE / EXCITATION GROUPS]"])
    for row in weakest_states:
        lines.append(" ".join(
            f"{key}={fmt(row[key])}" for key in (
                "condition", "replicate", "state", "excitation", "records", "recall",
                "acceptance_rate", "accepted_recall", "median_score_retention",
                "mean_margin_ratio", "false_negatives",
            )
        ))
    lines.extend(["", "[WEAKEST DAMAGE SETUPS]"])
    for row in weakest_setups:
        lines.append(" ".join(
            f"{key}={fmt(row[key])}" for key in (
                "condition", "replicate", "setup", "records", "recall",
                "acceptance_rate", "accepted_recall", "median_score_retention",
                "false_negatives",
            )
        ))
    lines.extend([
        "", "INTERPRETATION:",
        "- low recall + low retention: corruption removes damage evidence; this is the core gap.",
        "- low recall + retention near 1 on clean: detector/threshold is weak even without corruption.",
        "- low acceptance + high accepted recall: abstention helps but coverage cost is high.",
        "- low accepted recall: reliability ranking fails to reject the harmful cases.",
    ])
    log = "\n".join(lines) + "\n"
    (args.output_dir / "diagnostic.log").write_text(log, encoding="utf-8")
    print(log)
    print(f"Z24_GAP_DIAGNOSTIC_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
