from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def safe_metrics(labels: pd.Series, scores: pd.Series, predictions: pd.Series) -> dict[str, float]:
    y = labels.to_numpy(dtype=int)
    s = scores.to_numpy(dtype=float)
    p = predictions.to_numpy(dtype=bool)
    if len(y) == 0:
        return {key: float("nan") for key in ("auroc", "recall", "specificity", "balanced_accuracy", "f1")}
    auroc = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")
    recall = float(recall_score(y, p, zero_division=0)) if np.any(y == 1) else float("nan")
    specificity = float(recall_score(1 - y, ~p, zero_division=0)) if np.any(y == 0) else float("nan")
    balanced = float((recall + specificity) / 2.0) if np.isfinite(recall) and np.isfinite(specificity) else float("nan")
    return {
        "auroc": auroc,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "f1": float(f1_score(y, p, zero_division=0)) if np.any(y == 1) else float("nan"),
    }


def find_z24_manifest(root: Path | None) -> Path | None:
    if root is None or not root.exists():
        return None
    candidates = []
    for path in root.rglob("*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=2).columns)
        except Exception:
            continue
        if {"record_id", "label"}.issubset(columns) and ({"role", "split"} & columns):
            candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        candidates.sort(key=lambda path: ("manifest" not in path.name.lower(), len(str(path))))
        return candidates[0]
    return None


def add_manifest_columns(scores: pd.DataFrame, manifest_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    if manifest_path is None:
        return scores, []
    manifest = pd.read_csv(manifest_path)
    wanted = [
        column for column in (
            "record_id", "scenario", "evaluation_cohort", "layout_id", "setup", "role", "split"
        ) if column in manifest.columns
    ]
    if "record_id" not in wanted:
        return scores, []
    extra = [column for column in wanted if column != "record_id" and column not in scores.columns]
    if not extra:
        return scores, []
    merged = scores.merge(manifest[["record_id", *extra]].drop_duplicates("record_id"), on="record_id", how="left")
    return merged, extra


def write_log(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")
    print(text)


def format_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "(no rows)"
    shown = frame if columns is None else frame[[column for column in columns if column in frame.columns]]
    return shown.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def diagnose_z24(scores_path: Path, manifest_path: Path | None, output_dir: Path):
    frame = pd.read_csv(scores_path)
    required = {"record_id", "label", "condition", "replicate", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Z24 record_scores missing columns: {sorted(missing)}")
    frame, manifest_columns = add_manifest_columns(frame, manifest_path)
    frame["prediction"] = frame.score > 1.0
    if "accepted" in frame.columns:
        frame["accepted"] = as_bool(frame.accepted)
    else:
        frame["accepted"] = True
    frame["error"] = frame.prediction != frame.label.astype(bool)
    frame["error_type"] = np.where(
        (frame.label == 1) & ~frame.prediction, "false_negative",
        np.where((frame.label == 0) & frame.prediction, "false_positive", "correct"),
    )

    overall_rows = []
    for (condition, replicate), group in frame.groupby(["condition", "replicate"], sort=True):
        metrics = safe_metrics(group.label, group.score, group.prediction)
        accepted = group[group.accepted]
        accepted_metrics = safe_metrics(accepted.label, accepted.score, accepted.prediction)
        overall_rows.append({
            "condition": condition,
            "replicate": int(replicate),
            "records": len(group),
            "healthy": int((group.label == 0).sum()),
            "damage": int((group.label == 1).sum()),
            **metrics,
            "acceptance_rate": float(group.accepted.mean()),
            "healthy_acceptance": float(group.loc[group.label == 0, "accepted"].mean()),
            "damage_acceptance": float(group.loc[group.label == 1, "accepted"].mean()),
            **{f"accepted_{key}": value for key, value in accepted_metrics.items()},
        })
    overall = pd.DataFrame(overall_rows)

    subgroup_rows = []
    grouping_columns = [column for column in ("excitation", "scenario", "evaluation_cohort") if column in frame.columns]
    for group_column in grouping_columns:
        for (condition, replicate, value), group in frame.groupby(["condition", "replicate", group_column], dropna=False):
            metrics = safe_metrics(group.label, group.score, group.prediction)
            subgroup_rows.append({
                "group_type": group_column,
                "group": str(value),
                "condition": condition,
                "replicate": int(replicate),
                "records": len(group),
                "healthy": int((group.label == 0).sum()),
                "damage": int((group.label == 1).sum()),
                "detection_rate": float(group.prediction.mean()),
                "median_score": float(group.score.median()),
                "false_negatives": int(((group.label == 1) & ~group.prediction).sum()),
                "false_positives": int(((group.label == 0) & group.prediction).sum()),
                "acceptance_rate": float(group.accepted.mean()),
                **metrics,
            })
    subgroups = pd.DataFrame(subgroup_rows)

    clean = frame[(frame.condition == "clean") & (frame.replicate == 0)].copy()
    failures = frame[frame.error].sort_values(["condition", "replicate", "error_type", "score"])
    clean_failures = clean[clean.error].sort_values(["error_type", "score"])
    condition_summary = overall.groupby("condition", as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        acceptance_rate_mean=("acceptance_rate", "mean"),
        accepted_auroc_mean=("accepted_auroc", "mean"),
        accepted_recall_mean=("accepted_recall", "mean"),
    )

    overall.to_csv(output_dir / "z24_condition_replicate_metrics.csv", index=False)
    condition_summary.to_csv(output_dir / "z24_condition_summary.csv", index=False)
    subgroups.to_csv(output_dir / "z24_subgroup_metrics.csv", index=False)
    failures.to_csv(output_dir / "z24_error_records.csv", index=False)

    clean_metrics = safe_metrics(clean.label, clean.score, clean.prediction)
    imbalance = int((clean.label == 1).sum()) / max(int((clean.label == 0).sum()), 1)
    lines = [
        "Z24 DIAGNOSTIC LOG",
        "=" * 80,
        f"source={scores_path}",
        f"manifest={manifest_path}",
        f"manifest_columns_added={manifest_columns}",
        f"clean_records={len(clean)} healthy={(clean.label == 0).sum()} damage={(clean.label == 1).sum()} damage/healthy={imbalance:.1f}",
        "",
        "[1] CLEAN OVERALL",
        json.dumps(clean_metrics, indent=2),
        "",
        "[2] CONDITION SUMMARY",
        format_frame(condition_summary),
        "",
        "[3] CLEAN SUBGROUPS",
        format_frame(subgroups[(subgroups.condition == "clean") & (subgroups.replicate == 0)]),
        "",
        "[4] CLEAN ERROR RECORDS",
        format_frame(clean_failures, [
            "record_id", "label", "excitation", "scenario", "evaluation_cohort",
            "score", "prediction", "error_type", "reliability", "accepted",
        ]),
        "",
        "[5] AUTOMATIC GAP FLAGS",
    ]
    if imbalance >= 10:
        lines.append(f"- SEVERE CLASS IMBALANCE: damage/healthy={imbalance:.1f}; specificity and its CI are controlled by very few healthy records.")
    if clean_metrics["specificity"] >= 0.95 and clean_metrics["recall"] < 0.60:
        lines.append("- CONSERVATIVE THRESHOLD GAP: false alarms are suppressed, but most damage records remain below threshold.")
    if clean_metrics["auroc"] - clean_metrics["balanced_accuracy"] >= 0.05:
        lines.append("- RANKING-TO-ALARM GAP: ranking is materially better than thresholded balanced accuracy.")
    if not subgroups.empty and "excitation" in grouping_columns:
        clean_exc = subgroups[(subgroups.condition == "clean") & (subgroups.replicate == 0) & (subgroups.group_type == "excitation")]
        finite = clean_exc.auroc.dropna()
        if len(finite) >= 2 and finite.max() - finite.min() >= 0.10:
            lines.append(f"- EXCITATION DOMAIN GAP: clean subgroup AUROC range={finite.max() - finite.min():.3f}.")
    lines.append("- Reliability is useful only when accepted metrics improve at comparable coverage; inspect class-conditional acceptance before claiming benefit.")
    write_log(output_dir / "Z24_DIAGNOSTIC.log", lines)
    return frame, overall, subgroups, clean_metrics


def diagnose_hell(scores_path: Path, method: str, output_dir: Path):
    frame = pd.read_csv(scores_path)
    required = {"record_id", "label", "state", "regime", "score", "alarm"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"HellBridge record_scores missing columns: {sorted(missing)}")
    if "method" in frame.columns:
        frame = frame[frame.method == method].copy()
    elif "aggregation" in frame.columns:
        frame = frame[frame.aggregation == method].copy()
    if frame.empty:
        raise ValueError(f"No HellBridge rows for method={method}")
    frame["prediction"] = as_bool(frame.alarm)
    frame["error"] = frame.prediction != frame.label.astype(bool)
    frame["error_type"] = np.where(
        (frame.label == 1) & ~frame.prediction, "false_negative",
        np.where((frame.label == 0) & frame.prediction, "false_positive", "correct"),
    )
    overall_metrics = safe_metrics(frame.label, frame.score, frame.prediction)

    state_rows = []
    healthy = frame[frame.state == "UDS"]
    for state in sorted((value for value in frame.state.unique() if value != "UDS"), key=lambda value: int(value[2:])):
        damage = frame[frame.state == state]
        subset = pd.concat([healthy, damage])
        state_rows.append({
            "state": state,
            "damage_records": len(damage),
            "auroc_vs_uds": float(roc_auc_score(subset.label, subset.score)),
            "damage_detection_rate": float(damage.prediction.mean()),
            "false_negatives": int((~damage.prediction).sum()),
            "median_damage_score": float(damage.score.median()),
            "min_damage_score": float(damage.score.min()),
            "max_damage_score": float(damage.score.max()),
        })
    states = pd.DataFrame(state_rows)

    regime_rows = []
    for regime, group in frame.groupby("regime", sort=True):
        metrics = safe_metrics(group.label, group.score, group.prediction)
        regime_rows.append({
            "regime": regime,
            "records": len(group),
            "healthy": int((group.label == 0).sum()),
            "damage": int((group.label == 1).sum()),
            "false_negatives": int(((group.label == 1) & ~group.prediction).sum()),
            "false_positives": int(((group.label == 0) & group.prediction).sum()),
            "median_healthy_score": float(group.loc[group.label == 0, "score"].median()),
            "median_damage_score": float(group.loc[group.label == 1, "score"].median()),
            **metrics,
        })
    regimes = pd.DataFrame(regime_rows)
    failures = frame[frame.error].sort_values(["error_type", "regime", "state", "score"])

    states.to_csv(output_dir / "hell_state_metrics.csv", index=False)
    regimes.to_csv(output_dir / "hell_regime_metrics.csv", index=False)
    failures.to_csv(output_dir / "hell_error_records.csv", index=False)

    lines = [
        "HELLBRIDGE DIAGNOSTIC LOG",
        "=" * 80,
        f"source={scores_path}",
        f"method={method}",
        f"records={len(frame)} healthy={(frame.label == 0).sum()} damage={(frame.label == 1).sum()}",
        "",
        "[1] OVERALL",
        json.dumps(overall_metrics, indent=2),
        "",
        "[2] DAMAGE STATE DIAGNOSTICS",
        format_frame(states),
        "",
        "[3] REGIME DIAGNOSTICS",
        format_frame(regimes),
        "",
        "[4] ERROR RECORDS",
        format_frame(failures, [
            "record_id", "state", "regime", "position", "excitation", "axis",
            "score", "conformal_p", "prediction", "error_type",
        ]),
        "",
        "[5] AUTOMATIC GAP FLAGS",
    ]
    if overall_metrics["specificity"] >= 0.85 and overall_metrics["recall"] < 0.50:
        lines.append("- LOW-SENSITIVITY GAP: most damage records are missed while false alarms remain limited.")
    if overall_metrics["auroc"] - overall_metrics["balanced_accuracy"] >= 0.10:
        lines.append("- STRONG RANKING-TO-ALARM GAP: record scores rank damage better than the operational threshold detects it.")
    finite = regimes.auroc.dropna()
    if len(finite) >= 2 and finite.max() - finite.min() >= 0.20:
        lines.append(f"- REGIME DOMAIN GAP: AUROC range={finite.max() - finite.min():.3f}; excitation/axis-specific behavior dominates pooled performance.")
    weak_states = states[states.auroc_vs_uds < 0.70].state.tolist()
    if weak_states:
        lines.append(f"- WEAK DAMAGE STATES (AUROC<0.70): {weak_states}.")
    zero_detection = states[states.damage_detection_rate == 0].state.tolist()
    if zero_detection:
        lines.append(f"- ZERO-DETECTION STATES: {zero_detection}.")
    if (regimes.specificity < 1.0).any():
        lines.append(f"- FALSE ALARMS CONCENTRATED IN: {regimes.loc[regimes.specificity < 1.0, 'regime'].tolist()}.")
    write_log(output_dir / "HELLBRIDGE_DIAGNOSTIC.log", lines)
    return frame, states, regimes, overall_metrics


def cross_bridge_report(z24_metrics: dict, hell_metrics: dict, output_dir: Path) -> None:
    summary = pd.DataFrame([
        {"dataset": "Z24", **z24_metrics},
        {"dataset": "HellBridge", **hell_metrics},
    ])
    summary.to_csv(output_dir / "cross_bridge_summary.csv", index=False)
    lines = [
        "CROSS-BRIDGE GAP REPORT",
        "=" * 80,
        format_frame(summary),
        "",
        "INTERPRETATION",
        "- If both AUROCs are moderate but recall is low, the common weakness is sensitivity/calibration rather than one dataset alone.",
        "- If HellBridge varies strongly by regime while Z24 varies by excitation, domain normalization is the next representation gap.",
        "- If Z24 reliability improves corruption results but HellBridge reliability is untested, external reliability transfer is the main unfinished claim.",
        "- Do not optimize on pooled labels. Use these logs to predeclare one next hypothesis and retain a locked confirmation split/dataset.",
    ]
    write_log(output_dir / "CROSS_BRIDGE_GAPS.log", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z24-record-scores", type=Path, required=True)
    parser.add_argument("--hell-record-scores", type=Path, required=True)
    parser.add_argument("--z24-input-root", type=Path)
    parser.add_argument("--z24-manifest", type=Path)
    parser.add_argument("--hell-method", default="median_score")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/cross_bridge_diagnostic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.z24_manifest or find_z24_manifest(args.z24_input_root)
    _, _, _, z24_clean_metrics = diagnose_z24(args.z24_record_scores, manifest, args.output_dir)
    _, _, _, hell_metrics = diagnose_hell(args.hell_record_scores, args.hell_method, args.output_dir)
    cross_bridge_report(z24_clean_metrics, hell_metrics, args.output_dir)

    artifact = {
        "z24_record_scores": str(args.z24_record_scores),
        "z24_manifest": str(manifest) if manifest else None,
        "hell_record_scores": str(args.hell_record_scores),
        "hell_method": args.hell_method,
        "selection_uses_damage": False,
        "purpose": "diagnosis only; no model or threshold selection is performed",
    }
    (args.output_dir / "diagnostic_protocol.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"CROSS_BRIDGE_DIAGNOSTIC_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

