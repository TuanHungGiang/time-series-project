from __future__ import annotations

"""CLMM V22 objective ablation with validation-only architecture selection.

Two adapters share the same frozen source USAD, identical initialization and
identical hard-mask random stream:

* CLMM-R  : masked cross-sensor reconstruction only;
* CLMM-RS : reconstruction plus frozen-USAD score consistency.

Both branches and the lower score tail are fixed.  The winning adapter is
selected globally using random-50 validation AUC (BA, recall and FPR are
predeclared tie-breakers) before any test metric is computed.  Labelled
validation is then used to calibrate operating thresholds under the same FPR
cap for source USAD and the selected CLMM adapter.  Test labels are evaluation
only.  Targeted-50 remains a stress test.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_care_usad_v18 as care  # noqa: E402
import run_z24_clmm_v19 as v19  # noqa: E402
import run_z24_clmm_v21 as v21  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax,
    Normalizer,
    fit_usad,
    publication_metrics,
    score_usad,
    seed_everything,
)
from run_z24_mdr_v16 import threshold_under_fpr_cap  # noqa: E402
from run_z24_source_baselines import (  # noqa: E402
    load_manifest,
    read_stable_vertical,
    record_windows,
    robust_score_reference,
    split_training_windows,
    stable_sensor_map,
)


CONDITIONS = care.CONDITIONS
CANDIDATES = ("clmm_r", "clmm_rs")


def candidate_frame(
    method,
    rows,
    scores,
    counts,
    reference,
    parameters,
):
    standardized = care.standardized(scores, reference)
    upper, two = care.aggregate_standardized(standardized, counts)
    frame = pd.DataFrame(rows)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two
    frame["parameters"] = parameters
    return frame


def fit_candidate(
    fit_scaled,
    importance,
    source,
    source_reference,
    args,
    device,
    seed,
    score_weight,
):
    seed_everything(seed)
    return v21.fit_adapter(
        fit_scaled,
        importance,
        source,
        source_reference,
        args.adapter_epochs,
        args.batch_size,
        args.embedding,
        args.heads,
        args.hard_candidates,
        0.0,
        score_weight,
        device,
        seed,
    )


def run_group(args, group, labels, group_index, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]

    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    normalizer = Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit)
    internal_scaled = scale.transform(internal)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels),
            args.raw_window,
            args.window_stride,
            "raw_temporal",
            args.downsample,
            128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed = args.seed + group_index * 1009
    seed_everything(seed)
    source = fit_usad(
        fit_scaled, args.epochs, args.batch_size, device,
    )
    source_reference = robust_score_reference(
        score_usad(source, internal_scaled, args.batch_size, device),
    )
    adapter_seed = seed + 97
    adapters = {
        "clmm_r": fit_candidate(
            fit_scaled,
            importance,
            source,
            source_reference,
            args,
            device,
            adapter_seed,
            0.0,
        ),
        "clmm_rs": fit_candidate(
            fit_scaled,
            importance,
            source,
            source_reference,
            args,
            device,
            adapter_seed,
            args.score_weight,
        ),
    }
    references = {}
    for offset, (method, adapter) in enumerate(adapters.items()):
        reference, _ = care.conditional_reference(
            adapter,
            source,
            internal,
            scale,
            importance,
            args.batch_size,
            device,
            seed + 193 + offset,
        )
        references[method] = reference

    source_parameters = care.count_parameters(source)
    candidate_parameters = {
        method: source_parameters + care.count_parameters(adapter)
        for method, adapter in adapters.items()
    }

    frames = []
    for records in (validation, test):
        for condition, kind, rate, replicate in care.SCENARIOS:
            values, available, counts, rows = care.build_cohort(
                records,
                cache,
                condition,
                kind,
                rate,
                replicate,
                importance,
            )
            scaled_values = scale.transform(values)
            source_scores = score_usad(
                source, scaled_values, args.batch_size, device,
            )
            frames.append(care.source_frame(
                rows,
                source_scores,
                counts,
                source_reference,
                source_parameters,
            ))

            for method, adapter in adapters.items():
                completed, _ = care.complete(
                    adapter,
                    scaled_values,
                    available,
                    args.batch_size,
                    device,
                )
                scores = score_usad(
                    source, completed, args.batch_size, device,
                )
                reference = (
                    source_reference if condition == "clean"
                    else references[method]
                )
                frames.append(candidate_frame(
                    method,
                    rows,
                    scores,
                    counts,
                    reference,
                    candidate_parameters[method],
                ))

    result = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del source, adapters
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validation_policy(frame, method, condition, fpr_cap):
    current = frame[
        (frame.split == "val")
        & (frame.method == method)
        & (frame.condition == condition)
    ].copy()
    threshold, metric = threshold_under_fpr_cap(
        current.label.to_numpy(),
        current.score_lower.to_numpy(),
        fpr_cap,
    )
    return threshold, metric, len(current)


def validation_ablation(frame, args):
    rows = []
    for method in ("source_usad", *CANDIDATES):
        for condition in CONDITIONS:
            threshold, metric, records = validation_policy(
                frame, method, condition, args.fpr_cap,
            )
            rows.append({
                "method": method,
                "condition": condition,
                "score_tail": "lower",
                "threshold": threshold,
                "fpr_cap": args.fpr_cap,
                "validation_records": records,
                **metric,
            })
    return pd.DataFrame(rows)


def select_candidate(validation_metrics):
    random50 = validation_metrics[
        (validation_metrics.condition == "sensor_0.50")
        & validation_metrics.method.isin(CANDIDATES)
    ].copy()
    random50["selection_key"] = list(zip(
        random50.auc,
        random50.ba,
        random50.recall,
        -random50.fpr,
    ))
    selected = random50.sort_values(
        ["auc", "ba", "recall", "fpr", "method"],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    return str(selected.method), random50.drop(columns="selection_key")


def test_method(frame, method, output_method, condition, policy_row):
    source_method = "source_usad" if condition == "clean" else method
    current = frame[
        (frame.split == "test")
        & (frame.method == source_method)
        & (frame.condition == condition)
    ].copy()
    rows, scores = [], []
    threshold = float(policy_row.threshold)
    for replicate, cohort in current.groupby("replicate"):
        rows.append({
            "method": output_method,
            "source_method": source_method,
            "condition": condition,
            "replicate": int(replicate),
            "score_tail": "lower",
            "threshold": threshold,
            **publication_metrics(
                cohort.label,
                cohort.score_lower,
                threshold,
            ),
            "records": len(cohort),
            "healthy_records": int((cohort.label == 0).sum()),
            "damage_records": int((cohort.label == 1).sum()),
            "coverage_mean": float(cohort.coverage.mean()),
            "parameters": int(cohort.parameters.iloc[0]),
        })
        scored = cohort.copy()
        scored["output_method"] = output_method
        scored["score"] = scored.score_lower
        scored["threshold"] = threshold
        scored["prediction"] = (
            scored.score >= threshold
        ).astype(int)
        scores.append(scored)
    return rows, scores


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


def paired_effects(metrics, selected):
    keys = ["condition", "replicate"]
    wide = metrics[
        metrics.method.isin(["source_usad", selected])
    ].pivot_table(
        index=keys,
        columns="method",
        values=["auc", "ba", "recall", "fpr", "f1"],
    )
    rows = []
    for index, values in wide.iterrows():
        row = dict(zip(keys, index))
        for metric in ("auc", "ba", "recall", "fpr", "f1"):
            row[f"delta_{metric}"] = float(
                values[(metric, selected)]
                - values[(metric, "source_usad")]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def finish(args, frame):
    validation_metrics = validation_ablation(frame, args)
    selected, selection_table = select_candidate(validation_metrics)
    print(
        f"VALIDATION_LOCKED_SELECTION selected={selected}",
        flush=True,
    )

    metric_rows, score_rows = [], []
    # Source and both candidates are evaluated for the ablation table only
    # after the global architecture selection above has been locked.
    for method in ("source_usad", *CANDIDATES):
        output_method = method
        for condition in CONDITIONS:
            policy_method = "source_usad" if condition == "clean" else method
            policy_row = validation_metrics[
                (validation_metrics.method == policy_method)
                & (validation_metrics.condition == condition)
            ].iloc[0]
            rows, scores = test_method(
                frame,
                method,
                output_method,
                condition,
                policy_row,
            )
            metric_rows.extend(rows)
            score_rows.extend(scores)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_rows, ignore_index=True)
    summary = summarize(metrics)
    effects = paired_effects(metrics, selected)
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
        "selected_candidate": selected,
        "selection_dataset": "labelled validation only",
        "selection_primary_metric": "random50 validation AUC",
        "promote_selected_to_three_seed": bool(all(criteria.values())),
        "criteria": criteria,
        "test_used_for_selection": False,
        "targeted50_role": "stress_test_only",
    }

    metrics.to_csv(args.output_dir / "ablation_test_metrics.csv", index=False)
    scores.to_csv(args.output_dir / "ablation_record_scores.csv", index=False)
    validation_metrics.to_csv(
        args.output_dir / "validation_ablation.csv", index=False,
    )
    selection_table.to_csv(
        args.output_dir / "validation_selection_table.csv", index=False,
    )
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    effects.to_csv(args.output_dir / "selected_paired_effects.csv", index=False)
    effect_summary.to_csv(
        args.output_dir / "selected_effect_summary.csv", index=False,
    )
    (args.output_dir / "selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    pivot = summary[
        summary.condition == "sensor_0.50"
    ].set_index("method")[["auc_mean", "ba_mean", "recall_mean"]]
    axis = pivot.plot.bar(
        figsize=(10, 5), ylim=(0, 1.02), rot=0, grid=True,
    )
    axis.set_ylabel("Record-level metric")
    axis.set_title("CLMM V22 random-50 objective ablation")
    axis.figure.tight_layout()
    axis.figure.savefig(
        args.output_dir / "clmm_v22_ablation.png", dpi=220,
    )
    plt.close(axis.figure)

    print("\n=== VALIDATION ABLATION ===")
    print(validation_metrics.to_string(index=False))
    print("\n=== VALIDATION SELECTION TABLE ===")
    print(selection_table.to_string(index=False))
    print("\n=== TEST ABLATION SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\n=== SELECTED {selected} MINUS SOURCE-USAD ===")
    print(effect_summary.to_string(index=False))
    print("\n=== SELECTION DECISION ===")
    print(json.dumps(decision, indent=2))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270819)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=4)
    parser.add_argument("--score-weight", type=float, default=0.10)
    parser.add_argument("--fpr-cap", type=float, default=0.20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} seed={args.seed} "
        f"source_epochs={args.epochs} adapter_epochs={args.adapter_epochs}",
        flush=True,
    )

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
        run_group(args, group, labels, index, device, path)

    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files, got {len(paths)}")
    frame = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True,
    )
    decision = finish(args, frame)
    protocol = {
        "dataset": "Z24",
        "method": "CLMM V22 objective ablation",
        "candidates": {
            "clmm_r": "reconstruction only",
            "clmm_rs": "reconstruction plus USAD-score consistency",
        },
        "shared_source_usad": True,
        "identical_adapter_initialization": True,
        "identical_mask_random_stream": True,
        "fixed_tail": "lower",
        "candidate_selection": "random50 labelled-validation AUC",
        "threshold_calibration": "labelled validation under FPR cap",
        "test_used_for_selection": False,
        "seed": args.seed,
        "decision": decision,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"CLMM_V22_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

