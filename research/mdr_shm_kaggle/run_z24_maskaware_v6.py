from __future__ import annotations

"""One-seed Z24 mask-aware representation screening.

``compact_raw`` is the clean-only Compact USAD control.  The proposed
``maskaware_consistency`` uses the same compact USAD dimensions, preceded by a
mask-aware cross-sensor recovery adapter.  It is trained on paired clean and
synthetically masked healthy views with missing-channel recovery and latent
consistency losses.  Test data never fit or select any component.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, publication_metrics, seed_everything,
)
from run_z24_mir_usad_v0 import scenarios, sensor_corrupt  # noqa: E402
from run_z24_recovery_kd_v3 import (  # noqa: E402
    CompactUSAD, RecoveryAdapter, anomaly_components, count_parameters,
    fit_compact_usad, score_compact, usad_losses,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map, threshold_from_labeled_validation,
)

METHODS = ("compact_raw", "maskaware_consistency")


class MaskAwareStudent(nn.Module):
    def __init__(self, channels, length, recovery_hidden, hidden1, hidden2, latent):
        super().__init__()
        self.channels = channels
        self.recovery = RecoveryAdapter(channels, recovery_hidden)
        self.usad = CompactUSAD(channels * length, hidden1, hidden2, latent)

    def parts(self, values, available):
        completed, predicted = self.recovery(values, available)
        latent, first, second, cycle = self.usad.parts(completed)
        return completed, predicted, latent, first, second, cycle


def paired_mask(batch, channels, importance, device):
    """20% clean, 40% random loss and 40% critical-sensor loss."""
    available = torch.ones(batch, channels, device=device)
    targeted = torch.as_tensor(
        np.argsort(importance)[::-1].copy(), dtype=torch.long, device=device,
    )
    modes = torch.rand(batch, device=device)
    random_order = torch.rand(batch, channels, device=device).argsort(1)
    for index in range(batch):
        if modes[index] < 0.20:
            continue
        rate = 0.30 if torch.rand((), device=device) < 0.5 else 0.50
        count = min(channels - 1, max(1, int(round(rate * channels))))
        selected = (
            random_order[index, :count]
            if modes[index] < 0.60 else targeted[:count]
        )
        available[index, selected] = 0.0
    return available


def missing_recovery_loss(clean, predicted, available):
    missing = 1.0 - available[:, :, None]
    denominator = (missing.sum() * clean.shape[-1]).clamp_min(1.0)
    return ((clean - predicted).pow(2) * missing).sum() / denominator


def latent_consistency(clean_latent, masked_latent, available):
    changed = available.mean(1) < 1.0 - 1e-8
    if not changed.any():
        return masked_latent.sum() * 0.0
    anchor = F.normalize(clean_latent[changed].detach(), dim=1)
    student = F.normalize(masked_latent[changed], dim=1)
    return (1.0 - (anchor * student).sum(1)).mean()


def fit_maskaware(
    fit, importance, epochs, batch_size, recovery_hidden,
    hidden1, hidden2, latent, recovery_weight, consistency_weight, device,
):
    channels, length = fit.shape[1:]
    model = MaskAwareStudent(
        channels, length, recovery_hidden, hidden1, hidden2, latent,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit)), batch_size=batch_size, shuffle=True,
    )
    shared = list(model.recovery.parameters()) + list(model.usad.encoder.parameters())
    parameters1 = shared + list(model.usad.decoder1.parameters())
    parameters2 = shared + list(model.usad.decoder2.parameters())
    optimizer1 = torch.optim.Adam(parameters1, lr=1e-3)
    optimizer2 = torch.optim.Adam(parameters2, lr=1e-3)

    for epoch in range(1, epochs + 1):
        logs = []
        model.train()
        for (clean,) in loader:
            clean = clean.to(device)
            available = paired_mask(len(clean), channels, importance, device)
            observed = clean * available[:, :, None]
            full_mask = torch.ones_like(available)

            optimizer1.zero_grad(set_to_none=True)
            clean_completed, _, clean_latent, clean_first, clean_second, clean_cycle = (
                model.parts(clean, full_mask)
            )
            completed, predicted, masked_latent, first, second, cycle = model.parts(
                observed, available,
            )
            clean_loss1, _ = usad_losses(
                clean_completed, clean_first, clean_second, clean_cycle, epoch,
            )
            masked_loss1, _ = usad_losses(completed, first, second, cycle, epoch)
            recovery = missing_recovery_loss(clean, predicted, available)
            consistency = latent_consistency(clean_latent, masked_latent, available)
            loss1 = (
                0.5 * (clean_loss1 + masked_loss1)
                + recovery_weight * recovery
                + consistency_weight * consistency
            )
            loss1.backward()
            nn.utils.clip_grad_norm_(parameters1, 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            clean_completed, _, clean_latent, clean_first, clean_second, clean_cycle = (
                model.parts(clean, full_mask)
            )
            completed, predicted, masked_latent, first, second, cycle = model.parts(
                observed, available,
            )
            _, clean_loss2 = usad_losses(
                clean_completed, clean_first, clean_second, clean_cycle, epoch,
            )
            _, masked_loss2 = usad_losses(completed, first, second, cycle, epoch)
            recovery2 = missing_recovery_loss(clean, predicted, available)
            consistency2 = latent_consistency(clean_latent, masked_latent, available)
            loss2 = (
                0.5 * (clean_loss2 + masked_loss2)
                + 0.5 * recovery_weight * recovery2
                + 0.5 * consistency_weight * consistency2
            )
            loss2.backward()
            nn.utils.clip_grad_norm_(parameters2, 5.0)
            optimizer2.step()
            logs.append({
                "loss": float((loss1 + loss2).detach().cpu()),
                "recovery": float(recovery.detach().cpu()),
                "consistency": float(consistency.detach().cpu()),
            })

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            row = pd.DataFrame(logs).mean()
            print(
                f"  maskaware epoch={epoch}/{epochs} loss={row.loss:.6f} "
                f"recovery={row.recovery:.6f} consistency={row.consistency:.6f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def score_maskaware(model, values, available, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(available)),
        batch_size=batch_size,
    )
    scores = []
    for batch, mask in loader:
        batch, mask = batch.to(device), mask.to(device)
        completed, _, _, first, _, cycle = model.parts(batch, mask)
        _, score = anomaly_components(completed, first, cycle)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def build_values(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, metadata = [], [], [], []
    for row in records.itertuples():
        values, available = sensor_corrupt(
            cache[str(row.record_id)], kind, rate, str(row.record_id),
            replicate, importance,
        )
        arrays.append(values)
        masks.append(available)
        counts.append(len(values))
        metadata.append({
            "record_id": row.record_id,
            "split": row.split,
            "label": int(row.label),
            "state": row.state,
            "setup": row.setup,
            "excitation": row.excitation,
            "condition": condition,
            "replicate": replicate,
            "rate": rate,
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, metadata


def make_frame(method, scores, counts, metadata, reference, available, parameters):
    upper, two_sided = aggregate_record_scores(scores, counts, reference)
    frame = pd.DataFrame(metadata)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two_sided
    frame["windows"] = counts
    frame["parameters"] = parameters
    window_coverage = available.mean(1)
    start, coverage = 0, []
    for count in counts:
        coverage.append(float(window_coverage[start:start + count].mean()))
        start += count
    frame["coverage"] = coverage
    return frame


def group_file(output, setup, excitation):
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def run_group(group, labels, seed, group_index, args, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_by_record = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_by_record)
    normalizer = Normalizer.fit(fit_raw)
    fit_normalized = normalizer.transform(fit_raw)
    internal_normalized = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scaler = FitMinMax(fit_normalized)
    fit = scaler.transform(fit_normalized).astype(np.float32)
    internal = scaler.transform(internal_normalized).astype(np.float32)
    print(f"  fit={tuple(fit.shape)} internal={tuple(internal.shape)}", flush=True)

    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    base_seed = seed + group_index * 1009
    seed_everything(base_seed)
    compact = fit_compact_usad(
        fit, args.epochs, args.batch_size,
        args.hidden1, args.hidden2, args.latent, device,
    )
    compact_internal = score_compact(compact, internal, args.batch_size, device)

    seed_everything(base_seed + 97)
    maskaware = fit_maskaware(
        fit, importance, args.epochs, args.batch_size, args.recovery_hidden,
        args.hidden1, args.hidden2, args.latent,
        args.recovery_weight, args.consistency_weight, device,
    )
    internal_mask = np.ones(internal.shape[:2], dtype=np.float32)
    maskaware_internal = score_maskaware(
        maskaware, internal, internal_mask, args.batch_size, device,
    )
    references = {
        "compact_raw": robust_score_reference(compact_internal),
        "maskaware_consistency": robust_score_reference(maskaware_internal),
    }
    parameters = {
        "compact_raw": count_parameters(compact),
        "maskaware_consistency": count_parameters(maskaware),
    }
    print(f"  parameters={parameters}", flush=True)

    cohorts = [(validation, "clean", "clean", 0.0, 0)]
    cohorts.extend([
        (test, condition, kind, rate, replicate)
        for condition, kind, rate, replicate in scenarios()
    ])
    frames = []
    for records, condition, kind, rate, replicate in cohorts:
        values, available, counts, metadata = build_values(
            records, cache, condition, kind, rate, replicate, importance,
        )
        scaled = scaler.transform(values).astype(np.float32)
        raw_score = score_compact(compact, scaled, args.batch_size, device)
        proposed_score = score_maskaware(
            maskaware, scaled, available.astype(np.float32),
            args.batch_size, device,
        )
        frames.extend([
            make_frame(
                "compact_raw", raw_score, counts, metadata,
                references["compact_raw"], available, parameters["compact_raw"],
            ),
            make_frame(
                "maskaware_consistency", proposed_score, counts, metadata,
                references["maskaware_consistency"], available,
                parameters["maskaware_consistency"],
            ),
        ])
    result = pd.concat(frames, ignore_index=True)
    result["seed"] = seed
    result["stable_sensors"] = len(labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del compact, maskaware
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish(output, frame):
    metric_rows, scored = [], []
    for method in METHODS:
        validation = frame[(frame.method == method) & (frame.split == "val")].copy()
        test = frame[(frame.method == method) & (frame.split == "test")].copy()
        candidates = []
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_from_labeled_validation(
                validation.label, validation[f"score_{tail}"],
            )
            candidates.append(((metric["ba"], metric["auc"], metric["f1"]), tail, threshold))
        _, tail, threshold = max(candidates, key=lambda item: item[0])
        test["score"] = test[f"score_{tail}"]
        test["threshold"] = threshold
        test["score_tail"] = tail
        scored.append(test)
        rows, clean = [], None
        for (condition, replicate), current in test.groupby(["condition", "replicate"]):
            row = {
                "method": method,
                "condition": condition,
                "replicate": int(replicate),
                "score_tail": tail,
                "threshold": threshold,
                **publication_metrics(current.label, current.score, threshold),
                "coverage_mean": float(current.coverage.mean()),
                "parameters": int(current.parameters.iloc[0]),
            }
            rows.append(row)
            if condition == "clean":
                clean = row
        for row in rows:
            row["delta_auc"] = row["auc"] - clean["auc"]
            row["delta_ba"] = row["ba"] - clean["ba"]
            row["delta_fpr"] = row["fpr"] - clean["fpr"]
        metric_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(scored, ignore_index=True)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        delta_auc_mean=("delta_auc", "mean"),
        delta_ba_mean=("delta_ba", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"), parameters=("parameters", "mean"),
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(11, 5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("Z24 mask-aware healthy-only representation")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "maskaware_auc.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== ONE-SEED SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


def decision(summary):
    table = summary.set_index(["method", "condition"])
    corruptions = ["sensor_0.30", "sensor_0.50", "targeted_0.50"]
    effects = {}
    for condition in ["clean", *corruptions]:
        proposed = table.loc[("maskaware_consistency", condition)]
        baseline = table.loc[("compact_raw", condition)]
        effects[condition] = {
            "delta_auc": float(proposed.auc_mean - baseline.auc_mean),
            "delta_ba": float(proposed.ba_mean - baseline.ba_mean),
            "delta_fpr": float(proposed.fpr_mean - baseline.fpr_mean),
            "delta_recall": float(proposed.recall_mean - baseline.recall_mean),
        }
    promote = bool(
        np.mean([effects[c]["delta_auc"] for c in corruptions]) > 0.0
        and np.mean([effects[c]["delta_ba"] for c in corruptions]) > 0.0
        and effects["targeted_0.50"]["delta_auc"] >= 0.0
        and effects["clean"]["delta_auc"] >= -0.02
    )
    return {
        "effects_proposed_minus_raw": effects,
        "promote_to_three_seeds": promote,
        "decision": (
            "PROMOTE mask-aware consistency recovery to a three-seed pilot."
            if promote else
            "STOP after one seed; representation training did not jointly improve corrupted AUC and BA."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--recovery-hidden", type=int, default=48)
    parser.add_argument("--recovery-weight", type=float, default=0.20)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} methods={METHODS} "
        f"seed={args.seed} epochs={args.epochs}", flush=True,
    )
    started = time.time()
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, *key)
        if path.exists():
            print(f"SKIP complete cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(group, labels, args.seed, group_index, args, device, path)

    paths = [group_file(args.output_dir, *key) for key in sorted(sensor_map)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(METHODS) * (132 + 132 * len(scenarios()))
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(frame)}")
    keys = ["method", "record_id", "split", "condition", "replicate"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate method/record/condition keys")
    summary = finish(args.output_dir, frame)
    screening = decision(summary)
    (args.output_dir / "decision.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8",
    )
    protocol = {
        "experiment": "Z24 mask-aware recovery and latent consistency v6",
        "seed": args.seed, "epochs": args.epochs,
        "fit": "healthy train only",
        "selection": "clean labeled validation only",
        "test_used_for_fit_or_selection": False,
        "methods": list(METHODS),
        "training_masks": "20% clean, 40% random, 40% critical; 30/50% loss",
        "loss_weights": {
            "recovery": args.recovery_weight,
            "consistency": args.consistency_weight,
        },
        "conditions": scenarios(), "seconds": time.time() - started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print("\n=== SCREENING DECISION ===", flush=True)
    print(json.dumps(screening, indent=2), flush=True)
    print(f"MASKAWARE_V6_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

