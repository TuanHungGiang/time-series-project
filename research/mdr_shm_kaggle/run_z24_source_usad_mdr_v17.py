from __future__ import annotations

"""MDR-SHM V17: missing-sensor adaptation of the official/source USAD.

The control arm is the vendored EURECOM USAD implementation.  The proposed
arm starts from the exact same fitted USAD weights, then performs healthy-only
sensor-mask denoising.  Clean inference is forced to the original source USAD;
for missing-sensor regimes validation selects either source USAD or the
denoising branch and fixes the score tail and threshold before test.
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
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_missingness_baselines as missing  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, fit_usad, publication_metrics, score_usad, seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map,
)
from run_z24_mdr_v16 import threshold_under_fpr_cap, FPR_CAP  # noqa: E402
from third_party.usad import UsadModel  # noqa: E402


CONDITIONS = (
    "clean", "sensor_0.10", "sensor_0.30", "sensor_0.50",
    "targeted_0.30", "targeted_0.50",
)
SCENARIOS = (
    ("clean", "clean", 0.0, 0),
    ("sensor_0.10", "sensor", 0.10, 0),
    ("sensor_0.30", "sensor", 0.30, 0),
    ("sensor_0.50", "sensor", 0.50, 0),
    ("targeted_0.30", "targeted", 0.30, 0),
    ("targeted_0.50", "targeted", 0.50, 0),
)
METHODS = ("source_usad", "masked_denoising_usad")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def usad_target_losses(model, inputs, targets, epoch):
    latent = model.encoder(inputs)
    first = model.decoder1(latent)
    second = model.decoder2(latent)
    cycle = model.decoder2(model.encoder(first))
    weight = 1.0 / float(epoch)
    loss1 = weight * (targets - first).pow(2).mean()
    loss1 = loss1 + (1.0 - weight) * (targets - cycle).pow(2).mean()
    loss2 = weight * (targets - second).pow(2).mean()
    loss2 = loss2 - (1.0 - weight) * (targets - cycle).pow(2).mean()
    return loss1, loss2


def random_sensor_mask(batch, channels, importance_order, rng):
    availability = np.ones((batch, channels), dtype=np.float32)
    rates = (0.10, 0.30, 0.50)
    for index in range(batch):
        rate = rates[int(rng.integers(0, len(rates)))]
        count = min(channels - 1, max(1, int(round(rate * channels))))
        if rng.random() < 0.35:
            selected = importance_order[-count:]
        else:
            selected = rng.permutation(channels)[:count]
        availability[index, selected] = 0.0
    return availability


def fit_masked_usad(
    source: UsadModel,
    fit_scaled: np.ndarray,
    importance: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    denoise_weight: float,
):
    windows, channels, length = fit_scaled.shape
    model = UsadModel(channels * length, max(5, channels * length // 8)).to(device)
    model.load_state_dict(source.state_dict())
    flat = torch.from_numpy(fit_scaled.reshape(windows, -1))
    loader = DataLoader(TensorDataset(flat), batch_size=batch_size, shuffle=True)
    optimizer1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=2e-4,
    )
    optimizer2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=2e-4,
    )
    fill = torch.as_tensor(
        fit_scaled.mean(axis=(0, 2), keepdims=False),
        dtype=torch.float32,
        device=device,
    ).view(1, channels, 1)
    order = np.argsort(importance)
    rng = np.random.default_rng(seed)
    clean_weight = 1.0 - denoise_weight
    for epoch in range(1, epochs + 1):
        losses = []
        for (clean_flat,) in loader:
            clean_flat = clean_flat.to(device)
            clean = clean_flat.view(-1, channels, length)
            available_np = random_sensor_mask(len(clean), channels, order, rng)
            available = torch.as_tensor(available_np, device=device).unsqueeze(-1)
            masked = available * clean + (1.0 - available) * fill
            masked_flat = masked.reshape(len(clean), -1)

            optimizer1.zero_grad(set_to_none=True)
            clean1, _ = usad_target_losses(model, clean_flat, clean_flat, epoch)
            denoise1, _ = usad_target_losses(model, masked_flat, clean_flat, epoch)
            loss1 = clean_weight * clean1 + denoise_weight * denoise1
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            _, clean2 = usad_target_losses(model, clean_flat, clean_flat, epoch)
            _, denoise2 = usad_target_losses(model, masked_flat, clean_flat, epoch)
            loss2 = clean_weight * clean2 + denoise_weight * denoise2
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer2.step()
            losses.append(float((loss1 + loss2).detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  masked-USAD epoch={epoch}/{epochs} "
                f"loss={np.mean(losses):.6f}", flush=True,
            )
    return model.eval()


@torch.no_grad()
def score_masked_usad(
    model, values, availability, scale, fill_scaled, batch_size, device,
):
    scaled = scale.transform(values)
    filled = (
        availability[:, :, None] * scaled
        + (1.0 - availability[:, :, None]) * fill_scaled
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(filled), torch.from_numpy(availability.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    scores = []
    for batch, observed in loader:
        batch = batch.to(device)
        observed = observed.to(device)
        flat = batch.reshape(len(batch), -1)
        first = model.decoder1(model.encoder(flat))
        cycle = model.decoder2(model.encoder(first))
        error = 0.5 * (flat - first).pow(2) + 0.5 * (flat - cycle).pow(2)
        error = error.view(len(batch), batch.shape[1], batch.shape[2])
        weighted = error * observed.unsqueeze(-1)
        denominator = observed.sum(1).clamp_min(1.0) * batch.shape[2]
        scores.extend((weighted.sum((1, 2)) / denominator).cpu().numpy())
    return np.asarray(scores)


def build_cohort(records, cache, condition, kind, rate, replicate, importance):
    arrays, availability, counts, rows = [], [], [], []
    for row in records.itertuples():
        clean = cache[str(row.record_id)]
        current = missing.corrupt(
            clean, kind, rate, str(row.record_id), condition, replicate, importance,
        )
        observed = (~np.all(np.isclose(current, 0.0, atol=1e-8), axis=2)).astype(np.float32)
        arrays.append(current)
        availability.append(observed)
        counts.append(len(current))
        rows.append({
            "record_id": row.record_id, "split": row.split,
            "label": int(row.label), "state": row.state,
            "setup": row.setup, "excitation": row.excitation,
            "condition": condition, "replicate": replicate,
            "coverage": float(observed.mean()),
        })
    return np.concatenate(arrays), np.concatenate(availability), counts, rows


def add_record_scores(rows, method, scores, counts, reference, parameters):
    upper, two_sided = aggregate_record_scores(scores, counts, reference)
    frame = pd.DataFrame(rows)
    frame["method"] = method
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper, float)
    frame["score_two_sided"] = two_sided
    frame["parameters"] = parameters
    return frame


def run_group(args, group, labels, group_index, seed, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    normalizer = missing.Normalizer.fit(fit_raw)
    fit = normalizer.transform(fit_raw)
    internal = normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled = scale.transform(fit)
    internal_scaled = scale.transform(internal)
    fill_scaled = fit_scaled.mean(axis=(0, 2), keepdims=True).transpose(0, 1, 2)
    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed_everything(seed + group_index * 1009)
    source = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
    source_reference = robust_score_reference(
        score_usad(source, internal_scaled, args.batch_size, device)
    )
    seed_everything(seed + group_index * 1009 + 97)
    robust = fit_masked_usad(
        source, fit_scaled, importance, args.mask_epochs, args.batch_size,
        device, seed + group_index * 1009 + 97, args.denoise_weight,
    )
    all_available = np.ones(internal.shape[:2], dtype=np.float32)
    robust_reference = robust_score_reference(score_masked_usad(
        robust, internal, all_available, scale, fill_scaled,
        args.batch_size, device,
    ))
    parameters = {
        "source_usad": count_parameters(source),
        "masked_denoising_usad": count_parameters(robust),
    }
    frames = []
    for records in (validation, test):
        for condition, kind, rate, replicate in SCENARIOS:
            values, available, counts, rows = build_cohort(
                records, cache, condition, kind, rate, replicate, importance,
            )
            source_scores = score_usad(
                source, scale.transform(values), args.batch_size, device,
            )
            robust_scores = score_masked_usad(
                robust, values, available, scale, fill_scaled,
                args.batch_size, device,
            )
            frames.append(add_record_scores(
                rows, "source_usad", source_scores, counts,
                source_reference, parameters["source_usad"],
            ))
            frames.append(add_record_scores(
                rows, "masked_denoising_usad", robust_scores, counts,
                robust_reference, parameters["masked_denoising_usad"],
            ))
    result = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del source, robust
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose(validation):
    candidates = []
    for method in validation.method.unique():
        current = validation[validation.method == method]
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_under_fpr_cap(
                current.label, current[f"score_{tail}"], FPR_CAP,
            )
            key = (metric["ba"], metric["auc"], metric["recall"], -metric["fpr"])
            candidates.append((key, method, tail, threshold, metric))
    return max(candidates, key=lambda item: item[0])


def evaluate(frame, method, tail, threshold, condition, output_method):
    current = frame[(frame.method == method) & (frame.condition == condition)].copy()
    current["score"] = current[f"score_{tail}"]
    current["output_method"] = output_method
    current["source_method"] = method
    current["score_tail"] = tail
    current["threshold"] = threshold
    rows = []
    for replicate, cohort in current.groupby("replicate", sort=False):
        rows.append({
            "method": output_method, "source_method": method,
            "condition": condition, "replicate": int(replicate),
            "score_tail": tail, "threshold": threshold,
            **publication_metrics(cohort.label, cohort.score, threshold),
            "coverage_mean": float(cohort.coverage.mean()),
            "parameters": int(cohort.parameters.iloc[0]),
        })
    return current, rows


def finish(output, frame):
    validation = frame[frame.split == "val"]
    test = frame[frame.split == "test"]
    policies, metrics, scores = [], [], []
    for condition in CONDITIONS:
        current_val = validation[validation.condition == condition]
        source_choice = choose(current_val[current_val.method == "source_usad"])
        if condition == "clean":
            proposed_choice = source_choice
            reason = "forced_original_usad_clean"
        else:
            proposed_choice = choose(current_val)
            reason = "validation_locked_missing_sensor_branch"
        for name, choice in (
            ("source_usad", source_choice), ("proposed_v17", proposed_choice),
        ):
            _, method, tail, threshold, val_metric = choice
            scored, rows = evaluate(test, method, tail, threshold, condition, name)
            scores.append(scored)
            metrics.extend(rows)
            policies.append({
                "output_method": name, "condition": condition,
                "selected_method": method, "score_tail": tail,
                "threshold": threshold, "reason": reason,
                **{f"validation_{key}": value for key, value in val_metric.items()},
            })
    metrics = pd.DataFrame(metrics)
    scores = pd.concat(scores, ignore_index=True)
    policy = pd.DataFrame(policies)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ba_mean=("ba", "mean"),
        recall_mean=("recall", "mean"), specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"), precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"), parameters=("parameters", "mean"),
    )
    baseline = summary[summary.method == "source_usad"].set_index("condition")
    proposed = summary[summary.method == "proposed_v17"].set_index("condition")
    effects = proposed[["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]].subtract(
        baseline[["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]]
    )
    effects.columns = ["delta_auc", "delta_ba", "delta_recall", "delta_fpr"]
    effects = effects.reset_index()
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    policy.to_csv(output / "validation_policy.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "effects_vs_original_usad.csv", index=False)
    decision = {
        "promote_to_three_seed": bool(
            effects[effects.condition.str.startswith("sensor_")].delta_auc.mean() > 0.0
            and effects[effects.condition.str.startswith("sensor_")].delta_ba.mean() > 0.0
            and float(effects.loc[effects.condition == "clean", "delta_auc"].iloc[0]) >= 0.0
        ),
        "comparison": "proposed_v17 minus official/source USAD",
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )
    pivot = summary.pivot(index="condition", columns="method", values="auc_mean")
    axis = pivot.plot.bar(figsize=(12, 5.5), ylim=(0, 1.02), rot=20, grid=True)
    axis.set_ylabel("Record-level AUROC")
    axis.set_title("MDR-SHM V17 versus official/source USAD")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "v17_vs_source_usad.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== SOURCE USAD VS PROPOSED V17 ===")
    print(summary.to_string(index=False))
    print("\n=== PROPOSED MINUS SOURCE USAD ===")
    print(effects.to_string(index=False))
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261219)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mask-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--denoise-weight", type=float, default=0.70)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "groups").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.input_root)
    sensor_map = stable_sensor_map(manifest)
    used = manifest[manifest.split.isin(["train", "val", "test"])]
    print(
        f"device={device} cells={len(sensor_map)} seed={args.seed} "
        f"source_epochs={args.epochs} mask_epochs={args.mask_epochs}", flush=True,
    )
    for group_index, (key, labels) in enumerate(sorted(sensor_map.items())):
        path = group_file(args.output_dir, key[0], key[1])
        if path.exists():
            print(f"SKIP cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(args, group, labels, group_index, args.seed, device, path)
    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files, got {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    finish(args.output_dir, frame)
    protocol = {
        "dataset": "Z24", "method": "MDR-SHM V17 source-USAD anchored",
        "baseline": "vendored official EURECOM USAD",
        "fit": "healthy train only",
        "adaptation": "healthy-only random and targeted sensor-mask denoising",
        "selection": "condition-matched labeled validation only",
        "test_used_for_selection": False,
        "clean_branch": "forced original source USAD",
        "seed": args.seed, "source_epochs": args.epochs,
        "mask_epochs": args.mask_epochs, "denoise_weight": args.denoise_weight,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"SOURCE_USAD_MDR_V17_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

