from __future__ import annotations

"""CARE-USAD V18 screening for severe 50% sensor unavailability.

The vendored official USAD is fitted exactly as the control and then frozen.
A lightweight cross-sensor attention adapter is fitted on healthy data only.
It learns random-50% and criticality-targeted-50% completion.  Missing-regime
scores use mask-conditioned healthy references, while clean inference is
forced to the untouched source USAD.  Validation locks branch/tail/threshold;
test labels are evaluation-only.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_missingness_baselines as missing  # noqa: E402
from run_hellbridge_source_baselines import (  # noqa: E402
    FitMinMax, Normalizer, fit_usad, publication_metrics, score_usad,
    seed_everything,
)
from run_z24_source_baselines import (  # noqa: E402
    aggregate_record_scores, load_manifest, read_stable_vertical,
    record_windows, robust_score_reference, split_training_windows,
    stable_sensor_map,
)
from run_z24_mdr_v16 import FPR_CAP, threshold_under_fpr_cap  # noqa: E402


CONDITIONS = ("clean", "sensor_0.50", "targeted_0.50")
SCENARIOS = (
    ("clean", "clean", 0.0, 0),
    *(("sensor_0.50", "sensor", 0.50, replicate) for replicate in range(5)),
    ("targeted_0.50", "targeted", 0.50, 0),
)
CANDIDATES = ("source_usad", "imputed_usad", "care_fusion")


class CrossSensorAdapter(nn.Module):
    def __init__(self, channels: int, length: int, embedding: int, heads: int):
        super().__init__()
        self.channels = channels
        self.length = length
        self.encoder = nn.Sequential(
            nn.Linear(length, 128), nn.GELU(), nn.Linear(128, embedding),
        )
        self.sensor_embedding = nn.Parameter(torch.randn(channels, embedding) * 0.02)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, embedding))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding, nhead=heads, dim_feedforward=embedding * 4,
            dropout=0.10, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(embedding, 128), nn.GELU(), nn.Linear(128, length),
        )

    def forward(self, values: torch.Tensor, available: torch.Tensor):
        encoded = self.encoder(values) + self.sensor_embedding.unsqueeze(0)
        token = self.missing_token + self.sensor_embedding.unsqueeze(0)
        encoded = available.unsqueeze(-1) * encoded + (1.0 - available.unsqueeze(-1)) * token
        return self.decoder(self.context(encoded))


def count_parameters(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def group_file(output: Path, setup: int, excitation: str) -> Path:
    return output / "groups" / f"setup{setup:02d}_{excitation}.csv"


def masks_50(batch: int, channels: int, importance: np.ndarray, rng, targeted_probability=0.5):
    count = min(channels - 1, max(1, int(round(0.50 * channels))))
    order = np.argsort(importance)
    available = np.ones((batch, channels), dtype=np.float32)
    for index in range(batch):
        if rng.random() < targeted_probability:
            selected = order[-count:]
        else:
            selected = rng.permutation(channels)[:count]
        available[index, selected] = 0.0
    return available


def loss_per_sample(adapter, clean, available):
    masked = clean * available.unsqueeze(-1)
    reconstruction = adapter(masked, available)
    missing = 1.0 - available
    missing_denominator = (missing.sum(1) * clean.shape[2]).clamp_min(1.0)
    missing_error = (
        (reconstruction - clean).pow(2) * missing.unsqueeze(-1)
    ).sum((1, 2)) / missing_denominator
    observed_denominator = (available.sum(1) * clean.shape[2]).clamp_min(1.0)
    observed_error = (
        (reconstruction - clean).pow(2) * available.unsqueeze(-1)
    ).sum((1, 2)) / observed_denominator
    return missing_error + 0.10 * observed_error


def hardest_mask(adapter, clean, importance, rng, candidates):
    options, errors = [], []
    with torch.no_grad():
        for _ in range(candidates):
            current = torch.as_tensor(
                masks_50(len(clean), clean.shape[1], importance, rng),
                device=clean.device,
            )
            options.append(current)
            errors.append(loss_per_sample(adapter, clean, current))
    selection = torch.stack(errors).argmax(0)
    stacked = torch.stack(options)
    indices = torch.arange(len(clean), device=clean.device)
    return stacked[selection, indices]


def fit_adapter(fit_scaled, importance, epochs, batch_size, embedding, heads,
                candidates, device, seed):
    _, channels, length = fit_scaled.shape
    adapter = CrossSensorAdapter(channels, length, embedding, heads).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_scaled)),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        adapter.train()
        values = []
        for (clean,) in loader:
            clean = clean.to(device)
            available = hardest_mask(adapter, clean, importance, rng, candidates)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_per_sample(adapter, clean, available).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            values.append(float(loss.detach().cpu()))
        current = float(np.mean(values))
        if current < best:
            best = current
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"  CARE adapter epoch={epoch}/{epochs} loss={current:.6f}", flush=True)
    adapter.load_state_dict(best_state)
    return adapter.eval()


@torch.no_grad()
def complete(adapter, scaled, availability, batch_size, device):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled.astype(np.float32)),
            torch.from_numpy(availability.astype(np.float32)),
        ), batch_size=batch_size,
    )
    completed, consistency = [], []
    for values, observed in loader:
        values, observed = values.to(device), observed.to(device)
        reconstruction = adapter(values * observed.unsqueeze(-1), observed)
        output = observed.unsqueeze(-1) * values + (1.0 - observed.unsqueeze(-1)) * reconstruction
        denominator = (observed.sum(1) * values.shape[2]).clamp_min(1.0)
        error = (
            (reconstruction - values).pow(2) * observed.unsqueeze(-1)
        ).sum((1, 2)) / denominator
        completed.append(output.cpu().numpy())
        consistency.append(error.cpu().numpy())
    return np.concatenate(completed), np.concatenate(consistency)


def standardized(scores, reference):
    values = np.log1p(np.maximum(np.asarray(scores, float), 0.0))
    return (values - reference[0]) / reference[1]


def aggregate_standardized(values, counts):
    upper, two, start = [], [], 0
    for count in counts:
        current = values[start:start + count]
        upper.append(float(np.median(current)))
        two.append(float(np.median(np.abs(current))))
        start += count
    if start != len(values):
        raise RuntimeError("Score/count mismatch")
    return upper, two


def build_cohort(records, cache, condition, kind, rate, replicate, importance):
    arrays, masks, counts, rows = [], [], [], []
    for row in records.itertuples():
        clean = cache[str(row.record_id)]
        current = missing.corrupt(
            clean, kind, rate, str(row.record_id), condition, replicate, importance,
        )
        observed = (~np.all(np.isclose(current, 0.0, atol=1e-8), axis=2)).astype(np.float32)
        arrays.append(current)
        masks.append(observed)
        counts.append(len(current))
        rows.append({
            "record_id": row.record_id, "split": row.split,
            "label": int(row.label), "state": row.state,
            "setup": row.setup, "excitation": row.excitation,
            "condition": condition, "replicate": replicate,
            "coverage": float(observed.mean()),
        })
    return np.concatenate(arrays), np.concatenate(masks), counts, rows


def source_frame(rows, scores, counts, reference, parameters):
    upper, two = aggregate_record_scores(scores, counts, reference)
    frame = pd.DataFrame(rows)
    frame["method"] = "source_usad"
    frame["score_upper"] = upper
    frame["score_lower"] = -np.asarray(upper)
    frame["score_two_sided"] = two
    frame["parameters"] = parameters
    return frame


def proposed_frames(rows, usad_scores, consistency, counts, usad_reference,
                    consistency_reference, parameters, fusion_weight):
    usad_z = standardized(usad_scores, usad_reference)
    consistency_z = standardized(consistency, consistency_reference)
    outputs = []
    for method, values in (
        ("imputed_usad", usad_z),
        ("care_fusion", usad_z + fusion_weight * consistency_z),
    ):
        upper, two = aggregate_standardized(values, counts)
        frame = pd.DataFrame(rows)
        frame["method"] = method
        frame["score_upper"] = upper
        frame["score_lower"] = -np.asarray(upper)
        frame["score_two_sided"] = two
        frame["parameters"] = parameters
        outputs.append(frame)
    return outputs


def conditional_reference(adapter, source, internal, scale, importance,
                          batch_size, device, seed):
    rng = np.random.default_rng(seed)
    usad_values, consistency_values = [], []
    scaled = scale.transform(internal)
    for targeted_probability in (0.0, 0.5, 1.0):
        available = masks_50(
            len(internal), internal.shape[1], importance, rng, targeted_probability,
        )
        completed, consistency = complete(
            adapter, scaled, available, batch_size, device,
        )
        usad_values.append(score_usad(source, completed, batch_size, device))
        consistency_values.append(consistency)
    return (
        robust_score_reference(np.concatenate(usad_values)),
        robust_score_reference(np.concatenate(consistency_values)),
    )


def run_group(args, group, labels, group_index, device, output):
    train = group[(group.split == "train") & (group.label == 0)]
    validation = group[group.split == "val"]
    test = group[group.split == "test"]
    train_windows = [
        record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        ) for row in train.itertuples()
    ]
    fit_raw, internal_raw = split_training_windows(train_windows)
    normalizer = Normalizer.fit(fit_raw)
    fit, internal = normalizer.transform(fit_raw), normalizer.transform(internal_raw)
    importance = np.mean(np.square(fit_raw), axis=(0, 2))
    scale = FitMinMax(fit)
    fit_scaled, internal_scaled = scale.transform(fit), scale.transform(internal)
    cache = {}
    for row in pd.concat([validation, test]).itertuples():
        raw = record_windows(
            read_stable_vertical(Path(row.path), labels), args.raw_window,
            args.window_stride, "raw_temporal", args.downsample, 128,
        )
        cache[str(row.record_id)] = normalizer.transform(raw).astype(np.float32)

    seed = args.seed + group_index * 1009
    seed_everything(seed)
    source = fit_usad(fit_scaled, args.epochs, args.batch_size, device)
    source_reference = robust_score_reference(
        score_usad(source, internal_scaled, args.batch_size, device)
    )
    seed_everything(seed + 97)
    adapter = fit_adapter(
        fit_scaled, importance, args.adapter_epochs, args.batch_size,
        args.embedding, args.heads, args.hard_candidates, device, seed + 97,
    )
    conditional_usad, conditional_consistency = conditional_reference(
        adapter, source, internal, scale, importance,
        args.batch_size, device, seed + 193,
    )
    source_parameters = count_parameters(source)
    proposed_parameters = source_parameters + count_parameters(adapter)
    frames = []
    for records in (validation, test):
        for condition, kind, rate, replicate in SCENARIOS:
            values, available, counts, rows = build_cohort(
                records, cache, condition, kind, rate, replicate, importance,
            )
            baseline_scores = score_usad(
                source, scale.transform(values), args.batch_size, device,
            )
            frames.append(source_frame(
                rows, baseline_scores, counts, source_reference, source_parameters,
            ))
            completed, consistency = complete(
                adapter, scale.transform(values), available,
                args.batch_size, device,
            )
            imputed_scores = score_usad(source, completed, args.batch_size, device)
            reference_u = source_reference if condition == "clean" else conditional_usad
            reference_c = robust_score_reference(np.zeros(max(2, len(consistency)))) if condition == "clean" else conditional_consistency
            frames.extend(proposed_frames(
                rows, imputed_scores, consistency, counts,
                reference_u, reference_c, proposed_parameters, args.fusion_weight,
            ))
    result = pd.concat(frames, ignore_index=True)
    result["policy_parameters"] = proposed_parameters
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    del source, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose(validation):
    choices = []
    for method in validation.method.unique():
        current = validation[validation.method == method]
        for tail in ("upper", "lower", "two_sided"):
            threshold, metric = threshold_under_fpr_cap(
                current.label, current[f"score_{tail}"], FPR_CAP,
            )
            key = (metric["ba"], metric["auc"], metric["recall"], -metric["fpr"])
            choices.append((key, method, tail, threshold, metric))
    return max(choices, key=lambda item: item[0])


def evaluate(test, condition, choice, output_method):
    _, source_method, tail, threshold, _ = choice
    current = test[(test.condition == condition) & (test.method == source_method)].copy()
    current["score"] = current[f"score_{tail}"]
    rows = []
    for replicate, cohort in current.groupby("replicate"):
        rows.append({
            "method": output_method, "source_method": source_method,
            "condition": condition, "replicate": int(replicate),
            "score_tail": tail, "threshold": threshold,
            **publication_metrics(cohort.label, cohort.score, threshold),
            "coverage_mean": float(cohort.coverage.mean()),
            "parameters": int(
                cohort.policy_parameters.iloc[0]
                if output_method == "care_usad_v18"
                else cohort.parameters.iloc[0]
            ),
        })
    return current, rows


def finish(output, frame):
    validation, test = frame[frame.split == "val"], frame[frame.split == "test"]
    policies, metrics, score_frames = [], [], []
    for condition in CONDITIONS:
        current = validation[validation.condition == condition]
        baseline = choose(current[current.method == "source_usad"])
        proposed = baseline if condition == "clean" else choose(current)
        for output_method, choice in (("source_usad", baseline), ("care_usad_v18", proposed)):
            scored, rows = evaluate(test, condition, choice, output_method)
            score_frames.append(scored)
            metrics.extend(rows)
            _, selected, tail, threshold, val_metric = choice
            policies.append({
                "output_method": output_method, "condition": condition,
                "selected_method": selected, "score_tail": tail,
                "threshold": threshold,
                **{f"validation_{key}": value for key, value in val_metric.items()},
            })
    metrics = pd.DataFrame(metrics)
    scores = pd.concat(score_frames, ignore_index=True)
    policy = pd.DataFrame(policies)
    summary = metrics.groupby(["method", "condition"], as_index=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        ba_mean=("ba", "mean"), recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"), fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"), f1_mean=("f1", "mean"),
        parameters=("parameters", "mean"),
    )
    base = summary[summary.method == "source_usad"].set_index("condition")
    prop = summary[summary.method == "care_usad_v18"].set_index("condition")
    columns = ["auc_mean", "ba_mean", "recall_mean", "fpr_mean"]
    effects = prop[columns].subtract(base[columns])
    effects.columns = ["delta_auc", "delta_ba", "delta_recall", "delta_fpr"]
    effects = effects.reset_index()
    promote = bool(
        float(effects.loc[effects.condition == "sensor_0.50", "delta_auc"].iloc[0]) > 0
        and float(effects.loc[effects.condition == "sensor_0.50", "delta_ba"].iloc[0]) > 0
        and float(effects.loc[effects.condition == "targeted_0.50", "delta_auc"].iloc[0]) >= 0
        and float(effects.loc[effects.condition == "targeted_0.50", "delta_ba"].iloc[0]) >= 0
        and float(effects.loc[effects.condition == "clean", "delta_auc"].iloc[0]) == 0
    )
    decision = {
        "promote_to_three_seed": promote,
        "criterion": "random50 AUC/BA positive; targeted50 AUC/BA nonnegative; clean preserved",
    }
    metrics.to_csv(output / "metrics.csv", index=False)
    scores.to_csv(output / "record_scores.csv", index=False)
    policy.to_csv(output / "validation_policy.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    effects.to_csv(output / "effects_vs_source_usad.csv", index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    pivot = summary.pivot(index="condition", columns="method", values="ba_mean")
    axis = pivot.plot.bar(figsize=(10, 5), ylim=(0, 1.02), rot=15, grid=True)
    axis.set_ylabel("Record-level balanced accuracy")
    axis.set_title("CARE-USAD V18 severe sensor-loss screening")
    axis.figure.tight_layout()
    axis.figure.savefig(output / "care_usad_v18.png", dpi=220)
    plt.close(axis.figure)
    print("\n=== CARE-USAD V18 SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== CARE-USAD V18 MINUS SOURCE USAD ===")
    print(effects.to_string(index=False))
    print("\n=== VALIDATION-LOCKED POLICY ===")
    print(policy.to_string(index=False))
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270103)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-window", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-candidates", type=int, default=3)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
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
        path = group_file(args.output_dir, key[0], key[1])
        if path.exists():
            print(f"SKIP cell={key}", flush=True)
            continue
        print(f"cell={key} sensors={len(labels)}", flush=True)
        group = used[(used.setup == key[0]) & (used.excitation == key[1])]
        run_group(args, group, labels, index, device, path)
    paths = sorted((args.output_dir / "groups").glob("*.csv"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 group files, got {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    finish(args.output_dir, frame)
    protocol = {
        "dataset": "Z24", "method": "CARE-USAD V18",
        "baseline": "vendored official EURECOM USAD",
        "conditions": list(CONDITIONS),
        "adapter_fit": "healthy-only random/targeted hard 50% sensor masks",
        "source_usad_frozen_during_adapter_fit": True,
        "clean_branch": "forced untouched source USAD",
        "selection": "condition-matched labeled validation only",
        "test_used_for_selection": False,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )
    print(f"CARE_USAD_V18_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

