"""Local TimesNet smoke benchmark and inference ablations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import MethodType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.hellbridge import Normalizer, development_split, load_windows
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, split_training_windows, stable_sensor_map
from .experiment import anchor_periods, score_rows
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .models.backbone import fit_timesnet, reconstruct, seed_everything
from .robust_pooling import load
from .score_calibration import location_scale

SEED, MAX_WINDOWS, BATCH = 20270905, 8, 8
METHODS = ("native_self", "anchored_self", "uniform_self", "anchored_cross")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


def subset(x):
    return x[np.linspace(0, len(x) - 1, min(len(x), MAX_WINDOWS), dtype=int)]


def uniform_forward(layer, x):
    batch, length, width = x.shape
    outputs = []
    for frequency in layer.period_bins:
        period = length // frequency
        padded_length = ((length + period - 1) // period) * period
        padded = torch.nn.functional.pad(x, (0, 0, 0, padded_length - length))
        grid = padded.reshape(batch, padded_length // period, period, width).permute(0, 3, 1, 2).contiguous()
        result = layer.conv(grid).permute(0, 2, 3, 1).reshape(batch, padded_length, width)
        outputs.append(result[:, :length])
    return x + torch.stack(outputs, dim=-1).mean(dim=-1)


def cross_forward(model, batch, keep):
    # The hidden target has no window mean/scale. Predict it in train-scaled units.
    visible = torch.where(keep, batch, 0.)
    mean = visible.mean(1, keepdim=True)
    scale = (visible.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    scale = torch.where(keep, scale, 1.)
    encoded = model.enc_embedding((visible - mean) / scale, None)
    for layer in model.model:
        encoded = model.layer_norm(layer(encoded))
    return model.projection(encoded) * scale + mean


class Trial:
    def __init__(self, fit, reference, seed, epochs):
        self.scaler = Normalizer.fit(fit)
        fit = self.scaler.transform(fit)
        seed_everything(seed)
        self.model = fit_timesnet(fit, epochs, BATCH, DEVICE)
        self.anchors = anchor_periods(self.model, torch.as_tensor(fit, device=DEVICE))
        self.full = np.ones(fit.shape[1], dtype=bool)
        refs, _ = self.score(self.scaler.transform(reference), self.full, capture=False)
        self.refs = {k: location_scale(v) for k, v in refs.items()}

    @torch.no_grad()
    def score(self, values, mask, capture):
        x = torch.as_tensor(values, device=DEVICE)
        clean_mask = torch.as_tensor(mask, device=DEVICE)
        visible = torch.where(clean_mask.view(1, -1, 1), x, 0.)
        layers = list(self.model.model)
        layer_calls = [layer.forward for layer in layers]
        logged = {}
        active_logs = []

        def capture_stage(name, kind, anchors=None):
            def hook(_layer, args, result):
                if kind == "weight":
                    spectrum = torch.fft.rfft(args[0], dim=1).abs().mean(-1)
                    result = torch.softmax(spectrum[:, list(anchors)], dim=1)
                active_logs.append((name, result.detach().cpu().numpy()))
            return hook

        results = {}
        for method in METHODS:
            for layer, anchors, forward in zip(layers, self.anchors, layer_calls, strict=True):
                layer.period_bins = None if method == "native_self" else anchors
                layer.forward = MethodType(uniform_forward, layer) if method == "uniform_self" else forward
            handles = []
            if capture and method == "anchored_self":
                handles.append(self.model.enc_embedding.register_forward_hook(capture_stage("embedding", "activation")))
                for i, (layer, anchors) in enumerate(zip(layers, self.anchors, strict=True), 1):
                    handles.append(layer.register_forward_hook(capture_stage(f"block{i}", "activation")))
                    handles.append(layer.register_forward_hook(capture_stage(f"period_weight{i}", "weight", anchors)))
                handles.append(self.model.projection.register_forward_hook(capture_stage("projection", "activation")))
            chunks = []
            try:
                for target, observed in zip(x.split(BATCH), visible.split(BATCH), strict=True):
                    if method == "anchored_cross":
                        target = target.transpose(1, 2)
                        summed = torch.zeros(len(target), device=DEVICE)
                        ids = np.flatnonzero(mask)
                        for probe in (ids[::3], ids[1::3], ids[2::3]):
                            if not len(probe):
                                continue
                            context = clean_mask.clone()
                            context[probe] = False
                            pred = cross_forward(self.model, target, context.view(1, 1, -1))
                            summed += (pred[:, :, probe] - target[:, :, probe]).square().mean(1).sum(1)
                        error = summed / len(ids)
                    else:
                        prediction = reconstruct(self.model, observed)
                        error = (prediction[:, clean_mask] - target[:, clean_mask]).square().mean((1, 2))
                    chunks.append(error.cpu().numpy())
            finally:
                for handle in handles:
                    handle.remove()
            results[method] = np.log1p(np.concatenate(chunks).astype(np.float64))
        for layer, anchors, forward in zip(layers, self.anchors, layer_calls, strict=True):
            layer.period_bins = anchors
            layer.forward = forward
        for key, value in active_logs:
            logged.setdefault(key, []).append(value)
        return results, {key: np.concatenate(value) for key, value in logged.items()}

    def predict(self, transformed, mask, capture=False):
        results, logs = self.score(transformed, mask, capture)
        return {k: (v - self.refs[k][0]) / self.refs[k][1] for k, v in results.items()}, logs


def shifts(rows, clean, masked, labels, group, rep):
    labels = np.asarray(labels)
    for name, base in clean.items():
        delta = masked[name] - base
        if name.startswith("period_weight"):
            size = np.abs(delta).sum(-1) / 2
        else:
            axes = tuple(range(1, base.ndim))
            size = np.sqrt(np.mean(delta ** 2, axis=axes)) / np.maximum(np.sqrt(np.mean(base ** 2, axis=axes)), 1e-6)
        for label in np.unique(labels):
            selected = size[labels == label]
            rows.append(dict(group=group, stage=name, label=int(label), replicate=rep,
                             window_count=len(selected), mean_shift=float(selected.mean())))


def finish(dataset, rows, logs, anchors, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"{dataset}_scores.csv.gz", index=False)
    pd.DataFrame(logs).to_csv(out / f"{dataset}_layer_shifts.csv", index=False)
    pd.DataFrame(anchors).to_csv(out / f"{dataset}_anchors.csv", index=False)
    val = frame[frame.split.eq("val")]
    metrics, states = [], []
    for method, group in frame[frame.split.eq("test")].groupby("method"):
        sign = 1
        clean_val = val[val.method.eq(method) & val.condition.eq("clean")]
        if dataset == "Z24" and roc_auc_score(clean_val.label, clean_val.score) < .5:
            sign = -1
        clean = group[group.condition.eq("clean")]
        clean_rank = clean.set_index("sample_id").score * sign
        for (condition, rep), test in group.groupby(["condition", "replicate"]):
            threshold = threshold_from_healthy(val.loc[
                val.method.eq(method) & val.condition.eq(condition)
                & val.replicate.eq(rep) & val.label.eq(0), "score"
            ].to_numpy() * sign, .90)
            y, s = test.label.to_numpy(), test.score.to_numpy() * sign
            clean_aligned = clean_rank.loc[test.sample_id].to_numpy()
            base_good = clean_aligned[y == 1, None] > clean_aligned[y == 0][None, :]
            mask_bad = s[y == 1, None] <= s[y == 0][None, :]
            fpr_limit = 0.0 if dataset == "Z24" else 0.05
            metrics.append(dict(method=method, condition=condition, replicate=rep, sign=sign,
                                recall_low_fpr=recall_at_fpr(y, s, fpr_limit),
                                clean_correct_pairs_reversed=(base_good & mask_bad).sum() / max(1, base_good.sum()),
                                **publication_metrics(y, s, threshold)))
            for state, damage in test[test.label.eq(1)].groupby("state"):
                d = damage.score.to_numpy() * sign
                healthy = s[y == 0]
                states.append(dict(method=method, condition=condition, replicate=rep, state=state,
                                   auc=roc_auc_score(np.r_[np.zeros(len(healthy)), np.ones(len(d))], np.r_[healthy, d]),
                                   recall=(d >= threshold).mean()))
    measures = ["auc", "ba", "recall", "fpr", "recall_low_fpr", "clean_correct_pairs_reversed"]
    series = pd.DataFrame(metrics)
    series.to_csv(out / f"{dataset}_metrics_by_mask.csv", index=False)
    pd.DataFrame(states).to_csv(out / f"{dataset}_damage_states.csv", index=False)
    summary = series.groupby(["method", "condition"])[measures].mean().reset_index()
    summary.to_csv(out / f"{dataset}_summary.csv", index=False)
    print(dataset, summary.round(4).to_string(index=False), flush=True)


def run_hellbridge(h5, out, epochs):
    split_dir = out / "hellbridge_split"
    subprocess.run([
        sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(h5),
        "--output-dir", str(split_dir), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
        "--seed", "20261113", "--strict-paper-counts",
    ], check=True)
    split = split_dir / "sample_manifest.csv"
    frame = pd.read_csv(split).sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, val, test = development_split(frame)
    raw = load_windows(h5, frame, 128)
    trial = Trial(raw[fit], raw[val], SEED, epochs)
    joined = np.r_[val, test]
    values = trial.scaler.transform(raw[joined])
    rows, log = [], []
    rng = np.random.default_rng(20261119)
    cases = [("clean", 0, trial.full)] + [
        ("sensor_0.50", rep, sensor_mask(40, .5, rng.permutation(40))) for rep in range(3)
    ]
    for condition, rep, mask in cases:
        scores, stages = trial.predict(values, mask, capture=True)
        if condition == "clean":
            clean = stages
        else:
            shifts(log, clean, stages, frame.loc[joined, "label"], "P2_NM_Z", rep)
        for role, ids, start, stop in [("val", val, 0, len(val)), ("test", test, len(val), len(joined))]:
            picked = frame.loc[ids]
            before = len(rows)
            score_rows(rows, {k:v[start:stop] for k,v in scores.items()},
                       picked.sample_id, picked.label, role, condition, rep)
            states = picked.set_index("sample_id").state
            for entry in rows[before:]:
                entry["state"] = states.loc[entry["sample_id"]]
        print(f"HellBridge {condition} mask={rep}", flush=True)
    finish("HellBridge", rows, log, [dict(group="P2_NM_Z", block=i, bins=str(b)) for i,b in enumerate(trial.anchors,1)], out)


def run_z24(root, out, epochs):
    frame = load_manifest(root)
    rows, log, anchors = [], [], []
    for index, ((setup, exc), channels) in enumerate(sorted(stable_sensor_map(frame).items())):
        group = frame[frame.setup.eq(setup) & frame.excitation.eq(exc)]
        key = f"{setup:02d}/{exc.upper()}"
        print(f"Z24 {key} group={index+1}/18", flush=True)
        fit, ref = split_training_windows([load(row, channels) for row in group[group.split.eq("train")].itertuples()])
        trial = Trial(subset(fit), subset(ref), SEED + 1009 * index, epochs)
        anchors.extend(dict(group=key, block=i, bins=str(b)) for i,b in enumerate(trial.anchors,1))
        for row in group[group.split.isin(["val", "test"])].itertuples():
            raw = trial.scaler.transform(subset(load(row, channels)))
            score, clean = trial.predict(raw, trial.full, capture=row.split == "test")
            cases = [("clean", 0, score)]
            if row.split == "test" or row.label == 0:
                for rep in range(3):
                    order = stable_rng(str(row.record_id), "sensor", rep).permutation(len(channels))
                    kept = sensor_mask(len(channels), .50, order)
                    score, stage = trial.predict(raw, kept, capture=row.split == "test")
                    cases.append(("sensor_0.50", rep, score))
                    if row.split == "test":
                        shifts(log, clean, stage, np.full(len(raw), row.label), key, rep)
            for condition, rep, score in cases:
                before = len(rows)
                score_rows(rows, {m:[float(np.median(s))] for m,s in score.items()},
                           [row.record_id], [row.label], row.split, condition, rep)
                for entry in rows[before:]:
                    entry["state"] = row.state
        del trial
    finish("Z24", rows, log, anchors, out)


def plot_diagnostics(out):
    methods = ("native_self", "anchored_self", "uniform_self", "anchored_cross")
    labels = ("Gốc", "Neo chu kỳ", "Trọng số đều", "Dự đoán chéo*")
    colors = ("#3f6075", "#007e78", "#8b718f", "#bd5c35")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    fig.suptitle("TimesNet gặp khó ở đâu khi mất sensor?", fontsize=19, weight="bold")
    ax = axes[0, 0]
    z = pd.read_csv(out / "Z24_scores.csv.gz")
    metric_z = pd.read_csv(out / "Z24_metrics_by_mask.csv")
    z = z[z.split.eq("test") & z.method.eq("anchored_self")]
    sign = int(metric_z.loc[metric_z.method.eq("anchored_self"), "sign"].iloc[0])
    worst = metric_z[metric_z.method.eq("anchored_self") & metric_z.condition.eq("sensor_0.50")]
    rep = int(worst.sort_values(["recall_low_fpr", "replicate"]).iloc[0].replicate)
    clean = z[z.condition.eq("clean")].set_index("sample_id")
    masked = z[z.condition.eq("sensor_0.50") & z.replicate.eq(rep)].set_index("sample_id")
    joined = clean[["label", "score"]].join(masked[["score"]], rsuffix="_masked", validate="one_to_one")
    joined[["score", "score_masked"]] *= sign
    pairs = joined[joined.label.eq(0)].reset_index().merge(
        joined[joined.label.eq(1)].reset_index(), how="cross", suffixes=("_healthy", "_damage"),
    )
    pairs["clean_gap"] = pairs.score_damage - pairs.score_healthy
    pairs["mask_gap"] = pairs.score_masked_damage - pairs.score_masked_healthy
    flipped = pairs[pairs.clean_gap.gt(0) & pairs.mask_gap.lt(0)]
    if flipped.empty:
        ax.text(.5, .5, "Không có cặp đảo hạng ở lượt này", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        pair = flipped.assign(change=flipped.clean_gap.abs() + flipped.mask_gap.abs()).sort_values("change").iloc[len(flipped)//2]
        for tag, label, color in [("healthy", "Healthy", colors[0]), ("damage", "Damage", colors[3])]:
            ax.plot([0, 1], [pair[f"score_{tag}"], pair[f"score_masked_{tag}"]], "o-", lw=2.3,
                    label=label, color=color)
        ax.set(title=f"Z24 · một cặp đảo hạng (mask {rep})", ylabel="Score hư hỏng đã định hướng",
               xticks=[0,1], xticklabels=["Đủ sensor","Mất 50% sensor"], xlim=(-.15, 1.15))
        ax.set_yscale("symlog", linthresh=1)
        ax.legend(frameon=False, loc="best")
        pd.DataFrame([pair]).to_csv(out / "plotted_z24_pair.csv", index=False)

    ax = axes[0, 1]
    scores = pd.read_csv(out / "HellBridge_damage_states.csv")
    scores = scores[scores.method.eq("native_self")]
    state = scores.pivot_table(index="state", columns="condition", values="auc") * 100
    x = np.arange(len(state))
    ax.bar(x-.18, state["clean"], .34, color=colors[0], label="Đủ sensor")
    ax.bar(x+.18, state["sensor_0.50"], .34, color=colors[1], label="Mất 50% sensor")
    ax.axhline(50, color="#85929c", lw=1, ls="--")
    ax.set(title="HellBridge · có trạng thái yếu ngay từ clean", ylabel="AUC so với healthy (%)",
           xticks=x, xticklabels=state.index, ylim=(0, 105))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    stages = ["embedding", "block1", "block2", "projection"]
    stage_labels = ["Embedding", "TimesBlock 1", "TimesBlock 2", "Projection"]
    for bridge, color in [("Z24", colors[0]), ("HellBridge", colors[1])]:
        data = pd.read_csv(out / f"{bridge}_layer_shifts.csv")
        data = data[data.stage.isin(stages)].assign(weighted=lambda d: d.mean_shift * d.window_count)
        sums = data.groupby("stage")[["weighted", "window_count"]].sum().reindex(stages)
        ax.plot(range(len(stages)), 100 * sums.weighted / sums.window_count, "o-", color=color, label=bridge)
    ax.set(title="Mất sensor làm activation đổi từ embedding", ylabel="Độ dịch RMS tương đối (%)",
           xticks=range(len(stages)), xticklabels=stage_labels, ylim=(0, 75))
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for i, (bridge, suffix) in enumerate([("Z24", "0 FP"), ("HellBridge", "FPR ≤ 5%")]):
        metric = pd.read_csv(out / f"{bridge}_metrics_by_mask.csv")
        values = metric[metric.condition.eq("sensor_0.50")].groupby("method").recall_low_fpr.mean()
        for j, (method, label, color) in enumerate(zip(methods, labels, colors, strict=True)):
            height = 100 * values[method]
            ax.bar(i + (j-1.5)*.19, height, .17, color=color, label=label if i == 0 else None)
            ax.text(i + (j-1.5)*.19, height + .65, f"{height:.1f}", ha="center", fontsize=8)
    ax.set(title="Đổi cách suy luận chưa giúp đồng thời hai cầu", ylabel="Recall ở vùng ít báo động giả (%)",
           xticks=[0,1], xticklabels=["Z24 · 0 FP", "HellBridge · FPR ≤ 5%"], ylim=(0, 60))
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    fig.supxlabel("* Dự đoán chéo bằng trọng số tự tái tạo đã đóng băng · 1 seed · 3 mask mất 50%",
                  fontsize=10)
    fig.savefig(out / "timesnet_failure_modes.png", dpi=170)
    fig.savefig(out / "timesnet_failure_modes.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path)
    parser.add_argument("--hell-h5", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        if args.z24_root is None or args.hell_h5 is None or args.epochs < 1:
            parser.error("A run requires both datasets and at least one epoch")
        (out / "protocol.json").write_text(json.dumps(dict(
            seed=SEED, epochs=args.epochs, device=str(DEVICE), max_windows_for_z24=MAX_WINDOWS,
            mask_replicates=[0, 1, 2], native_fft_batch=BATCH, score_variants=METHODS,
            orientation="Z24 labeled clean validation; HellBridge upper",
            notes="healthy fit; frozen-weight inference ablations; 50% loss; diagnostic protocol",
        ), indent=2), encoding="utf-8")
        run_hellbridge(args.hell_h5, out, args.epochs)
        run_z24(args.z24_root, out, args.epochs)
    plot_diagnostics(out)
    print("TIMESNET_FAILURE_VISUALS_PASS", flush=True)


if __name__ == "__main__":
    main()
