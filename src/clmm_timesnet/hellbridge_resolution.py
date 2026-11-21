"""Controlled input-resolution ablation on HellBridge."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.hellbridge import Normalizer, development_split, load_windows
from .experiment import MASK_REPLICATES, anchor_periods, test_masks
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .robust_pooling import channel_errors
from .score_calibration import fit_mask_aware


LENGTHS = (128, 256, 512)
CONDITIONS = ("clean", "sensor_0.50")
FIELDS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "fp", "tp", "recall_at_fpr_05")


def score_resolution(scores, metadata, threshold, length, condition, rep):
    y = metadata.label.to_numpy()
    healthy = scores[y == 0]
    alert = scores >= threshold
    limit = np.sort(healthy)[::-1][int(np.floor(0.05 * len(healthy)))]
    group = dict(length=length, condition=condition, replicate=rep)
    overall = dict(
        **group, threshold=threshold, **publication_metrics(y, scores, threshold),
        fp=int((alert & (y == 0)).sum()), tp=int((alert & (y == 1)).sum()),
        recall_at_fpr_05=recall_at_fpr(y, scores, 0.05),
    )
    by_state = []
    for state in sorted(metadata.loc[metadata.label.eq(1), "state"].unique()):
        member = metadata.state.eq(state).to_numpy()
        state_scores = scores[member]
        by_state.append(dict(
            **group, state=state, windows=int(member.sum()),
            auc=roc_auc_score(
                np.r_[np.zeros(len(healthy)), np.ones(len(state_scores))],
                np.r_[healthy, state_scores],
            ),
            recall=alert[member].mean(),
            recall_at_fpr_05=(state_scores > limit).mean(),
        ))
    return overall, by_state


def plot(out, overall, states):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    panels = [("auc", "ROC-AUC"), ("recall_at_fpr_05", "Recall at test ROC FPR ≤ 5%"),
              ("recall", "Recall at validation Q90"), ("fpr", "FPR at validation Q90")]
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for condition, label, color in [("clean", "Clean", "#526780"),
                                         ("sensor_0.50", "50% sensor loss", "#00796b")]:
            stats = overall[overall.condition.eq(condition)].groupby("length")[metric].agg(["mean", "min", "max"]) * 100
            x = stats.index.to_numpy()
            ax.plot(x, stats["mean"], "o-", color=color, label=label)
            ax.fill_between(x, stats["min"], stats["max"], color=color, alpha=.14)
        ax.set(title=title, xlabel="Points per window", ylabel="%", xticks=sorted(overall.length.unique()))
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("HellBridge input resolution · bands show mask min–max")
    fig.savefig(out / "01_resolution_metrics.png", dpi=180)
    fig.savefig(out / "01_resolution_metrics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for row, condition in enumerate(CONDITIONS):
        for col, (metric, label) in enumerate([("auc", "ROC-AUC"), ("recall", "Recall at Q90")]):
            ax = axes[row, col]
            values = states[states.condition.eq(condition)].pivot_table(index="state", columns="length", values=metric) * 100
            img = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
            for i, line in enumerate(values.to_numpy()):
                for j, value in enumerate(line):
                    ax.text(j, i, f"{value:.0f}", va="center", ha="center",
                            color="white" if value > 62 else "#20272f")
            ax.set(title=f"{'Clean' if row == 0 else '50% sensor loss'} · {label}",
                   xticks=range(len(values.columns)), xticklabels=values.columns,
                   yticks=range(len(values.index)), yticklabels=values.index, xlabel="Points per window")
    fig.colorbar(img, ax=axes, label="%")
    fig.savefig(out / "02_damage_state_resolution.png", dpi=180)
    fig.savefig(out / "02_damage_state_resolution.pdf")
    plt.close(fig)


def run_resolution(args, manifest, indices, cases, length, device):
    fit, cal, test = indices
    raw = load_windows(args.hell_h5, manifest, length)
    scaler = Normalizer.fit(raw[fit])
    data = scaler.transform(raw)
    del raw
    model = fit_mask_aware(data[fit], device, args.seed, args.epochs)
    anchor_periods(model, torch.as_tensor(data[fit], device=device))
    combined = data[np.r_[cal, test]]
    state_metadata = manifest.loc[test]
    metrics, states, score_tables = [], [], []
    for condition, rep, kept in cases:
        residuals = channel_errors(model, combined, kept, device)
        raw_score = np.log1p(residuals.mean(axis=1))
        test_score = raw_score[len(cal):]
        threshold = threshold_from_healthy(raw_score[:len(cal)], 0.90)
        overall, by_state = score_resolution(test_score, state_metadata, threshold, length, condition, rep)
        metrics.append(overall)
        states.extend(by_state)
        score_tables.append(state_metadata[["sample_id", "record_id", "state", "label"]].assign(
            length=length, condition=condition, replicate=rep, score=test_score,
            threshold=threshold, alarm=test_score >= threshold,
        ))
        print(f"points={length} condition={condition} mask={rep} AUC={overall['auc']:.4f} "
              f"recall={overall['recall']:.4f} FPR={overall['fpr']:.4f}", flush=True)
    return pd.DataFrame(metrics), pd.DataFrame(states), pd.concat(score_tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270905)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=LENGTHS)
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths) or args.epochs < 1:
        parser.error("Provide distinct lengths and at least one epoch")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split = out / "hellbridge_split"
    subprocess.run([sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(args.hell_h5),
                    "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
                    "--seed", "20261113", "--strict-paper-counts"], check=True)
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    indices = development_split(manifest)
    if manifest.loc[indices[2], "label"].value_counts().to_dict() != {0: 80, 1: 640}:
        raise ValueError("Unexpected test balance")
    rng = np.random.default_rng(20261119)
    orders = [rng.permutation(40) for _ in range(MASK_REPLICATES)]
    cases = [case for case in test_masks(40, orders) if case[0] in CONDITIONS]
    pd.DataFrame([dict(condition=c, replicate=r,
                       dropped_sensors=" ".join(f"AL{i+1:02d}" for i in np.flatnonzero(~m)))
                  for c, r, m in cases]).to_csv(out / "masks.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics, all_states, all_scores = [], [], []
    for length in args.lengths:
        print(f"=== INPUT [N, 40, {length}] on {device} ===", flush=True)
        metrics, states, scores = run_resolution(args, manifest, indices, cases, length, device)
        all_metrics.append(metrics)
        all_states.append(states)
        all_scores.append(scores)
        metrics.to_csv(out / f"metrics_{length}.csv", index=False)
    overall = pd.concat(all_metrics, ignore_index=True)
    states = pd.concat(all_states, ignore_index=True)
    overall.to_csv(out / "metrics_by_mask.csv", index=False)
    overall.groupby(["condition", "length"], as_index=False)[list(FIELDS)].mean().to_csv(
        out / "summary.csv", index=False,
    )
    states.to_csv(out / "damage_by_mask.csv", index=False)
    states.groupby(["state", "condition", "length"], as_index=False)[
        ["auc", "recall", "recall_at_fpr_05"]
    ].mean().to_csv(out / "damage_summary.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(out / "test_scores.csv.gz", index=False)
    (out / "protocol.json").write_text(json.dumps(dict(
        seed=args.seed, epochs=args.epochs, raw_points=1024, raw_hz=100, channels=40, lengths=list(args.lengths),
        mask_seed=20261119, conditions=list(CONDITIONS), healthy_fit=60, healthy_calibration=20,
        healthy_test=80, damage_test=640, preprocessing="linear detrend then Fourier resample",
        score="log1p mean observed channel MSE; upper-tail", threshold="healthy validation Q90 per mask",
    ), indent=2), encoding="utf-8")
    plot(out, overall, states)
    print("HELLBRIDGE_RESOLUTION_PASS", flush=True)


if __name__ == "__main__":
    main()
