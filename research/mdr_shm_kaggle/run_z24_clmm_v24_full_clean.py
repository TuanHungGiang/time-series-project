from __future__ import annotations

"""CLMM V24: evaluate the full reconstruction path on complete Z24 data.

The frozen CLMM-RS implementation preserves every observed channel, so its
normal clean-data path is exactly Source-USAD.  This ablation deliberately
feeds complete records through the entire cross-sensor adapter and then into
the frozen USAD detector.  It answers a narrow question: can the learned
adapter also act as a useful clean-data denoiser?

Training remains healthy-only.  Source-USAD and full-reconstruction CLMM use
separate validation-calibrated thresholds under the same fixed FPR cap.  Test
labels are used only for evaluation.  Ten independent training seeds are
aggregated with a paired seed bootstrap.  The experiment is resumable.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_z24_care_usad_v18 as care  # noqa: E402
import run_z24_clmm_v21 as v21  # noqa: E402


DEFAULT_SEEDS = (
    20270901,
    20270913,
    20270929,
    20271011,
    20271027,
    20271109,
    20271123,
    20271207,
    20271219,
    20280103,
)
METRICS = ("auc", "ap", "ba", "recall", "specificity", "fpr", "precision", "f1")


def full_reconstruction_complete(adapter, scaled, availability, batch_size, device):
    """Use the adapter output itself only when every sensor is available."""
    if not np.all(np.asarray(availability) > 0.5):
        return ORIGINAL_COMPLETE(
            adapter, scaled, availability, batch_size, device,
        )

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled.astype(np.float32)),
            torch.from_numpy(availability.astype(np.float32)),
        ),
        batch_size=batch_size,
    )
    reconstructed, consistency = [], []
    adapter.eval()
    with torch.no_grad():
        for values, observed in loader:
            values = values.to(device)
            observed = observed.to(device)
            prediction = adapter(
                values * observed.unsqueeze(-1), observed,
            )
            denominator = (
                observed.sum(1) * values.shape[2]
            ).clamp_min(1.0)
            error = (
                (prediction - values).pow(2) * observed.unsqueeze(-1)
            ).sum((1, 2)) / denominator
            reconstructed.append(prediction.cpu().numpy())
            consistency.append(error.cpu().numpy())
    return np.concatenate(reconstructed), np.concatenate(consistency)


def full_clean_selected_method(output_method: str, condition: str):
    del condition
    return "source_usad" if output_method == "source_usad" else "imputed_usad"


ORIGINAL_COMPLETE = care.complete


def worker(args):
    care.complete = full_reconstruction_complete
    v21.selected_method = full_clean_selected_method
    forwarded = [
        "run_z24_clmm_v21.py",
        "--input-root", str(args.input_root),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--adapter-epochs", str(args.adapter_epochs),
        "--batch-size", str(args.batch_size),
        "--raw-window", str(args.raw_window),
        "--window-stride", str(args.window_stride),
        "--downsample", str(args.downsample),
        "--embedding", str(args.embedding),
        "--heads", str(args.heads),
        "--hard-candidates", str(args.hard_candidates),
        "--score-weight", str(args.score_weight),
        "--fpr-cap", str(args.fpr_cap),
    ]
    sys.argv = forwarded
    v21.main()

    protocol_path = args.output_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "method": "CLMM V24 full-clean reconstruction ablation",
        "clean_path": "complete signal -> cross-sensor adapter -> frozen USAD",
        "observed_channels_preserved_on_clean": False,
        "purpose": "test whether the adapter is also a clean-data denoiser",
    })
    protocol_path.write_text(
        json.dumps(protocol, indent=2), encoding="utf-8",
    )


def complete_seed(path: Path):
    required = ("metrics.csv", "validation_policy.csv", "protocol.json")
    return all((path / name).exists() for name in required)


def run_seed(args, seed: int):
    output = args.output_dir / f"seed_{seed}"
    if complete_seed(output):
        print(f"SKIP completed seed={seed} output={output}", flush=True)
        return
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--input-root", str(args.input_root),
        "--output-dir", str(output),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--adapter-epochs", str(args.adapter_epochs),
        "--batch-size", str(args.batch_size),
        "--raw-window", str(args.raw_window),
        "--window-stride", str(args.window_stride),
        "--downsample", str(args.downsample),
        "--embedding", str(args.embedding),
        "--heads", str(args.heads),
        "--hard-candidates", str(args.hard_candidates),
        "--score-weight", str(args.score_weight),
        "--fpr-cap", str(args.fpr_cap),
    ]
    print(f"\n=== FULL-CLEAN SEED {seed} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not complete_seed(output):
        raise RuntimeError(f"Incomplete output for seed {seed}: {output}")


def paired_seed_bootstrap(effects, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        column = f"delta_{metric}"
        values = effects[column].to_numpy(dtype=float)
        draws = rng.choice(
            values, size=(samples, len(values)), replace=True,
        ).mean(axis=1)
        rows.append({
            "metric": column,
            "estimate": float(values.mean()),
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "bootstrap_samples": samples,
            "bootstrap_unit": "training_seed",
        })
    return pd.DataFrame(rows)


def finish(args):
    metric_frames, policy_frames = [], []
    for seed in args.seeds:
        root = args.output_dir / f"seed_{seed}"
        metrics = pd.read_csv(root / "metrics.csv")
        metrics = metrics[metrics.condition.eq("clean")].copy()
        metrics.insert(0, "seed", seed)
        metrics["method"] = metrics.method.replace({
            "clmm_v21": "clmm_full_clean",
        })
        metric_frames.append(metrics)

        policy = pd.read_csv(root / "validation_policy.csv")
        policy = policy[policy.condition.eq("clean")].copy()
        policy.insert(0, "seed", seed)
        policy["output_method"] = policy.output_method.replace({
            "clmm_v21": "clmm_full_clean",
        })
        policy_frames.append(policy)

    metrics = pd.concat(metric_frames, ignore_index=True)
    policies = pd.concat(policy_frames, ignore_index=True)
    summary = metrics.groupby("method", as_index=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ap_mean=("ap", "mean"), ap_std=("ap", "std"),
        ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        specificity_mean=("specificity", "mean"),
        fpr_mean=("fpr", "mean"),
        precision_mean=("precision", "mean"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        seeds=("seed", "nunique"),
    )

    wide = metrics.pivot_table(
        index="seed", columns="method", values=list(METRICS),
    )
    effect_rows = []
    for seed, values in wide.iterrows():
        row = {"seed": int(seed)}
        for metric in METRICS:
            row[f"delta_{metric}"] = float(
                values[(metric, "clmm_full_clean")]
                - values[(metric, "source_usad")]
            )
        effect_rows.append(row)
    effects = pd.DataFrame(effect_rows).sort_values("seed")
    bootstrap = paired_seed_bootstrap(
        effects, args.bootstrap_samples, args.analysis_seed,
    )

    metrics.to_csv(args.output_dir / "clean_all_metrics.csv", index=False)
    policies.to_csv(
        args.output_dir / "clean_validation_policies.csv", index=False,
    )
    summary.to_csv(args.output_dir / "clean_ten_seed_summary.csv", index=False)
    effects.to_csv(args.output_dir / "clean_paired_seed_effects.csv", index=False)
    bootstrap.to_csv(
        args.output_dir / "clean_paired_bootstrap_ci.csv", index=False,
    )

    auc_ci = bootstrap.set_index("metric").loc["delta_auc"]
    ba_ci = bootstrap.set_index("metric").loc["delta_ba"]
    decision = {
        "experiment": "full CLMM adapter on complete clean records",
        "clean_auc_improved_in_mean": bool(auc_ci.estimate > 0),
        "clean_auc_ci_lower_above_zero": bool(auc_ci.ci_lower > 0),
        "clean_ba_improved_in_mean": bool(ba_ci.estimate > 0),
        "clean_ba_ci_lower_above_zero": bool(ba_ci.ci_lower > 0),
        "promotion_rule": (
            "promote only if validation supports the branch and paired test "
            "CIs do not show a clean-performance loss"
        ),
        "test_used_for_training_or_threshold": False,
    }
    (args.output_dir / "clean_full_model_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )

    print("\n=== FULL-MODEL CLEAN PERFORMANCE (10 SEEDS) ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED EFFECTS BY SEED (FULL CLMM - SOURCE USAD) ===")
    print(effects.to_string(index=False))
    print("\n=== PAIRED SEED BOOTSTRAP 95% CI ===")
    print(bootstrap.to_string(index=False))
    print("\n=== DECISION ===")
    print(json.dumps(decision, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20270901)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
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
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20261118)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("Exactly ten unique seeds are required")
    print(
        "CLMM V24 full-clean reconstruction ablation "
        f"seeds={args.seeds}",
        flush=True,
    )
    for seed in args.seeds:
        run_seed(args, seed)
    finish(args)
    print(f"CLMM_V24_FULL_CLEAN_PASS output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

