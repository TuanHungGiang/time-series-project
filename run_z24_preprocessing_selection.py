"""Leak-controlled selection of spatial preprocessing for Z24.

This is step 1 of the proposed-method roadmap.  Damage labels are never used
to fit PSD statistics, representations, the Procrustes adapter, autoencoder,
or thresholds.  Validation damage is used only to select one preprocessing
variant.  The locked test is scored only for that selected variant.

The experiment deliberately excludes paired-domain refinement and alternative
calibrators so that the only changing factor is sensor pooling.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch import nn

import run_z24_healthy_cpc_benchmark as control
import run_z24_spatial_compact_cpc as spatial
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal
from run_z24_procrustes_ae import (
    SourceAE, encode, fit_rotation, reconstruction_errors, train_source_ae,
)


VARIANTS = (
    "mean",
    "attention",
    "mean_attention_max",
    "mean_attention_avgtopk",
    "mean_attention_trimmed_topk",
)


class SensorPoolingCPC(nn.Module):
    """CPC encoder whose only ablated component is sensor aggregation.

    AvgTopK and trimmed TopK rank whole sensors with the learned attention
    score, then gather complete sensor vectors.  Consequently, trimming drops
    one sensor identity consistently across all latent dimensions.  This is
    intentionally different from the legacy feature-wise activation TopK.
    """

    def __init__(self, features: int, latent: int, context: int,
                 prediction_steps: int, pooling: str,
                 topk_fraction: float):
        super().__init__()
        if pooling not in VARIANTS:
            raise ValueError(f"Unknown pooling variant: {pooling}")
        self.pooling = pooling
        self.topk_fraction = topk_fraction
        self.prediction_steps = prediction_steps
        self.sensor_encoder = nn.Sequential(
            nn.Linear(features, 128), nn.ELU(), nn.Linear(128, latent),
            nn.LayerNorm(latent),
        )
        self.attention = nn.Sequential(
            nn.Linear(latent, max(latent // 2, 1)), nn.Tanh(),
            nn.Linear(max(latent // 2, 1), 1),
        )
        components = 3 if pooling.startswith("mean_attention_") else 1
        self.pool_projection = nn.Sequential(
            nn.Linear(components * latent, latent), nn.ELU(),
            nn.LayerNorm(latent),
        )
        self.context = nn.GRU(latent, context, batch_first=True)
        self.predictors = nn.ModuleList(
            [nn.Linear(context, latent) for _ in range(prediction_steps)]
        )

    @staticmethod
    def _masked_mean(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        return (z * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)

    def _ranked_sensor_pool(self, z: torch.Tensor, mask: torch.Tensor,
                            logits: torch.Tensor, trim_first: bool) -> torch.Tensor:
        """Select sensor identities, optionally drop the strongest, and mean."""
        sensor_count = mask.sum(dim=2)
        minimum = 3 if trim_first else 1
        k = torch.ceil(sensor_count * self.topk_fraction).to(torch.long)
        k = torch.maximum(k, torch.full_like(k, minimum))
        k = torch.minimum(k, sensor_count)
        ranked_indices = torch.argsort(logits, dim=2, descending=True)
        ranked_z = torch.gather(
            z, 2, ranked_indices.unsqueeze(-1).expand_as(z)
        )
        ranks = torch.arange(z.shape[2], device=z.device).view(1, 1, -1)
        keep = ranks < k.unsqueeze(-1)
        if trim_first:
            keep = keep & (ranks > 0)
        weights = keep.unsqueeze(-1).to(z.dtype)
        return (ranked_z * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.)

    def sensor_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.sensor_encoder(x)
        mean = self._masked_mean(z, mask)
        logits = self.attention(z).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=2).unsqueeze(-1)
        attended = (weights * z).sum(dim=2)

        if self.pooling == "mean":
            components = [mean]
        elif self.pooling == "attention":
            components = [attended]
        elif self.pooling == "mean_attention_max":
            maximum = z.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(z.dtype).min
            ).max(dim=2).values
            components = [mean, attended, maximum]
        elif self.pooling == "mean_attention_avgtopk":
            topk = self._ranked_sensor_pool(z, mask, logits, False)
            components = [mean, attended, topk]
        else:
            trimmed = self._ranked_sensor_pool(z, mask, logits, True)
            components = [mean, attended, trimmed]
        return self.pool_projection(torch.cat(components, dim=-1))

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.sensor_pool(x, mask)
        context, _ = self.context(pooled)
        return context

    def predictive_loss(self, x: torch.Tensor, mask: torch.Tensor,
                        temperature: float) -> torch.Tensor:
        z = F.normalize(self.sensor_pool(x, mask), dim=-1)
        context, _ = self.context(z)
        losses = []
        for index, step in enumerate(range(1, self.prediction_steps + 1)):
            prediction = F.normalize(
                self.predictors[index](context[:, :-step]), dim=-1
            )
            target = z[:, step:]
            p = prediction.reshape(-1, prediction.shape[-1])
            q = target.reshape(-1, target.shape[-1])
            logits = p @ q.T / temperature
            labels = torch.arange(len(p), device=x.device)
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


def record_error(model, values: np.ndarray, device: torch.device,
                 rotation: np.ndarray | None = None,
                 quantile: float = .95) -> float:
    errors = reconstruction_errors(model, values, device, rotation)
    return float(np.quantile(errors, quantile))


@torch.no_grad()
def permutation_audit(model: SensorPoolingCPC, features: int,
                      device: torch.device) -> float:
    """Return the maximum change after permuting complete sensor identities."""
    values = torch.linspace(-1., 1., steps=2 * 5 * features,
                            device=device).reshape(1, 2, 5, features)
    mask = torch.ones((1, 2, 5), dtype=torch.bool, device=device)
    permutation = torch.tensor([2, 4, 0, 3, 1], device=device)
    original = model.sensor_pool(values, mask)
    changed = model.sensor_pool(values[:, :, permutation],
                                mask[:, :, permutation])
    return float((original - changed).abs().max().cpu())


def fit_downstream(cache, parts, args, device, seed):
    """Fit a fixed AE+Procrustes detector and healthy-only thresholds."""
    source_train, target_train, source_val, target_val, _, _ = parts
    control.seed_all(seed + 100_000)
    scaled, embedding_center, embedding_scale = control.prepare_variant(
        cache, source_train
    )
    source_x = np.concatenate([scaled[x] for x in source_train.record_id])
    target_x = np.concatenate([scaled[x] for x in target_train.record_id])
    source_val_x = np.concatenate([scaled[x] for x in source_val.record_id])
    ae_args = SimpleNamespace(
        latent=args.ae_latent, dropout=args.dropout, lr=args.ae_lr,
        weight_decay=args.weight_decay, epochs=args.downstream_epochs,
        patience=args.downstream_patience, batch_size=args.batch_size,
    )
    model, history = train_source_ae(source_x, source_val_x, ae_args, device)
    rotation = fit_rotation(
        encode(model, source_x, device), encode(model, target_x, device)
    )

    source_calibration = {
        row.record_id: record_error(model, scaled[row.record_id], device)
        for row in source_val.itertuples()
    }
    target_calibration = {
        row.record_id: record_error(
            model, scaled[row.record_id], device, rotation
        ) for row in target_val.itertuples()
    }
    thresholds = {
        "source": max(max(source_calibration.values()), np.finfo(float).tiny),
        "target": max(max(target_calibration.values()), np.finfo(float).tiny),
    }
    evaluator = {
        "scaled": scaled,
        "embedding_center": embedding_center,
        "embedding_scale": embedding_scale,
        "ae_state": {key: value.detach().cpu().clone()
                     for key, value in model.state_dict().items()},
        "features": int(source_x.shape[1]),
        "rotation": rotation,
        "thresholds": thresholds,
        "healthy_calibration": {
            "source": source_calibration, "target": target_calibration,
        },
        "epochs_run": int(history.epoch.max()),
    }
    return evaluator, history


def restore_ae(evaluator, args, device):
    model = SourceAE(evaluator["features"], args.ae_latent,
                     args.dropout).to(device)
    model.load_state_dict(evaluator["ae_state"])
    model.eval()
    return model


def score_rows(rows, evaluator, args, device, split):
    model = restore_ae(evaluator, args, device)
    output = []
    for row in rows.itertuples():
        domain = modal.domain_for_state(row.state)
        rotation = evaluator["rotation"] if domain == "target" else None
        raw = record_error(model, evaluator["scaled"][row.record_id],
                           device, rotation)
        threshold = evaluator["thresholds"][domain]
        # Do not evaluate a healthy validation record against a threshold that
        # includes its own error. Damage validation and the locked test use all
        # three healthy calibration records from their matching domain.
        calibration = evaluator["healthy_calibration"][domain]
        if split == "validation" and int(row.label) == 0 and \
                row.record_id in calibration:
            leave_one_out = [value for record_id, value in calibration.items()
                             if record_id != row.record_id]
            threshold = max(max(leave_one_out), np.finfo(float).tiny)
        score = raw / threshold
        output.append({
            "record_id": row.record_id, "state": row.state,
            "scenario": row.scenario, "excitation": row.excitation,
            "setup": int(row.setup), "label": int(row.label),
            "split": split, "domain": domain, "raw_score": raw,
            "domain_threshold": threshold,
            "score": score, "alarm": score > 1.,
        })
    return pd.DataFrame(output)


def aggregate_metrics(frame: pd.DataFrame, group: str) -> dict:
    result = common.metrics(frame.label, frame.score, 1.)
    return {"evaluation": group, **result}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    return seeds


def make_figures(validation_metrics, aggregate, ensemble_scores, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    variants = list(aggregate.preprocessing)
    axes[0, 0].boxplot([
        validation_metrics.loc[
            validation_metrics.preprocessing == name, "auroc"
        ].values for name in variants
    ], labels=variants, showmeans=True)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set(title="Validation AUROC across seeds", ylim=(0, 1))
    axes[0, 1].bar(variants, aggregate.auroc_mean)
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set(title="Selection metric: mean validation AUROC", ylim=(0, 1))

    fpr, tpr, _ = roc_curve(ensemble_scores.label, ensemble_scores.score)
    metric = common.metrics(ensemble_scores.label, ensemble_scores.score, 1.)
    axes[1, 0].plot(fpr, tpr, label=f"AUROC={metric['auroc']:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].legend()
    axes[1, 0].set(title="Selected preprocessing: locked-test ROC",
                   xlabel="False-positive rate", ylabel="True-positive rate")
    ConfusionMatrixDisplay(
        confusion_matrix(ensemble_scores.label, ensemble_scores.alarm,
                         labels=[0, 1]),
        display_labels=["Healthy", "Damage"],
    ).plot(ax=axes[1, 1], cmap="Blues", colorbar=False)
    axes[1, 1].set_title("Selected preprocessing: seed-ensemble confusion")
    fig.tight_layout()
    fig.savefig(output / "preprocessing_selection.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def make_training_curves(history: pd.DataFrame, selected: str,
                         output: Path) -> None:
    chosen = history[history.preprocessing == selected]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specifications = (
        ("cpc", "train_objective", "healthy_val_objective", "CPC"),
        ("compact_vicreg", "train_objective", "healthy_val_objective",
         "Compact VICReg"),
        ("downstream_ae", "train_loss", "source_val_loss", "Downstream AE"),
    )
    for axis, (stage, train_column, val_column, title) in zip(axes, specifications):
        stage_rows = chosen[chosen.reported_stage == stage]
        for seed, group in stage_rows.groupby("seed"):
            axis.plot(group.epoch, group[train_column], alpha=.75,
                      label=f"train seed {seed}")
            axis.plot(group.epoch, group[val_column], "--", alpha=.9,
                      label=f"healthy val seed {seed}")
        axis.set(title=title, xlabel="Epoch", ylabel="Objective")
        axis.grid(alpha=.25)
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Selected preprocessing training curves: {selected}")
    fig.tight_layout()
    fig.savefig(output / "selected_training_curves.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/z24_preprocessing_selection"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--fs", type=float, default=50.)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--psd-bins", type=int, default=96)
    parser.add_argument("--fmin", type=float, default=3.)
    parser.add_argument("--fmax", type=float, default=30.)
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--segment-stride", type=int, default=4)
    parser.add_argument("--prediction-steps", type=int, default=3)
    parser.add_argument("--cpc-latent", type=int, default=32)
    parser.add_argument("--cpc-context", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--aug-noise", type=float, default=.02)
    parser.add_argument("--mask-probability", type=float, default=.03)
    parser.add_argument("--cpc-epochs", type=int, default=50)
    parser.add_argument("--cpc-patience", type=int, default=10)
    parser.add_argument("--cpc-lr", type=float, default=1e-3)
    parser.add_argument("--compact-epochs", type=int, default=30)
    parser.add_argument("--compact-patience", type=int, default=8)
    parser.add_argument("--compact-lr", type=float, default=3e-4)
    parser.add_argument("--compact-weight", type=float, default=1.)
    parser.add_argument("--consistency-weight", type=float, default=.5)
    parser.add_argument("--compact-cpc-weight", type=float, default=.2)
    parser.add_argument("--variance-weight", type=float, default=.25)
    parser.add_argument("--covariance-weight", type=float, default=.05)
    parser.add_argument("--variance-target", type=float, default=1.)
    parser.add_argument("--topk-fraction", type=float, default=.25)
    parser.add_argument("--downstream-epochs", type=int, default=100)
    parser.add_argument("--downstream-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ae-latent", type=int, default=4)
    parser.add_argument("--ae-lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=.2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choices={VARIANTS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.split == "train") & (frame.state == "01")].copy()
    target_train = frame[(frame.split == "train") & (frame.state == "08")].copy()
    source_val = frame[(frame.split == "val") & (frame.state == "01")].copy()
    target_val = frame[(frame.split == "val") & (frame.state == "08")].copy()
    val_rows = frame[frame.split == "val"].copy()
    test_rows = frame[frame.split == "test"].copy()
    parts = (source_train, target_train, source_val, target_val,
             val_rows, test_rows)
    counts = tuple(map(len, parts))
    expected = (12, 12, 3, 3, 132, 132)
    if counts != expected:
        raise ValueError(f"Unexpected split {counts}; expected {expected}")
    print(f"device={device} split={counts} seeds={seeds} variants={variants}",
          flush=True)
    print("damage_used_for_gradient=False damage_val_used_for_selection=True "
          "test_used_for_selection=False", flush=True)

    started = time.time()
    raw = {}
    for index, row in enumerate(frame.itertuples(), 1):
        raw[row.record_id] = spatial.spatial_psd(
            row.path, args.fs, args.window, args.stride, args.psd_bins,
            args.fmin, args.fmax,
        )
        if index % 24 == 0 or index == len(frame):
            print(f"spatial PSD={index}/{len(frame)}", flush=True)
    psd_center, psd_scale = spatial.fit_spatial_robust(
        [raw[key] for key in source_train.record_id]
    )
    scaled = spatial.scale_spatial(raw, psd_center, psd_scale)
    max_sensors = max(scaled[key].shape[1] for key in source_train.record_id)
    unseen_widths = {key: value.shape[1] for key, value in scaled.items()
                     if value.shape[1] > max_sensors}
    if unseen_widths:
        raise ValueError("Evaluation contains more sensors than healthy train: "
                         f"{unseen_widths}")
    train, train_mask = spatial.spatial_segments(
        [scaled[key] for key in source_train.record_id], args.context_length,
        args.segment_stride, max_sensors,
    )
    healthy_validation, healthy_validation_mask = spatial.spatial_segments(
        [scaled[key] for key in source_val.record_id], args.context_length,
        args.segment_stride, max_sensors,
    )

    validation_metrics = []
    validation_scores = []
    histories = []
    evaluators = {}
    representation_states = {}
    pooling_audits = []
    for variant in variants:
        for seed in seeds:
            print(f"\n=== preprocessing={variant} seed={seed} ===", flush=True)
            control.seed_all(seed)
            model = SensorPoolingCPC(
                args.psd_bins, args.cpc_latent, args.cpc_context,
                args.prediction_steps, variant, args.topk_fraction,
            ).to(device)
            parameters = sum(value.numel() for value in model.parameters())
            permutation_difference = permutation_audit(
                model, args.psd_bins, device
            )
            if permutation_difference > 1e-5:
                raise RuntimeError(
                    f"Pooling is not permutation invariant: {variant} "
                    f"difference={permutation_difference}"
                )
            pooling_audits.append({
                "preprocessing": variant, "seed": seed,
                "permutation_max_abs_difference": permutation_difference,
                "passed": True,
            })
            model, first_history = spatial.train_spatial(
                model, train, train_mask, healthy_validation,
                healthy_validation_mask, args, device, args.cpc_epochs,
                args.cpc_patience, args.cpc_lr,
            )
            healthy_center = spatial.spatial_center(
                model, train, train_mask, args.batch_size, device
            )
            model, second_history = spatial.train_spatial(
                model, train, train_mask, healthy_validation,
                healthy_validation_mask, args, device, args.compact_epochs,
                args.compact_patience, args.compact_lr, healthy_center, True,
            )
            representation = spatial.embed_spatial(
                model, scaled, max_sensors, device
            )
            evaluator, ae_history = fit_downstream(
                representation, parts, args, device, seed
            )
            scores = score_rows(val_rows, evaluator, args, device, "validation")
            metric = aggregate_metrics(scores, "validation")
            validation_metrics.append({
                "preprocessing": variant, "seed": seed,
                "parameters": parameters, "representation_epochs":
                int(first_history.epoch.max() + second_history.epoch.max()),
                "downstream_epochs": evaluator["epochs_run"], **metric,
            })
            scores.insert(0, "preprocessing", variant)
            scores.insert(1, "seed", seed)
            validation_scores.append(scores)
            for stage_history, stage in ((first_history, "cpc"),
                                         (second_history, "compact_vicreg"),
                                         (ae_history, "downstream_ae")):
                stage_history = stage_history.copy()
                stage_history.insert(0, "preprocessing", variant)
                stage_history.insert(1, "seed", seed)
                stage_history["reported_stage"] = stage
                histories.append(stage_history)
            evaluators[(variant, seed)] = evaluator
            representation_states[(variant, seed)] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print(f"validation AUROC={metric['auroc']:.4f} "
                  f"BA={metric['balanced_accuracy']:.4f} "
                  f"recall={metric['recall']:.4f} "
                  f"specificity={metric['specificity']:.4f}", flush=True)

    validation_metrics_frame = pd.DataFrame(validation_metrics)
    validation_scores_frame = pd.concat(validation_scores, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True, sort=False)
    aggregate = validation_metrics_frame.groupby("preprocessing", as_index=False).agg(
        seeds=("seed", "nunique"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        recall_mean=("recall", "mean"), specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"), parameters=("parameters", "first"),
    )
    aggregate = aggregate.sort_values(
        ["auroc_mean", "balanced_accuracy_mean", "specificity_mean"],
        ascending=False,
    ).reset_index(drop=True)
    selected = str(aggregate.iloc[0].preprocessing)
    aggregate["selected"] = aggregate.preprocessing == selected
    print(f"\nSELECTED FROM VALIDATION ONLY: {selected}", flush=True)

    test_metrics = []
    test_scores = []
    for seed in seeds:
        evaluator = evaluators[(selected, seed)]
        scores = score_rows(test_rows, evaluator, args, device, "locked_test")
        metric = aggregate_metrics(scores, "locked_test")
        test_metrics.append({"preprocessing": selected, "seed": seed, **metric})
        scores.insert(0, "preprocessing", selected)
        scores.insert(1, "seed", seed)
        test_scores.append(scores)
    test_metrics_frame = pd.DataFrame(test_metrics)
    test_scores_frame = pd.concat(test_scores, ignore_index=True)
    ensemble = test_scores_frame.groupby(
        ["record_id", "state", "scenario", "excitation", "setup", "label"],
        as_index=False,
    ).agg(score=("score", "mean"), raw_score=("raw_score", "mean"))
    ensemble["alarm"] = ensemble.score > 1.
    ensemble_metrics = common.metrics(ensemble.label, ensemble.score, 1.)
    state_scores = test_scores_frame.groupby(
        ["seed", "state", "scenario", "label"], as_index=False
    ).agg(records=("record_id", "size"), score_median=("score", "median"),
          alarm_rate=("alarm", "mean"))
    ensemble_states = ensemble.groupby(
        ["state", "scenario", "label"], as_index=False
    ).agg(records=("record_id", "size"), score_median=("score", "median"),
          alarm_rate=("alarm", "mean"))

    validation_metrics_frame.to_csv(args.output / "validation_metrics_by_seed.csv",
                                    index=False)
    validation_scores_frame.to_csv(args.output / "validation_record_scores.csv",
                                   index=False)
    aggregate.to_csv(args.output / "preprocessing_ranking.csv", index=False)
    history_frame.to_csv(args.output / "training_histories.csv", index=False)
    pd.DataFrame(pooling_audits).to_csv(
        args.output / "pooling_permutation_audit.csv", index=False
    )
    test_metrics_frame.to_csv(args.output / "selected_test_metrics_by_seed.csv",
                              index=False)
    test_scores_frame.to_csv(args.output / "selected_test_record_scores.csv",
                             index=False)
    ensemble.to_csv(args.output / "selected_test_seed_ensemble_scores.csv",
                    index=False)
    state_scores.to_csv(args.output / "selected_test_state_scores_by_seed.csv",
                        index=False)
    ensemble_states.to_csv(args.output / "selected_test_state_scores_ensemble.csv",
                           index=False)
    pd.DataFrame([{ "preprocessing": selected, **ensemble_metrics }]).to_csv(
        args.output / "selected_test_seed_ensemble_metrics.csv", index=False
    )
    make_figures(validation_metrics_frame, aggregate, ensemble, args.output)
    make_training_curves(history_frame, selected, args.output)

    checkpoint = {
        "selected_preprocessing": selected,
        "seeds": seeds,
        "representation_states": {
            str(seed): representation_states[(selected, seed)] for seed in seeds
        },
        "downstream": {
            str(seed): {key: value for key, value in evaluators[(selected, seed)].items()
                        if key != "scaled"}
            for seed in seeds
        },
        "psd_center": psd_center, "psd_scale": psd_scale,
        "max_sensors": max_sensors, "parameters": vars(args),
    }
    torch.save(checkpoint, args.output / "selected_preprocessing.pt")
    metadata = {
        "step": "preprocessing selection",
        "selection_rule": "highest mean validation AUROC; ties: balanced accuracy then specificity",
        "selected_preprocessing": selected,
        "damage_used_for_gradient_scaling_alignment_threshold": False,
        "damage_validation_used_for_selection": True,
        "locked_test_used_for_selection": False,
        "healthy_threshold": "maximum q95 record error over three healthy validation records per domain; leave-one-record-out for healthy validation scoring",
        "source_domain": "S01-S07", "target_domain": "S08-S17",
        "topk_rule": "rank whole sensors by learned attention; fixed 25%; trimmed variant drops strongest sensor identity",
        "legacy_featurewise_topk_included": False,
        "elapsed_seconds": time.time() - started,
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nVALIDATION RANKING\n" + aggregate.to_string(index=False), flush=True)
    print("\nSELECTED LOCKED-TEST METRICS BY SEED\n" +
          test_metrics_frame.to_string(index=False), flush=True)
    print("\nSELECTED SEED-ENSEMBLE TEST RESULT\n" +
          pd.DataFrame([{ "preprocessing": selected,
                         **ensemble_metrics }]).to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

