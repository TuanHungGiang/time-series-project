"""Strict healthy-only spatial Compact-CPC ablation for Z24.

This benchmark extends ``run_z24_healthy_cpc_benchmark.py`` without changing
that control experiment.  It tests, one change at a time:

1. sensor-set CPC with learned attention plus mean/max pooling;
2. healthy compactness and view consistency;
3. variance/covariance anti-collapse regularisation; and
4. fusion with healthy-train modal coordinates.

All representation/scaler/anchor fitting uses S01/train.  Early stopping and
threshold calibration use S01/val.  Damage is evaluation-only.  S08 healthy is
used only by the downstream Orthogonal-Procrustes alignment, exactly as in the
paper-derived backbone.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import detrend, welch
from scipy.stats import genpareto
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_z24_healthy_cpc_benchmark as control
import run_z24_strict_psd_scatterad as common
import z24_modal_transfer_common as modal
from run_z24_procrustes_ae import (
    encode, fit_rotation, reconstruction_errors, train_source_ae,
)


def spatial_psd(path: Path, fs: float, window: int, stride: int, bins: int,
                fmin: float, fmax: float) -> np.ndarray:
    """Return [time, sensor, frequency] energy-normalised log-Welch PSD."""
    values = common.vertical_values(path)
    grid = np.linspace(fmin, fmax, bins)
    result = []
    for start in range(0, len(values) - window + 1, stride):
        sensors = []
        for channel in range(values.shape[1]):
            signal = detrend(values[start:start + window, channel], type="linear")
            frequency, power = welch(signal, fs=fs, nperseg=min(512, window),
                                     noverlap=min(256, window // 2))
            keep = (frequency >= fmin) & (frequency <= fmax)
            shape = np.interp(grid, frequency[keep], power[keep])
            shape /= max(float(np.trapezoid(shape, grid)), np.finfo(float).tiny)
            sensors.append(np.log10(shape + np.finfo(float).tiny))
        result.append(np.stack(sensors))
    if len(result) < 8:
        raise ValueError(f"Too few PSD windows: {path}")
    return np.asarray(result, np.float32)


def fit_spatial_robust(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([x.reshape(-1, x.shape[-1]) for x in parts])
    center = np.median(pooled, axis=0)
    scale = 1.4826 * np.median(np.abs(pooled - center), axis=0)
    positive = scale[scale > 1e-6]
    floor = max(float(np.median(positive)) * .02 if len(positive) else 0., 1e-4)
    return center.astype(np.float32), np.maximum(scale, floor).astype(np.float32)


def scale_spatial(cache, center, scale):
    return {key: np.clip((value - center[None, None, :]) /
                         scale[None, None, :], -10., 10.).astype(np.float32)
            for key, value in cache.items()}


def spatial_segments(parts: list[np.ndarray], length: int, stride: int,
                     max_sensors: int) -> tuple[np.ndarray, np.ndarray]:
    samples, masks = [], []
    for value in parts:
        if value.shape[1] > max_sensors:
            raise ValueError(f"Record has {value.shape[1]} sensors, max={max_sensors}")
        for start in range(0, len(value) - length + 1, stride):
            block = value[start:start + length]
            padded = np.zeros((length, max_sensors, value.shape[-1]), np.float32)
            mask = np.zeros((length, max_sensors), bool)
            padded[:, :value.shape[1]] = block
            mask[:, :value.shape[1]] = True
            samples.append(padded); masks.append(mask)
    if not samples:
        raise ValueError("No spatial CPC segments; reduce --context-length")
    return np.stack(samples), np.stack(masks)


class SpatialCPC(nn.Module):
    """Permutation-invariant sensor encoder retaining local max deviations."""
    def __init__(self, features: int, latent: int, context: int,
                 prediction_steps: int):
        super().__init__(); self.prediction_steps = prediction_steps
        self.sensor_encoder = nn.Sequential(
            nn.Linear(features, 128), nn.ELU(), nn.Linear(128, latent),
            nn.LayerNorm(latent),
        )
        self.attention = nn.Sequential(nn.Linear(latent, latent // 2), nn.Tanh(),
                                       nn.Linear(latent // 2, 1))
        self.pool_projection = nn.Sequential(nn.Linear(3 * latent, latent),
                                             nn.ELU(), nn.LayerNorm(latent))
        self.context = nn.GRU(latent, context, batch_first=True)
        self.predictors = nn.ModuleList([nn.Linear(context, latent)
                                         for _ in range(prediction_steps)])

    def sensor_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.sensor_encoder(x)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=2).clamp_min(1)
        mean = (z * valid).sum(dim=2) / count
        maximum = z.masked_fill(~valid, torch.finfo(z.dtype).min).max(dim=2).values
        logits = self.attention(z).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=2).unsqueeze(-1)
        attended = (weights * z).sum(dim=2)
        return self.pool_projection(torch.cat([mean, maximum, attended], dim=-1))

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.sensor_pool(x, mask)
        context, _ = self.context(pooled)
        return context

    def predictive_loss(self, x: torch.Tensor, mask: torch.Tensor,
                        temperature: float) -> torch.Tensor:
        z = F.normalize(self.sensor_pool(x, mask), dim=-1)
        context, _ = self.context(z); losses = []
        for index, step in enumerate(range(1, self.prediction_steps + 1)):
            predicted = F.normalize(self.predictors[index](context[:, :-step]), dim=-1)
            target = z[:, step:]
            p = predicted.reshape(-1, predicted.shape[-1])
            q = target.reshape(-1, target.shape[-1])
            logits = p @ q.T / temperature
            labels = torch.arange(len(p), device=x.device)
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


def spatial_view(x: torch.Tensor, mask: torch.Tensor, noise: float,
                 mask_probability: float) -> torch.Tensor:
    view = x + noise * torch.randn_like(x) * mask.unsqueeze(-1)
    if mask_probability > 0:
        frequency_mask = (torch.rand(x.shape[0], 1, 1, x.shape[-1], device=x.device)
                          < mask_probability)
        view = view.masked_fill(frequency_mask, 0.)
    return view * mask.unsqueeze(-1)


def spatial_objective(model, x, mask, args, center=None,
                      anti_collapse=False):
    a = spatial_view(x, mask, args.aug_noise, args.mask_probability)
    b = spatial_view(x, mask, args.aug_noise, args.mask_probability)
    predictive = .5 * (model.predictive_loss(a, mask, args.temperature) +
                        model.predictive_loss(b, mask, args.temperature))
    if center is None:
        return predictive, {"predictive": predictive.detach()}
    ca, cb = model.embed(a, mask), model.embed(b, mask)
    compact = .5 * (((ca - center.view(1, 1, -1)) ** 2).mean() +
                    ((cb - center.view(1, 1, -1)) ** 2).mean())
    consistency = 1. - F.cosine_similarity(ca, cb, dim=-1).mean()
    variance = torch.zeros((), device=x.device); covariance = variance
    if anti_collapse:
        flat = torch.cat([ca.reshape(-1, ca.shape[-1]),
                          cb.reshape(-1, cb.shape[-1])], dim=0)
        variance = F.relu(args.variance_target -
                          torch.sqrt(flat.var(dim=0) + 1e-4)).mean()
        centered = flat - flat.mean(dim=0)
        cov = centered.T @ centered / max(len(flat) - 1, 1)
        covariance = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / cov.shape[0]
    loss = (args.compact_cpc_weight * predictive + args.compact_weight * compact +
            args.consistency_weight * consistency +
            (args.variance_weight * variance + args.covariance_weight * covariance
             if anti_collapse else 0.))
    return loss, {"predictive": predictive.detach(), "compact": compact.detach(),
                  "consistency": consistency.detach(), "variance": variance.detach(),
                  "covariance": covariance.detach()}


@torch.no_grad()
def spatial_validation(model, arrays, masks, args, device, center=None,
                       anti_collapse=False):
    model.eval(); values = []
    data = TensorDataset(torch.from_numpy(arrays), torch.from_numpy(masks))
    for x, mask in DataLoader(data, batch_size=args.batch_size):
        loss, _ = spatial_objective(model, x.to(device), mask.to(device), args,
                                    center, anti_collapse)
        values.append(float(loss.cpu()))
    return float(np.mean(values))


def train_spatial(model, train, train_mask, validation, validation_mask,
                  args, device, epochs, patience, lr, center=None,
                  anti_collapse=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(train), torch.from_numpy(train_mask))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); totals = []
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = spatial_objective(model, x, mask, args, center,
                                              anti_collapse)
            loss.backward(); optimizer.step(); totals.append(float(loss.detach().cpu()))
        value = spatial_validation(model, validation, validation_mask, args,
                                   device, center, anti_collapse)
        history.append({"epoch": epoch, "train_objective": np.mean(totals),
                        "healthy_val_objective": value,
                        "stage": "anti_collapse" if anti_collapse else
                                 ("compact" if center is not None else "cpc")})
        if value < best - 1e-5:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"spatial stage={history[-1]['stage']} epoch={epoch}/{epochs} "
                  f"train={np.mean(totals):.6f} val={value:.6f}", flush=True)
        if stale >= patience:
            break
    if state is None:
        raise RuntimeError("Spatial CPC produced no checkpoint")
    model.load_state_dict(state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def spatial_center(model, arrays, masks, batch_size, device):
    model.eval(); values = []
    data = TensorDataset(torch.from_numpy(arrays), torch.from_numpy(masks))
    for x, mask in DataLoader(data, batch_size=batch_size):
        values.append(model.embed(x.to(device), mask.to(device)).reshape(
            -1, model.context.hidden_size))
    center = torch.cat(values).mean(dim=0)
    eps = 1e-3
    center[(center.abs() < eps) & (center < 0)] = -eps
    center[(center.abs() < eps) & (center >= 0)] = eps
    return center.detach()


@torch.no_grad()
def embed_spatial(model, cache, max_sensors, device):
    model.eval(); result = {}
    for key, value in cache.items():
        padded = np.zeros((1, len(value), max_sensors, value.shape[-1]), np.float32)
        mask = np.zeros((1, len(value), max_sensors), bool)
        padded[0, :, :value.shape[1]] = value; mask[0, :, :value.shape[1]] = True
        result[key] = model.embed(torch.from_numpy(padded).to(device),
                                  torch.from_numpy(mask).to(device))[0].cpu().numpy().astype(np.float32)
    return result


def paired_healthy_segments(cache, source_rows, target_rows, length, stride,
                            max_sensors):
    """Match only healthy train records sharing excitation and setup.

    Pairing is fixed entirely by manifest metadata.  No damage state or score
    participates.  Segment indices are paired only within a matched record;
    the refinement loss compares segment distributions rather than assuming
    sample-level phase synchronisation across campaigns.
    """
    target_index = {(str(row.excitation).lower(), int(row.setup)): row.record_id
                    for row in target_rows.itertuples()}
    source_parts, source_masks, target_parts, target_masks, audit = [], [], [], [], []
    for row in source_rows.itertuples():
        key = (str(row.excitation).lower(), int(row.setup))
        if key not in target_index:
            continue
        source_x, source_mask = spatial_segments([cache[row.record_id]], length,
                                                 stride, max_sensors)
        target_id = target_index[key]
        target_x, target_mask = spatial_segments([cache[target_id]], length,
                                                 stride, max_sensors)
        count = min(len(source_x), len(target_x))
        source_parts.append(source_x[:count]); source_masks.append(source_mask[:count])
        target_parts.append(target_x[:count]); target_masks.append(target_mask[:count])
        audit.append({"excitation": key[0], "setup": key[1],
                      "source_record": row.record_id, "target_record": target_id,
                      "paired_segments": count})
    if not source_parts:
        raise ValueError("No matched S01/S08 healthy train pairs")
    return (np.concatenate(source_parts), np.concatenate(source_masks),
            np.concatenate(target_parts), np.concatenate(target_masks),
            pd.DataFrame(audit))


def _off_diagonal_covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    covariance = x.T @ x / max(len(x) - 1, 1)
    return (covariance - torch.diag(torch.diag(covariance))).pow(2).sum() / covariance.shape[0]


def paired_distribution_loss(source_context, target_context, args):
    """VICReg-style healthy domain invariance without temporal phase pairing."""
    source_summary = source_context.mean(dim=1)
    target_summary = target_context.mean(dim=1)
    invariance = (1. - F.cosine_similarity(source_summary, target_summary, dim=-1).mean() +
                  .1 * F.mse_loss(source_summary, target_summary))
    combined = torch.cat([source_context.reshape(-1, source_context.shape[-1]),
                          target_context.reshape(-1, target_context.shape[-1])])
    variance = F.relu(args.variance_target -
                      torch.sqrt(combined.var(dim=0) + 1e-4)).mean()
    covariance = _off_diagonal_covariance(combined)
    return invariance, variance, covariance


@torch.no_grad()
def paired_validation_loss(model, source, source_mask, target, target_mask,
                           args, device):
    """Healthy-only validation distribution discrepancy; setup pairing not required."""
    model.eval()

    def contexts(values, masks):
        output = []
        data = TensorDataset(torch.from_numpy(values), torch.from_numpy(masks))
        for x, mask in DataLoader(data, batch_size=args.batch_size):
            output.append(model.embed(x.to(device), mask.to(device)).mean(dim=1))
        return torch.cat(output)

    source_z, target_z = contexts(source, source_mask), contexts(target, target_mask)
    mean_loss = F.mse_loss(source_z.mean(dim=0), target_z.mean(dim=0))
    source_centered = source_z - source_z.mean(dim=0)
    target_centered = target_z - target_z.mean(dim=0)
    source_cov = source_centered.T @ source_centered / max(len(source_z) - 1, 1)
    target_cov = target_centered.T @ target_centered / max(len(target_z) - 1, 1)
    covariance_loss = F.mse_loss(source_cov, target_cov)
    return float((mean_loss + .1 * covariance_loss).cpu())


def train_paired_vicreg(model, paired, source_validation, target_validation,
                        args, device):
    """Remove lifting-domain information using matched healthy train pairs."""
    sx, sm, tx, tm = paired
    sv, svm = source_validation; tv, tvm = target_validation
    dataset = TensorDataset(torch.from_numpy(sx), torch.from_numpy(sm),
                            torch.from_numpy(tx), torch.from_numpy(tm))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.paired_lr,
                                 weight_decay=args.weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.paired_epochs + 1):
        model.train(); totals, invariances = [], []
        for source_x, source_mask, target_x, target_mask in loader:
            source_x, source_mask = source_x.to(device), source_mask.to(device)
            target_x, target_mask = target_x.to(device), target_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            source_view = spatial_view(source_x, source_mask, args.aug_noise,
                                       args.mask_probability)
            target_view = spatial_view(target_x, target_mask, args.aug_noise,
                                       args.mask_probability)
            source_context = model.embed(source_view, source_mask)
            target_context = model.embed(target_view, target_mask)
            invariance, variance, covariance = paired_distribution_loss(
                source_context, target_context, args)
            predictive = .5 * (
                model.predictive_loss(source_view, source_mask, args.temperature) +
                model.predictive_loss(target_view, target_mask, args.temperature))
            loss = (args.paired_domain_weight * invariance +
                    args.paired_cpc_weight * predictive +
                    args.variance_weight * variance +
                    args.covariance_weight * covariance)
            loss.backward(); optimizer.step()
            totals.append(float(loss.detach().cpu()))
            invariances.append(float(invariance.detach().cpu()))
        validation = paired_validation_loss(model, sv, svm, tv, tvm, args, device)
        history.append({"epoch": epoch, "train_objective": np.mean(totals),
                        "paired_invariance": np.mean(invariances),
                        "healthy_domain_val_loss": validation})
        if validation < best - 1e-6:
            best, state, stale = validation, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"paired-vicreg epoch={epoch}/{args.paired_epochs} "
                  f"train={np.mean(totals):.6f} val-domain={validation:.6f}", flush=True)
        if stale >= args.paired_patience:
            break
    if state is None:
        raise RuntimeError("Paired VICReg produced no checkpoint")
    model.load_state_dict(state)
    return model, pd.DataFrame(history)


def nuisance_projection(cache, source_rows, target_rows, rank):
    """Project out paired healthy S08-S01 difference directions."""
    target_index = {(str(row.excitation).lower(), int(row.setup)): row.record_id
                    for row in target_rows.itertuples()}
    differences = []
    pair_count = 0
    for row in source_rows.itertuples():
        key = (str(row.excitation).lower(), int(row.setup))
        if key not in target_index:
            continue
        source = cache[row.record_id]
        target = interpolate_rows(cache[target_index[key]], len(source))
        differences.append(target - source); pair_count += 1
    if not differences:
        raise ValueError("No healthy pairs for nuisance projection")
    matrix = np.concatenate(differences)
    # Do not mean-centre: the mean lifting direction is itself nuisance.
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank].T.astype(np.float32)
    projection = np.eye(basis.shape[0], dtype=np.float32) - basis @ basis.T
    source_center = np.mean(np.concatenate([cache[x] for x in source_rows.record_id]),
                            axis=0).astype(np.float32)
    transformed = {key: ((value - source_center) @ projection).astype(np.float32)
                   for key, value in cache.items()}
    energy = singular ** 2
    audit = {"rank": rank, "healthy_pair_count": pair_count,
             "explained_nuisance_energy": float(energy[:rank].sum() /
                                                 max(energy.sum(), 1e-12)),
             "singular_values": singular[:min(8, len(singular))].tolist()}
    return transformed, basis, audit


def interpolate_rows(value: np.ndarray, length: int) -> np.ndarray:
    if len(value) == length:
        return value.astype(np.float32)
    old = np.linspace(0., 1., len(value)); new = np.linspace(0., 1., length)
    return np.stack([np.interp(new, old, value[:, j])
                     for j in range(value.shape[1])], axis=1).astype(np.float32)


def modal_cache(frame, source_train, args):
    spectra, frequency = {}, None
    for row in frame.itertuples():
        f, spectrum = modal.record_spectra(row.path, args.fs, args.modal_window,
                                           args.modal_stride, args.fmin,
                                           min(args.modal_fmax, args.fmax))
        if frequency is None:
            frequency = f
        spectra[row.record_id] = spectrum
    anchors = modal.learn_anchors([spectra[x] for x in source_train.record_id],
                                  frequency, args.modes, args.mode_separation)
    features = {key: modal.track_modes_centroid(frequency, value, anchors,
                                                 args.tracking_radius)
                for key, value in spectra.items()}
    center, scale = modal.robust_scaler(np.concatenate(
        [features[x] for x in source_train.record_id]))
    return {key: modal.scale(value, center, scale) for key, value in features.items()}, anchors


def evt_threshold(errors: np.ndarray, tail_quantile: float, alpha: float):
    """Healthy-only peaks-over-threshold GPD; empirical fallback is audited."""
    errors = np.asarray(errors, float)
    boundary = float(np.quantile(errors, tail_quantile))
    excess = errors[errors > boundary] - boundary
    tail_probability = max(1. - tail_quantile, np.finfo(float).eps)
    if len(excess) < 8 or not np.any(excess > 0):
        return float(np.quantile(errors, 1. - alpha)), "empirical_fallback"
    try:
        shape, _, scale = genpareto.fit(excess, floc=0.)
        conditional = 1. - min(max(alpha / tail_probability, 1e-6), .999999)
        threshold = boundary + float(genpareto.ppf(conditional, shape, loc=0., scale=scale))
        if not np.isfinite(threshold):
            raise ValueError("non-finite EVT threshold")
        return threshold, "gpd_pot"
    except Exception:
        return float(np.quantile(errors, 1. - alpha)), "empirical_fallback"


def score_rows(rows, cache, scorer, threshold, split):
    output = []
    for row in rows.itertuples():
        score = float(np.quantile(scorer(cache[row.record_id],
                                         modal.domain_for_state(row.state)), .95))
        output.append({"record_id": row.record_id, "state": row.state,
                       "scenario": row.scenario, "excitation": row.excitation,
                       "setup": int(row.setup), "label": int(row.label),
                       "split": split, "score": score, "alarm": score > threshold})
    return pd.DataFrame(output)


def procrustes_decisions(variant, cache, parts, args, device):
    # Paired ablation: identical downstream initialisation/shuffle per feature
    # variant, so representation changes are not confounded by AE RNG order.
    control.seed_all(args.seed)
    source_train, target_train, source_val, target_val, test_rows = parts
    scaled, _, _ = control.prepare_variant(cache, source_train)
    sx = np.concatenate([scaled[x] for x in source_train.record_id])
    tx = np.concatenate([scaled[x] for x in target_train.record_id])
    sv = np.concatenate([scaled[x] for x in source_val.record_id])
    tv = np.concatenate([scaled[x] for x in target_val.record_id])
    ae_args = SimpleNamespace(latent=args.ae_latent, dropout=args.dropout,
                              lr=args.ae_lr, weight_decay=args.weight_decay,
                              epochs=args.downstream_epochs,
                              patience=args.downstream_patience,
                              batch_size=args.batch_size)
    model, history = train_source_ae(sx, sv, ae_args, device)
    rotation = fit_rotation(encode(model, sx, device), encode(model, tx, device))
    calibration = reconstruction_errors(model, sv, device)
    target_calibration = reconstruction_errors(model, tv, device, rotation)
    empirical = float(np.quantile(calibration, args.threshold_quantile))
    evt, evt_fit = evt_threshold(calibration, args.evt_tail_quantile, args.evt_alpha)

    # Procrustes creates separate source/target score distributions.  A single
    # source threshold confounds residual domain scale with damage.  Calibrate
    # each scale using only its matching healthy validation partition, then
    # express every error as a ratio to the healthy q99.  Damage remains fully
    # locked and the common decision boundary becomes one.
    domain_thresholds = {
        "source": empirical,
        "target": float(np.quantile(target_calibration, args.threshold_quantile)),
    }

    def scorer(x, domain):
        return reconstruction_errors(model, x, device,
                                     rotation if domain == "target" else None)

    summaries, scores = [], []
    for decision, threshold, fit in (("empirical_q99", empirical, "empirical"),
                                     ("healthy_EVT", evt, evt_fit)):
        test = score_rows(test_rows, scaled, scorer, threshold, "locked_test")
        metrics = common.metrics(test.label, test.score, threshold)
        summaries.append({"backbone": "MSSP2026_AE_OrthogonalProcrustes",
                          "preprocessing": variant, "decision": decision,
                          "threshold_fit": fit, "epochs_run": int(history.epoch.max()),
                          **metrics})
        test.insert(0, "preprocessing", variant); test.insert(1, "decision", decision)
        scores.append(test)

    def normalized_scorer(x, domain):
        return scorer(x, domain) / max(domain_thresholds[domain],
                                       np.finfo(float).tiny)

    calibrated = score_rows(test_rows, scaled, normalized_scorer, 1., "locked_test")
    calibrated_metrics = common.metrics(calibrated.label, calibrated.score, 1.)
    summaries.append({"backbone": "MSSP2026_AE_OrthogonalProcrustes",
                      "preprocessing": variant,
                      "decision": "domain_healthy_q99",
                      "threshold_fit": "S01-val source + S08-val target",
                      "epochs_run": int(history.epoch.max()),
                      "source_threshold": domain_thresholds["source"],
                      "target_threshold": domain_thresholds["target"],
                      **calibrated_metrics})
    calibrated.insert(0, "preprocessing", variant)
    calibrated.insert(1, "decision", "domain_healthy_q99")
    calibrated["raw_domain_threshold"] = calibrated.state.map(
        lambda state: domain_thresholds[modal.domain_for_state(state)])
    scores.append(calibrated)
    return summaries, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("/kaggle/working/z24_spatial_compact_cpc"))
    p.add_argument("--fs", type=float, default=50.)
    p.add_argument("--window", type=int, default=2048); p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--psd-bins", type=int, default=96); p.add_argument("--fmin", type=float, default=3.)
    p.add_argument("--fmax", type=float, default=30.); p.add_argument("--context-length", type=int, default=12)
    p.add_argument("--segment-stride", type=int, default=4); p.add_argument("--prediction-steps", type=int, default=3)
    p.add_argument("--cpc-latent", type=int, default=32); p.add_argument("--cpc-context", type=int, default=32)
    p.add_argument("--temperature", type=float, default=.1); p.add_argument("--aug-noise", type=float, default=.02)
    p.add_argument("--mask-probability", type=float, default=.03)
    p.add_argument("--cpc-epochs", type=int, default=50); p.add_argument("--cpc-patience", type=int, default=10)
    p.add_argument("--cpc-lr", type=float, default=1e-3)
    p.add_argument("--compact-epochs", type=int, default=30); p.add_argument("--compact-patience", type=int, default=8)
    p.add_argument("--compact-lr", type=float, default=3e-4); p.add_argument("--compact-weight", type=float, default=1.)
    p.add_argument("--consistency-weight", type=float, default=.5); p.add_argument("--compact-cpc-weight", type=float, default=.2)
    p.add_argument("--variance-weight", type=float, default=.25); p.add_argument("--covariance-weight", type=float, default=.05)
    p.add_argument("--variance-target", type=float, default=1.)
    p.add_argument("--paired-epochs", type=int, default=30); p.add_argument("--paired-patience", type=int, default=8)
    p.add_argument("--paired-lr", type=float, default=2e-4)
    p.add_argument("--paired-domain-weight", type=float, default=1.)
    p.add_argument("--paired-cpc-weight", type=float, default=.2)
    p.add_argument("--modal-window", type=int, default=8192); p.add_argument("--modal-stride", type=int, default=4096)
    p.add_argument("--modal-fmax", type=float, default=20.); p.add_argument("--modes", type=int, default=6)
    p.add_argument("--mode-separation", type=float, default=.5); p.add_argument("--tracking-radius", type=float, default=.45)
    p.add_argument("--downstream-epochs", type=int, default=100); p.add_argument("--downstream-patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ae-latent", type=int, default=4); p.add_argument("--ae-lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=.2); p.add_argument("--threshold-quantile", type=float, default=.99)
    p.add_argument("--evt-tail-quantile", type=float, default=.8); p.add_argument("--evt-alpha", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    control.seed_all(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = common.resolve_manifest(args.manifest, args.data_root)
    source_train = frame[(frame.split == "train") & (frame.state == "01")].copy()
    target_train = frame[(frame.split == "train") & (frame.state == "08")].copy()
    source_val = frame[(frame.split == "val") & (frame.state == "01")].copy()
    target_val = frame[(frame.split == "val") & (frame.state == "08")].copy()
    test_rows = frame[frame.split == "test"].copy()
    counts = tuple(map(len, (source_train, target_train, source_val, target_val, test_rows)))
    if counts != (12, 12, 3, 3, 132):
        raise ValueError(f"Unexpected split: {counts}")
    print(f"device={device} split={counts} damage_used_for_preprocessing=False", flush=True)

    raw = {}
    for index, row in enumerate(frame.itertuples(), 1):
        raw[row.record_id] = spatial_psd(row.path, args.fs, args.window, args.stride,
                                         args.psd_bins, args.fmin, args.fmax)
        if index % 24 == 0 or index == len(frame):
            print(f"spatial PSD={index}/{len(frame)}", flush=True)
    center, scale = fit_spatial_robust([raw[x] for x in source_train.record_id])
    scaled = scale_spatial(raw, center, scale)
    # Architecture/padding width is fixed from healthy training metadata only.
    max_sensors = max(scaled[x].shape[1] for x in source_train.record_id)

    # Re-run the published current control in the same process/seed so every
    # delta is paired fairly with the new spatial variants.
    median_raw = {key: np.median(value, axis=1).astype(np.float32)
                  for key, value in raw.items()}
    median_scaled, median_center, median_scale = control.prepare_variant(
        median_raw, source_train)
    median_train = control.segments([median_scaled[x] for x in source_train.record_id],
                                    args.context_length, args.segment_stride)
    median_validation = control.segments([median_scaled[x] for x in source_val.record_id],
                                         args.context_length, args.segment_stride)
    median_model, median_h1 = control.train_cpc(median_train, median_validation,
                                               args, device)
    median_healthy = control.healthy_center(median_model, median_train,
                                           args.batch_size, device)
    median_model, median_h2 = control.train_compact_cpc(
        median_model, median_train, median_validation, median_healthy, args, device)
    median_compact = control.embed_records(median_model, median_scaled, device)
    median_h1.to_csv(args.output / "median_control_cpc_history.csv", index=False)
    median_h2.to_csv(args.output / "median_control_compact_history.csv", index=False)

    train, train_mask = spatial_segments([scaled[x] for x in source_train.record_id],
                                         args.context_length, args.segment_stride, max_sensors)
    validation, validation_mask = spatial_segments([scaled[x] for x in source_val.record_id],
                                                    args.context_length, args.segment_stride, max_sensors)
    started = time.time()
    model = SpatialCPC(args.psd_bins, args.cpc_latent, args.cpc_context,
                       args.prediction_steps).to(device)
    model, h1 = train_spatial(model, train, train_mask, validation, validation_mask,
                              args, device, args.cpc_epochs, args.cpc_patience,
                              args.cpc_lr)
    spatial_cpc = embed_spatial(model, scaled, max_sensors, device)
    initial_state = copy.deepcopy(model.state_dict())
    healthy = spatial_center(model, train, train_mask, args.batch_size, device)

    compact_model = copy.deepcopy(model)
    compact_model, h2 = train_spatial(compact_model, train, train_mask, validation,
                                      validation_mask, args, device, args.compact_epochs,
                                      args.compact_patience, args.compact_lr, healthy, False)
    spatial_compact = embed_spatial(compact_model, scaled, max_sensors, device)

    anti_model = SpatialCPC(args.psd_bins, args.cpc_latent, args.cpc_context,
                            args.prediction_steps).to(device)
    anti_model.load_state_dict(initial_state)
    anti_model, h3 = train_spatial(anti_model, train, train_mask, validation,
                                   validation_mask, args, device, args.compact_epochs,
                                   args.compact_patience, args.compact_lr, healthy, True)
    spatial_anti = embed_spatial(anti_model, scaled, max_sensors, device)
    paired_sx, paired_sm, paired_tx, paired_tm, pair_audit = paired_healthy_segments(
        scaled, source_train, target_train, args.context_length,
        args.segment_stride, max_sensors)
    target_validation, target_validation_mask = spatial_segments(
        [scaled[x] for x in target_val.record_id], args.context_length,
        args.segment_stride, max_sensors)
    paired_model = copy.deepcopy(anti_model)
    paired_model, h4 = train_paired_vicreg(
        paired_model, (paired_sx, paired_sm, paired_tx, paired_tm),
        (validation, validation_mask),
        (target_validation, target_validation_mask), args, device)
    paired_cache = embed_spatial(paired_model, scaled, max_sensors, device)
    nuisance_r1, nuisance_basis_r1, nuisance_audit_r1 = nuisance_projection(
        paired_cache, source_train, target_train, rank=1)
    nuisance_r2, nuisance_basis_r2, nuisance_audit_r2 = nuisance_projection(
        paired_cache, source_train, target_train, rank=2)
    pair_audit.to_csv(args.output / "paired_healthy_manifest.csv", index=False)
    (args.output / "nuisance_projection_audit.json").write_text(
        json.dumps({"rank1": nuisance_audit_r1, "rank2": nuisance_audit_r2},
                   indent=2), encoding="utf-8")
    pd.concat([h1, h2, h3], ignore_index=True).to_csv(
        args.output / "spatial_training_history.csv", index=False)
    h4.to_csv(args.output / "paired_vicreg_history.csv", index=False)

    modes, anchors = modal_cache(frame, source_train, args)
    fused = {key: np.concatenate([value, interpolate_rows(modes[key], len(value))], axis=1)
             for key, value in spatial_anti.items()}
    variants = {"median_compact_control": median_compact,
                "spatial_cpc": spatial_cpc,
                "spatial_compact": spatial_compact,
                "spatial_vicreg_compact": spatial_anti,
                "spatial_vicreg_modal_fusion": fused,
                "paired_healthy_vicreg": paired_cache,
                "paired_vicreg_nuisance_r1": nuisance_r1,
                "paired_vicreg_nuisance_r2": nuisance_r2}
    parts = (source_train, target_train, source_val, target_val, test_rows)
    summaries, scores = [], []
    for name, cache in variants.items():
        rows, detail = procrustes_decisions(name, cache, parts, args, device)
        summaries.extend(rows); scores.extend(detail)
        selected = next(x for x in rows if x["decision"] == "domain_healthy_q99")
        print(f"done {name}/domain-calibrated: AUROC={selected['auroc']:.4f} "
              f"BA={selected['balanced_accuracy']:.4f} "
              f"specificity={selected['specificity']:.4f}", flush=True)
    summary = pd.DataFrame(summaries)
    detail = pd.concat(scores, ignore_index=True)
    summary.to_csv(args.output / "metrics.csv", index=False)
    detail.to_csv(args.output / "locked_test_scores.csv", index=False)
    empirical_rows = summary[summary.decision == "empirical_q99"].copy()
    baseline = empirical_rows[empirical_rows.preprocessing ==
                              "median_compact_control"].iloc[0]
    deltas = []
    for row in empirical_rows.itertuples():
        deltas.append({"preprocessing": row.preprocessing,
                       "delta_auroc": row.auroc - baseline.auroc,
                       "delta_balanced_accuracy": row.balanced_accuracy - baseline.balanced_accuracy,
                       "delta_f1": row.f1 - baseline.f1,
                       "delta_specificity": row.specificity - baseline.specificity})
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(args.output / "ablation_deltas_vs_current.csv", index=False)
    domain_rows = summary[summary.decision == "domain_healthy_q99"].copy()
    domain_control = domain_rows[domain_rows.preprocessing ==
                                 "median_compact_control"].iloc[0]
    domain_deltas = []
    for row in domain_rows.itertuples():
        domain_deltas.append({"preprocessing": row.preprocessing,
                              "delta_auroc": row.auroc - domain_control.auroc,
                              "delta_balanced_accuracy": row.balanced_accuracy - domain_control.balanced_accuracy,
                              "delta_f1": row.f1 - domain_control.f1,
                              "delta_specificity": row.specificity - domain_control.specificity})
    pd.DataFrame(domain_deltas).to_csv(
        args.output / "domain_calibrated_deltas_vs_current.csv", index=False)
    states = detail.groupby(["preprocessing", "decision", "state", "scenario", "label"],
                            as_index=False).agg(records=("record_id", "size"),
                                                score_median=("score", "median"),
                                                alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= .5
    states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    torch.save({"spatial_cpc": initial_state,
                "spatial_compact": compact_model.state_dict(),
                "spatial_vicreg_compact": anti_model.state_dict(),
                "paired_healthy_vicreg": paired_model.state_dict(),
                "healthy_center": healthy.cpu(), "psd_center": center,
                "psd_scale": scale, "modal_anchors": anchors,
                "median_psd_center": median_center,
                "median_psd_scale": median_scale,
                "nuisance_basis_rank1": nuisance_basis_r1,
                "nuisance_basis_rank2": nuisance_basis_r2,
                "max_sensors": max_sensors, "args": vars(args)},
               args.output / "spatial_compact_cpc.pt")
    metadata = {
        "protocol": "representation/scalers/anchors S01 train; early stop and thresholds S01 val; damage evaluation only",
        "s08_use": "downstream Procrustes healthy alignment only, not preprocessing",
        "domain_calibration": "S01-val calibrates source; S08-val calibrates target; damage excluded",
        "paired_invariance": "S01/S08 healthy train pairs sharing excitation+setup; damage excluded",
        "nuisance_projection": {"rank1": nuisance_audit_r1,
                                "rank2": nuisance_audit_r2},
        "variants": list(variants), "elapsed_seconds": time.time() - started,
        "warning": "proposed equation adaptation; not author code",
        "parameters": vars(args),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("\nRESULTS\n" + summary.to_string(index=False), flush=True)
    print("\nDELTAS VS CURRENT COMPACT-CPC\n" + delta_frame.to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

