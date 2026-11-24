"""TimesBlock 2 scoring with healthy mask references and observed spectral pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

from .data.hellbridge import development_split
from .data.masking import sensor_mask, stable_rng
from .data.prepare_hellbridge import main as prepare_hellbridge
from .data.z24 import load_manifest, split_training_windows, stable_sensor_map
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .timesnet_native_1024 import Detector, LENGTH, SEED, load_hellbridge, load_z24, stage_features, subset

# Use shared bands within the Z24 antialiasing cutoff.
BANDS = ((1, 4), (4, 8), (8, 16), (16, 22), (22, 30))
RATES = (0.10, 0.30, 0.50)
METHODS = (
    "timesnet_native", "latent_clean_ref", "latent_matched_center",
    "latent_matched_ref", "spectral_pairs", "clmm_mask_pair",
)
FIELDS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_low_fpr", "rank_reversal")


def location_scale(values, floor):
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    return center, np.maximum(np.maximum(mad, 0.25 * values.std(axis=0)), floor)


def band_power(raw):
    # Native acceleration after the shared linear detrend, before sensor-wise scaling.
    frequency, psd = welch(raw, fs=100.0, nperseg=512, noverlap=256, detrend=False, axis=-1)
    return np.stack([
        psd[..., (frequency >= lo) & (frequency < hi)].sum(axis=-1, dtype=np.float64)
        * (frequency[1] - frequency[0])
        for lo, hi in BANDS
    ], axis=-1)


class MaskPairBlock:
    """A frozen-backbone feature block, fitted only on held-back healthy windows."""

    columns = METHODS

    def __init__(self, detector, reference):
        self.detector = detector
        self.reference = detector.scaler.transform(reference)
        self.full = detector.full
        self.left, self.right = np.triu_indices(self.full.size, k=1)
        reference_power = band_power(reference)
        self.power_floor = max(float(np.median(reference_power)) * 1e-8, 1e-20)
        self.ref_ratios = self.ratios(self.log_power(reference_power))
        self.pair_location, self.pair_scale = location_scale(self.ref_ratios, floor=0.05)
        self.cache = {}
        self.clean_latent, _, _ = self.reference_for(self.full)

    def log_power(self, power):
        return np.log(power + self.power_floor)

    def ratios(self, log_power):
        return log_power[:, self.left, :] - log_power[:, self.right, :]

    def infer_latent(self, data, kept):
        features = []

        def record_features(_layer, _inputs, output):
            features.append(stage_features(output.detach().cpu().numpy()))

        hook = self.detector.model.model[1].register_forward_hook(record_features)
        try:
            native, _ = self.detector.score(data, kept)
        finally:
            hook.remove()
        return native, np.concatenate(features)

    def reference_for(self, kept):
        key = kept.tobytes()
        if key not in self.cache:
            _, latent = self.infer_latent(self.reference, kept)
            pairs = kept[self.left] & kept[self.right]
            if not pairs.any():
                raise ValueError("At least two surviving sensors are required for spectral pairs")
            pooled = self.pool_pairs(self.ref_ratios, pairs)
            self.cache[key] = (
                location_scale(latent, floor=1e-3), pairs, location_scale(pooled, floor=0.10),
            )
        return self.cache[key]

    def pool_pairs(self, ratios, valid):
        residual = (ratios[:, valid] - self.pair_location[valid]) / self.pair_scale[valid]
        return np.median(np.abs(residual), axis=1)

    def scores(self, normalized, ratios, kept):
        native, latent = self.infer_latent(normalized, kept)
        (center, scale), valid, (pair_center, pair_scale) = self.reference_for(kept)
        clean_center, clean_scale = self.clean_latent
        clean_latent = (latent - clean_center) / clean_scale
        centered_latent = latent - center
        matched_center_latent = centered_latent / clean_scale
        matched_latent = centered_latent / scale
        pooled_pairs = self.pool_pairs(ratios, valid)
        pair_features = (pooled_pairs - pair_center) / pair_scale
        # One weighted deviation head over the latent and spectral features.
        # Assign equal total weight to the two feature groups by construction.
        latent_distance = np.abs(matched_latent).mean(axis=1)
        pair_distance = np.maximum(pair_features, 0.0).mean(axis=1)
        clmm_score = 0.5 * (latent_distance + pair_distance)
        return np.stack((native, np.abs(clean_latent).mean(axis=1),
                         np.abs(matched_center_latent).mean(axis=1),
                         latent_distance, pair_distance, clmm_score), axis=1)

    def prepare(self, raw):
        return self.detector.scaler.transform(raw), self.ratios(self.log_power(band_power(raw)))


def cases(channels, mask_key, replicates):
    yield "clean", 0, np.ones(channels, dtype=bool)
    for rep in replicates:
        order = stable_rng(mask_key, "spectral-pair-trial", rep).permutation(channels)
        for rate in RATES:
            yield f"sensor_{rate:.2f}", rep, sensor_mask(channels, rate, order)


def save_cases(block, raw, metadata, mask_key, args, rows, masks, group, record_level=False, clean_only=False):
    inputs = block.prepare(raw)
    sample_rows = list(metadata.itertuples())
    replicates = () if clean_only else args.replicates
    for condition, rep, kept in cases(raw.shape[1], mask_key, replicates):
        scores = block.scores(*inputs, kept=kept)
        if record_level:
            scores = np.median(scores, axis=0, keepdims=True)
        masks.append(dict(group=group, mask_key=mask_key, split=sample_rows[0].split,
                          condition=condition, replicate=rep, channels=len(kept),
                          kept_indices=" ".join(map(str, np.flatnonzero(kept)))))
        for row, values in zip(sample_rows, scores, strict=True):
            rows.append(dict(sample_id=row.sample_id, record_id=row.record_id, group=group,
                             split=row.split, state=row.state, label=row.label,
                             condition=condition, replicate=rep, **dict(zip(block.columns, values, strict=True))))


def fit_mask_pair(fit, reference, seed, args, device, _group):
    return MaskPairBlock(Detector(fit, reference, seed, args, device), reference)


def run_z24(args, device, make_block=fit_mask_pair):
    manifest = load_manifest(args.z24_root).assign(sample_id=lambda f: f.record_id)
    rows, masks, fits = [], [], []
    channel_map = stable_sensor_map(manifest)
    for index, ((setup, excitation), channels) in enumerate(channel_map.items()):
        group = f"{setup:02d}/{excitation.upper()}"
        print(f"Z24 {index + 1}/{len(channel_map)} {group} sensors={len(channels)}", flush=True)
        cohort = manifest.loc[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]
        train = cohort[cohort.split.eq("train")]
        fit, ref = split_training_windows([
            load_z24(Path(row.path), channels) for row in train.itertuples()
        ])
        fit, ref = subset(fit, args.z24_windows), subset(ref, args.z24_windows)
        seed = args.seed + 1009 * index
        fits.append(dict(group=group, seed=seed, channels=channels, fit_windows=len(fit),
                         reference_windows=len(ref), healthy_records=train.record_id.to_list()))
        block = make_block(fit, ref, seed, args, device, group)
        evaluation = cohort[cohort.split.isin(["val", "test"])]
        for idx, row in evaluation.iterrows():
            raw = subset(load_z24(Path(row.path), channels), args.z24_windows)
            save_cases(block, raw, cohort.loc[[idx]], row.record_id, args, rows, masks, group,
                       record_level=True, clean_only=row.split == "val" and row.label == 1)
        print(f"  evaluated={len(evaluation)}", flush=True)
        del block
    return rows, masks, fits


def run_hellbridge(args, device, out, make_block=fit_mask_pair):
    split = out / "split"
    prepare_hellbridge([
        "--h5", str(args.hell_h5),
        "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
        "--seed", "20261113", "--strict-paper-counts",
    ])
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, cal, test = development_split(manifest)
    # From each six-window healthy fit block, reserve one window for reference.
    # This keeps the original twenty threshold-calibration windows separate.
    fit_blocks = fit.reshape(-1, 6)
    ref, fit = fit_blocks[:, -1], fit_blocks[:, :-1].ravel()
    raw = load_hellbridge(args.hell_h5, manifest)
    group = "P2_NM_Z"
    fits = [dict(group=group, seed=args.seed, channels=[f"AL{i:02d}/z" for i in range(1, 41)],
                 fit_windows=len(fit), reference_windows=len(ref),
                 fit_ids=manifest.loc[fit].sample_id.to_list(),
                 reference_ids=manifest.loc[ref].sample_id.to_list())]
    print(f"HellBridge {group} fit={len(fit)} reference={len(ref)} val={len(cal)} test={len(test)}", flush=True)
    block = make_block(raw[fit], raw[ref], args.seed, args, device, group)
    rows, masks = [], []
    for role, ids in (("val", cal), ("test", test)):
        save_cases(block, raw[ids], manifest.loc[ids].assign(split=role), group, args, rows, masks, group)
        print(f"  scored {role} windows={len(ids)}", flush=True)
    return rows, masks, fits


def evaluate(dataset, rows, masks, fits, out, methods=METHODS, share_native_direction=False):
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    key = ["condition", "replicate"]
    if scores.duplicated([*key, "split", "sample_id"]).any() or not np.isfinite(scores[list(methods)]).all().all():
        raise ValueError(f"Invalid scores for {dataset}")
    clean_val = scores.loc[scores.split.eq("val") & scores.condition.eq("clean")]
    # Retain the native Z24 runner's validation-only choice of score direction.
    native_sign = 1
    if dataset == "Z24" and roc_auc_score(clean_val.label, clean_val.timesnet_native) < 0.5:
        native_sign = -1
    scores["native_raw"] = scores.timesnet_native
    oriented_methods = list(methods) if share_native_direction else ["timesnet_native"]
    scores[oriented_methods] *= native_sign
    scores.to_csv(out / "scores.csv.gz", index=False)
    pd.DataFrame(masks).to_csv(out / "masks.csv", index=False)
    (out / "fit_groups.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    cal = scores.loc[scores.split.eq("val") & scores.label.eq(0)]
    test = scores.loc[scores.split.eq("test")]
    clean = test.loc[test.condition.eq("clean")].set_index("sample_id")
    clean_cal = cal.loc[cal.condition.eq("clean")]
    low_fpr = 0.0 if dataset == "Z24" else 0.05
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    expected_cal = 6 if dataset == "Z24" else 20
    metrics = []
    for (condition, rep), trial in test.groupby(key):
        mask_cal = cal.loc[cal.condition.eq(condition) & cal.replicate.eq(rep)]
        if trial.label.value_counts().to_dict() != expected or len(mask_cal) != expected_cal:
            raise ValueError(f"Unexpected test or calibration counts for {dataset}/{condition}/{rep}")
        if mask_cal.label.ne(0).any() or set(trial.sample_id) != set(clean.index):
            raise ValueError(f"Invalid calibration labels or test pairing for {dataset}/{condition}/{rep}")
        y = trial.label.to_numpy()
        for method in methods:
            result = trial[method].to_numpy()
            aligned_clean = clean.loc[trial.sample_id, method].to_numpy()
            formerly_correct = aligned_clean[y == 1, None] > aligned_clean[y == 0][None, :]
            now_wrong = result[y == 1, None] <= result[y == 0][None, :]
            for policy, calibration in (("clean_q90", clean_cal), ("masked_q90", mask_cal)):
                threshold = threshold_from_healthy(calibration[method].to_numpy(), 0.90)
                metrics.append(dict(
                    dataset=dataset, method=method, condition=condition, replicate=rep, policy=policy,
                    threshold=threshold, score_sign=native_sign if method in oriented_methods else 1,
                    healthy=expected[0], damage=expected[1],
                    fp=int(((result >= threshold) & (y == 0)).sum()), low_fpr_limit=low_fpr,
                    recall_low_fpr=recall_at_fpr(y, result, low_fpr),
                    rank_reversal=float((formerly_correct & now_wrong).sum() / max(1, formerly_correct.sum())),
                    **publication_metrics(y, result, threshold),
                ))
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics_by_mask.csv", index=False)
    return table


def method_deltas(metrics, method, comparators):
    pairing = ["dataset", "condition", "replicate", "policy"]
    candidate = metrics.loc[metrics.method.eq(method)].set_index(pairing)
    deltas = []
    for comparator in comparators:
        baseline = metrics.loc[metrics.method.eq(comparator)].set_index(pairing)
        if not baseline.index.equals(candidate.index):
            raise ValueError(f"Different mask indices for {method} and {comparator}")
        delta = candidate[list(FIELDS)] - baseline[list(FIELDS)]
        deltas.append(delta.add_prefix("delta_").reset_index().assign(comparator=comparator))
    return pd.concat(deltas, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--z24-windows", type=int, default=8)
    parser.add_argument("--mask-reps", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.z24_windows, args.mask_reps) < 1:
        parser.error("epochs, batch-size, z24-windows and mask-reps must be positive")
    args.replicates = list(range(300, 300 + args.mask_reps))
    args.layer_probe = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        z24_root=str(args.z24_root), hellbridge_h5=str(args.hell_h5), seed=args.seed,
        window_points=LENGTH, sample_rate_hz=100.0, epochs=args.epochs, batch_size=args.batch_size,
        z24_windows_per_record=args.z24_windows, device=str(device), methods=METHODS,
        rates=(0.0, *RATES), masked_replicates=args.replicates, bands_hz=BANDS,
        detrend="linear", normalization="train healthy per sensor for TimesNet; raw detrended units for Welch",
        z24_protocol="Locked record split, fixed stable channel roster from manifest; median across windows",
        hellbridge_protocol="P2_NM_Z, 50 fit/10 reference/20 calibration/720 test windows; "
                            "healthy test windows share source recordings with development",
        mask_scope="Z24 per record, HellBridge shared per replicate",
        score_orientation="upper for the new heads and native HellBridge; native Z24 chooses direction "
                          "from clean labeled validation as in the previous benchmark",
        reference="held-back healthy, exact mask for latent and matching available pairs",
        latent_ablation=dict(
            latent_clean_ref="clean center, clean scale",
            latent_matched_center="exact-mask center, clean scale",
            latent_matched_ref="exact-mask center, exact-mask scale",
        ),
        head="equal total latent/pair weight; two-sided latent distance, upper pair deviations",
        calibration="90th percentile of clean or condition-matched healthy validation scores",
        fft="native TimesNet batch-adaptive FFT; evaluation batches of at most batch_size",
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"device={device} seed={args.seed} mask_reps={args.replicates}", flush=True)
    hell = out / "HellBridge"
    z24 = out / "Z24"
    all_metrics = pd.concat([
        evaluate("HellBridge", *run_hellbridge(args, device, hell), hell),
        evaluate("Z24", *run_z24(args, device), z24),
    ], ignore_index=True)
    groups = ["dataset", "method", "condition", "policy"]
    summary = all_metrics.groupby(groups, sort=False)[list(FIELDS)].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    method_deltas(all_metrics, "clmm_mask_pair", METHODS[:-1]).to_csv(out / "paired_deltas.csv", index=False)
    method_deltas(all_metrics, "latent_matched_center", (
        "latent_clean_ref", "latent_matched_ref", "timesnet_native",
    )).to_csv(out / "latent_center_deltas.csv", index=False)
    print(summary.loc[summary.policy.eq("clean_q90")].round(4).to_string(index=False), flush=True)
    print(f"CLMM_MASK_PAIR_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
