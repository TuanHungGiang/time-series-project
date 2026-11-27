"""Benchmark one PCA-SPE detector per measurement group under nested sensor loss."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from clmm_timesnet.data.hellbridge import development_split, Normalizer
from clmm_timesnet.data.masking import sensor_mask, stable_rng
from clmm_timesnet.data.prepare_hellbridge import main as prepare_hellbridge
from clmm_timesnet.data.z24 import load_manifest, read_stable_vertical, split_training_windows
from clmm_timesnet.metrics import publication_metrics, threshold_from_healthy


POINTS = 1024
RATES = (0, 10, 30, 50, 70, 90)
METRICS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "diagnostic_recall")
REPLICATES = 30


def select_windows(windows, count=8):
    return windows[np.linspace(0, len(windows) - 1, count, dtype=int)]


def nested_keep(sites, measured, key, rate, replicate):
    if rate == 0:
        return measured.copy()
    available = np.flatnonzero(np.bincount(sites[measured], minlength=sites.max()+1))
    rng = stable_rng(key, "pca-1024-whole-location", replicate)
    order = rng.permutation(len(available))
    removed = available[~sensor_mask(len(available), rate / 100, order)]
    return measured & ~np.isin(sites, removed)


class PCASPE:
    def __init__(self, fit, reference):
        self.normalizer = Normalizer.fit(fit)
        norm = self.normalizer.transform(fit)
        self.model = PCA(n_components=.95, svd_solver="full").fit(norm.reshape(len(norm), -1))
        f = self.normalizer.transform(reference).reshape(len(reference), -1)
        error = np.log1p(self.explicit_spe(f))
        self.center = float(np.median(error))
        self.scale = float(max(1.4826 * np.median(np.abs(error - self.center)), .25 * np.std(error), 1e-4))

    def orient_neutral(self, scores):
        return (np.log1p(np.maximum(scores, 0.0)) - self.center) / self.scale

    def explicit_spe(self, flat):
        reconstruction = self.model.inverse_transform(self.model.transform(flat))
        return np.mean((flat - reconstruction) ** 2, axis=1)

    def terms(self, raw):
        # Decompose squared distance and projected energy by axis. This is
        # algebraically the full zero-filled PCA-SPE reconstruction error.
        x = self.normalizer.transform(raw)
        basis = self.model.components_.reshape(-1, *x.shape[1:])
        mean = self.model.mean_.reshape(x.shape[1:])
        shifted_energy = np.einsum("nct,nct->nc", x, x, dtype=np.float64)
        shifted_energy -= 2 * np.einsum("nct,ct->nc", x, mean, dtype=np.float64)
        constant = np.sum(mean.astype(np.float64) ** 2)
        axes_projection = np.matmul(
            np.ascontiguousarray(x.transpose(1, 0, 2)),
            np.ascontiguousarray(basis.transpose(1, 2, 0)),
        ).transpose(1, 0, 2)
        mean_projection = self.model.mean_ @ self.model.components_.T
        width = self.model.n_features_in_

        def score(kept):
            norm = constant + shifted_energy[:, kept].sum(axis=1)
            projection = axes_projection[:, kept].sum(axis=1, dtype=np.float64) - mean_projection
            energy = (norm - np.einsum("nk,nk->n", projection, projection)) / width
            return self.orient_neutral(energy)

        for keep in (np.ones(x.shape[1], dtype=bool), np.arange(x.shape[1]) % 2 == 0):
            sample = x[:min(4, len(x))].copy()
            sample[:, ~keep] = 0
            expected = self.orient_neutral(self.explicit_spe(sample.reshape(len(sample), -1)))
            np.testing.assert_allclose(score(keep)[:len(sample)], expected, atol=.006, rtol=.001)
        return score

    def score_columns(self, raw, present):
        score = self.terms(raw)

        def columns(keep):
            return {"spe_score": score(keep)}

        return columns


def read_z24_windows(path, layout, labels):
    present = np.array([name in labels for name in layout.channels], dtype=bool)
    if not present.any() or labels.difference(layout.channels):
        raise ValueError(f"Unexpected sensor roster in {path}")
    raw = read_stable_vertical(path, np.array(layout.channels)[present].tolist())
    count = raw.shape[1] // POINTS
    raw = raw[:, :count * POINTS].reshape(raw.shape[0], count, POINTS).transpose(1, 0, 2)
    aligned = np.zeros((count, len(layout.channels), POINTS), dtype=np.float32)
    aligned[:, present] = detrend(raw, axis=-1, type="linear")
    return aligned, present


def mask_summary(names, channel_sites, present, keep):
    seen = set(channel_sites[present])
    retained = set(channel_sites[keep])
    dropped = sorted(seen - retained)
    return dict(
        available_sensors=len(seen), dropped_sensors=len(dropped), actual_sensor_loss=len(dropped) / len(seen),
        natural_missing_axes=int((~present).sum()), dropped_axes=int((present & ~keep).sum()),
        actual_axis_loss=float((present & ~keep).sum() / present.sum()),
        dropped_sensor_ids="|".join(names[i] for i in dropped),
    )


def fit_z24_group(train, group_name, detector_cls=PCASPE):
    from clmm_timesnet.z24_full_sensors import Layout, response_labels

    paths = [Path(path) for path in train.path]
    labels = [response_labels(p) for p in paths]
    layout = Layout.from_training(labels)
    fit, reference = split_training_windows([
        read_z24_windows(p, layout, available)[0] for p, available in zip(paths, labels, strict=True)
    ])
    detector = detector_cls(select_windows(fit), select_windows(reference))
    metadata = dict(dataset="Z24", group=group_name, channels=list(layout.channels),
                    sensors=list(layout.sites), fit_records=train.record_id.tolist(),
                    pca_components=int(detector.model.n_components_))
    return layout, detector, metadata


def z24_scores(root, detector_cls=PCASPE):
    from clmm_timesnet.z24_full_sensors import response_labels

    manifest = load_manifest(root)
    if manifest.record_id.duplicated().any():
        raise ValueError("Z24: duplicate records")
    counts = manifest.groupby(["split", "label"]).size().to_dict()
    if counts != {("train", 0): 24, ("val", 0): 6, ("val", 1): 126, ("test", 0): 6, ("test", 1): 126}:
        raise ValueError(counts)
    damage = manifest.loc[manifest.label.eq(1)]
    if set(damage.loc[damage.split.eq("val"), "state"]) & set(damage.loc[damage.split.eq("test"), "state"]):
        raise ValueError("Z24: a damage state occurs in validation and test")
    all_scores, masks, inventory = [], [], []
    for (setup, exc), group in manifest.groupby(["setup", "excitation"], sort=True):
        group_name = f"{setup:02d}/{exc.upper()}"
        layout, detector, metadata = fit_z24_group(group.loc[group.split.eq("train")], group_name, detector_cls)
        inventory.append(metadata)
        print(f"Z24 {group_name} | {len(layout.sites)} sensors, {len(layout.channels)} channels", flush=True)
        for row in group.loc[group.split.isin(["val", "test"])].itertuples():
            p = Path(row.path)
            raw, present = read_z24_windows(p, layout, response_labels(p))
            score = detector.score_columns(select_windows(raw), present)
            base = dict(dataset="Z24", group=group_name, split=row.split, sample_id=row.record_id,
                        record_id=row.record_id, state=str(row.state), label=row.label)
            for rate in RATES if row.split == "test" or row.label == 0 else (0,):
                for replicate in (0,) if rate == 0 else range(1, REPLICATES + 1):
                    keep = nested_keep(layout.site_of_axis, present, row.record_id, rate, replicate)
                    scores = {name: float(np.median(values)) for name, values in score(keep).items()}
                    all_scores.append(dict(**base, loss=rate, replicate=replicate, **scores))
                    masks.append(dict(dataset="Z24", key=row.record_id, loss=rate, replicate=replicate,
                                      **mask_summary(layout.sites, layout.site_of_axis, present, keep)))
    return pd.DataFrame(all_scores), masks, inventory


def hell_axes(record):
    channels = []
    for field, prefixes in (("acceleration", ("AG", "AL")), ("strain", ("SB", "SC"))):
        for sensor, group in sorted(record[field].items()):
            if sensor.startswith(prefixes):
                for axis, data in sorted(group.items()):
                    if isinstance(data, h5py.Dataset) and data.ndim == 1:
                        channels.append((field, sensor, axis))
    return tuple(channels)


def hell_scores(h5_path, out, detector_cls=PCASPE):
    split = out / "hellbridge_split"
    prepare_hellbridge([
        "--h5", str(h5_path), "--output-dir", str(split), "--profile", "svendsen2022",
        "--regime", "P2_NM_Z", "--seed", "20261113", "--strict-paper-counts",
    ])
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    fit, calibration, test = development_split(manifest)
    fit_blocks = fit.reshape(-1, 6)
    fit, reference = fit_blocks[:, :-1].ravel(), fit_blocks[:, -1]
    with h5py.File(h5_path, "r") as source:
        healthy = manifest.loc[fit, "record_id"].unique()
        axes = hell_axes(source[healthy[0]])
        if len(axes) != 90 or any(hell_axes(source[name]) != axes for name in manifest.record_id.unique()):
            raise ValueError("Unexpected or inconsistent HellBridge structural response channels")
        names = tuple(dict.fromkeys(sensor for _, sensor, _ in axes))
        sites = np.array([names.index(sensor) for _, sensor, _ in axes])
        raw = np.empty((len(manifest), len(axes), POINTS), dtype=np.float32)
        for record, rows in manifest.groupby("record_id"):
            print(f"HellBridge {record} | windows={len(rows)}, sensors={len(names)}, channels={len(axes)}", flush=True)
            group = source[record]
            if int(group.attrs["sample_rate"]) != 100:
                raise ValueError(f"Unexpected sample rate {group.attrs['sample_rate']!r}")
            sample_idx = rows.start.to_numpy()[:, None] + np.arange(POINTS)
            for j, (field, sensor, axis) in enumerate(axes):
                full = np.asarray(group[field][sensor][axis], dtype=np.float32)
                raw[rows.index, j] = detrend(full[sample_idx], axis=-1, type="linear")
    detector = detector_cls(raw[fit], raw[reference])
    idx = np.r_[calibration, test]
    present = np.ones(len(axes), dtype=bool)
    score = detector.score_columns(raw[idx], present)
    meta = manifest.loc[idx, ["sample_id", "record_id", "state", "label"]].reset_index(drop=True)
    meta = meta.assign(dataset="HellBridge", group="P2_NM_Z",
                       split=np.where(np.arange(len(idx)) < len(calibration), "val", "test"))
    all_scores, masks = [], []
    for rate in RATES:
        for replicate in (0,) if rate == 0 else range(1, REPLICATES + 1):
            keep = nested_keep(sites, present, "HellBridge/P2_NM_Z", rate, replicate)
            all_scores.append(meta.assign(loss=rate, replicate=replicate, **score(keep)))
            masks.append(dict(dataset="HellBridge", key="P2_NM_Z", loss=rate, replicate=replicate,
                              **mask_summary(names, sites, present, keep)))
    inventory = [dict(dataset="HellBridge", group="P2_NM_Z",
                      channels=["/".join(axis) for axis in axes], sensors=list(names),
                      pca_components=int(detector.model.n_components_), fit_windows=len(fit),
                      reference_windows=len(reference), calibration_windows=len(calibration),
                      fit_sample_ids=manifest.loc[fit, "sample_id"].tolist())]
    return pd.concat(all_scores, ignore_index=True), masks, inventory


def pair_credit(y, values):
    diff = values[y == 1, None] - values[y == 0][None, :]
    return (diff > 0).astype(float) + .5 * (diff == 0)


def measure(data, name, prior_sign=None):
    is_z24 = name == "Z24"
    validation = data.loc[data.split.eq("val") & data.loss.eq(0)]
    clean_validation_auc = roc_auc_score(validation.label, validation.spe_score) if is_z24 else None
    # Preserve the previous locked Z24 validation protocol and also report
    # the conventional healthy-only upper-tail direction for comparison.
    val_sign = prior_sign if prior_sign is not None else (-1 if is_z24 and clean_validation_auc < .5 else 1)
    orientations = {"upper": 1, "prior_protocol": val_sign}
    cal = data.loc[data.split.eq("val") & data.label.eq(0)]
    trials = data.loc[data.split.eq("test")]
    clean = trials.loc[trials.loss.eq(0)].set_index("sample_id", verify_integrity=True)
    rows = []
    for (loss, replicate), trial in trials.groupby(["loss", "replicate"], sort=True):
        ids = trial.sample_id
        if len(set(ids)) != len(ids) or set(ids) != set(clean.index):
            raise ValueError(f"Different test cohort for {name}/{loss}/{replicate}")
        y = trial.label.to_numpy()
        want = {0: 6, 1: 126} if is_z24 else {0: 80, 1: 640}
        if trial.label.value_counts().to_dict() != want:
            raise ValueError(f"Unexpected test classes {name}")
        mask_cal = cal.loc[cal.loss.eq(loss) & cal.replicate.eq(replicate), "spe_score"].to_numpy()
        clean_cal = cal.loc[cal.loss.eq(0), "spe_score"].to_numpy()
        if len(mask_cal) != len(clean_cal) or len(mask_cal) != (6 if is_z24 else 20):
            raise ValueError(f"Invalid calibration cohort {name}")
        for orientation, sign in orientations.items():
            score = sign * trial.spe_score.to_numpy()
            before = pair_credit(y, sign * clean.loc[ids, "spe_score"].to_numpy())
            after = pair_credit(y, score)
            rank_loss = np.maximum(before - after, 0).mean()
            rank_gain = np.maximum(after - before, 0).mean()
            fpr, recall, _ = roc_curve(y, score)
            for policy, values in (("mask_matched", mask_cal), ("clean_frozen", clean_cal)):
                threshold = threshold_from_healthy(sign * values, .90 if is_z24 else .95)
                metrics = publication_metrics(y, score, threshold)
                np.testing.assert_allclose(metrics["auc"], after.mean(), atol=1e-12)
                pred = score >= threshold
                rows.append(dict(
                    dataset=name, orientation=orientation, score_sign=sign,
                    val_clean_auc_upper=clean_validation_auc, loss=loss, replicate=replicate,
                    policy=policy, threshold=threshold, rank_credit_lost=rank_loss, rank_credit_gained=rank_gain,
                    fp=int((pred & (y == 0)).sum()), tp=int((pred & (y == 1)).sum()),
                    diagnostic_recall=float(recall[fpr <= (0.0 if is_z24 else .05)].max()), **metrics,
                ))
    return pd.DataFrame(rows)


def report(metrics, out):
    rng = np.random.default_rng(20261127)
    keys = ["dataset", "orientation", "policy"]
    fields = [*METRICS, "rank_credit_lost", "rank_credit_gained"]
    grouped = metrics.groupby([*keys, "loss"])[fields].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join(col) for col in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(out / "mean_sd_by_mask.csv", index=False)
    rows = []
    for label, group in metrics.groupby(keys, sort=False):
        clean = group.loc[group.loss.eq(0)].iloc[0]
        for metric in METRICS:
            previous = np.full(REPLICATES, clean[metric])
            previous_loss = 0
            for level in RATES[1:]:
                curr = group.loc[group.loss.eq(level)].sort_values("replicate")[metric].to_numpy()
                if len(curr) != REPLICATES:
                    raise ValueError(f"Missing mask trials for {label}/{metric}/{level}")
                comparisons = {0: curr - clean[metric]}
                if previous_loss:
                    comparisons[previous_loss] = curr - previous
                for comparator, diffs in comparisons.items():
                    indices = rng.integers(0, REPLICATES, size=(4000, REPLICATES))
                    lo, hi = np.quantile(diffs[indices].mean(axis=1), [.025, .975])
                    rows.append(dict(
                        **dict(zip(keys, label, strict=True)), metric=metric, loss=level, reference_loss=comparator,
                        delta_mean=diffs.mean(), delta_sd=diffs.std(ddof=1), ci95_mask_low=lo, ci95_mask_high=hi,
                        masks_lower=int((diffs < -1e-12).sum()), masks_equal=int((np.abs(diffs) <= 1e-12).sum()),
                        masks_higher=int((diffs > 1e-12).sum()), masks=REPLICATES,
                    ))
                previous, previous_loss = curr, level
    pd.DataFrame(rows).to_csv(out / "paired_deltas.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24", required=True, type=Path)
    parser.add_argument("--hell", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(json.dumps(dict(
        baseline="PCA-SPE, deterministic; 95% explained variance; full SVD; flattened window",
        samples=POINTS, sample_rate_hz=100, normalization="per-axis healthy fit after linear detrending",
        masking="zero fill after standardization, loss of all axes at each sampled device; one nested order per replicate",
        roster="Z24 27/33 structural acceleration axes; HellBridge 75 acceleration and 15 strain response axes",
        replicates=REPLICATES, loss_rates=list(RATES), z24_evaluation="source record, median of eight windows",
        hell_evaluation="nonoverlapping test windows, healthy recordings shared across development and test",
        orientation="prior protocol: Z24 uses clean labeled validation orientation, HellBridge uses upper tail; upper as sensitivity",
        bootstrap="paired mask-order uncertainty only, conditional on fixed dataset split",
    ), indent=2), encoding="utf-8")
    combined = []
    with threadpool_limits(limits=4):
        for name, rows in (("Z24", lambda: z24_scores(args.z24)),
                           ("HellBridge", lambda: hell_scores(args.hell, args.out))):
            data, masks, inventory = rows()
            data.to_csv(args.out / f"{name}_scores.csv.gz", index=False)
            pd.DataFrame(masks).to_csv(args.out / f"{name}_masks.csv.gz", index=False)
            (args.out / f"{name}_sensor_fit_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            metrics = measure(data, name)
            metrics.to_csv(args.out / f"{name}_metrics_by_mask.csv", index=False)
            combined.append(metrics)
            print(f"{name} COMPLETE", flush=True)
    report(pd.concat(combined, ignore_index=True), args.out)
    print(f"PCA_SENSOR_LOSS_COMPLETE {args.out}", flush=True)


if __name__ == "__main__":
    main()
