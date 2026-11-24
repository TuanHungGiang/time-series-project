"""Exploratory analysis of native Z24 and HellBridge acceleration windows."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

from .data.hellbridge import read_al40
from .data.masking import sensor_mask, stable_rng
from .data.z24 import load_manifest, read_stable_vertical, stable_sensor_map


FS, LENGTH = 100, 1024
BANDS = (0.5, 6.25, 12.5, 25.0, 50.0)
COLORS = {0: "#28628b", 1: "#b24c35"}


def spread_indices(count, limit):
    if count <= 0:
        raise ValueError("A record has no complete windows")
    return np.linspace(0, count - 1, min(count, limit), dtype=int)


def features(raw):
    x = detrend(raw, axis=-1, type="linear")
    f, psd = welch(x, fs=FS, nperseg=256, noverlap=128, detrend=False, axis=-1)
    total = np.maximum(psd[..., f >= BANDS[0]].sum(-1), np.finfo(psd.dtype).tiny)
    fractions = np.stack([
        psd[..., (f >= low) & ((f <= high) if high == BANDS[-1] else (f < high))].sum(-1)
        / total
        for low, high in zip(BANDS[:-1], BANDS[1:], strict=True)
    ], axis=-1)
    relative_psd = psd / total[..., None]
    spectrum = np.median(relative_psd, axis=(0, 1))
    valid = f >= BANDS[0]
    peak = f[valid][int(spectrum[valid].argmax())]
    flat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    covariance = flat @ flat.T
    scale = np.sqrt(np.maximum(covariance.diagonal(), np.finfo(np.float32).tiny))
    correlation = (covariance / scale[:, None] / scale[None, :]).clip(-1, 1)
    off_diagonal = np.triu_indices(x.shape[1], 1)
    summary = {
        "windows_visualized": len(x), "sensors": x.shape[1],
        "rms_median_sensor_window": float(np.median(np.sqrt(np.mean(x ** 2, axis=-1)))),
        "dominant_hz": float(peak), "median_abs_pair_correlation": float(np.median(np.abs(correlation[off_diagonal]))),
    }
    summary.update({
        f"median_energy_share_{low:g}_{high:g}_hz": float(np.median(fractions[..., i]))
        for i, (low, high) in enumerate(zip(BANDS[:-1], BANDS[1:], strict=True))
    })
    return f, spectrum, correlation, x[0], summary


def z24_data(root, limit, out):
    manifest = load_manifest(root)
    layouts = stable_sensor_map(manifest)
    layout_rows = [
        dict(setup=setup, excitation=exc, channels=len(labels), sensor_ids=" ".join(labels))
        for (setup, exc), labels in sorted(layouts.items())
    ]
    pd.DataFrame(layout_rows).to_csv(out / "Z24_sensor_layouts.csv", index=False)
    manifest.groupby(["split", "label", "excitation"], as_index=False).size().to_csv(
        out / "Z24_split_counts.csv", index=False,
    )
    example_group = manifest[
        manifest.setup.eq(3) & manifest.excitation.eq("fvt") & manifest.split.eq("test")
    ]
    examples = set(example_group.sort_values("record_id").groupby("label").head(1).record_id)
    rows, curves, cases = [], [], {}
    for index, row in enumerate(manifest.itertuples(), 1):
        labels = layouts[(row.setup, row.excitation)]
        signals = read_stable_vertical(Path(row.path), labels)
        count = signals.shape[1] // LENGTH
        indices = spread_indices(count, limit)
        raw = np.stack([signals[:, start*LENGTH:(start+1)*LENGTH] for start in indices])
        f, psd, corr, trace, info = features(raw)
        rows.append(dict(dataset="Z24", record_id=row.record_id, split=row.split, state=row.state,
                         label=row.label, excitation=row.excitation.upper(), setup=row.setup,
                         raw_samples=signals.shape[1], **info))
        curves.append(psd)
        if row.record_id in examples:
            cases[int(row.label)] = (row.record_id, labels, corr, trace)
        if index % 48 == 0 or index == len(manifest):
            print(f"Z24 records {index}/{len(manifest)}", flush=True)
    return f, pd.DataFrame(rows), np.stack(curves), cases


def hellbridge_data(h5, limit, out, source_manifest):
    if source_manifest is None:
        split = out / "hellbridge_split"
        subprocess.run([
            sys.executable, "-m", "clmm_timesnet.data.prepare_hellbridge", "--h5", str(h5),
            "--output-dir", str(split), "--profile", "svendsen2022", "--regime", "P2_NM_Z",
            "--seed", "20261113", "--strict-paper-counts",
        ], check=True)
        source_manifest = split / "sample_manifest.csv"
    manifest = pd.read_csv(source_manifest)
    manifest.groupby(["role", "state"], as_index=False).size().to_csv(out / "HellBridge_split_counts.csv", index=False)
    labels = [f"AL{i:02d}/z" for i in range(1, 41)]
    rows, curves, cases = [], [], {}
    with h5py.File(h5, "r") as source:
        for record, group in manifest.groupby("record_id", sort=True):
            state = group.state.iloc[0]
            label = int(group.label.iloc[0])
            signal = read_al40(source[record])
            ordered = group.sort_values("start")
            picked = ordered.iloc[spread_indices(len(ordered), limit)]
            raw = np.stack([signal[:, row.start:row.stop] for row in picked.itertuples()])
            if raw.shape[1:] != (40, LENGTH):
                raise ValueError(f"Unexpected window shape for {record}: {raw.shape}")
            f, psd, corr, trace, info = features(raw)
            rows.append(dict(dataset="HellBridge", record_id=record, split="protocol_windows", state=state,
                             label=label, excitation="P2_NM_Z", setup=2, raw_samples=signal.shape[1], **info))
            curves.append(psd)
            if state == "DS5" or (state == "UDS" and 0 not in cases):
                cases[label] = (record, labels, corr, trace)
            print(f"HellBridge {record} windows={len(raw)}", flush=True)
    return f, pd.DataFrame(rows), np.stack(curves), cases


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def make_figures(data, out):
    f, z, z_psd, z_cases = data["Z24"]
    hf, hb, h_psd, h_cases = data["HellBridge"]
    if not np.array_equal(f, hf):
        raise ValueError("Frequency axes differ")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), layout="constrained")
    for ax, (bridge, cases) in zip(axes, [("Z24 · setup 03 FVT", z_cases), ("HellBridge · P2_NM_Z", h_cases)], strict=True):
        for label, (record, labels, _, trace) in cases.items():
            sensor = len(labels)//2
            ax.plot(np.arange(LENGTH)/FS, trace[sensor], lw=.65, alpha=.85, color=COLORS[label],
                    label=("Healthy" if label == 0 else "Damage") + f" · {record}")
        ax.set(title=f"{bridge} · {labels[sensor]}", xlabel="Time (s)", ylabel="Detrended acceleration (recorded units)")
        ax.grid(alpha=.19)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Example native windows · 1024 time points at 100 Hz")
    save(fig, out, "01_waveforms")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), layout="constrained")
    for ax, exc in zip(axes[:2], ("AVT", "FVT"), strict=True):
        for label in (0, 1):
            group = z.excitation.eq(exc) & z.label.eq(label)
            lower, center, upper = np.quantile(z_psd[group], [.25, .50, .75], axis=0)
            ax.plot(f, center, color=COLORS[label], label=f"{'Healthy' if label == 0 else 'Damage'} · n={group.sum()} records")
            ax.fill_between(f, np.maximum(lower, 1e-12), np.maximum(upper, 1e-12), color=COLORS[label], alpha=.16)
        ax.set(title=f"Z24 · {exc}")
        ax.legend(frameon=False, fontsize=7)
    ax = axes[2]
    selected = ["UDS", "DS2", "DS5"]
    for state, color in zip(selected, ("#28628b", "#00796b", "#b24c35"), strict=True):
        ax.plot(f, np.median(h_psd[hb.state.eq(state)], axis=0), color=color, label=state)
    ax.legend(frameon=False, fontsize=8)
    ax.set(title="HellBridge · selected states")
    for ax in axes:
        ax.set(xlabel="Frequency (Hz)", xlim=(.5, 47), yscale="log")
        ax.axvline(6.25, lw=.8, color=".35", ls="--")
        ax.axvline(25, lw=.8, color=".35", ls=":")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Relative Welch spectrum")
    fig.suptitle("Frequency content · Z24 bands show between-record interquartile ranges")
    save(fig, out, "02_spectra")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.9), layout="constrained")
    mask_rows = []
    for i, (bridge, cases) in enumerate([("Z24", z_cases), ("HellBridge", h_cases)]):
        _, labels, normal, _ = cases[0]
        damage_record, _, damage, _ = cases[1]
        if bridge == "Z24":
            order = stable_rng(damage_record, "sensor", 0).permutation(len(labels))
        else:
            order = np.random.default_rng(20261119).permutation(len(labels))
        kept = sensor_mask(len(labels), .5, order)
        available = kept[:, None] & kept[None, :]
        delta = np.abs(damage - normal)
        show = np.ma.masked_where(~available, delta)
        for j, (values, title) in enumerate([
            (normal, f"{bridge} · Healthy correlation"),
            (damage, f"{bridge} · Damage correlation"),
            (show, f"{bridge} · Change on surviving pairs"),
        ]):
            ax = axes[i, j]
            if j < 2:
                obj = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                cmap = plt.get_cmap("magma").copy()
                cmap.set_bad("#d5d9de")
                obj = ax.imshow(values, cmap=cmap, vmin=0, vmax=2)
            step = 4 if len(labels) > 20 else 2
            ticks = range(0, len(labels), step)
            ax.set(title=title, xticks=ticks, yticks=ticks,
                   xticklabels=[labels[k] for k in ticks], yticklabels=[labels[k] for k in ticks])
            ax.tick_params(axis="x", labelrotation=65, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            fig.colorbar(obj, ax=ax, shrink=.73)
        mask_rows.append(dict(dataset=bridge, healthy_example=cases[0][0], damage_example=damage_record,
                              sensor_count=len(labels), kept_count=int(kept.sum()),
                              kept_pairs=int(kept.sum() * (kept.sum()-1)//2),
                              dropped_sensors=" ".join(np.asarray(labels)[~kept])))
    fig.suptitle("Spatial information lost when whole sensor channels are removed")
    save(fig, out, "03_sensor_relationships")
    pd.DataFrame(mask_rows).to_csv(out / "illustrated_masks.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hell-manifest", type=Path)
    parser.add_argument("--z24-windows", type=int, default=8)
    parser.add_argument("--hell-windows", type=int, default=16)
    args = parser.parse_args()
    if min(args.z24_windows, args.hell_windows) < 1:
        parser.error("Window limits must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "Z24": z24_data(args.z24_root, args.z24_windows, out),
        "HellBridge": hellbridge_data(args.hell_h5, args.hell_windows, out, args.hell_manifest),
    }
    all_rows = pd.concat([entry[1] for entry in data.values()], ignore_index=True)
    all_rows.to_csv(out / "per_record_features.csv", index=False)
    numeric = ["sensors", "raw_samples", "dominant_hz", "median_abs_pair_correlation",
               *[field for field in all_rows if field.startswith("median_energy_share_")]]
    group_rows = all_rows.groupby(["dataset", "excitation", "label"], as_index=False)
    rollup = group_rows[numeric].median().merge(
        group_rows.size().rename(columns={"size":"records"}),
        on=["dataset", "excitation", "label"], validate="one_to_one",
    )
    rollup.to_csv(out / "dataset_summary.csv", index=False)
    make_figures(data, out)
    (out / "eda_protocol.json").write_text(json.dumps(dict(
        sampling_hz=FS, points_per_window=LENGTH, resampling=False, z24_windows_per_record=args.z24_windows,
        hellbridge_windows_per_record=args.hell_windows, welch_segment_points=256,
        bands_hz=BANDS, mode="descriptive, uses labels from included splits, no model fitted",
    ), indent=2), encoding="utf-8")
    print(rollup.round(4).to_string(index=False), flush=True)
    print("BRIDGE_DATASET_EDA_PASS", flush=True)


if __name__ == "__main__":
    main()
