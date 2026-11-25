"""Save source arrays, plot coordinates and diagrams for the sensor-outage sweep."""

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .data.hellbridge import read_al40
from .data.z24 import load_manifest, read_stable_vertical
from .targeted_checkpoints import load_checkpoint


def save_figure(figure, path):
    for extension in (".png", ".pdf"):
        figure.savefig(path.with_suffix(extension), dpi=180)
    plt.close(figure)


def schematic(title, boxes, path):
    figure, axis = plt.subplots(figsize=(15, 2.35), layout="constrained")
    axis.set_axis_off()
    positions = np.linspace(.07, .93, len(boxes))
    for position, label in zip(positions, boxes, strict=True):
        axis.text(position, .47, label, ha="center", va="center", fontsize=9,
                  bbox=dict(boxstyle="round,pad=.55", facecolor="#eef3f8", edgecolor="#49718b"))
    for left, right in zip(positions[:-1], positions[1:], strict=True):
        axis.annotate("", xy=(right - .076, .47), xytext=(left + .076, .47),
                      arrowprops=dict(arrowstyle="->", color="#344e60"))
    axis.set_title(title, fontsize=14)
    save_figure(figure, path)


def export_inputs(dataset, groups, orders, args, out):
    z24 = dataset == "Z24"
    if z24:
        load_manifest(args.z24_root).drop(columns="path").to_csv(out / "split_membership.csv", index=False)
    inventory = []
    for group in groups:
        n_val = group.metadata.split.eq("val").sum()
        inventory.append(dict(
            group=group.name, channels=len(group.sensors), sensor_order=" ".join(group.sensors),
            raw_sampling_hz=100, raw_window_points=1024, model_points=group.train.shape[-1],
            model_sampling_hz=50 if z24 else 12.5, fit_windows=len(group.train),
            rank_reference_windows=len(group.reference), val_windows=sum(group.counts[:n_val]),
            test_windows=sum(group.counts[n_val:]), val_units=n_val, test_units=len(group.counts) - n_val,
            unit="record" if z24 else "window",
        ))
    data = pd.DataFrame(inventory)
    data.to_csv(out / "group_inventory.csv", index=False)
    group = next(group for group in groups
                 if group.metadata.loc[group.metadata.split.eq("test"), "label"].nunique() == 2)
    selected = group.metadata.loc[group.metadata.split.eq("test")].groupby("label").head(1)
    selected = selected.sort_values("label")
    offsets = np.r_[0, np.cumsum(group.counts)]
    normalized = group.values[offsets[selected.index]].copy()
    prepared = normalized * group.normalizer.scale + group.normalizer.mean
    if z24:
        raw = np.stack([read_stable_vertical(Path(row.path), group.sensors)[:, :1024]
                        for row in selected.itertuples()])
    else:
        with h5py.File(args.hell_h5, "r") as h5:
            raw = np.stack([read_al40(h5[row.record_id])[:, row.start:row.stop]
                            for row in selected.itertuples()])
    count = max(1, len(group.sensors) // 2)
    kept = np.ones(len(group.sensors), dtype=bool)
    kept[orders[group.name][:count]] = False
    masked = normalized.copy()
    masked[:, ~kept] = 0
    hz = 50 if z24 else 12.5
    time = np.arange(prepared.shape[-1]) / hz
    frequencies, psd = welch(prepared, fs=hz, axis=-1, nperseg=min(256, prepared.shape[-1]))
    np.save(out / "example_preprocessed.npy", prepared, allow_pickle=False)
    np.savez_compressed(
        out / "example_inputs.npz", group=np.array(group.name), sample_id=selected.sample_id.astype(str).to_numpy(dtype=str),
        sensors=np.array(group.sensors), label=selected.label.to_numpy(), kept=kept, raw=raw,
        raw_time_s=np.arange(raw.shape[-1]) / 100, model_time_s=time, model_hz=np.array(hz),
        normalized=normalized, masked=masked, psd_frequency_hz=frequencies, psd=psd,
        scaler_mean=group.normalizer.mean, scaler_scale=group.normalizer.scale,
    )
    selected.drop(columns="path", errors="ignore").assign(group=group.name).to_csv(
        out / "example_samples.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    channel = int(np.flatnonzero(kept)[0])
    for i, label in enumerate(("Healthy", "Damage")):
        axes[0, 0].plot(np.arange(raw.shape[-1]) / 100, raw[i, channel], lw=0.8, label=label)
        axes[0, 1].plot(time, normalized[i, channel], lw=0.8, label=label)
        axes[1, 0].semilogy(frequencies, np.maximum(psd[i, channel], 1e-30), label=label)
    axes[0, 0].set(title="Raw window", xlabel="Time (s)", ylabel="Acceleration (source units)")
    axes[0, 1].set(title="Input after preprocessing", xlabel="Time (s)", ylabel="Training z-score")
    axes[1, 0].set(title="Power spectral density", xlabel="Frequency (Hz)", ylabel="PSD")
    axes[0, 0].legend()
    axes[1, 1].imshow(np.broadcast_to(kept[:, None], (len(kept), len(time))), aspect="auto",
                     cmap="Blues", vmin=0, vmax=1, interpolation="nearest", extent=(0, 10.24, len(kept), 0))
    axes[1, 1].set(title=f"Mask · {count} of {len(kept)} removed", xlabel="Time (s)", ylabel="Channel index")
    figure.suptitle(f"{dataset} · {group.name} · {group.sensors[channel]}")
    save_figure(figure, out / "waveform_examples")

    channel_min, channel_max = int(data.channels.min()), int(data.channels.max())
    channels = str(channel_min) if channel_min == channel_max else f"{channel_min}–{channel_max}"
    boxes = [
        "Raw vibration\n100 Hz\nrecordings",
        f"Align sensors by group\n{channels} channels\n1,024 samples/window",
        f"Linear detrend + resample\n{int(data.model_points.iloc[0])} samples/window\nhealthy-fit z-score",
        "Clean healthy fit\ntrained checkpoint",
        "Test sensor masks\none at a time + cumulative",
        f"{'Record' if z24 else 'Window'} scores\nclean-calibrated alarm\nAUC, AP, BA, recall, FPR, F1",
    ]
    schematic(f"{dataset} · Sensor-outage benchmark", boxes, out / "input_pipeline")


def export_scores(frame, max_steps, calibration, run):
    tail, threshold = calibration["score_tail"], calibration["threshold"]
    frame = frame.loc[frame.split.eq("test")]
    clean = frame.loc[frame.design.eq("clean")]
    cumulative = frame.loc[frame.design.eq("cumulative")]
    total = clean.group.map(max_steps).add(1).sum()
    outputs, curves, confusions = [], [], []
    steps = [int(key) for key in cumulative.key.unique()]
    midpoint = min(steps, key=lambda step: abs(clean.group.map(
        {group: min(step, limit) for group, limit in max_steps.items()}).sum() / total - .5))
    scenarios = {"clean": clean, f"cumulative_{midpoint}": cumulative.loc[
        cumulative.removed.eq(cumulative.group.map(max_steps).clip(upper=midpoint))]}
    for condition, rows in scenarios.items():
        scores = rows[tail].to_numpy()
        labels = rows.label.to_numpy()
        predictions = scores >= threshold
        percent = 100 * rows.removed.sum() / total
        export = rows[["sample_id", "record_id", "state", "group", "label", "removed"]].copy()
        outputs.append(export.assign(condition=condition, removed_percent=percent, score=scores,
                                     threshold=threshold, predicted_damage=predictions))
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(labels, scores)
        curves.extend([pd.DataFrame(dict(condition=condition, curve="ROC", x=fpr, y=tpr)),
                       pd.DataFrame(dict(condition=condition, curve="PR", x=recall, y=precision))])
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        confusions.append(dict(condition=condition, removed_percent=percent, tn=tn, fp=fp, fn=fn, tp=tp))
    evidence = pd.concat(outputs, ignore_index=True)
    evidence.to_csv(run / "plot_scores_clean_and_midloss.csv", index=False)
    evidence.groupby(["condition", "state", "label"]).agg(
        samples=("score", "size"), score_median=("score", "median"), alarm_rate=("predicted_damage", "mean")
    ).to_csv(run / "scores_by_state.csv")
    pd.concat(curves, ignore_index=True).to_csv(run / "roc_pr_points.csv", index=False)
    pd.DataFrame(confusions).to_csv(run / "confusion_counts.csv", index=False)


def export_checkpoint_index(dataset, out, weights, metrics_paths, groups):
    rows = []
    for metrics in metrics_paths:
        tag = metrics.parent.name
        directory = weights / dataset / tag
        destination = out / "checkpoint_metadata" / tag
        destination.mkdir(parents=True, exist_ok=True)
        for group in groups:
            file = directory / f"{group.name}.pt"
            config = file.with_suffix(".json")
            metadata = json.loads(config.read_text(encoding="utf-8"))
            rows.append(dict(dataset=dataset, model=metadata["model"], seed=metadata["seed"],
                             group=group.name, checkpoint=file.relative_to(weights).as_posix(),
                             size_bytes=file.stat().st_size, parameters=metadata["parameter_count"],
                             channels=len(metadata["sensors"]), model_points=metadata["points"]))
            (destination / config.name).write_bytes(config.read_bytes())
        (destination / "calibration.json").write_bytes((directory / "calibration.json").read_bytes())
    index = pd.DataFrame(rows)
    index.to_csv(out / "checkpoint_index.csv", index=False)
    native = index.loc[index.model.eq("timesnet")]
    if not native.empty:
        file = Path(native.checkpoint.iloc[0])
        meta = json.loads((weights / file.with_suffix(".json")).read_text(encoding="utf-8"))
        config = meta["model_config"]
        boxes = [
            f"Window [B, T, C]\nT = {meta['points']}\nC per sensor group",
            f"Normalize + embed\n[B, T, {config['d_model']}]",
            f"{config['e_layers']} × TimesBlock\nFFT top-{config['top_k']} + 2D conv\nresidual + LayerNorm",
            "Project to C channels\nrestore input scale\n[B, T, C]",
            "Mean squared error\nover time and channels\n[B] window scores",
            "Median across windows\n[B → record score]" if dataset == "Z24" else "Window score\nclean-calibrated alarm",
        ]
        schematic(f"{dataset} · Native TimesNet baseline", boxes, out / "timesnet_architecture")


def plot_discrimination(dataset, out, metrics_paths):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for metrics in metrics_paths:
        model, seed = metrics.parent.name.rsplit("__", 1)
        table = pd.read_csv(metrics.parent / "roc_pr_points.csv")
        for row, (condition, cohort) in enumerate(table.groupby("condition", sort=False)):
            for col, curve in enumerate(("ROC", "PR")):
                points = cohort.loc[cohort.curve.eq(curve)]
                axes[row, col].plot(points.x, points.y, lw=1.3, label=f"{model} ({seed})")
                axes[row, col].set(title=f"{curve} · {'clean' if row == 0 else 'approximately 50% loss'}",
                                   xlabel="FPR" if curve == "ROC" else "Recall",
                                   ylabel="Recall" if curve == "ROC" else "Precision", xlim=(0, 1), ylim=(0, 1.02))
                axes[row, col].grid(alpha=.2)
    for axis in axes[:, 0]:
        axis.plot([0, 1], [0, 1], "--", lw=.7, color="grey")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if len(labels) <= 12:
        axes[1, 1].legend(handles, labels, fontsize=7, loc="lower left")
    figure.suptitle(f"{dataset} · Score discrimination")
    save_figure(figure, out / "roc_pr_clean_and_midloss")


@torch.no_grad()
def export_timesnet_probe(dataset, out, weights, metrics_paths, source):
    native_runs = [path for path in metrics_paths if path.parent.name.startswith("timesnet__")]
    if not native_runs:
        return
    with np.load(out / "example_inputs.npz", allow_pickle=False) as data:
        inputs = {condition: data[condition] for condition in ("normalized", "masked")}
        group, sample_ids, labels = str(data["group"]), data["sample_id"], data["label"]
        kept, rate = data["kept"], float(data["model_hz"])
    for metrics in native_runs:
        directory = metrics.parent
        path = directory / "timesnet_example_layers.npz"
        if path.is_file():
            continue
        checkpoint = weights / dataset / directory.name / f"{group}.pt"
        _, scorer = load_checkpoint(checkpoint, source, torch.device("cpu"))
        network = scorer.state
        captures, bins = {}, {}

        def store_activation(name):
            def capture(_, _inputs, output):
                captures[name] = output.detach().numpy().copy()
            return capture

        def store_bins(name):
            def capture(block, arguments):
                spectrum = torch.fft.rfft(arguments[0], dim=1).abs().mean(dim=(0, 2))
                spectrum[0] = 0.0
                bins[name] = torch.topk(spectrum, block.k).indices.numpy()
            return capture

        handles = [network.enc_embedding.register_forward_hook(store_activation("embedding")),
                   network.projection.register_forward_hook(store_activation("projection"))]
        for index, block in enumerate(network.model, 1):
            name = f"timesblock_{index}"
            handles.extend([block.register_forward_pre_hook(store_bins(name)),
                            block.register_forward_hook(store_activation(name))])
        saved, detected = {}, []
        try:
            for condition, values in inputs.items():
                batch = torch.from_numpy(values)
                reconstruction = source.model_forward("timesnet", network, batch)
                saved.update({f"{condition}_{key}": value for key, value in captures.items()})
                saved[f"{condition}_reconstruction"] = reconstruction.numpy()
                for block, frequencies in bins.items():
                    for rank, frequency in enumerate(frequencies, 1):
                        detected.append(dict(condition=condition, layer=block, rank=rank, fft_bin=frequency,
                                             frequency_hz=frequency * rate / batch.shape[-1]))
        finally:
            for handle in handles:
                handle.remove()
        shifts = []
        for key in (*captures, "reconstruction"):
            clean, masked = (saved[f"{prefix}_{key}"] for prefix in ("normalized", "masked"))
            for j, sample_id in enumerate(sample_ids):
                shifts.append(dict(sample_id=sample_id, label=labels[j], layer=key, output_shape=str(clean[j].shape),
                                   rms_shift=float(np.sqrt(np.mean((clean[j] - masked[j]) ** 2)))))
        pd.DataFrame(shifts).to_csv(directory / "timesnet_example_activation_shift.csv", index=False)
        pd.DataFrame(detected).to_csv(directory / "timesnet_example_fft_bins.csv", index=False)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **saved, sample_ids=sample_ids, labels=labels, kept=kept)
        temporary.replace(path)


def merge_reports(reports, out, require_all=False):
    from .targeted_baselines import MODELS, SEEDS, atomic_csv, summarize

    reports = [path.resolve() for path in reports]
    out = out.resolve()
    if not reports or len(set(reports)) != len(reports) or out in reports:
        raise ValueError("Provide distinct source reports and a different output directory")
    protocol = (reports[0] / "protocol.json").read_bytes()
    if any((path / "protocol.json").read_bytes() != protocol for path in reports):
        raise ValueError("Source reports have different experiment protocols")

    summaries = []
    for dataset in ("Z24", "HellBridge"):
        directories = [path / dataset for path in reports
                       if (path / dataset / "metrics_by_seed.csv").is_file()]
        if not directories:
            if require_all:
                raise ValueError(f"Missing completed reports for {dataset}")
            continue
        sensor_order = (directories[0] / "sensor_removal_order.csv").read_bytes()
        if any((directory / "sensor_removal_order.csv").read_bytes() != sensor_order for directory in directories):
            raise ValueError(f"{dataset} sensor-removal orders differ")
        paths = [directory / "metrics_by_seed.csv" for directory in directories]
        coverage = pd.concat([pd.read_csv(path, usecols=["model", "seed", "design", "key"], dtype={"key": str})
                              for path in paths], ignore_index=True)
        if coverage.duplicated().any():
            raise ValueError(f"{dataset} contains duplicate model/seed/condition rows")
        if require_all:
            observed = set(coverage.loc[coverage.design.eq("clean"), ["model", "seed"]].itertuples(index=False, name=None))
            expected = {(model, seed) for model in MODELS
                        for seed in (SEEDS[:1] if model == "pca_spe" else SEEDS)}
            if observed != expected:
                raise ValueError(f"{dataset} seed coverage differs: missing={sorted(expected-observed)}")
        target = out / dataset
        target.mkdir(parents=True, exist_ok=True)
        (target / "sensor_removal_order.csv").write_bytes(sensor_order)
        summaries.append(summarize(target, dataset, paths))
    if not summaries:
        raise ValueError("No completed dataset metrics found")
    (out / "protocol.json").write_bytes(protocol)
    (out / "report_sources.json").write_text(json.dumps([str(path) for path in reports], indent=2), encoding="utf-8")
    atomic_csv(pd.concat(summaries, ignore_index=True), out / "summary_mean_sd.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge completed seed shards and regenerate aggregate figures.")
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    merge_reports(args.reports, args.output_dir, args.require_all)


if __name__ == "__main__":
    main()
