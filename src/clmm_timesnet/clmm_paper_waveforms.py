"""Export two paired reconstruction examples per bridge from one seed."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .clmm_paper_score_figures import save_figure
from .data.hellbridge import development_split
from .data.prepare_hellbridge import main as prepare_hellbridge
from .data.z24 import load_manifest, split_training_windows, stable_sensor_map
from .mask_pair_block import cases
from .post_block_audit import ResidualAudit
from .timesnet_native_1024 import LENGTH, load_hellbridge, load_z24, subset
from .whitening_multiseed_kaggle import SEEDS
from .whitening_two_bridges_kaggle import INPUT, Z24_INPUT

REPLICATE = 300
CONDITION = "sensor_0.50"


class WaveformModel(ResidualAudit):
    def reference_audit(self, reference, group, args):
        # Weight fitting is complete. This figure uses reconstruction outputs.
        pass


def loss_mask(channels, key):
    return next(mask for condition, rep, mask in cases(channels, key, [REPLICATE])
                if condition == CONDITION and rep == REPLICATE)


def select_z24(root):
    manifest = load_manifest(root)
    for index, ((setup, excitation), sensors) in enumerate(stable_sensor_map(manifest).items()):
        group = manifest.loc[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]
        test = group.loc[group.split.eq("test")]
        if set(test.label) == {0, 1}:
            healthy_fit = group.loc[group.split.eq("train")]
            fit, ref = split_training_windows([load_z24(Path(row.path), sensors)
                                               for row in healthy_fit.itertuples()])
            examples = []
            for label in (0, 1):
                row = test.loc[test.label.eq(label)].sort_values(["state", "record_id"]).iloc[0]
                sequence = load_z24(Path(row.path), sensors)
                ids = np.linspace(0, len(sequence) - 1, min(len(sequence), 8), dtype=int)
                batch = sequence[ids]
                middle = len(batch) // 2
                examples.append(dict(
                    label=label, state=str(row.state), sample_id=row.record_id, data=batch,
                    within_batch=middle, window_index=int(ids[middle]), condition=CONDITION,
                    mask=loss_mask(len(sensors), row.record_id),
                ))
            return subset(fit, 8), subset(ref, 8), f"{setup:02d}/{excitation.upper()}", (
                SEEDS[0] + 1009 * index), sensors, examples
    raise ValueError("No Z24 test cohort includes both healthy and damaged examples")


def select_hellbridge(h5, out):
    split_dir = out / "hellbridge_split"
    prepare_hellbridge([
        "--h5", str(h5), "--output-dir", str(split_dir), "--profile", "svendsen2022",
        "--regime", "P2_NM_Z", "--seed", "20261113", "--strict-paper-counts",
    ])
    manifest = pd.read_csv(split_dir / "sample_manifest.csv").sort_values(
        ["record_id", "start"]).reset_index(drop=True)
    fit, _, test = development_split(manifest)
    blocks = fit.reshape(-1, 6)
    ref, fit = blocks[:, -1], blocks[:, :-1].ravel()
    selected = [fit, ref]
    examples = []
    sensors = [f"AL{i:02d}" for i in range(1, 41)]
    mask = loss_mask(len(sensors), "P2_NM_Z")
    for label in (0, 1):
        cohort = manifest.loc[test]
        group = cohort.loc[cohort.label.eq(label)]
        record = group.record_id.iloc[0]
        group = group.loc[group.record_id.eq(record)]
        row = group.iloc[len(group) // 2]
        index = int(np.flatnonzero(test == row.name)[0])
        batch_start = index // 8 * 8
        selected.append(test[batch_start:batch_start + 8])
        examples.append(dict(
            label=label, state=str(row.state), sample_id=row.sample_id,
            within_batch=index - batch_start, window_index=int(row.window_index),
            condition=CONDITION, mask=mask,
        ))
    raw = load_hellbridge(h5, manifest.loc[np.concatenate(selected)].reset_index(drop=True))
    split_at = np.cumsum([len(item) for item in selected[:-1]])
    healthy_fit, healthy_ref, healthy, damage = np.split(raw, split_at)
    for item, batch in zip(examples, (healthy, damage), strict=True):
        item["data"] = batch
    return healthy_fit, healthy_ref, "P2_NM_Z", SEEDS[0], sensors, examples


def traces(dataset, data, args, device):
    fit, ref, group, seed, sensors, examples = data
    print(f"{dataset} | fitting group {group} | seed {seed}", flush=True)
    model = WaveformModel(fit, ref, seed, args, device, group)
    first_shared = int(np.flatnonzero(examples[0]["mask"] & examples[1]["mask"])[0])
    output = []
    for example in examples:
        mask_array = example["mask"]
        mask = torch.as_tensor(mask_array, device=device)
        kept_position = int(np.flatnonzero(np.flatnonzero(mask_array) == first_shared)[0])
        values = torch.as_tensor(model.prepare(example["data"])[0], device=device)
        with torch.no_grad():
            base, _ = model.predict_observed(values, mask, enabled=False)
            corrected, _ = model.predict_observed(values, mask, enabled=True)
        index = example["within_batch"]
        obs = values[index, first_shared].cpu().numpy()
        output.append(pd.DataFrame(dict(
            dataset=dataset, group=group, effective_seed=seed, state=example["state"],
            label=example["label"], sample_id=example["sample_id"], window_index=example["window_index"],
            sensor=sensors[first_shared], dropped=int((~mask_array).sum()), total=len(mask_array),
            replicate=REPLICATE, time_s=np.arange(LENGTH) / 100,
            observed=obs, base=base[index, kept_position].cpu().numpy(),
            corrected=corrected[index, kept_position].cpu().numpy(),
        )))
    return pd.concat(output, ignore_index=True)


def draw(table, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), layout="constrained", sharex=True)
    for row, bridge in enumerate(("Z24", "HellBridge")):
        for label, class_name in enumerate(("Healthy", "Damage")):
            ax = axes[row, label]
            part = table.loc[table.dataset.eq(bridge) & table.label.eq(label)]
            first = part.iloc[0]
            for col, title, color, style in zip(
                ("observed", "base", "corrected"), ("Observed", "Base", "Corrected"),
                ("#24384e", "#a76a44", "#167477"), ("-", "--", "-"), strict=True,
            ):
                ax.plot(part.time_s, part[col], color=color, linestyle=style, linewidth=1.0, label=title,
                        alpha=.87)
            ax.set(title=f"{bridge} · {class_name} · {first.sensor} (state {first.state})",
                   ylabel="Standardized acceleration", xlabel="Time (s)")
            ax.grid(axis="y", alpha=.19)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Observed-channel reconstructions · 50% nominal sensor loss · mask 300")
    save_figure(fig, out, "observed_base_corrected_waveforms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, default=Z24_INPUT)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU on Kaggle for the waveform reconstruction run.")
    for path in (args.z24_root / "metadata/records_manifest.csv", args.hell_h5):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args.epochs, args.adapter_epochs, args.batch_size = 5, 8, 8
    args.layer_probe, args.strength, args.train_log = False, .5, []
    z24 = traces("Z24", select_z24(args.z24_root), args, device)
    hell = traces("HellBridge", select_hellbridge(args.hell_h5, out), args, device)
    result = pd.concat([z24, hell], ignore_index=True)
    result.to_csv(out / "waveform_examples.csv.gz", index=False)
    draw(result, out)
    print(f"CLMM_PAPER_WAVEFORMS_READY {out}", flush=True)


if __name__ == "__main__":
    main()
