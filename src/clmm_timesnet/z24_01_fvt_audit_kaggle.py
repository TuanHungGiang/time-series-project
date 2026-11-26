"""Separate score composition, input masking, and adapter effects on Z24 01-FVT."""

import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


GROUP = "01-FVT"
WATCH = "z24_s08_fvt_setup01"
STEPS = {30: 5, 50: 8}
BRANCHES = ("full_all", "full_retained", "masked_off", "masked_on")


@torch.inference_mode()
def measure(head, source, data, kept, full_only=False):
    device = head.detector.device
    mask = torch.as_tensor(kept, device=device)
    data = torch.as_tensor(data, device=device)
    model = head.detector.model
    rows, by_sensor = [], []
    for chunk in data.split(head.detector.batch_size):
        clean = (chunk - source.model_forward("timesnet", model, chunk)).square()
        retained = clean[:, mask]
        if full_only:
            off, on = retained, retained
        else:
            masked = torch.where(mask.view(1, -1, 1), chunk, 0.0)
            prediction = source.model_forward("timesnet", model, masked)
            off = (chunk[:, mask] - prediction[:, mask]).square()
            on, _ = head.residual(chunk, mask, enabled=True)
        rows.append(torch.stack([x.mean((1, 2)) for x in (clean, retained, off, on)], dim=1).cpu().numpy())
        # Columns 1 and 2 keep NaN at removed channels, except for the all-sensor clean error.
        errors = torch.full((len(chunk), 3, len(kept)), torch.nan, dtype=chunk.dtype, device=device)
        errors[:, 0] = clean.mean(-1)
        errors[:, 1, mask] = off.mean(-1)
        errors[:, 2, mask] = on.mean(-1)
        by_sensor.append(errors.cpu().numpy())
    return np.concatenate(rows).astype(np.float64), np.concatenate(by_sensor).astype(np.float64)


def changes(errors):
    return np.diff(errors, axis=1)


def summarize_watch(errors, reference):
    from clmm_timesnet.conditional_gain_kaggle import robust_scale

    # The existing benchmark selected the lower reconstruction tail on clean validation.
    centers_scales = [robust_scale(np.log1p(reference[:, i]), 1e-6) for i in range(len(BRANCHES))]
    median_log_errors = np.median(np.log1p(errors), axis=0)
    rows = []
    for column, (center, scale) in enumerate(centers_scales):
        rows.append(dict(branch=BRANCHES[column], raw_mse_median=np.median(errors[:, column]),
                         raw_mse_mean=errors[:, column].mean(), ref_log_center=center,
                         ref_log_scale=scale, lower_score=(center - median_log_errors[column]) / scale))
    step = changes(errors)
    attribution = dict(
        composition_mean=step[:, 0].mean(), masking_mean=step[:, 1].mean(), adapter_mean=step[:, 2].mean(),
        total_mean=(errors[:, -1] - errors[:, 0]).mean(), composition_median=np.median(step[:, 0]),
        masking_median=np.median(step[:, 1]), adapter_median=np.median(step[:, 2]),
        adapter_error_reduction_window_fraction=np.mean(step[:, 2] < 0),
    )
    np.testing.assert_allclose(sum(attribution[k] for k in ("composition_mean", "masking_mean", "adapter_mean")),
                               attribution["total_mean"], rtol=1e-10, atol=1e-12)
    return rows, attribution


def main():
    from clmm_timesnet.laplacian_kaggle import BASELINE_COMMIT, SEED, WORK, Z24, checkout
    from clmm_timesnet.targeted_baselines import archive_functions, atomic_csv, load_z24
    from clmm_timesnet.targeted_gain_head import GAIN_WEIGHT, STRENGTH, fit_or_restore_head, sensor_orders, targeted_protocol

    if not (Z24 / "metadata/records_manifest.csv").is_file():
        raise FileNotFoundError(f"Attach the Z24 Kaggle dataset at {Z24}")
    os.chdir(WORK)
    root = WORK / f"clmm_z24_01_fvt_audit_v1_{SEED}"
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    existing = [WORK / f"{name}_{SEED}" for name in (
        "clmm_threshold_masks_v1", "clmm_conditional_gain_v1", "clmm_laplacian_v1", "clmm_old_gain_diagnostics",
    )]
    archive = next((p / "source_baselines" for p in existing if (p / "source_baselines/.git").is_dir()),
                   root / "source_baselines")
    checkout(archive, "codex/mdr-shm-kaggle", BASELINE_COMMIT)
    functions = archive_functions(archive / "research/mdr_shm_kaggle")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    index, group = next((i, g) for i, g in enumerate(load_z24(Z24, 0)) if g.name == GROUP)
    if len(group.sensors) != 17:
        raise ValueError(f"Expected 17 sensors in {GROUP}")
    args = SimpleNamespace(seed=SEED, epochs=30, adapter_epochs=8, batch_size=16, smoke_window_limit=0,
                           strength=STRENGTH, gain_weight=GAIN_WEIGHT, center_gain=False,
                           baseline_runs=None, previous_clmm_weights=None)
    args.protocol = targeted_protocol(args)
    args.previous_clmm_weights = next((p / "weights" for p in existing
                                      if (p / "reports/Z24/sensor_removal_order.csv").is_file()), None)
    order = sensor_orders("Z24", [group], args, out, functions[0])[GROUP]
    args.previous_clmm_weights = next((p / "weights" for p in existing if all(
        (p / "weights" / "Z24" / str(SEED) / f"{GROUP}{suffix}.pt").is_file()
        for suffix in ("", "_adapter")
    )), None)
    weights = root / "weights" / "Z24" / str(SEED)
    weights.mkdir(parents=True, exist_ok=True)
    _, head, _ = fit_or_restore_head("Z24", group, index, SEED, args, functions, device, weights)
    validation = group.metadata.loc[group.metadata.split.eq("val")]
    hit = np.flatnonzero(validation.sample_id.eq(WATCH) & validation.label.eq(0))
    if len(hit) != 1:
        raise ValueError(f"Expected one healthy validation record named {WATCH}")
    edges = np.r_[0, np.cumsum(group.counts[:len(validation)])]
    span = slice(edges[hit[0]], edges[hit[0] + 1])
    val = group.values[:edges[-1]]
    source = functions[-1]
    protocol = dict(seed=SEED, root=str(Z24), group=GROUP, watch_record=WATCH, steps=STEPS, baseline_commit=BASELINE_COMMIT,
                    score_tail="lower selected from clean validation in the previous run", order=[group.sensors[i] for i in order],
                    checkpoints=args.protocol, scored_cohorts="healthy reference and validation only")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"{GROUP} | seed={SEED} | device={device} | observed batch geometry={val.shape}", flush=True)

    summary, deltas, windows, sensors = [], [], [], []
    for requested, count in {0: 0, **STEPS}.items():
        kept = np.ones(len(group.sensors), dtype=bool)
        kept[order[:count]] = False
        ref_errors, _ = measure(head, source, group.reference, kept, count == 0)
        val_errors, sensor_errors = measure(head, source, val, kept, count == 0)
        watched = val_errors[span]
        per_sensor = sensor_errors[span]
        key = dict(sample_id=WATCH, requested_loss=requested, sensors_removed=count,
                   actual_group_loss=100 * count / len(kept),
                   removed_sensors=" ".join(np.asarray(group.sensors)[~kept]))
        branches, attribution = summarize_watch(watched, ref_errors)
        summary.extend(dict(**key, **branch) for branch in branches)
        deltas.append(dict(**key, **attribution))
        for j, row in enumerate(watched):
            windows.append(dict(**key, window_index=j, **dict(zip(BRANCHES, row, strict=True))))
        for i, name in enumerate(group.sensors):
            row = dict(**key, sensor=name, retained=bool(kept[i]), full_mse=per_sensor[:, 0, i].mean())
            if kept[i]:
                row.update(masked_off_mse=per_sensor[:, 1, i].mean(), masked_on_mse=per_sensor[:, 2, i].mean(),
                           masking_delta=(per_sensor[:, 1, i] - per_sensor[:, 0, i]).mean(),
                           adapter_delta=(per_sensor[:, 2, i] - per_sensor[:, 1, i]).mean())
            sensors.append(row)
        print(f"loss={requested}% ({count}/{len(kept)} locally) | removed {key['removed_sensors']}", flush=True)

    for name, rows in (("branch_summary", summary), ("error_attribution", deltas),
                       ("per_window_mse", windows), ("per_sensor_mse", sensors)):
        atomic_csv(pd.DataFrame(rows), out / f"{name}.csv")
    bundle = shutil.make_archive(str(root / "z24_01_fvt_audit"), "zip", root_dir=out)
    print(f"CLMM_01_FVT_AUDIT_PASS reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
