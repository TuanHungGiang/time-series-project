"""Trace the original CLMM gain head at the benchmark's clean and cumulative masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from .metrics import publication_metrics, recall_at_fpr
from .post_block_audit import relative_rms
from .post_block_head import DEVIATION_CAP
from .targeted_baselines import (
    SEEDS, archive_functions, atomic_csv, clean_calibration, load_hellbridge, load_z24,
)
from .targeted_checkpoints import write_json
from .targeted_gain_head import (
    GAIN_WEIGHT, STRENGTH, fit_or_restore_head, score_columns, sensor_orders, targeted_protocol,
)

STAGES = ("embedding", "timesblock1", "timesblock2_pre_adapter", "after_adapter", "layernorm2", "projection")
TAILS = ("upper", "lower", "two_sided")
RATES = (0, 10, 30, 50)


def loss_steps(groups):
    counts = [int(g.metadata.split.eq("test").sum()) for g in groups]
    sizes = [len(g.sensors) for g in groups]
    total = sum(c * size for c, size in zip(counts, sizes, strict=True))
    choices = [(k, 100 * sum(c * min(k, size - 1) for c, size in zip(counts, sizes, strict=True)) / total)
               for k in range(max(sizes))]
    return {rate: min(choices, key=lambda item: (abs(item[1] - rate), item[1])) for rate in RATES}


def aggregate_windows(values, counts, function=np.median):
    return [float(function(part, axis=0)) for part in np.split(values, np.cumsum(counts)[:-1])]


@torch.no_grad()
def trace_layers(head, data, kept):
    model, device = head.detector.model, head.detector.device
    values = torch.as_tensor(data, device=device)
    mask = torch.as_tensor(kept, device=device)
    stages, bins, raw, normalized = {}, {}, [], []
    probe, periods, activations = {}, {}, {}
    norm_calls = 0

    def reduce(output):
        return torch.cat((output.mean(1), output.std(1, unbiased=False)), dim=1)

    def save(name, observed=False):
        def hook(_module, _inputs, output):
            probe[name] = reduce(output[:, :, mask] if observed else output)
            if name in ("timesblock2_pre_adapter", "after_adapter"):
                activations[name] = output
        return hook

    def norm_hook(_module, _inputs, output):
        nonlocal norm_calls
        norm_calls += 1
        probe["timesblock1" if norm_calls == 1 else "layernorm2"] = reduce(output)
        if norm_calls == 2:
            activations["layernorm2"] = output

    def fft(index):
        def hook(block, inputs):
            spectrum = torch.fft.rfft(inputs[0], dim=1).abs().mean(0).mean(-1)
            spectrum[0] = 0.0
            periods[index] = spectrum.topk(block.k).indices.expand(len(inputs[0]), -1)
        return hook

    handles = [
        model.enc_embedding.register_forward_hook(save("embedding")),
        model.model[1].register_forward_hook(save("timesblock2_pre_adapter")),
        head.adapter.register_forward_hook(save("after_adapter")),
        model.layer_norm.register_forward_hook(norm_hook),
        model.projection.register_forward_hook(save("projection", observed=True)),
        *[block.register_forward_pre_hook(fft(index)) for index, block in enumerate(model.model, 1)],
    ]
    try:
        for chunk in values.split(head.detector.batch_size):
            norm_calls = 0
            probe.clear()
            periods.clear()
            activations.clear()
            head.residual(chunk, mask, enabled=True)
            if len(probe) != len(STAGES) or norm_calls != 2:
                raise RuntimeError(f"Expected two TimesBlocks; captured {list(probe)} and {norm_calls} norms")
            for name, feature in probe.items():
                stages.setdefault(name, []).append(feature.cpu().numpy())
            for index, top in periods.items():
                bins.setdefault(index, []).append(top.cpu().numpy())
            before, after = (activations[name] for name in ("timesblock2_pre_adapter", "after_adapter"))
            layer = model.layer_norm
            baseline_norm = F.layer_norm(before, layer.normalized_shape, layer.weight, layer.bias, layer.eps)
            raw.append(relative_rms(before, after).cpu().numpy())
            normalized.append(relative_rms(baseline_norm, activations["layernorm2"]).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return (
        {key: np.concatenate(parts) for key, parts in stages.items()},
        {key: np.concatenate(parts) for key, parts in bins.items()},
        np.concatenate(raw), np.concatenate(normalized),
    )


def probe_score(features, reference):
    center = np.median(reference, axis=0)
    mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
    scale = np.maximum(np.maximum(mad, .25 * reference.std(axis=0)), 1e-3)
    return (np.abs(features - center).astype(np.float64) / scale).mean(axis=1)


def run_group(dataset, group, order, index, args, functions, device, out, steps):
    scores_path = out / "groups" / f"{group.name}.csv.gz"
    if scores_path.is_file():
        print(f"SKIP {dataset}/{group.name}", flush=True)
        return pd.read_csv(scores_path)
    weights = args.weights_dir / dataset / str(args.seed)
    weights.mkdir(parents=True, exist_ok=True)
    _, aggregate, _, _, source = functions
    scorer, head, norm = fit_or_restore_head(dataset, group, index, args.seed, args, functions, device, weights)
    n_val = int(group.metadata.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    test = group.metadata.iloc[n_val:].drop(columns="path", errors="ignore").copy()
    counts = group.counts[n_val:]
    values = group.values[offset:]
    val = group.metadata.iloc[:n_val].drop(columns="path", errors="ignore").assign(
        **score_columns(dataset, scorer(group.values[:offset]), group.counts[:n_val], norm, aggregate),
        requested_loss=0, actual_loss=0., group=group.name,
    )
    rows = [val]
    for rate, (step, percent) in steps.items():
        removed = order[:min(step, len(order) - 1)]
        kept = np.ones(len(group.sensors), dtype=bool)
        kept[removed] = False
        reference, _, _, _ = trace_layers(head, group.reference, kept)
        features, periods, shift, after_norm = trace_layers(head, values, kept)
        if rate == 0:
            clean_periods = periods
            reconstruction = scorer(values)
            deviation = np.zeros(len(values))
        else:
            (_, reconstruction, deviation), _ = head.targeted_signals(source, values, group.reference, kept, False)
        base = score_columns(dataset, reconstruction, counts, norm, aggregate)
        final = base if rate == 0 else score_columns(
            dataset, reconstruction, counts, norm, aggregate, GAIN_WEIGHT * deviation,
        )
        row = test.assign(
            group=group.name, requested_loss=rate, actual_loss=percent,
            removed_sensors=" ".join(group.sensors[i] for i in removed),
            **final, **{f"reconstruction_{key}": value for key, value in base.items()},
            gain=aggregate_windows(deviation, counts), adapter_shift=aggregate_windows(shift, counts),
            after_norm_shift=aggregate_windows(after_norm, counts),
        )
        for stage in STAGES:
            row[stage] = aggregate_windows(probe_score(features[stage], reference[stage]), counts)
        for block in periods:
            changed = (np.sort(periods[block], axis=1) != np.sort(clean_periods[block], axis=1)).any(axis=1)
            row[f"fft_changed_{block}"] = aggregate_windows(changed, counts, np.mean)
        rows.append(row)
        print(f"  loss={rate}% actual={percent:.2f}% group={group.name} "
              f"fft_changed={row.fft_changed_1.mean():.3f}/{row.fft_changed_2.mean():.3f} "
              f"adapter_shift={row.adapter_shift.median():.4f} after_norm={row.after_norm_shift.median():.4f}",
              flush=True)
    result = pd.concat(rows, ignore_index=True)
    atomic_csv(result, scores_path)
    for name, entries in (("adapter_training", args.train_log), ("gain_reference", args.reference_log)):
        if entries:
            atomic_csv(pd.DataFrame(entries), out / "groups" / f"{group.name}_{name}.csv")
    return result


def summarize(dataset, records, select, out, args):
    val = records.loc[records.split.eq("val")]
    test = records.loc[records.split.eq("test")].copy()
    tail, threshold = clean_calibration(dataset, val, select)
    low_fpr = 0. if dataset == "Z24" else .05
    test = test.assign(score=test[tail], reconstruction_score=test[f"reconstruction_{tail}"])
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    metrics, probes, cohorts, head_diagnostics = [], [], [], []
    diagnostic_stages = (*STAGES, "reconstruction_score", "gain", "score")
    for rate, rows in test.groupby("requested_loss"):
        if rows.label.value_counts().to_dict() != expected or rows.sample_id.duplicated().any():
            raise ValueError(f"Unexpected test cohort: {dataset}/{rate}")
        y = rows.label.to_numpy()
        key = dict(dataset=dataset, requested_loss=rate, actual_loss=rows.actual_loss.iloc[0])
        metrics.append({**key, **publication_metrics(y, rows.score, threshold),
                        "recall_low_fpr": recall_at_fpr(y, rows.score, low_fpr)})
        for stage in diagnostic_stages:
            values = rows[stage].to_numpy()
            probes.append(dict(**key, stage=stage, auc=roc_auc_score(y, values),
                               recall_low_fpr=recall_at_fpr(y, values, low_fpr),
                               healthy_median=np.median(values[y == 0]), damage_median=np.median(values[y == 1])))
        for label, part in rows.groupby("label"):
            score_delta = part.score - part.reconstruction_score
            cohorts.append(dict(**key, label=label, records=len(part),
                                fft_changed_1=part.fft_changed_1.mean(), fft_changed_2=part.fft_changed_2.mean(),
                                adapter_shift=part.adapter_shift.median(),
                                after_norm_shift=part.after_norm_shift.median()))
            head_diagnostics.append(dict(
                **key, label=label, records=len(part), median_gain=part.gain.median(),
                gain_at_cap_rate=np.isclose(part.gain, DEVIATION_CAP).mean(),
                median_score_increment=score_delta.median(),
                crossed_threshold_after_gain=((part.reconstruction_score < threshold) & (part.score >= threshold)).sum(),
            ))
    metrics, probes, cohorts = map(pd.DataFrame, (metrics, probes, cohorts))
    probes["auc_delta_vs_clean"] = probes.auc - probes.stage.map(
        probes.loc[probes.requested_loss.eq(0)].set_index("stage").auc)
    columns = ["requested_loss", "actual_loss", "auc", "ba", "recall", "fpr", "recall_low_fpr"]
    print(f"\n{dataset} OLD CLMM | score_tail={tail} | frozen_threshold={threshold:.6f}", flush=True)
    print(metrics[columns].round(4).to_string(index=False), flush=True)
    print(f"\n{dataset} LAYER PROBES (descriptive, matched healthy reference)", flush=True)
    points = probes.pivot(index="stage", columns="requested_loss", values="auc").reindex(diagnostic_stages)
    print(points.round(4).to_string(), flush=True)
    print(cohorts.round(4).to_string(index=False), flush=True)
    head_diagnostics = pd.DataFrame(head_diagnostics)
    print(f"\n{dataset} GAIN HEAD | saturation and threshold crossings by class", flush=True)
    print(head_diagnostics.round(4).to_string(index=False), flush=True)
    for name, frame in (("old_clmm_metrics", metrics), ("layer_probes", probes),
                        ("activation_changes", cohorts), ("gain_head_diagnostics", head_diagnostics),
                        ("test_record_diagnostics", test)):
        atomic_csv(frame, out / f"{name}.csv")
    atomic_csv(val, out / "calibration_records.csv")
    write_json(args.weights_dir / dataset / str(args.seed) / "calibration.json",
               dict(score_tail=tail, threshold=threshold, healthy_quantile=.90,
                    score_unit="record" if dataset == "Z24" else "window"))
    return metrics, probes, cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("z24-root", "hell-h5", "baseline-dir", "output", "weights-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("baseline-runs", "previous-clmm-weights"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--seed", type=int, default=SEEDS[0], choices=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=["Z24", "HellBridge"], choices=["Z24", "HellBridge"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=8)
    parser.add_argument("--smoke-window-limit", type=int, default=0)
    args = parser.parse_args()
    args.batch_size, args.strength, args.gain_weight, args.center_gain = 16, STRENGTH, GAIN_WEIGHT, False
    args.protocol = targeted_protocol(args)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output / "diagnostic_protocol.json"
    protocol = dict(seed=args.seed, z24_root=str(args.z24_root), hell_h5=str(args.hell_h5),
                    diagnostic="old gain head only", rates=list(RATES), backbone=args.protocol)
    if provenance.is_file() and json.loads(provenance.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Use a fresh output directory: {args.output}")
    write_json(provenance, protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    functions = archive_functions(args.baseline_dir)
    summaries = []
    for dataset in args.datasets:
        out = args.output / dataset
        (out / "groups").mkdir(parents=True, exist_ok=True)
        groups = (list(load_z24(args.z24_root, args.smoke_window_limit)) if dataset == "Z24"
                  else [load_hellbridge(args.hell_h5, out, args.smoke_window_limit)])
        orders = sensor_orders(dataset, groups, args, out, functions[0])
        steps = loss_steps(groups)
        records = pd.concat([run_group(dataset, group, orders[group.name], index, args, functions, device, out, steps)
                             for index, group in enumerate(groups)], ignore_index=True)
        summaries.append(summarize(dataset, records, functions[3], out, args))
    for name, tables in zip(("old_clmm_metrics", "layer_probes", "activation_changes"), zip(*summaries), strict=True):
        atomic_csv(pd.concat(tables, ignore_index=True), args.output / f"{name}.csv")
    print(f"OLD_CLMM_DIAGNOSTIC_PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
