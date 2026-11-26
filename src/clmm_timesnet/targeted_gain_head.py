"""Evaluate the post-TimesBlock-2 residual adapter and gain head on baseline masks."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from clmm_timesnet import targeted_artifacts as artifacts, targeted_checkpoints as checkpoints
from clmm_timesnet.models.backbone import seed_everything
from clmm_timesnet.post_block_audit import PostBlockAdapter
from clmm_timesnet.post_block_head import GainHeadAudit, gain_deviation
from clmm_timesnet.targeted_baselines import (
    METRICS, SEEDS, archive_functions, atomic_csv, build_plan, condition_plan, evaluate,
    load_hellbridge, load_z24, summarize,
)

METHODS = ("timesnet", "post_block_residual", "clmm_gain_head")
CENTERED_METHOD = "clmm_centered_gain_head"
STRENGTH, GAIN_WEIGHT = .50, .25


class TargetedHead(GainHeadAudit):
    def __init__(self, scorer, fit, reference, seed, args, group, state_path, prior_adapter=None):
        self.detector = SimpleNamespace(device=scorer.device, model=scorer.state.requires_grad_(False).eval(),
                                        batch_size=args.batch_size)
        seed_everything(seed + 7001)
        width = scorer.state.configs.d_model
        self.adapter = PostBlockAdapter(width, fit.shape[1], STRENGTH).to(scorer.device)
        self.extra_centers = {}
        restored_from = state_path if state_path.is_file() else prior_adapter
        if restored_from is not None:
            state = torch.load(restored_from, map_location="cpu", weights_only=True)
            previous = {key: value for key, value in state["protocol"].items() if key != "gain_centering"}
            current = {key: value for key, value in args.protocol.items() if key != "gain_centering"}
            if previous != current:
                raise ValueError(f"Adapter configuration differs: {restored_from}")
            self.adapter.load_state_dict(state["adapter"])
            self.adapter.eval().requires_grad_(False)
            self.centers = {n: tensor.numpy() for n, tensor in state["gain_centers"].items()}
            self.scale = state["gain_scale"].numpy()
            print(f"  adapter restored from {restored_from}", flush=True)
        else:
            self.fit(fit, seed, args, group)
            self.reference_audit(reference, group, args)
            state = {
                "adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                "gain_centers": {k: torch.from_numpy(v) for k, v in self.centers.items()},
                "gain_scale": torch.from_numpy(self.scale),
                "protocol": args.protocol,
            }
        if not state_path.is_file():
            temporary = state_path.with_suffix(".tmp")
            torch.save({**state, "protocol": args.protocol}, temporary)
            temporary.replace(state_path)

    def prepare(self, values):
        return (values,)

    @torch.no_grad()
    def mask_reference(self, reference, kept, center_gain):
        missing = int((~kept).sum())
        gains = None
        if missing in self.centers:
            center = self.centers[missing][kept]
        if missing not in self.centers or center_gain:
            ref = torch.as_tensor(reference, device=self.detector.device)
            mask = torch.as_tensor(kept, device=self.detector.device)
            parts = [self.batch_signals(chunk, mask)[2] for chunk in ref.split(self.detector.batch_size)]
            gains = np.concatenate(parts)
        if missing not in self.centers:
            center = np.median(gains, axis=0)
            self.extra_centers[tuple(kept)] = center
        offset = float(np.median(gain_deviation(gains, center, self.scale[kept]))) if center_gain else 0.0
        return center, offset

    @torch.no_grad()
    def targeted_signals(self, source, values, reference, kept, center_gain):
        device = self.detector.device
        mask = torch.as_tensor(kept, device=device)
        values = torch.as_tensor(values, device=device)
        center, offset = self.mask_reference(reference, kept, center_gain)
        outputs = []
        for chunk in values.split(self.detector.batch_size):
            masked = torch.where(mask.view(1, -1, 1), chunk, 0.0)
            prediction = source.model_forward("timesnet", self.detector.model, masked)
            off = (masked - prediction).square()
            on, _ = self.residual(chunk, mask, enabled=True)
            raw_native = off.mean((1, 2)).cpu().numpy().astype(np.float64)
            raw_adapter = on.mean((1, 2)).cpu().numpy().astype(np.float64)
            gain = (np.log1p(off[:, mask].mean(-1).cpu().numpy().astype(np.float64))
                    - np.log1p(on.mean(-1).cpu().numpy().astype(np.float64)))
            deviation = gain_deviation(gain, center, self.scale[kept])
            outputs.append(np.column_stack((raw_native, raw_adapter, deviation)))
        return np.concatenate(outputs).T, offset


def score_columns(dataset, raw, counts, norm, aggregate, bonus=None):
    if dataset == "HellBridge":
        upper = raw if bonus is None else np.expm1(np.log1p(raw) + norm[1] * bonus)
        return dict(upper=upper, lower=-upper, two_sided=np.abs(upper))
    if bonus is None:
        upper, two_sided = aggregate(raw, counts, norm)
        return dict(upper=upper, lower=-np.asarray(upper), two_sided=two_sided)
    values = (np.log1p(raw) - norm[0]) / norm[1]
    offsets = np.cumsum(counts)[:-1]
    return {name: [np.median(part) for part in np.split(points, offsets)]
            for name, points in (("upper", values + bonus), ("lower", -values + bonus),
                                 ("two_sided", np.abs(values) + bonus))}


def run_group(dataset, group, order, index, seed, args, functions, device, run, weights):
    _, aggregate, reference, _, source = functions
    cache_file = run / f"{group.name}.csv.gz"
    own_weights = weights / f"{group.name}.pt"
    adapter_path = weights / f"{group.name}_adapter.pt"
    if cache_file.exists() and own_weights.is_file() and adapter_path.is_file():
        print(f"SKIP {dataset}/{group.name}/{seed}", flush=True)
        return pd.read_csv(cache_file, dtype={"key": str})
    print(f"{dataset} seed={seed} group={group.name} fit={group.train.shape} eval={group.values.shape}", flush=True)
    effective_seed = seed + (1009 * index if dataset == "Z24" else 0)
    prior = None
    if args.baseline_runs:
        prior = (args.baseline_runs / f"targeted_checkpoint_seed{SEEDS.index(seed):02d}" / "weights"
                 / dataset / f"timesnet__{seed}" / own_weights.name)
    prior_adapter = None
    if args.previous_clmm_weights is not None and not own_weights.is_file():
        previous = args.previous_clmm_weights / dataset / str(seed)
        if (previous / own_weights.name).is_file() and (previous / adapter_path.name).is_file():
            prior = previous / own_weights.name
            prior_adapter = previous / adapter_path.name
    target = own_weights if own_weights.is_file() else prior
    if target is not None and target.is_file():
        state, scorer = checkpoints.load_checkpoint(target, source, device)
        meta = state["metadata"]
        expected = (dataset, group.name, seed, effective_seed, args.epochs, args.batch_size, group.sensors)
        fields = ("dataset", "group", "seed", "effective_seed", "epochs", "batch_size", "sensors")
        if (tuple(meta[key] for key in fields) != expected or meta["model"] != "timesnet"
                or meta["protocol"]["smoke_window_limit"] != args.smoke_window_limit):
            raise ValueError(f"Incompatible backbone {target}")
        np.testing.assert_array_equal(state["normalizer_mean"].numpy(), group.normalizer.mean)
        np.testing.assert_array_equal(state["normalizer_scale"].numpy(), group.normalizer.scale)
        checkpoints.verify_checkpoint(state, scorer)
        del state
        print(f"  backbone restored from {target}", flush=True)
    else:
        seed_everything(effective_seed)
        scorer = checkpoints.fit_scorer(source, "timesnet", group.train, args.epochs, args.batch_size, device)
    norm = reference(scorer(group.reference))
    if not own_weights.exists():
        checkpoints.write_checkpoint(own_weights, scorer, group, dataset, seed, effective_seed,
                                     args.epochs, norm if dataset == "Z24" else None, args.protocol)
    args.train_log, args.reference_log = [], []
    head = TargetedHead(scorer, group.train, group.reference, effective_seed, args, group.name,
                        adapter_path, prior_adapter=prior_adapter)
    for filename, entries in (("adapter_training", args.train_log), ("gain_reference", args.reference_log)):
        if entries:
            atomic_csv(pd.DataFrame(entries), run / f"{group.name}_{filename}.csv")

    n_val = int(group.metadata.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    raw = np.concatenate((scorer(group.values[:offset]), scorer(group.values[offset:]))).astype(np.float64)
    clean = score_columns(dataset, raw, group.counts, norm, aggregate)
    clean_metadata = group.metadata.drop(columns="path", errors="ignore")
    test = clean_metadata.iloc[n_val:]
    counts = group.counts[n_val:]
    cache, rows = {(): (clean,) * len(args.methods)}, []
    gain_offsets = []
    for design, key, removed in condition_plan(dataset, group.sensors, order):
        mask = tuple(sorted(removed))
        if mask not in cache:
            kept = np.ones(len(group.sensors), dtype=bool)
            kept[list(mask)] = False
            (native, adapted, deviation), gain_offset = head.targeted_signals(
                source, group.values[offset:], group.reference, kept, args.center_gain,
            )
            cache[mask] = (
                score_columns(dataset, native, counts, norm, aggregate),
                score_columns(dataset, adapted, counts, norm, aggregate),
                score_columns(dataset, adapted, counts, norm, aggregate, GAIN_WEIGHT * deviation),
            )
            if args.center_gain:
                bonus = GAIN_WEIGHT * (deviation - gain_offset)
                cache[mask] += (score_columns(dataset, adapted, counts, norm, aggregate, bonus),)
                gain_offsets.append(dict(
                    group=group.name, sensors_removed=len(mask), healthy_windows=len(group.reference),
                    removed_sensors=" ".join(group.sensors[i] for i in mask),
                    healthy_gain_median=gain_offset, score_offset=GAIN_WEIGHT * gain_offset,
                ))
        metadata = test if mask else clean_metadata
        for method, scores in zip(args.methods, cache[mask], strict=True):
            if not all(np.isfinite(values).all() for values in scores.values()):
                raise ValueError(f"Nonfinite scores: {dataset}/{group.name}/{method}/{key}")
            rows.append(metadata.assign(group=group.name, design=design, key=key, removed=len(mask),
                                        model=method, **scores))
    frame = pd.concat(rows, ignore_index=True)
    if head.extra_centers:
        pd.DataFrame([
            dict(removed_sensors=" ".join(name for name, ok in zip(group.sensors, kept) if not ok),
                 retained_sensor=name, gain_center=float(center))
            for kept, centers in head.extra_centers.items()
            for name, center in zip(np.asarray(group.sensors)[np.asarray(kept)], centers, strict=True)
        ]).to_csv(run / f"{group.name}_additional_mask_references.csv", index=False)
    if gain_offsets:
        atomic_csv(pd.DataFrame(gain_offsets), run / f"{group.name}_healthy_gain_offsets.csv")
    atomic_csv(frame, cache_file)
    print(f"DONE {dataset}/{group.name}/{seed}", flush=True)
    return frame


def run_dataset(dataset, groups, args, device, functions):
    out = args.output / dataset
    out.mkdir(parents=True, exist_ok=True)
    reference_dirs = []
    if args.previous_clmm_weights is not None:
        reference_dirs.append(args.previous_clmm_weights.parent / "reports")
    if args.baseline_runs is not None:
        reference_dirs.append(args.baseline_runs / "targeted_checkpoint_seed00" / "reports")
    orders_available = (root / dataset / "sensor_removal_order.csv" for root in reference_dirs)
    order_path = next((path for path in orders_available if path.is_file()), None)
    if order_path is not None:
        order_table = pd.read_csv(order_path)
        orders = {}
        for group in groups:
            labels = order_table.loc[order_table.group.eq(group.name)].sort_values("rank").sensor.tolist()
            if len(labels) != len(group.sensors) or set(labels) != set(group.sensors):
                raise ValueError(f"Incompatible sensor order for {dataset}/{group.name}: {order_path}")
            orders[group.name] = [group.sensors.index(sensor) for sensor in labels]
        atomic_csv(order_table, out / "sensor_removal_order.csv")
        print(f"{dataset} sensor order restored from {order_path}", flush=True)
    else:
        orders = build_plan(groups, functions[0], out)
    artifacts.export_inputs(dataset, groups, orders, args, out)
    masks = []
    for group in groups:
        for design, key, removed in condition_plan(dataset, group.sensors, orders[group.name]):
            masks.append(dict(group=group.name, design=design, key=key, removed=len(removed),
                              removed_sensors=" ".join(group.sensors[i] for i in removed),
                              retained_sensors=" ".join(s for i, s in enumerate(group.sensors) if i not in removed)))
    manifest = pd.DataFrame(masks)
    manifest_file = out / "mask_manifest.csv"
    if manifest_file.is_file() and not pd.read_csv(manifest_file, dtype={"key": str},
                                                  keep_default_na=False).equals(manifest):
        raise ValueError(f"Sensor masks changed. Use a fresh output directory: {out}")
    atomic_csv(manifest, manifest_file)
    max_steps = {group.name: len(group.sensors) - 1 for group in groups}
    paths, paired = [], []
    for seed in args.seeds:
        run = out / "groups" / str(seed)
        weights = args.weights_dir / dataset / str(seed)
        run.mkdir(parents=True, exist_ok=True)
        weights.mkdir(parents=True, exist_ok=True)
        frames = pd.concat([
            run_group(dataset, group, orders[group.name], index, seed, args, functions, device, run, weights)
            for index, group in enumerate(groups)
        ], ignore_index=True)
        results = {}
        calibrations = {}
        for method in args.methods:
            scores = frames.loc[frames.model.eq(method)]
            metrics = evaluate(dataset, scores, max_steps, functions[3]).assign(model=method, seed=seed)
            results[method] = metrics
            calibration = dict(score_tail=metrics.score_tail.iloc[0], threshold=float(metrics.threshold.iloc[0]),
                               healthy_quantile=.9, score_unit="record" if dataset == "Z24" else "window")
            calibrations[method] = calibration
            target = out / "runs" / f"{method}__{seed}"
            target.mkdir(parents=True, exist_ok=True)
            path = target / "metrics.csv"
            artifacts.export_scores(scores, max_steps, calibration, target)
            atomic_csv(metrics, path)
            paths.append(path)
        if any(rule != calibrations["timesnet"] for rule in calibrations.values()):
            raise ValueError(f"Clean calibration differs across branches: {dataset}/{seed}")
        checkpoints.write_json(weights / "calibration.json", calibrations["timesnet"])
        keys = ["design", "key"]
        final_method = args.methods[-1]
        candidate = results[final_method].set_index(keys)
        for comparator in args.methods[:-1]:
            base = results[comparator].set_index(keys)
            delta = candidate[list(METRICS)].sub(base[list(METRICS)]).add_prefix("delta_")
            paired.append(delta.assign(dataset=dataset, seed=seed, method=final_method, comparator=comparator,
                                       removed_percent=base.removed_percent).reset_index())
    combined = pd.concat(paired, ignore_index=True)
    atomic_csv(combined, out / "paired_deltas_by_seed.csv")
    artifacts.plot_discrimination(dataset, out, [path for path in paths if path.parent.name.endswith(f"__{args.seeds[0]}")])
    return summarize(out, dataset, paths), combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", required=True, type=Path)
    parser.add_argument("--hell-h5", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--baseline-runs", type=Path)
    parser.add_argument("--previous-clmm-weights", type=Path)
    parser.add_argument("--center-gain", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", default=[SEEDS[0]], choices=SEEDS, type=int)
    parser.add_argument("--datasets", nargs="+", default=["Z24", "HellBridge"], choices=["Z24", "HellBridge"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--adapter-epochs", type=int, default=8)
    parser.add_argument("--smoke-window-limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.adapter_epochs) < 1 or args.smoke_window_limit not in (0, *range(4, 65)):
        parser.error("Invalid epoch count or smoke window limit")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Seeds and datasets must be unique")
    args.batch_size, args.strength, args.gain_weight = 16, STRENGTH, GAIN_WEIGHT
    args.methods = (*METHODS, CENTERED_METHOD) if args.center_gain else METHODS
    args.output.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.protocol = dict(
        design="targeted post-block gain head", epochs=args.epochs, adapter_epochs=args.adapter_epochs,
        batch_size=args.batch_size, smoke_window_limit=args.smoke_window_limit, strength=args.strength,
        gain_weight=args.gain_weight, z24_points=512, hellbridge_points=128,
        gain_reference="healthy per-count reference at original 10/30/50%; exact-mask healthy median otherwise",
        hellbridge_reference="20 healthy validation windows; no test windows",
        score_tail="shared native TimesNet direction and Q90 from clean, fixed across all masks",
        z24_two_sided="median(abs(standardized adapted MSE) + gain bonus) over windows",
        hellbridge_gain="monotone transform back to native window MSE scale",
    )
    if args.center_gain:
        args.protocol["gain_centering"] = "subtract the healthy aggregate gain median under the identical mask"
    for folder in (args.output, args.weights_dir):
        protocol = folder / "protocol.json"
        if protocol.is_file() and json.loads(protocol.read_text(encoding="utf-8")) != args.protocol:
            raise ValueError(f"Use a fresh output directory: {folder}")
        checkpoints.write_json(protocol, args.protocol)
    archive = archive_functions(args.baseline_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries, pairs = [], []
    for dataset in args.datasets:
        groups = (list(load_z24(args.z24_root, args.smoke_window_limit)) if dataset == "Z24"
                  else [load_hellbridge(args.hell_h5, args.output / dataset, args.smoke_window_limit)])
        summary, paired = run_dataset(dataset, groups, args, device, archive)
        summaries.append(summary)
        pairs.append(paired)
    atomic_csv(pd.concat(summaries, ignore_index=True), args.output / "summary_mean_sd.csv")
    atomic_csv(pd.concat(pairs, ignore_index=True), args.output / "paired_deltas_by_seed.csv")
    print(f"CLMM_TARGETED_PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
