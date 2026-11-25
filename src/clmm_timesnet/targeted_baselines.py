"""Paired single-sensor and cumulative-outage benchmark on Z24 and HellBridge."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data.hellbridge import Normalizer, development_split, load_windows
from .data.prepare_hellbridge import main as prepare_hellbridge
from .data.z24 import load_manifest, read_stable_vertical, record_windows, split_training_windows, stable_sensor_map
from .metrics import publication_metrics, recall_at_fpr, threshold_from_healthy
from .models.backbone import seed_everything

MODELS = ("pca_spe", "lstm_ae", "usad", "tranad", "timesnet", "moderntcn", "scatterad")
SEEDS = (20270901, 20270913, 20270929, 20271011, 20271027, 20271109, 20271123, 20271207, 20271219, 20280103)
METRICS = ("auc", "ap", "ba", "recall", "fpr", "precision", "f1", "recall_low_fpr")
SCORE_COLUMNS = ("upper", "lower", "two_sided")
Z24_EXAMPLE = ("441V", "233V")


@dataclass
class Group:
    name: str
    sensors: list
    train: np.ndarray
    reference: np.ndarray
    values: np.ndarray
    metadata: pd.DataFrame
    counts: list


def limit_windows(array, limit):
    if not limit or len(array) <= limit:
        return array
    return array[np.linspace(0, len(array) - 1, limit, dtype=int)]


def load_z24(root, limit):
    manifest = load_manifest(root)
    for (setup, excitation), sensors in stable_sensor_map(manifest).items():
        cohort = manifest[manifest.setup.eq(setup) & manifest.excitation.eq(excitation)]

        def windows(row):
            raw = read_stable_vertical(Path(row.path), sensors)
            return record_windows(raw, 1024, 1024, "raw_temporal", 2, 128)

        train, reference = split_training_windows([windows(row) for row in cohort[cohort.split.eq("train")].itertuples()])
        normalizer = Normalizer.fit(train)
        evaluation = pd.concat([cohort.loc[cohort.split.eq(role)] for role in ("val", "test")])
        arrays = [limit_windows(windows(row), limit) for row in evaluation.itertuples()]
        yield Group(
            f"{setup:02d}-{excitation.upper()}", sensors,
            normalizer.transform(limit_windows(train, limit)),
            normalizer.transform(limit_windows(reference, limit)),
            normalizer.transform(np.concatenate(arrays)),
            evaluation[["record_id", "split", "label"]].rename(columns={"record_id": "sample_id"}).reset_index(drop=True),
            [len(array) for array in arrays],
        )


def load_hellbridge(path, out, fit_limit):
    split = out / "split"
    prepare_hellbridge(["--h5", str(path), "--output-dir", str(split), "--strict-paper-counts"])
    manifest = pd.read_csv(split / "sample_manifest.csv").sort_values(["record_id", "start"]).reset_index(drop=True)
    train, calibration, test = development_split(manifest)
    raw = load_windows(path, manifest, 128)
    values = Normalizer.fit(raw[train]).transform(raw)
    metadata = pd.concat([manifest.loc[calibration].assign(split="val"),
                          manifest.loc[test].assign(split="test")])
    return Group("P2_NM_Z", [f"AL{i:02d}/z" for i in range(1, 41)], limit_windows(values[train], fit_limit),
                 values[calibration], values[metadata.index],
                 metadata[["sample_id", "split", "label"]].reset_index(drop=True), [1] * len(metadata))


def dropped(values, indices):
    masked = values.copy()
    masked[:, indices, :] = 0.0
    return masked


def build_plan(groups, fit_scorer, out):
    rows, orders = [], {}
    for group in groups:
        scorer = fit_scorer("pca_spe", group.train, 1, 16, torch.device("cpu"))
        clean = np.log1p(scorer(group.reference))
        impacts = np.array([
            np.quantile(np.abs(np.log1p(scorer(dropped(group.reference, [i]))) - clean), .90)
            for i in range(len(group.sensors))
        ])
        order = sorted(range(len(group.sensors)), key=lambda i: (-impacts[i], group.sensors[i]))
        orders[group.name] = order
        for rank, i in enumerate(order, 1):
            rows.append(dict(group=group.name, rank=rank, sensor=group.sensors[i],
                             reference_q90_score_shift=impacts[i],
                             remaining_after_step=" ".join(group.sensors[j] for j in order[rank:])))
    pd.DataFrame(rows).to_csv(out / "sensor_removal_order.csv", index=False)
    return orders


def archive_functions(directory):
    if not (directory / "run_z24_missingness_baselines.py").is_file():
        raise FileNotFoundError(directory / "run_z24_missingness_baselines.py")
    sys.path.insert(0, str(directory))
    from run_z24_missingness_baselines import fit_scorer
    from run_z24_source_baselines import aggregate_record_scores, robust_score_reference, threshold_from_labeled_validation
    return fit_scorer, aggregate_record_scores, robust_score_reference, threshold_from_labeled_validation


def atomic_csv(table, path):
    tmp = path.with_name(path.name + ".tmp")
    table.to_csv(tmp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    tmp.replace(path)


def fit_model(fit_scorer, model, values, epochs, batch_size, device):
    if model != "scatterad":
        return fit_scorer(model, values, epochs, batch_size, device)
    from run_hellbridge_source_baselines import fit_scatterad, score_scatterad
    network, graph = fit_scatterad(values, epochs, device)

    @torch.no_grad()
    def score(samples):
        score_parts = []
        for batch in torch.from_numpy(samples).split(batch_size):
            size, channels, length = batch.shape
            x = batch.to(device).transpose(1, 2).reshape(size * length, channels)
            edges = (graph[None] + (torch.arange(size, device=device) * length)[:, None, None])
            edges = edges.transpose(0, 1).reshape(2, -1)
            encoder = network.model.online_encoder
            hidden, _ = encoder(x, edges)
            hidden = hidden.reshape(size, length, -1)
            delta = (hidden[:, 1:] - hidden[:, :-1]).square().mean(-1)
            temporal = torch.nn.functional.pad(delta, (0, 1))
            centers = network.model.center.unsqueeze(0).expand(size, -1, -1)
            distance = torch.cdist(hidden, centers).min(dim=-1).values
            score_parts.append((distance + temporal).mean(1).cpu().numpy())
        return np.concatenate(score_parts)

    check = values[:min(3, len(values))]
    np.testing.assert_allclose(score(check), score_scatterad((network, graph), check, device),
                               rtol=5e-5, atol=5e-6)
    return score


def group_scores(dataset, group, order, args, functions, model, seed, index, device, path):
    fit_scorer, aggregate, reference, _ = functions
    if path.is_file():
        return pd.read_csv(path, dtype={"key": str})
    seed_everything(seed + (1009 * index if dataset == "Z24" else 0))
    print(f"{dataset} {model} seed={seed} group={group.name}", flush=True)
    scorer = fit_model(fit_scorer, model, group.train, args.epochs, args.batch_size, device)
    norm = reference(scorer(group.reference)) if dataset == "Z24" else None
    n_val = int(group.metadata.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    test = group.metadata.iloc[n_val:].reset_index(drop=True)
    arrays = []
    steps = [("clean", "0", [])]
    steps += [("single", sensor, [i]) for i, sensor in enumerate(group.sensors)]
    steps += [("cumulative", str(k), order[:k]) for k in range(1, len(order))]
    if dataset == "Z24":
        for step in range(1, len(Z24_EXAMPLE) + 1):
            labels = Z24_EXAMPLE[:step]
            indices = [i for i, label in enumerate(group.sensors) if label in labels]
            if indices:
                steps.append(("example", "+".join(labels), indices))
    cache = {}
    for design, key, removed in steps:
        mask = tuple(sorted(removed))
        if mask not in cache:
            if mask:
                score = scorer(dropped(group.values[offset:], list(mask)))
            else:
                # TimesNet selects periods within a batch. Keep clean test batch
                # boundaries identical to all corrupted test passes.
                score = np.concatenate((scorer(group.values[:offset]), scorer(group.values[offset:])))
            score = np.asarray(score, dtype=np.float64)
            expected = len(group.values) - (offset if mask else 0)
            if score.shape != (expected,) or not np.isfinite(score).all():
                raise ValueError(f"Invalid {dataset}/{group.name}/{model}/{seed}/{design}/{key} scores")
            counts = group.counts if not mask else group.counts[n_val:]
            if dataset == "Z24":
                upper, two_sided = aggregate(score, counts, norm)
                cache[mask] = dict(upper=upper, lower=-np.asarray(upper), two_sided=two_sided)
            else:
                cache[mask] = dict(upper=score, lower=-score, two_sided=np.abs(score))
        metadata = test if mask else group.metadata
        arrays.append(metadata.assign(group=group.name, design=design, key=key, removed=len(mask), **cache[mask]))
    result = pd.concat(arrays, ignore_index=True)
    atomic_csv(result, path)
    return result


def metric_row(labels, values, threshold, fpr_limit):
    return dict(**publication_metrics(labels, values, threshold),
                recall_low_fpr=recall_at_fpr(labels, values, fpr_limit))


def evaluate(dataset, frame, max_steps, select_threshold):
    clean = frame.loc[frame.design.eq("clean")]
    val = clean.loc[clean.split.eq("val")]
    test = clean.loc[clean.split.eq("test")].set_index("sample_id")
    expected = ({0: 6, 1: 126}, 6) if dataset == "Z24" else ({0: 80, 1: 640}, 20)
    if test.label.value_counts().to_dict() != expected[0] or val.label.eq(0).sum() != expected[1]:
        raise RuntimeError(f"Unexpected {dataset} calibration or test counts")
    if not test.index.is_unique:
        raise ValueError("Duplicate clean test sample IDs")
    tail = "upper"
    if dataset == "Z24":
        candidates = []
        for direction in SCORE_COLUMNS:
            _, metric = select_threshold(val.label, val[direction])
            candidates.append(((metric["ba"], metric["auc"], metric["f1"], metric["specificity"]), direction))
        tail = max(candidates, key=lambda pair: pair[0])[1]
    threshold = threshold_from_healthy(val.loc[val.label.eq(0), tail].to_numpy(), .90)
    fpr_limit = 0 if dataset == "Z24" else .05
    y = test.label.to_numpy()
    clean_scores = test[tail]
    clean_metrics = metric_row(y, clean_scores.to_numpy(), threshold, fpr_limit)
    rows = [dict(design="clean", key="0", removed_percent=0., affected=len(test),
                 healthy_affected=int(test.label.eq(0).sum()), **clean_metrics)]
    standalone = frame.loc[frame.design.isin(["single", "example"])]
    for (design, sensor), records in standalone.groupby(["design", "key"]):
        scores = clean_scores.copy()
        records = records.set_index("sample_id")
        if not records.index.is_unique:
            raise ValueError(f"Duplicate single-drop sample IDs for {sensor}")
        scores.loc[records.index] = records[tail]
        rows.append(dict(design=design, key=sensor, removed_percent=np.nan, affected=len(records),
                         healthy_affected=int(records.label.eq(0).sum()),
                         **metric_row(y, scores.to_numpy(), threshold, fpr_limit)))
    cumulative = frame.loc[frame.design.eq("cumulative")]
    total_channels = sum((max_k + 1) * int(test.group.eq(group).sum()) for group, max_k in max_steps.items())
    for step in range(1, max(max_steps.values()) + 1):
        selected = cumulative.loc[cumulative.removed.eq(cumulative.group.map(max_steps).clip(upper=step))].set_index("sample_id")
        if not selected.index.is_unique or set(selected.index) != set(test.index):
            raise ValueError(f"Incomplete or duplicated cumulative scores at {dataset}/{step}")
        scores = selected.loc[test.index, tail]
        rows.append(dict(design="cumulative", key=str(step), removed_percent=100 * selected.removed.sum() / total_channels,
                         affected=len(test), healthy_affected=int(test.label.eq(0).sum()),
                         **metric_row(y, scores.to_numpy(), threshold, fpr_limit)))
    result = pd.DataFrame(rows).assign(score_tail=tail, threshold=threshold)
    for metric in METRICS:
        result[f"delta_{metric}"] = result[metric] - clean_metrics[metric]
    result["step_delta_auc"] = np.nan
    result["step_delta_recall_low_fpr"] = np.nan
    curve = result.design.isin(["clean", "cumulative"])
    for metric in ("auc", "recall_low_fpr"):
        result.loc[curve, f"step_delta_{metric}"] = result.loc[curve, metric].diff()
    return result


def plot_results(summary, dataset, out):
    figures, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    curve = summary.loc[summary.design.isin(["clean", "cumulative"])]
    for axis, metric, title in zip(axes, ("auc", "recall_low_fpr"), ("ROC-AUC", "Recall at low FPR"), strict=True):
        for model, rows in curve.groupby("model", sort=False):
            rows = rows.sort_values("removed_percent")
            x, y = rows.removed_percent.to_numpy(), rows[f"{metric}_mean"].to_numpy()
            sd = rows[f"{metric}_std"].fillna(0).to_numpy()
            line, = axis.plot(x, y, marker=".", markersize=3, linewidth=1.4, label=model)
            axis.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=.09)
        axis.set(xlabel="Sensor channels removed (%)", ylabel=title, title=f"{dataset} · {title}", ylim=(0, 1))
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=8, loc="best")
    for extension in ("png", "pdf"):
        figures.savefig(out / f"cumulative_sensor_loss.{extension}", dpi=190)
    plt.close(figures)

    single = summary.loc[summary.design.eq("single")]
    pivots = []
    for metric in ("auc", "recall_low_fpr"):
        pivot = single.pivot(index="key", columns="model", values=f"delta_{metric}_mean")
        pivots.append(pivot.loc[pivot.mean(axis=1).sort_values().index])

    def heatmaps(start, limit):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
        for axis, pivot, title in zip(axes, pivots, ("Δ ROC-AUC", "Δ recall at low FPR"), strict=True):
            part = pivot.iloc[start:start + limit]
            absmax = max(.01, float(np.abs(pivot.to_numpy()).max()))
            heat = axis.imshow(part, cmap="RdBu", aspect="auto", vmin=-absmax, vmax=absmax)
            axis.set_xticks(np.arange(len(part.columns)), part.columns, rotation=35, ha="right")
            axis.set_yticks(np.arange(len(part.index)), part.index)
            axis.set(title=f"{dataset} · {title}", ylabel="Sensor removed")
            fig.colorbar(heat, ax=axis, shrink=.75)
        return fig

    fig = heatmaps(0, 18)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"single_sensor_impact.{extension}", dpi=190)
    plt.close(fig)
    with PdfPages(out / "single_sensor_all.pdf") as pdf:
        for start in range(0, len(pivots[0]), 24):
            page = heatmaps(start, 24)
            pdf.savefig(page)
            plt.close(page)


def summarize(out, dataset, paths):
    table = pd.concat([pd.read_csv(path, dtype={"key": str}) for path in paths], ignore_index=True)
    atomic_csv(table, out / "metrics_by_seed.csv")
    names = [*METRICS, *[f"delta_{metric}" for metric in METRICS], "step_delta_auc", "step_delta_recall_low_fpr"]
    by = table.groupby(["model", "design", "key", "removed_percent", "affected", "healthy_affected"], dropna=False)
    summary = by[names].agg(["mean", "std"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary["runs"] = by.seed.nunique()
    summary = summary.reset_index()
    atomic_csv(summary, out / "summary_mean_sd.csv")
    plot_results(summary, dataset, out)
    return summary.assign(dataset=dataset)


def run_dataset(dataset, groups, args, out, device, functions):
    out.mkdir(parents=True, exist_ok=True)
    orders = build_plan(groups, functions[0], out)
    max_steps = {g.name: len(g.sensors) - 1 for g in groups}
    files = []
    for model in args.models:
        seeds = args.seeds[:1] if model == "pca_spe" else args.seeds
        for seed in seeds:
            run = out / "runs" / f"{model}__{seed}"
            run.mkdir(parents=True, exist_ok=True)
            metrics = run / "metrics.csv"
            files.append(metrics)
            if metrics.is_file():
                print(f"SKIP {dataset}/{model}/{seed}", flush=True)
                continue
            frame = pd.concat([
                group_scores(dataset, group, orders[group.name], args, functions, model, seed, index, device,
                             run / f"{group.name}.csv.gz")
                for index, group in enumerate(groups)
            ], ignore_index=True)
            result = evaluate(dataset, frame, max_steps, functions[3]).assign(model=model, seed=seed)
            atomic_csv(result, metrics)
            print(f"DONE {dataset}/{model}/{seed}", flush=True)
    return summarize(out, dataset, files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z24-root", type=Path, required=True)
    parser.add_argument("--hell-h5", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("Z24", "HellBridge"), default=["Z24", "HellBridge"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-window-limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.smoke_window_limit not in (0, *range(4, 65)):
        parser.error("Use positive epochs and batch size; smoke-window-limit is 0 or in [4, 64]")
    if any(len(seq) != len(set(seq)) for seq in (args.datasets, args.models, args.seeds)):
        parser.error("Datasets, models and seeds must be unique")
    if "scatterad" in args.models:
        from torch_geometric.nn import GATConv  # noqa: F401
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source_files = ("run_z24_missingness_baselines.py", "run_z24_source_baselines.py",
                    "run_hellbridge_source_baselines.py", "run_hellbridge_tsad_baselines.py")
    paths = [args.baseline_dir / name for name in source_files]
    paths += sorted(p for p in (args.baseline_dir / "third_party").rglob("*.py"))
    source_hash = hashlib.sha256()
    for path in paths:
        source_hash.update(path.relative_to(args.baseline_dir).as_posix().encode())
        source_hash.update(path.read_bytes().replace(b"\r\n", b"\n"))
    protocol = dict(version=1, z24_root=str(args.z24_root), hell_h5=str(args.hell_h5), epochs=args.epochs,
                    batch_size=args.batch_size, smoke_window_limit=args.smoke_window_limit,
                    baseline_source_sha256=source_hash.hexdigest(),
                    training="healthy clean only; checkpoint reused across masks",
                    sensor_order="common per setup/excitation, descending q90 absolute score shift "
                                 "of PCA-SPE on held-out healthy development windows",
                    imputation="zero after training z-score (healthy mean)",
                    z24="1024 raw points, downsampled to 512; median record score; native tail "
                        "selected on labeled clean validation; roving sensor layouts",
                    hellbridge="P2_NM_Z; detrended/resampled to 128 following baseline protocol; "
                               "native upper-tail window score; shared source recordings for healthy development and test",
                    threshold="Q90 healthy clean validation frozen across all conditions",
                    evaluation="single physical channel wherever present; groupwise cumulative drop capped at one survivor",
                    z24_example="441V alone then 441V plus 233V wherever present; retrospective illustration",
                    pca="one deterministic fit; neural models use all requested seeds",
                    scatterad="reference training and scoring, batched over disconnected graphs with equivalence check")
    protocol_file = out / "protocol.json"
    if protocol_file.is_file() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Output directory contains a different protocol. Use a fresh output path.")
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    functions = archive_functions(args.baseline_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for dataset in args.datasets:
        target = out / dataset
        groups = (list(load_z24(args.z24_root, args.smoke_window_limit)) if dataset == "Z24"
                  else [load_hellbridge(args.hell_h5, target, args.smoke_window_limit)])
        summaries.append(run_dataset(dataset, groups, args, target, device, functions))
        del groups
    atomic_csv(pd.concat(summaries, ignore_index=True), out / "summary_mean_sd.csv")
    print(f"TARGETED_BASELINES_PASS output={out}", flush=True)


if __name__ == "__main__":
    main()
