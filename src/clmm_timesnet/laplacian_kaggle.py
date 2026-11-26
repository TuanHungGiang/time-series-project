from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20270901
DATASETS = ("Z24", "HellBridge")
RATES = (0, 30, 50)
WORK = Path("/kaggle/working")
Z24 = Path("/kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split")
HELL = Path("/kaggle/input/datasets/giangtuanhung/hell-bridge-dataset/data_100Hz.h5")
PREVIOUS_WEIGHTS = WORK / f"clmm_old_gain_diagnostics_{SEED}/weights"
ROOT = WORK / f"clmm_laplacian_v1_{SEED}"
URL = "https://github.com/TuanHungGiang/time-series-project.git"
MODEL_COMMIT = "3e5b550af76502f6ddeb6e5c65615d958263ad3a"
BASELINE_COMMIT = "9f6e24b785e51462c5bfc93e2a06d8ca0df1e2df"
RANKING_HEADS = ("original_clmm", "adapter_reconstruction", "original_gain", "gain_variance", "gain_laplacian")


def checkout(path, branch, commit):
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, URL, str(path)], check=True)
    git = ["git", "-C", str(path)]
    head = lambda: subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head() != commit:
        have_commit = subprocess.run([*git, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True)
        if have_commit.returncode:
            subprocess.run([*git, "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run([*git, "checkout", "--detach", commit], check=True)
    if head() != commit:
        raise RuntimeError(f"Unexpected commit in {path}")
    print(f"{branch} {commit}", flush=True)


def healthy_graph(train):
    channels = train.shape[1]
    samples = train.transpose(0, 2, 1).reshape(-1, channels).astype(np.float64)
    samples -= samples.mean(axis=0)
    covariance = samples.T @ samples
    sd = np.sqrt(covariance.diagonal())
    denom = sd[:, None] * sd[None, :]
    graph = np.divide(np.abs(covariance), denom, out=np.zeros_like(covariance), where=denom > 0)
    np.fill_diagonal(graph, 0.0)
    return graph


def energies(gain, reference_gain, weights):
    residual = gain - np.median(reference_gain, axis=0)
    variance = residual.var(axis=1)
    src, dst = np.nonzero(np.triu(weights, k=1))
    if not len(src):
        raise ValueError("The observed sensors have no nonzero healthy correlation edges")
    edge_weight = weights[src, dst]
    difference = residual[:, src] - residual[:, dst]
    energy = np.average(np.square(difference), axis=1, weights=edge_weight)
    return variance, energy


def healthy_standardize(values, reference):
    # Fit a separate healthy score reference for each group, mask and statistic.
    unit = max(float(np.median(reference)), 1e-10)
    log_ref = np.log1p(reference / unit)
    center = np.median(log_ref)
    scale = max(1.4826 * np.median(np.abs(log_ref - center)), .25 * log_ref.std(), .05)
    return (np.log1p(values / unit) - center) / scale


@torch.inference_mode()
def read_gain(head, source, data, kept):
    device = head.detector.device
    mask = torch.as_tensor(kept, device=device)
    data = torch.as_tensor(data, device=device)
    adapted_scores, gains = [], []
    for batch in data.split(head.detector.batch_size):
        masked = torch.where(mask.view(1, -1, 1), batch, 0.0)
        prediction = source.model_forward("timesnet", head.detector.model, masked)
        off = (masked[:, mask, :] - prediction[:, mask, :]).square()
        on, _ = head.residual(batch, mask, enabled=True)
        adapted_scores.append(on.mean((1, 2)).cpu().numpy().astype(np.float64))
        off_mean = off.mean(-1).cpu().numpy().astype(np.float64)
        on_mean = on.mean(-1).cpu().numpy().astype(np.float64)
        gains.append(np.log1p(off_mean) - np.log1p(on_mean))
    return np.concatenate(adapted_scores), np.concatenate(gains)


def run_group(dataset, group, order, index, steps, args, functions, device, out):
    from clmm_timesnet.old_gain_diagnostics import aggregate_windows
    from clmm_timesnet.post_block_head import DEVIATION_CAP, gain_deviation
    from clmm_timesnet.targeted_baselines import atomic_csv
    from clmm_timesnet.targeted_gain_head import GAIN_WEIGHT, fit_or_restore_head, score_columns

    table = out / "groups" / f"{group.name}.csv.gz"
    if table.is_file():
        print(f"{dataset}/{group.name} scores restored", flush=True)
        return pd.read_csv(table)

    _, aggregate, _, _, source = functions
    weights = ROOT / "weights" / dataset / str(args.seed)
    weights.mkdir(parents=True, exist_ok=True)
    scorer, head, norm = fit_or_restore_head(dataset, group, index, args.seed, args, functions, device, weights)
    graph = healthy_graph(group.train)
    n_val = int(group.metadata.split.eq("val").sum())
    offset = sum(group.counts[:n_val])
    counts = group.counts[n_val:]
    test_values = group.values[offset:]
    columns = group.metadata.drop(columns="path", errors="ignore")
    clean_val = score_columns(dataset, scorer(group.values[:offset]), group.counts[:n_val], norm, aggregate)
    results = [columns.iloc[:n_val].assign(group=group.name, requested_loss=0, actual_loss=0., **clean_val)]

    for rate in RATES:
        step, actual = steps[rate]
        kept = np.ones(len(group.sensors), dtype=bool)
        kept[order[:min(step, len(kept) - 1)]] = False
        missing = int((~kept).sum())
        names = " ".join(np.asarray(group.sensors)[~kept])
        weights_kept = graph[np.ix_(kept, kept)]
        edge_count = np.count_nonzero(np.triu(weights_kept, 1))
        if not missing:
            adapted = scorer(test_values)
            old_gain = var = lap = raw_var = raw_lap = np.zeros(len(test_values))
        else:
            adapted, gain = read_gain(head, source, test_values, kept)
            _, healthy_gain = read_gain(head, source, group.reference, kept)
            center = head.centers[missing][kept] if missing in head.centers else np.median(healthy_gain, axis=0)
            old_gain = gain_deviation(gain, center, head.scale[kept])
            raw_var, raw_lap = energies(gain, healthy_gain, weights_kept)
            ref_var, ref_lap = energies(healthy_gain, healthy_gain, weights_kept)
            var = healthy_standardize(raw_var, ref_var)
            lap = healthy_standardize(raw_lap, ref_lap)

        base = score_columns(dataset, adapted, counts, norm, aggregate)
        old = score_columns(dataset, adapted, counts, norm, aggregate, GAIN_WEIGHT * old_gain) if missing else base
        window_stats = dict(original_gain=old_gain, gain_variance=var, gain_laplacian=lap,
                            variance_raw=raw_var, laplacian_raw=raw_lap)
        output = {key: aggregate_windows(value, counts) for key, value in window_stats.items()}
        output["cap_fraction"] = aggregate_windows(np.isclose(old_gain, DEVIATION_CAP), counts, np.mean)
        results.append(columns.iloc[n_val:].assign(
            group=group.name, requested_loss=rate, actual_loss=actual, removed_sensors=names,
            kept_sensors=int(kept.sum()), graph_edges=int(edge_count), **old,
            **{f"adapter_{key}": values for key, values in base.items()}, **output,
        ))
        print(f"  loss={rate}% actual={actual:.2f}% | {kept.sum()} sensors | {edge_count} edges | "
              f"gain cap={np.isclose(old_gain, DEVIATION_CAP).mean():.1%}", flush=True)

    scores = pd.concat(results, ignore_index=True)
    atomic_csv(scores, table)
    return scores


def summarize(dataset, scores, out, select_threshold):
    from clmm_timesnet.metrics import publication_metrics, recall_at_fpr
    from clmm_timesnet.post_block_head import DEVIATION_CAP
    from clmm_timesnet.targeted_baselines import atomic_csv, clean_calibration

    val = scores.loc[scores.split.eq("val")]
    test = scores.loc[scores.split.eq("test")].copy()
    tail, threshold = clean_calibration(dataset, val, select_threshold)
    test["original_clmm"] = test[tail]
    test["adapter_reconstruction"] = test[f"adapter_{tail}"]
    expected = {0: 6, 1: 126} if dataset == "Z24" else {0: 80, 1: 640}
    calibration_healthy = 6 if dataset == "Z24" else 20
    if val.label.eq(0).sum() != calibration_healthy or set(test.requested_loss) != set(RATES):
        raise ValueError(f"Incorrect calibration cohort or mask coverage: {dataset}")
    clean_ids = test.loc[test.requested_loss.eq(0), ["sample_id", "label"]].set_index("sample_id").label.sort_index()
    metric_rows = []
    for rate, part in test.groupby("requested_loss"):
        if part.sample_id.duplicated().any() or part.label.value_counts().to_dict() != expected:
            raise ValueError(f"Incorrect test cohort: {dataset} loss={rate}")
        if not part.set_index("sample_id").label.sort_index().equals(clean_ids):
            raise ValueError(f"Test IDs or labels changed: {dataset} loss={rate}")
        labels = part.label.to_numpy()
        for method in RANKING_HEADS:
            values = part[method].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Invalid score: {dataset} {rate} {method}")
            valid = method in RANKING_HEADS[:2] or rate != 0
            ranking = dict(auc=np.nan, ap=np.nan, recall_at_fpr_00=np.nan,
                           recall_at_fpr_05=np.nan, recall_at_fpr_10=np.nan)
            if valid:
                ranking = dict(auc=roc_auc_score(labels, values), ap=average_precision_score(labels, values),
                               recall_at_fpr_00=recall_at_fpr(labels, values, .00),
                               recall_at_fpr_05=recall_at_fpr(labels, values, .05),
                               recall_at_fpr_10=recall_at_fpr(labels, values, .10))
            row = dict(dataset=dataset, seed=SEED, requested_loss=rate, actual_loss=part.actual_loss.iloc[0],
                       method=method, healthy=expected[0], damage=expected[1],
                       status="scored" if valid else "adapter identity on clean", **ranking)
            if method in RANKING_HEADS[:2]:
                metrics = publication_metrics(labels, values, threshold)
                row.update({f"clean_frozen_{key}": metrics[key] for key in ("ba", "recall", "fpr", "precision", "f1")})
            metric_rows.append(row)

    report = pd.DataFrame(metric_rows)
    baseline = report.loc[report.method.eq("original_clmm")].set_index("requested_loss")
    for key in ("auc", "recall_at_fpr_05", "recall_at_fpr_10"):
        report[f"delta_{key}_vs_original_clmm"] = report[key] - report.requested_loss.map(baseline[key])
    diagnostics = (test.groupby(["requested_loss", "label"], as_index=False)
                   .agg(samples=("sample_id", "size"), gain_cap_window_fraction=("cap_fraction", "mean"),
                        gain_cap_sample_fraction=("original_gain", lambda s: np.isclose(s, DEVIATION_CAP).mean()),
                        gain_median=("original_gain", "median"), variance_median=("gain_variance", "median"),
                        laplacian_median=("gain_laplacian", "median")))
    masks = test[["group", "requested_loss", "actual_loss", "removed_sensors", "kept_sensors", "graph_edges"]].drop_duplicates()
    for name, table in (("head_comparison", report), ("score_diagnostics", diagnostics),
                        ("test_scores", test), ("calibration_scores", val), ("mask_manifest", masks)):
        atomic_csv(table, out / f"{name}.csv")
    return report, diagnostics.assign(dataset=dataset)


def run():
    for dataset, path in (("Z24", Z24 / "metadata/records_manifest.csv"), ("HellBridge", HELL)):
        if dataset in DATASETS and not path.is_file():
            raise FileNotFoundError(f"Attach the {dataset} dataset: {path}")
    ROOT.mkdir(parents=True, exist_ok=True)
    repo, archive = ROOT / "source_main", ROOT / "source_baselines"
    checkout(repo, "main", MODEL_COMMIT)
    checkout(archive, "codex/mdr-shm-kaggle", BASELINE_COMMIT)
    sys.path.insert(0, str(repo / "src"))
    os.chdir(WORK)
    torch.set_num_threads(2)

    from clmm_timesnet.old_gain_diagnostics import loss_steps
    from clmm_timesnet.targeted_baselines import archive_functions, atomic_csv, load_hellbridge, load_z24
    from clmm_timesnet.targeted_gain_head import GAIN_WEIGHT, STRENGTH, sensor_orders, targeted_protocol

    args = SimpleNamespace(seed=SEED, epochs=30, adapter_epochs=8, batch_size=16, smoke_window_limit=0,
                           strength=STRENGTH, gain_weight=GAIN_WEIGHT, center_gain=False,
                           baseline_runs=None, previous_clmm_weights=PREVIOUS_WEIGHTS)
    args.protocol = targeted_protocol(args)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    configuration = dict(seed=SEED, rates=list(RATES), datasets=list(DATASETS), model_commit=MODEL_COMMIT,
                         baseline_commit=BASELINE_COMMIT, z24=str(Z24), hellbridge=str(HELL),
                         previous_weights=str(PREVIOUS_WEIGHTS), adapter=args.protocol,
                         masks="cumulative healthy PCA sensitivity from old CLMM diagnostic",
                         graph="absolute Pearson correlations from healthy training windows",
                         head_reference="exact-mask held-back healthy windows, log1p with robust normalization")
    config_path = out / "protocol.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
        raise ValueError(f"Change the output directory for a different run: {out}")
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | seed={SEED} | prior checkpoint directory={PREVIOUS_WEIGHTS}", flush=True)
    functions = archive_functions(archive / "research/mdr_shm_kaggle")
    reports, diagnostics = [], []
    for dataset in DATASETS:
        dataset_out = out / dataset
        (dataset_out / "groups").mkdir(parents=True, exist_ok=True)
        groups = list(load_z24(Z24, 0)) if dataset == "Z24" else [load_hellbridge(HELL, dataset_out, 0)]
        orders = sensor_orders(dataset, groups, args, dataset_out, functions[0])
        steps = loss_steps(groups)
        records = pd.concat([run_group(dataset, g, orders[g.name], i, steps, args, functions, device, dataset_out)
                             for i, g in enumerate(groups)], ignore_index=True)
        report, diagnostic = summarize(dataset, records, dataset_out, functions[3])
        reports.append(report)
        diagnostics.append(diagnostic)
        cols = ["requested_loss", "actual_loss", "method", "auc", "delta_auc_vs_original_clmm",
                "recall_at_fpr_05", "recall_at_fpr_10", "clean_frozen_fpr"]
        print(f"\n=== {dataset} ===", flush=True)
        display(report[cols].round(4))
        del groups, records

    atomic_csv(pd.concat(reports, ignore_index=True), out / "head_comparison.csv")
    atomic_csv(pd.concat(diagnostics, ignore_index=True), out / "score_diagnostics.csv")
    bundle = shutil.make_archive(str(ROOT / "clmm_laplacian_reports"), "zip", root_dir=out)
    display(FileLink(os.path.relpath(bundle)))
    print(f"CLMM_LAPLACIAN_PASS reports={out} weights={ROOT / 'weights'}", flush=True)


if __name__ == "__main__":
    run()
