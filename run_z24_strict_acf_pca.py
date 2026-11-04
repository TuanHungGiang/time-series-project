"""Restore the strict zero-target Z24 ACF/PCA baseline.

Only S01 fits preprocessing, PCA and the OOF threshold. Whole MAT records are
cross-fit groups. Validation damage states select q95 versus persistence-3;
S08 and locked damage states are opened only for the final report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

import run_z24_strict_psd_scatterad as common

VAL_DAMAGE = {"04", "05", "06", "10", "14", "16", "17"}
TEST_DAMAGE = {"03", "07", "09", "11", "12", "13", "15"}


def vertical_raw(path: Path) -> np.ndarray:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(x).strip() for x in mat["labelshulp"].tolist()]
    use = [i for i, name in enumerate(labels)
           if name.endswith("V") and name[0].isdigit() and not name.startswith(("R", "DP"))]
    values = detrend(np.asarray(mat["data"][:, use], dtype=np.float32), axis=0, type="linear")
    if not 15 <= values.shape[1] <= 17 or not np.isfinite(values).all():
        raise ValueError(f"Invalid vertical data in {path}: {values.shape}")
    return values.astype(np.float32)


def acf_features(path: Path, window: int, stride: int) -> np.ndarray:
    values = vertical_raw(path)
    chunks = np.stack([values[s:s + window] for s in range(0, len(values) - window + 1, stride)])
    centered = chunks - chunks.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2 * window, axis=1)
    corr = np.fft.irfft(fft * np.conj(fft), axis=1)[:, :257, :].real
    acf = (corr / (corr[:, :1, :] + 1e-12))[:, 1:257:4, :]
    quantiles = np.quantile(acf, [.25, .50, .75], axis=2)
    return np.transpose(quantiles, (1, 2, 0)).reshape(len(chunks), -1).astype(np.float32)


class Detector:
    def __init__(self, seed=42):
        self.scaler = RobustScaler(quantile_range=(10, 90)); self.pca = None; self.seed = seed
    def fit(self, x):
        z = self.scaler.fit_transform(x)
        self.pca = PCA(n_components=max(2, min(12, len(z) - 1, z.shape[1])),
                       random_state=self.seed).fit(z)
    def score(self, x):
        z = self.scaler.transform(x); rec = self.pca.inverse_transform(self.pca.transform(z))
        return np.mean((z - rec) ** 2, axis=1)


def max_run(x):
    best = cur = 0
    for value in x: cur = cur + 1 if value else 0; best = max(best, cur)
    return best


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_strict_acf_pca"))
    p.add_argument("--window", type=int, default=2048); p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--folds", type=int, default=6); p.add_argument("--threshold-quantile", type=float, default=.99)
    p.add_argument("--seed", type=int, default=42); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frame = common.resolve_manifest(args.manifest, args.data_root)
    cache = {}
    def features(row):
        if row.record_id not in cache: cache[row.record_id] = acf_features(row.path, args.window, args.stride)
        return cache[row.record_id]
    s01 = frame[frame.state == "01"].sort_values(["excitation", "setup"])
    parts, groups, identities = [], [], []
    for gid, row in enumerate(s01.itertuples()):
        x = features(row); parts.append(x); groups.extend([gid] * len(x)); identities.append((gid, row, len(x)))
    x_s01, groups = np.concatenate(parts), np.asarray(groups)
    oof = np.full(len(x_s01), np.nan); fold_rows = []
    for fold, (fit, held) in enumerate(GroupKFold(args.folds).split(x_s01, groups=groups), 1):
        model = Detector(args.seed + fold); model.fit(x_s01[fit]); oof[held] = model.score(x_s01[held])
        fold_rows.append({"fold": fold, "fit_windows": len(fit), "held_windows": len(held),
                          "held_record_groups": "|".join(map(str, sorted(set(groups[held])))),
                          "record_overlap": False})
    if not np.isfinite(oof).all(): raise RuntimeError("Incomplete OOF scores")
    threshold = float(np.quantile(oof, args.threshold_quantile)); final = Detector(args.seed); final.fit(x_s01)
    healthy_rows = []
    for gid, row, count in identities:
        scores = oof[groups == gid]
        healthy_rows.append({"record_id": row.record_id, "state": "01", "scenario": row.scenario,
                             "excitation": row.excitation, "setup": int(row.setup), "label": 0,
                             "windows": count, "q95_score": float(np.quantile(scores, .95)),
                             "max_run": max_run(scores > threshold)})
    def score_states(states, split):
        output = []
        for row in frame[frame.state.isin(states)].sort_values(["state", "excitation", "setup"]).itertuples():
            scores = final.score(features(row)); output.append(
                {"record_id": row.record_id, "state": row.state, "scenario": row.scenario,
                 "excitation": row.excitation, "setup": int(row.setup), "label": int(row.state != "08"),
                 "split": split, "windows": len(scores), "q95_score": float(np.quantile(scores, .95)),
                 "max_run": max_run(scores > threshold)})
        return output
    validation = pd.DataFrame(healthy_rows + score_states(VAL_DAMAGE, "validation_damage"))
    test = pd.DataFrame(score_states({"08"}, "locked_test_healthy") +
                        score_states(TEST_DAMAGE, "locked_test_damage"))
    rules = {"q95": lambda f: (f.q95_score > threshold).astype(int),
             "persistence3": lambda f: (f.max_run >= 3).astype(int)}
    candidates = []
    for name, rule in rules.items():
        result = common.metrics(validation.label, validation.q95_score, threshold)
        # Replace threshold-derived predictions with the candidate decision.
        pred = rule(validation).to_numpy(); y = validation.label.to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        ba = .5 * (tn / max(tn + fp, 1) + tp / max(tp + fn, 1))
        candidates.append({"decision": name, "tn": int(tn), "fp": int(fp), "fn": int(fn),
                           "tp": int(tp), "balanced_accuracy": float(ba), "auroc": result["auroc"]})
    selection = pd.DataFrame(candidates).sort_values(["balanced_accuracy", "auroc"], ascending=False)
    selected = str(selection.iloc[0].decision); test["alarm"] = rules[selected](test)
    y, pred = test.label.to_numpy(), test.alarm.to_numpy(); score = test.q95_score.to_numpy()
    record_metrics = common.metrics(y, score, threshold)
    # common.metrics used q95 threshold internally; overwrite confusion-dependent values for persistence.
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    record_metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                           "accuracy": float((pred == y).mean()),
                           "balanced_accuracy": float(.5 * (tn / max(tn + fp, 1) + recall)),
                           "precision": float(precision), "recall": float(recall),
                           "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                           "specificity": float(tn / max(tn + fp, 1))})
    states = test.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), alarm_fraction=("alarm", "mean"),
        score_median=("q95_score", "median")); states["alarm"] = states.alarm_fraction >= .5
    pd.DataFrame(fold_rows).to_csv(args.output / "crossfit_folds.csv", index=False)
    selection.to_csv(args.output / "validation_decision_selection.csv", index=False)
    test.to_csv(args.output / "locked_test_record_scores.csv", index=False)
    states.to_csv(args.output / "locked_test_state_scores.csv", index=False)
    summary = {"method": "strict_ACF_RobustPCA_restored", "selected_decision": selected,
               "threshold": threshold, "train_records": 18, "damage_used_for_fit": False,
               "s08_used_for_fit": False, **{f"record_{k}": v for k, v in record_metrics.items()}}
    pd.DataFrame([summary]).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps(
        {"protocol": "strict zero-target S01-only", "representation": "ACF sensor-set Q25/median/Q75",
         "detector": "RobustScaler + PCA reconstruction", "validation_candidates": candidates,
         "parameters": vars(args)}, indent=2, default=str), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); ConfusionMatrixDisplay(
        confusion_matrix(y, pred, labels=[0, 1]), display_labels=["Healthy", "Damage"]).plot(
            ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Locked test: {selected}"); axes[1].bar(states.state, states.alarm_fraction,
        color=["#2ca02c" if x == 0 else "#d62728" for x in states.label]); axes[1].axhline(.5,ls="--",color="black")
    axes[1].set(title="Alarm fraction by state", xlabel="State", ylabel="Alarm fraction", ylim=(0,1.05))
    fig.tight_layout(); fig.savefig(args.output / "strict_acf_pca_results.png", dpi=190); plt.close(fig)
    print("\nVALIDATION DECISION SELECTION\n" + selection.to_string(index=False), flush=True)
    print("\nRESULT\n" + pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__": main()

