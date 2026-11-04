"""Equation-based reproduction of the Binomial Distribution Classifier on Z24.

The damage indicator is the official ScatterAD anomaly score applied to the
same log-PSD adapter as ``run_z24_strict_psd_scatterad.py``.  The two-level
alert/alarm decision follows Kamali et al., MSSP 207 (2024) 110938:

1. a sensor-window is an alert when its DI exceeds a healthy tail threshold;
2. a record is an alarm when the binomial survival probability of observing
   at least that many alerts under the healthy rate is below ``alarm_alpha``.

Only S01 is used to fit models, score references, and alert probabilities.
S08 and all damage states are evaluation-only.  This is an equation-based Z24
adapter, not author code (no public author repository was found).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

import run_z24_strict_psd_scatterad as base


def bdc_score(scores: np.ndarray, alert_threshold: float, healthy_alert_rate: float):
    n = int(len(scores)); k = int(np.sum(scores > alert_threshold))
    log_p = float(binom.logsf(k - 1, n, healthy_alert_rate)) if k else 0.0
    if not np.isfinite(log_p): log_p = math.log(np.finfo(float).tiny)
    return k, n, float(k / n), float(-log_p / math.log(10.0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_scatterad_bdc"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--raw-window", type=int, default=2048)
    p.add_argument("--raw-stride", type=int, default=2048)
    p.add_argument("--psd-bins", type=int, default=96)
    p.add_argument("--fmin", type=float, default=3.0)
    p.add_argument("--fmax", type=float, default=20.0)
    p.add_argument("--alert-quantile", type=float, default=0.95)
    p.add_argument("--alarm-alpha", type=float, default=0.01)
    p.add_argument("--state-alarm-rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not 2 <= args.cv_folds <= 9: raise ValueError("--cv-folds must be in [2, 9]")
    if not 0 < args.alarm_alpha < 1: raise ValueError("--alarm-alpha must be in (0, 1)")
    args.output.mkdir(parents=True, exist_ok=True); base.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = base.resolve_manifest(args.manifest, args.data_root)
    train = frame[frame.state == "01"].sort_values(["setup", "excitation"]).copy()
    test = frame[frame.state != "01"].sort_values(["state", "setup", "excitation"]).copy()
    train["fold"] = (train.setup.astype(int) - 1) % args.cv_folds
    feature_args = (args.fs, args.raw_window, args.raw_stride,
                    args.psd_bins, args.fmin, args.fmax)
    cache = {}
    for i, row in enumerate(train.itertuples(), 1):
        cache[row.record_id] = base.psd_vectors(row.path, *feature_args)
        print(f"features S01 {i}/18 {row.record_id} {cache[row.record_id].shape}", flush=True)

    fold_models, history, oof_rows = [], [], []
    started = time.time()
    for fold in range(args.cv_folds):
        fit = train[train.fold != fold]; held = train[train.fold == fold]
        model, center, scale, batcher, hist = base.train_model(
            [cache[x] for x in fit.record_id], args, device, f"cv{fold}")
        history.extend(hist)
        fit_vector_scores = []
        for rid in fit.record_id:
            fit_vector_scores.append(base.vector_scores(
                model, cache[rid], center, scale, batcher, args.batch_size))
        fit_vector_scores = np.concatenate(fit_vector_scores)
        alert_threshold = float(np.quantile(fit_vector_scores, args.alert_quantile))
        p0 = float(np.mean(fit_vector_scores > alert_threshold))
        p0 = min(max(p0, 1.0 / (len(fit_vector_scores) + 1)), 1 - 1e-9)
        fold_models.append((fold, model, center, scale, batcher, alert_threshold, p0))
        for row in held.itertuples():
            scores = base.vector_scores(model, cache[row.record_id], center, scale,
                                        batcher, args.batch_size)
            k, n, rate, score = bdc_score(scores, alert_threshold, p0)
            oof_rows.append({"record_id": row.record_id, "fold": fold,
                             "setup": int(row.setup), "excitation": row.excitation,
                             "alerts": k, "windows": n, "alert_rate": rate,
                             "bdc_score": score,
                             "alarm": score > -math.log10(args.alarm_alpha)})
    oof = pd.DataFrame(oof_rows).sort_values("record_id")
    if len(oof) != 18 or oof.record_id.nunique() != 18:
        raise RuntimeError("Every S01 record must receive exactly one OOF BDC score")
    oof.to_csv(args.output / "s01_crossfit_bdc.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    print(f"OOF S01 alarms={int(oof.alarm.sum())}/18", flush=True)

    rows = []
    for i, row in enumerate(test.itertuples(), 1):
        features = base.psd_vectors(row.path, *feature_args)
        fold_details = []
        for fold, model, center, scale, batcher, threshold, p0 in fold_models:
            scores = base.vector_scores(model, features, center, scale,
                                        batcher, args.batch_size)
            k, n, rate, score = bdc_score(scores, threshold, p0)
            fold_details.append((fold, k, n, rate, score))
        ensemble_score = float(np.median([x[4] for x in fold_details]))
        item = {"record_id": row.record_id, "state": row.state,
                "scenario": row.scenario, "excitation": row.excitation,
                "setup": int(row.setup), "label": int(row.label),
                "bdc_score": ensemble_score,
                "alarm": ensemble_score > -math.log10(args.alarm_alpha)}
        for fold, k, n, rate, score in fold_details:
            item.update({f"fold{fold}_alerts": k, f"fold{fold}_windows": n,
                         f"fold{fold}_alert_rate": rate, f"fold{fold}_bdc_score": score})
        rows.append(item)
        if i % 18 == 0 or i == len(test): print(f"evaluated={i}/{len(test)}", flush=True)
    records = pd.DataFrame(rows); records.to_csv(args.output / "record_scores.csv", index=False)
    decision_threshold = -math.log10(args.alarm_alpha)
    record_result = base.metrics(records.label, records.bdc_score, decision_threshold)
    states = records.groupby(["state", "scenario", "label"], as_index=False).agg(
        records=("record_id", "size"), score=("bdc_score", "median"),
        alarm_rate=("alarm", "mean"))
    states["alarm"] = states.alarm_rate >= args.state_alarm_rate
    state_pred = states.alarm.astype(int).to_numpy(); state_y = states.label.to_numpy()
    tn, fp, fn, tp = confusion_matrix(state_y, state_pred, labels=[0, 1]).ravel()
    state_result = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                    "accuracy": float((state_pred == state_y).mean()),
                    "balanced_accuracy": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
                    "threshold": args.state_alarm_rate}
    states.to_csv(args.output / "state_scores.csv", index=False)
    result = {"method": "ScatterAD_logPSD_plus_BDC_MSSP2024_equation_reproduction",
              "epochs_per_model": args.epochs, "cv_folds": args.cv_folds,
              "train_records": 18, "damage_used_for_fit": False,
              "s08_used_for_fit": False, "oof_s01_alarms": int(oof.alarm.sum()),
              "seconds": time.time() - started,
              **{f"record_{k}": v for k, v in record_result.items()},
              **{f"state_{k}": v for k, v in state_result.items()}}
    pd.DataFrame([result]).to_csv(args.output / "metrics.csv", index=False)
    metadata = {"paper": "Kamali et al., Reducing false alarms in SHM systems by exploiting time information via BDC, MSSP 207 (2024) 110938",
                "doi": "10.1016/j.ymssp.2023.110938",
                "implementation_scope": "equation-based Z24 adapter; no public author code found",
                "damage_indicator": "official ScatterAD score on energy-normalised log-PSD sensor windows",
                "model_ensemble": "median BDC score from setup-grouped cross-fit models",
                "strict_protocol": "S01-only fit; S08 and damage evaluation-only",
                "parameters": vars(args)}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(oof.bdc_score, bins=10, alpha=.8, label="S01 OOF")
    axes[0].axvline(decision_threshold, color="red", ls="--", label="alarm threshold")
    axes[0].set(title="Leakage-free BDC calibration", xlabel="-log10 binomial p", ylabel="records")
    axes[0].legend(); axes[0].grid(alpha=.2)
    fpr, tpr, _ = roc_curve(records.label, records.bdc_score)
    axes[1].plot(fpr, tpr, label=f"AUROC={record_result['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="Record ROC", xlabel="FPR", ylabel="TPR"); axes[1].legend()
    ConfusionMatrixDisplay(confusion_matrix(state_y, state_pred, labels=[0, 1]),
                           display_labels=["Healthy", "Damage"]).plot(
                               ax=axes[2], cmap="Greens", colorbar=False)
    axes[2].set_title("State confusion matrix")
    fig.tight_layout(); fig.savefig(args.output / "bdc_results.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("\nRESULT\n" + pd.DataFrame([result]).to_string(index=False), flush=True)
    print("\nSTATE SCORES\n" + states.to_string(index=False), flush=True)
    print(f"outputs={args.output}", flush=True)


if __name__ == "__main__":
    main()

