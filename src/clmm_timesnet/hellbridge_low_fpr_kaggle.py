"""Limit HellBridge spectral reranking while preserving the original CLMM score."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .hellbridge_keep_gain_kaggle import validate_source
from .hellbridge_spectral_kaggle import INPUT, OUTPUT, SPECTRAL_WEIGHT, calibration_cutoff, write_calibration_reports


DESTINATION = OUTPUT.with_name("clmm_hellbridge_low_fpr_1024_seed20270905")
LEGACY = "clmm_gain_old"
UNBOUNDED = "clmm_gain_spectral"
SIGNED = "clmm_spectral_bounded_signed"
POSITIVE = "clmm_spectral_bounded_positive"
COMPARISONS = (UNBOUNDED, SIGNED, POSITIVE)
METHODS = ("timesnet_native", "adapter_mse", LEGACY, *COMPARISONS)
ALPHA = .05


def spectral_adjustments(legacy, bonus, missing):
    spectral = np.where(missing, bonus, 0.)
    standardized = spectral / SPECTRAL_WEIGHT
    return {
        UNBOUNDED: legacy + spectral,
        SIGNED: legacy + SPECTRAL_WEIGHT * np.tanh(standardized),
        POSITIVE: legacy + SPECTRAL_WEIGHT * np.tanh(np.maximum(standardized, 0.)),
    }


def roc_operating_threshold(labels, values):
    # For diagnostic recall on the test ROC only. Alerts use healthy validation.
    fpr, recall, thresholds = roc_curve(labels, values, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= ALPHA + 1e-12)
    return thresholds[eligible[np.argmax(recall[eligible])]]


def tail_audit(scores):
    changes, cohorts, windows, ranking = [], [], [], []
    for (condition, replicate), frame in scores.groupby(["condition", "replicate"]):
        calibration = frame.loc[frame.split.eq("val")]
        test = frame.loc[frame.split.eq("test")].copy()
        labels = test.label.to_numpy()
        if (len(calibration) != 20 or not calibration.label.eq(0).all()
                or test.label.value_counts().to_dict() != {0: 80, 1: 640}):
            raise ValueError(f"Unexpected threshold or test cohort: {condition}/{replicate}")
        healthy = test.label.eq(0).to_numpy()
        methods = (LEGACY, *COMPARISONS)
        validated = {m: calibration_cutoff(calibration[m].to_numpy(), ALPHA) for m in methods}
        roc_thresholds = {m: roc_operating_threshold(labels, test[m].to_numpy()) for m in METHODS}
        for method, threshold in roc_thresholds.items():
            values = test[method].to_numpy()
            alert = values >= threshold
            ranking.append(dict(condition=condition, replicate=replicate, method=method,
                                partial_auc_05=roc_auc_score(labels, values, max_fpr=ALPHA),
                                recall_at_fpr_05=alert[~healthy].mean(), fpr_at_roc_point=alert[healthy].mean()))
        empirical = {m: roc_thresholds[m] for m in methods}
        for policy, thresholds in (("mask_calibration_05", validated), ("test_roc_05_diagnostic", empirical)):
            before = test[LEGACY].ge(thresholds[LEGACY]).to_numpy()
            for method in COMPARISONS:
                after = test[method].ge(thresholds[method]).to_numpy()
                entered, exited = after & ~before, before & ~after
                meta = dict(condition=condition, replicate=replicate, method=method, policy=policy)
                changes.append(dict(
                    **meta, damage_gained=int(np.count_nonzero(entered & ~healthy)),
                    damage_lost=int(np.count_nonzero(exited & ~healthy)),
                    false_alarms_gained=int(np.count_nonzero(entered & healthy)),
                    false_alarms_removed=int(np.count_nonzero(exited & healthy)),
                    legacy_fpr=before[healthy].mean(), candidate_fpr=after[healthy].mean(),
                ))
                by_state = test.assign(legacy_alert=before, candidate_alert=after).groupby("state")
                for state, part in by_state:
                    cohorts.append(dict(**meta, state=state, windows=len(part),
                                        legacy_rate=part.legacy_alert.mean(), candidate_rate=part.candidate_alert.mean()))
                movers = test.loc[(entered | exited) & healthy, ["sample_id", "record_id", LEGACY, method]].copy()
                movers = movers.rename(columns={method: "candidate_score"})
                windows.append(movers.assign(change=np.where(entered[(entered | exited) & healthy], "entered", "exited"),
                                             **meta))
    return pd.DataFrame(changes), pd.DataFrame(cohorts), pd.concat(windows, ignore_index=True), pd.DataFrame(ranking)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hell-h5", type=Path, default=INPUT)
    parser.add_argument("--source-dir", type=Path, default=OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    cli = parser.parse_args()
    source, out = cli.source_dir, cli.output_dir
    if source.resolve() == out.resolve():
        raise ValueError("Use separate source and output directories")
    required = ("protocol.json", "scores.csv.gz", "masks.csv", "fit_groups.json")
    if any(not (source / name).is_file() for name in required):
        subprocess.run([sys.executable, "-u", "-m", "clmm_timesnet.hellbridge_spectral_kaggle",
                        "--hell-h5", str(cli.hell_h5), "--output-dir", str(source)], check=True)
    protocol, scores = validate_source(source)
    print(f"Scoring from saved reconstruction and spectral outputs: {source}", flush=True)
    bonus = scores.clmm_spectral.to_numpy() - scores.adapter_mse.to_numpy()
    improvements = spectral_adjustments(scores[LEGACY].to_numpy(), bonus, scores.condition.ne("clean").to_numpy())
    scores = scores.assign(**improvements)
    out.mkdir(parents=True, exist_ok=True)
    protocol.update(score_source=str(source), methods=list(METHODS), spectral_cap=SPECTRAL_WEIGHT,
                    bounded_forms=("signed tanh", "positive-only tanh"), score_selection="fixed before test evaluation",
                    diagnostic_alpha=ALPHA, clean_gate="zero spectral adjustment for complete input")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name in ("masks.csv", "fit_groups.json"):
        shutil.copy2(source / name, out / name)
    saved = out / "scores.csv.gz"
    scores.to_csv(saved, index=False)
    write_calibration_reports(saved, out, METHODS, candidate=POSITIVE)
    changes, states, healthy, ranking = tail_audit(scores)
    changes.to_csv(out / "low_fpr_alarm_changes.csv", index=False)
    states.to_csv(out / "low_fpr_state_recall.csv", index=False)
    healthy.to_csv(out / "healthy_tail_movers.csv", index=False)
    ranking.to_csv(out / "low_fpr_ranking.csv", index=False)
    bundle = shutil.make_archive(str(out) + "_report", "zip", root_dir=out)
    print(f"HELLBRIDGE_LOW_FPR_PASS reports={out} zip={bundle}", flush=True)


if __name__ == "__main__":
    main()
