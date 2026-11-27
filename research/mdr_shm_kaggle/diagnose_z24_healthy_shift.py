from __future__ import annotations

"""Healthy-only audit for Z24 preprocessing and conditioning choices.

Candidate representations and conditioning schemes are ranked exclusively by
leave-one-healthy-record-out stability on the development cohort (train plus
healthy validation). Locked-test healthy records are written to a separate
audit table and never influence the recommendation.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_z24_source_baselines import (  # noqa: E402
    load_manifest,
    parse_labels,
    read_stable_vertical,
)


SCHEMES = ("global", "excitation", "setup", "setup_excitation")
REPRESENTATIONS = ("amplitude", "spectral_shape", "combined")


def quantiles(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float64)


def record_features(signals: np.ndarray, fs: float, bins: int) -> dict[str, np.ndarray]:
    values = detrend(np.asarray(signals, dtype=np.float64), axis=1, type="linear")
    eps = np.finfo(np.float64).tiny

    rms = np.sqrt(np.mean(values**2, axis=1))
    derivative_rms = np.sqrt(np.mean(np.diff(values, axis=1) ** 2, axis=1))
    amplitude = np.concatenate([
        quantiles(np.log10(np.maximum(rms, eps))),
        quantiles(np.log10(np.maximum(derivative_rms, eps))),
    ])

    frequency, psd = welch(
        values,
        fs=fs,
        window="hann",
        nperseg=min(2048, values.shape[1]),
        noverlap=min(1024, values.shape[1] // 2),
        axis=1,
        detrend=False,
        scaling="density",
    )
    keep = (frequency >= 0.5) & (frequency <= 30.0)
    frequency, psd = frequency[keep], psd[:, keep]
    area = np.trapezoid(psd, frequency, axis=1)
    psd = psd / np.maximum(area[:, None], eps)
    grid = np.linspace(0.5, 30.0, bins)
    shape = np.stack([
        np.interp(grid, frequency, np.log10(np.maximum(sensor, eps)))
        for sensor in psd
    ])
    # Remove each sensor's broadband level. Only spectral shape remains.
    shape -= shape.mean(axis=1, keepdims=True)
    spectral_shape = np.concatenate([
        np.median(shape, axis=0),
        np.quantile(shape, 0.75, axis=0) - np.quantile(shape, 0.25, axis=0),
    ])

    return {
        "amplitude": amplitude,
        "spectral_shape": spectral_shape,
        "combined": np.concatenate([amplitude, spectral_shape]),
    }


def cohort_key(row: pd.Series, scheme: str) -> str:
    if scheme == "global":
        return "global"
    if scheme == "excitation":
        return str(row.excitation)
    if scheme == "setup":
        return f"setup{int(row.setup):02d}"
    if scheme == "setup_excitation":
        return f"setup{int(row.setup):02d}_{row.excitation}"
    raise KeyError(scheme)


def robust_distance(query: np.ndarray, reference: np.ndarray) -> float:
    center = np.median(reference, axis=0)
    mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
    standard = np.std(reference, axis=0, ddof=0)
    scale = np.maximum.reduce([mad, 0.25 * standard, np.full_like(mad, 1e-6)])
    return float(np.sqrt(np.mean(((query - center) / scale) ** 2)))


def score_queries(
    queries: pd.DataFrame,
    reference: pd.DataFrame,
    features: dict[str, dict[str, np.ndarray]],
    representation: str,
    scheme: str,
    leave_one_out: bool,
) -> pd.DataFrame:
    rows = []
    for _, query in queries.iterrows():
        key = cohort_key(query, scheme)
        peers = reference[
            reference.apply(lambda row: cohort_key(row, scheme) == key, axis=1)
        ]
        if leave_one_out:
            peers = peers[peers.record_id != query.record_id]
        peer_count = len(peers)
        distance = np.nan
        if peer_count >= 3:
            matrix = np.stack([
                features[str(record_id)][representation]
                for record_id in peers.record_id
            ])
            distance = robust_distance(
                features[str(query.record_id)][representation], matrix
            )
        rows.append({
            "record_id": query.record_id,
            "source_split": query.split,
            "setup": int(query.setup),
            "excitation": query.excitation,
            "representation": representation,
            "conditioning": scheme,
            "cohort": key,
            "fit_peers": peer_count,
            "healthy_distance": distance,
        })
    return pd.DataFrame(rows)


def summarize_development(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation, conditioning), group in frame.groupby(
        ["representation", "conditioning"], sort=False
    ):
        valid = group.healthy_distance.notna()
        distances = group.loc[valid, "healthy_distance"]
        rows.append({
            "representation": representation,
            "conditioning": conditioning,
            "development_records": len(group),
            "valid_records": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "minimum_fit_peers": int(group.fit_peers.min()),
            "median_fit_peers": float(group.fit_peers.median()),
            "median_healthy_distance": float(distances.median()) if len(distances) else np.nan,
            "q90_healthy_distance": float(distances.quantile(0.9)) if len(distances) else np.nan,
            "maximum_healthy_distance": float(distances.max()) if len(distances) else np.nan,
            "eligible": bool(valid.all() and group.fit_peers.min() >= 5),
        })
    summary = pd.DataFrame(rows)
    summary["development_rank"] = np.nan
    eligible = summary.eligible
    if eligible.any():
        # Primary criterion is tail stability; median distance breaks ties.
        ordered = summary.loc[eligible].sort_values(
            ["q90_healthy_distance", "median_healthy_distance"],
            ascending=True,
        )
        summary.loc[ordered.index, "development_rank"] = np.arange(1, len(ordered) + 1)
    return summary.sort_values(
        ["eligible", "development_rank", "q90_healthy_distance"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def make_figures(development: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    eligible = summary[summary.eligible].copy()
    if len(eligible):
        labels = eligible.representation + "\n" + eligible.conditioning
        figure, axis = plt.subplots(figsize=(12, 6))
        x = np.arange(len(eligible))
        axis.bar(x - 0.18, eligible.median_healthy_distance, 0.36, label="Median")
        axis.bar(x + 0.18, eligible.q90_healthy_distance, 0.36, label="90th percentile")
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel("Healthy-only robust distance")
        axis.set_title("Z24 development healthy stability (lower is better)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "development_healthy_stability.png", dpi=220, bbox_inches="tight")
        plt.close(figure)

    peer_table = development.groupby(
        ["representation", "conditioning"], as_index=False
    ).fit_peers.min()
    pivot = peer_table.pivot(index="representation", columns="conditioning", values="fit_peers")
    figure, axis = plt.subplots(figsize=(9, 4))
    image = axis.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.iloc[row, column]:.0f}", ha="center", va="center", color="white")
    axis.set_title("Minimum healthy development peers per candidate")
    figure.colorbar(image, ax=axis, label="Peers")
    figure.tight_layout()
    figure.savefig(output / "conditioning_peer_counts.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--spectral-bins", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.input_root)
    healthy = manifest[manifest.label == 0].copy().reset_index(drop=True)
    development = healthy[healthy.split.isin(["train", "val"])].copy()
    locked_test = healthy[healthy.split == "test"].copy()
    if len(development) != 30 or len(locked_test) != 6:
        raise RuntimeError(
            f"Expected 30 development and 6 locked-test healthy records; "
            f"got {len(development)} and {len(locked_test)}"
        )

    inventory = []
    feature_map: dict[str, dict[str, np.ndarray]] = {}
    for index, row in healthy.iterrows():
        labels = parse_labels(row.vertical_labels)
        signals = read_stable_vertical(Path(row.path), labels)
        feature_map[str(row.record_id)] = record_features(
            signals, args.fs, args.spectral_bins
        )
        inventory.append({
            "record_id": row.record_id,
            "split": row.split,
            "setup": int(row.setup),
            "excitation": row.excitation,
            "state": int(row.state),
            "sensors": len(labels),
            "samples": signals.shape[1],
        })
        print(
            f"healthy={index + 1}/{len(healthy)} record={row.record_id} "
            f"split={row.split} setup={row.setup} excitation={row.excitation}",
            flush=True,
        )
    pd.DataFrame(inventory).to_csv(args.output_dir / "healthy_record_inventory.csv", index=False)

    development_scores, test_scores = [], []
    for representation in REPRESENTATIONS:
        for scheme in SCHEMES:
            development_scores.append(score_queries(
                development, development, feature_map, representation, scheme, True
            ))
            test_scores.append(score_queries(
                locked_test, development, feature_map, representation, scheme, False
            ))
    development_scores = pd.concat(development_scores, ignore_index=True)
    test_scores = pd.concat(test_scores, ignore_index=True)
    summary = summarize_development(development_scores)

    development_scores.to_csv(
        args.output_dir / "development_healthy_cv.csv", index=False
    )
    summary.to_csv(args.output_dir / "development_candidate_summary.csv", index=False)
    test_scores.to_csv(args.output_dir / "locked_test_healthy_audit.csv", index=False)

    ranked = summary[summary.eligible].sort_values("development_rank")
    recommendation = {
        "selection_data": "train plus healthy validation only",
        "damage_labels_used": False,
        "locked_test_used_for_ranking": False,
        "recommended": (
            ranked.iloc[0][["representation", "conditioning"]].to_dict()
            if len(ranked) else None
        ),
        "eligibility_rule": "all development records scoreable and minimum five fit peers",
        "ranking_rule": "lowest development q90 healthy distance, then median distance",
    }
    (args.output_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    make_figures(development_scores, summary, args.output_dir)

    print("\n=== DEVELOPMENT-ONLY CANDIDATE SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== RECOMMENDATION (NO DAMAGE, NO LOCKED TEST SELECTION) ===")
    print(json.dumps(recommendation, indent=2))
    print(f"Z24_HEALTHY_SHIFT_AUDIT_PASS output={args.output_dir}")


if __name__ == "__main__":
    main()

