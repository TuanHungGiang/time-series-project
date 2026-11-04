"""Audit and visualize healthy/damage labels in the Z24 all-record split.

Z24 progressive-damage labels are *record-level structural states*.  There is
no annotated damage-onset timestamp inside a MAT file: every sample in one
record inherits the state in which that complete measurement was acquired.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Local development dependencies live beside the repository; Kaggle uses its
# normal site-packages.  Adding a missing directory is harmless there.
sys.path[:0] = [str(ROOT.parent / ".packages"), str(ROOT / ".packages")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch


SCENARIOS = {
    "01": "Initial undamaged reference",
    "02": "Pier-settlement system installation",
    "03": "Pier lowering 20 mm",
    "04": "Pier lowering 40 mm",
    "05": "Pier lowering 80 mm",
    "06": "Pier lowering 95 mm",
    "07": "Foundation tilt",
    "08": "Post-lifting reference",
    "09": "Concrete spalling 12 m²",
    "10": "Concrete spalling 24 m²",
    "11": "Abutment landslide",
    "12": "Concrete hinge failure",
    "13": "Two anchor heads failure",
    "14": "Four anchor heads failure",
    "15": "Two tendons ruptured",
    "16": "Four tendons ruptured",
    "17": "Six tendons ruptured",
}

CATEGORIES = [
    "train healthy", "val healthy", "test healthy",
    "train post-lifting reference", "val post-lifting reference", "test post-lifting reference",
    "val damage", "test damage", "excluded",
]
COLORS = ["#16803c", "#67bd63", "#b5df9d", "#087f8c", "#2ba7b2", "#83d1d5", "#f28e8e", "#c92c2c", "#9a9a9a"]
SHORT = ["TR-H", "VA-H", "TE-H", "TR-R8", "VA-R8", "TE-R8", "VA-D", "TE-D", "EX"]


def package_root(value: Path) -> Path:
    value = value.expanduser().resolve()
    candidates = [value, value.parent] if value.name.lower() in {"train", "val", "test"} else [value]
    for candidate in candidates:
        if (candidate / "metadata" / "records_manifest.csv").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find metadata/records_manifest.csv below {value} or its parent")


def read_manifest(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "metadata" / "records_manifest.csv", dtype={"state": str})
    frame["state"] = frame.state.str.zfill(2)
    frame["setup"] = frame.setup.astype(int)
    frame["path"] = frame.relative_path.map(lambda item: (root / str(item)).resolve())
    missing = frame.loc[~frame.path.map(Path.exists), "path"]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} MAT files are missing; first: {missing.iloc[0]}")
    return frame


def semantic_category(row: pd.Series) -> str:
    if row["split"] == "excluded":
        return "excluded"
    if row["state"] == "08":
        return f"{row['split']} post-lifting reference"
    return f"{row['split']} {'healthy' if int(row['label']) == 0 else 'damage'}"


def validate(frame: pd.DataFrame) -> list[str]:
    errors = []
    expected_labels = frame.state.map(lambda state: -1 if state == "02" else (0 if state in {"01", "08"} else 1))
    bad = frame[frame.label.astype(int) != expected_labels]
    if len(bad):
        errors.append(f"{len(bad)} records disagree with the locked state-to-label map")
    if (frame.loc[frame.split == "train", "label"].astype(int) != 0).any():
        errors.append("training contains damage")
    if frame.record_id.duplicated().any() or frame.path.duplicated().any():
        errors.append("duplicate record IDs or MAT paths")
    if len(frame) != 306:
        errors.append(f"expected 306 records, found {len(frame)}")
    return errors


def assignment_matrix(frame: pd.DataFrame, output: Path) -> None:
    columns = [(exc, setup) for exc in ("avt", "fvt") for setup in range(1, 10)]
    values = np.full((17, 18), np.nan)
    for i, state in enumerate([f"{x:02d}" for x in range(1, 18)]):
        for j, (excitation, setup) in enumerate(columns):
            row = frame[(frame.state == state) & (frame.excitation.str.lower() == excitation) & (frame.setup == setup)]
            if len(row) == 1:
                values[i, j] = CATEGORIES.index(semantic_category(row.iloc[0]))
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CATEGORIES) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(16, 9))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(18), [f"{e.upper()}{s:02d}" for e, s in columns], rotation=45, ha="right")
    ax.set_yticks(range(17), [f"S{s}: {SCENARIOS[s]}" for s in [f"{x:02d}" for x in range(1, 18)]])
    ax.set_title("Z24 record-level split and label audit (18 records per state)", weight="bold")
    ax.set_xlabel("Excitation and setup (each cell is one complete MAT record)")
    for i in range(17):
        for j in range(18):
            if np.isfinite(values[i, j]):
                k = int(values[i, j])
                ax.text(j, i, SHORT[k], ha="center", va="center", fontsize=6.5,
                        color="white" if k in {0, 3, 6, 7} else "black")
    cbar = fig.colorbar(image, ax=ax, ticks=range(len(CATEGORIES)), fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(CATEGORIES)
    fig.tight_layout()
    fig.savefig(output / "split_assignment_matrix.png", dpi=180)
    plt.close(fig)


def state_timeline(frame: pd.DataFrame, output: Path) -> None:
    states = [f"{x:02d}" for x in range(1, 18)]
    kind = {"01": 0, "02": 3, "08": 1}
    values = [kind.get(state, 2) for state in states]
    names = ["initial healthy", "post-lifting reference", "damage", "excluded transition"]
    colors = ["#16803c", "#1f9eaa", "#c92c2c", "#9a9a9a"]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(range(17), np.ones(17), color=[colors[v] for v in values], edgecolor="white", width=0.92)
    for i, state in enumerate(states):
        ax.text(i, 0.55, f"S{state}", rotation=90, ha="center", va="center", color="white", weight="bold")
        ax.text(i, -0.06, SCENARIOS[state], rotation=55, ha="right", va="top", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=n) for c, n in zip(colors, names)]
    ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set(xlim=(-0.6, 16.6), ylim=(0, 1), xticks=[], yticks=[],
           title="Z24 acquisition chronology: labels apply to whole records, not time segments")
    ax.title.set_y(1.18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.78, bottom=0.42)
    fig.savefig(output / "state_label_timeline.png", dpi=180)
    plt.close(fig)


def load_vertical(path: Path) -> tuple[np.ndarray, list[str]]:
    mat = loadmat(path, variable_names=["data", "labelshulp"])
    labels = [str(value).strip() for value in mat["labelshulp"].tolist()]
    indices = [i for i, label in enumerate(labels) if label.endswith("V") and label[0].isdigit()]
    return np.asarray(mat["data"][:, indices], dtype=np.float64), [labels[i] for i in indices]


def signal_and_psd(frame: pd.DataFrame, output: Path, excitation: str, setup: int,
                   seconds: float, fs: float) -> None:
    selected_states = ["01", "08", "03", "07", "09", "12", "15"]
    rows = []
    loaded = {}
    for state in selected_states:
        match = frame[(frame.state == state) & (frame.excitation.str.lower() == excitation) & (frame.setup == setup)]
        if len(match) != 1:
            continue
        row = match.iloc[0]
        data, labels = load_vertical(row.path)
        loaded[state] = (data, labels)
        rows.append(row)
    common = set.intersection(*(set(labels) for _, labels in loaded.values()))
    if not common:
        raise ValueError("No common vertical sensor exists across the selected records")
    sensor = sorted(common)[0]
    fig_signal, axes_signal = plt.subplots(len(rows), 1, figsize=(14, 11), sharex=True)
    fig_psd, axes_psd = plt.subplots(len(rows), 1, figsize=(14, 11), sharex=True)
    axes_signal = np.atleast_1d(axes_signal); axes_psd = np.atleast_1d(axes_psd)
    for ax_time, ax_psd, row in zip(axes_signal, axes_psd, rows):
        data, labels = loaded[row.state]
        series = data[:, labels.index(sensor)]
        count = min(len(series), max(1, int(seconds * fs)))
        start = max(0, (len(series) - count) // 2)
        segment = series[start:start + count]
        time = np.arange(count) / fs
        reference = row.state == "08"
        healthy = int(row.label) == 0
        color = "#1f9eaa" if reference else ("#16803c" if healthy else "#c92c2c")
        ax_time.plot(time, segment, color=color, lw=0.65)
        ax_time.set_facecolor("#e7f6f7" if reference else ("#ebf6ec" if healthy else "#fdecec"))
        ax_time.set_ylabel(f"S{row.state}", rotation=0, labelpad=20, weight="bold")
        freq, power = welch(series - np.mean(series), fs=fs, nperseg=min(4096, len(series)))
        ax_psd.semilogy(freq, power + np.finfo(float).tiny, color=color, lw=0.9)
        ax_psd.set_facecolor(ax_time.get_facecolor())
        ax_psd.set_ylabel(f"S{row.state}", rotation=0, labelpad=20, weight="bold")
    axes_signal[-1].set_xlabel("Time within record (s); this is not a healthy→damage transition")
    axes_psd[-1].set_xlabel("Frequency (Hz)")
    axes_psd[-1].set_xlim(0, min(30, fs / 2))
    fig_signal.suptitle(f"Same sensor/configuration waveform: {sensor}, {excitation.upper()} setup {setup:02d}", weight="bold")
    fig_psd.suptitle(f"Same sensor/configuration Welch PSD: {sensor}, {excitation.upper()} setup {setup:02d}", weight="bold")
    fig_signal.tight_layout(rect=(0, 0, 1, 0.97)); fig_psd.tight_layout(rect=(0, 0, 1, 0.97))
    fig_signal.savefig(output / "signal_examples.png", dpi=180)
    fig_psd.savefig(output / "psd_examples.png", dpi=180)
    plt.close(fig_signal); plt.close(fig_psd)


def score_audit(frame: pd.DataFrame, scores_path: Path, output: Path) -> None:
    scores = pd.read_csv(scores_path, dtype={"state": str})
    if "state" in scores:
        scores["state"] = scores.state.str.zfill(2)
    required = {"record_id", "score_median"}
    if not required <= set(scores):
        raise ValueError(f"Score CSV must include {sorted(required)}")
    score_columns = [column for column in scores if column.startswith("score_")]
    joined = frame.merge(scores[["record_id", *score_columns]], on="record_id", how="inner")
    if len(joined) == 0:
        raise ValueError("No record IDs from the score CSV match the manifest")
    joined.to_csv(output / "score_audit.csv", index=False)
    plot = joined.sort_values(["state", "excitation", "setup"]).copy()
    colors = ["#1f9eaa" if s == "08" else ("#16803c" if int(y) == 0 else "#c92c2c") for s, y in zip(plot.state, plot.label)]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.scatter(np.arange(len(plot)), plot.score_median, c=colors, s=18, alpha=0.8)
    healthy_val = plot[(plot.split == "val") & (plot.label.astype(int) == 0)]
    if len(healthy_val):
        threshold = healthy_val.score_median.max()
        ax.axhline(threshold, color="black", ls="--", lw=1.2, label=f"max validation-healthy threshold = {threshold:.3g}")
        worst = healthy_val.loc[healthy_val.score_median.idxmax()]
        x = plot.index.get_loc(worst.name)
        ax.annotate(f"threshold-setting record\n{worst.record_id}", (x, worst.score_median), xytext=(10, 15),
                    textcoords="offset points", arrowprops={"arrowstyle": "->"})
    boundaries = plot.groupby("state", sort=False).size().cumsum().to_numpy()[:-1] - 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="#dddddd", lw=0.6)
    centers = plot.groupby("state", sort=False).size().cumsum().to_numpy() - plot.groupby("state", sort=False).size().to_numpy() / 2
    ax.set_xticks(centers - 0.5, plot.state.drop_duplicates().map(lambda s: f"S{s}"))
    ax.set(title="Per-record anomaly scores by structural state", xlabel="State", ylabel="Median anomaly score")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(output / "record_score_audit.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="kaggle_z24_split root or its train directory")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/z24_split_audit"))
    parser.add_argument("--scores", type=Path, help="optional ScatterAD record_scores.csv")
    parser.add_argument("--excitation", choices=["avt", "fvt"], default="fvt")
    parser.add_argument("--setup", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    args = parser.parse_args()
    root = package_root(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = read_manifest(root)
    errors = validate(frame)
    frame["semantic_category"] = frame.apply(semantic_category, axis=1)
    frame.to_csv(args.output / "label_audit.csv", index=False)
    assignment_matrix(frame, args.output)
    state_timeline(frame, args.output)
    signal_and_psd(frame, args.output, args.excitation, args.setup, args.seconds, args.sample_rate)
    if args.scores:
        score_audit(frame, args.scores, args.output)
    counts = frame.groupby(["split", "label"]).size().to_dict()
    print("Z24 LABEL AUDIT")
    print("Labels are structural-state labels for complete MAT records; no within-record damage onset exists.")
    print(f"package_root={root}")
    print(f"records={len(frame)} counts={counts}")
    print("state01=initial healthy; state08=post-lifting reference (shown separately); state02=excluded transition")
    print(f"validation={'PASS' if not errors else 'FAIL: ' + '; '.join(errors)}")
    print(f"outputs={args.output.resolve()}")


if __name__ == "__main__":
    main()

