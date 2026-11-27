"""Plot recorded training losses from the matched baseline notebook or its log."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUN = re.compile(r"^(Z24|HellBridge) / (\d+) / ")
GROUP = re.compile(r"^(\S+) \| fit\.shape=")
METHOD = re.compile(r"^\s+fitting (\w+)$")
EPOCH = re.compile(r"^\s+epoch=(\d+)/(\d+) (\w*loss)=([\d.eE+-]+)$")
LABELS = {
    "lstm_ae": "LSTM-AE", "timesnet_native": "TimesNet", "tranad": "TranAD",
    "moderntcn": "ModernTCN", "scatterad": "ScatterAD", "cnn_ae_1d": "CNN-AE 1D",
}
EXPECTED_GROUPS = {"Z24": 18, "HellBridge": 1}


def read_log(path):
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".ipynb":
        return content
    notebook = json.loads(content)
    return "\n".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"] for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def parse(text):
    dataset = seed = group = model = None
    rows = []
    for line in text.splitlines():
        if match := RUN.match(line):
            dataset, seed = match.group(1), int(match.group(2))
            group = model = None
        elif match := GROUP.match(line):
            group, model = match.group(1), None
        elif match := METHOD.match(line):
            model = match.group(1)
        elif dataset and group and model in LABELS and (match := EPOCH.match(line)):
            epoch, total, objective, value = match.groups()
            rows.append(dict(dataset=dataset, seed=seed, group=group, model=model,
                             epoch=int(epoch), epochs=int(total), objective=objective, loss=float(value)))
    if not rows:
        raise ValueError("No matched-baseline training loss records were found in the supplied log.")
    return pd.DataFrame(rows).drop_duplicates(["dataset", "seed", "group", "model", "epoch"], keep="last")


def plot_curves(records, out):
    out.mkdir(parents=True, exist_ok=True)
    records.to_csv(out / "training_loss_by_group.csv", index=False)
    # Keep a common cohort of groups at every recorded epoch for each seed.
    keys = ["dataset", "model", "seed"]
    pieces = []
    for _, part in records.groupby(keys, sort=False):
        cohort = set.intersection(*(set(epoch.group) for _, epoch in part.groupby("epoch")))
        if cohort:
            pieces.append(part.loc[part.group.isin(cohort)])
    if not pieces:
        raise ValueError("No consistent fitted-group cohort across logged epochs.")
    summary = pd.concat(pieces).groupby([*keys, "epoch"], as_index=False).agg(
        loss=("loss", "mean"), groups=("group", "nunique"))
    summary.to_csv(out / "training_loss_by_seed.csv", index=False)
    epochs = range(1, int(records.epochs.max()) + 1)
    paths = []
    for dataset, bridge in summary.groupby("dataset"):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, layout="constrained")
        fig.suptitle(f"{dataset} | Healthy training loss")
        for (name, label), ax in zip(LABELS.items(), axes.flat, strict=True):
            model = bridge.loc[bridge.model.eq(name)]
            ax.set_title(label)
            ax.set_xticks(list(epochs))
            ax.grid(axis="y", alpha=0.22)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("ScatterAD objective" if name == "scatterad" else "MSE")
            for seed, curve in model.groupby("seed"):
                groups = int(curve.groups.iloc[0])
                ax.plot(list(epochs), curve.set_index("epoch").loss.reindex(epochs),
                        marker="o", markersize=3.5, label=f"{seed} ({groups}/{EXPECTED_GROUPS[dataset]} groups)")
            if not model.empty:
                ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(.5, .5, "No logged loss", ha="center", va="center", transform=ax.transAxes)
        path = out / f"training_loss_{dataset}.png"
        fig.savefig(path, dpi=170)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_curves(parse(read_log(args.log)), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
