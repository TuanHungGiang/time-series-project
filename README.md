# CLMM–TimesNet

**English** | [Tiếng Việt](README.vi.md)

Healthy-only bridge damage detection under whole-sensor loss. The active experiment compares a frozen TimesNet backbone with period anchoring, observed-only scoring, channel completion, cross-sensor prediction, observed-sensor attention, and a frequency loss. It uses Z24 and HellBridge, clean input and five nested masks at each of 10%, 30%, and 50% sensor loss. The first run uses one seed. Two-bridge results are pending.

| Path | Purpose |
|---|---|
| [src/clmm_timesnet/](src/clmm_timesnet/) | Installable model, dataset, metric, and experiment code |
| [configs/](configs/) | Dataset provenance and model configuration |
| [data/](data/) | Dataset access and preprocessing instructions |
| [experiments/](experiments/) | Per-dataset outputs and cross-dataset comparisons |
| [notebooks/](notebooks/) | Kaggle entry point and analysis |
| [docs/](docs/) | Protocol and bilingual documentation |
| [paper/](paper/) | Manuscript workspace |
| [website/](website/) | Experiment status dashboard |

## Run the experiment

Attach the two datasets and enable GPU and Internet on Kaggle. Open the [one-cell notebook](notebooks/analysis/clmm_timesnet_kaggle.ipynb). It clones the repository, logs the commit, and runs the package from a fresh output directory.

For a local run, use an environment with CUDA-enabled PyTorch if available.

```bash
python -m pip install -e .
python -m clmm_timesnet.experiment --config configs/experiments/clmm_timesnet.yaml   --z24-root data/z24/processed/kaggle_z24_split   --hell-h5 data/hellbridge/raw/data_100Hz.h5   --output-dir experiments/exp002_clmm_timesnet_run01
```

The runner creates the HellBridge split and writes a resolved configuration plus dataset-specific scores and metrics. The `summary/` directory holds cross-dataset metrics, paired changes, and the audit of batch-dependent scores.

See [the protocol](docs/en/experiment-protocol.md) and [the experiment record](experiments/exp001_clmm_timesnet/notes.md). The [legacy workflows](docs/en/legacy-workflows.md) document the older root scripts. The repository layout follows the [AI Research Template](https://github.com/Le-Xuan-Thang/ai-research-template). The vendored TimesNet port keeps its [upstream license](src/clmm_timesnet/models/timesnet/LICENSE).
