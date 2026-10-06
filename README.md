# z24-prism

Train [PRISM](https://github.com/fedezuc/PRISM) (a lightweight multivariate time-series classifier) on the
processed Z24 bridge dataset, [thanglexuan/Z24-dataset-processed](https://huggingface.co/datasets/thanglexuan/Z24-dataset-processed):
1530 samples x 27 accelerometer channels x 6000 time steps, 17 damage scenarios. The full 6000-sample
sequences are used (no downsampling).

## Splits

The 1530 samples are `17 scenarios x 9 setups x 10 segments`, where the 10 segments of a (scenario, setup)
pair are cut from one long recording. A random split would put neighbouring segments in both train and test,
so `run_prism_z24.py` splits by group:

| `--split` | train / val / test | what it measures |
|---|---|---|
| `segment` | segments 0-6 / 7 / 8-9 of every recording | optimistic: same recordings in every split |
| `setup` | setups 0-5 / 6 / 7-8 | legacy strict split: test setups are never seen in training |
| `setup_holdout` | setups 0-4 / 5-6 / 7-8 | recommended hard split with a larger validation set |
| `low_data` | setups 0-2 / 3-4 / 5-8 | low-data domain-generalisation stress test |

Sample index = `scenario*90 + setup*10 + segment`.

For all modes, validation and test probabilities are averaged across the available segments from
each original `(scenario, setup)` recording. Checkpoint selection and primary metrics are therefore
recording-level; segment-level metrics are saved only as secondary diagnostics. The grouped setup
modes contain no recording overlap between train, validation, and test.

## Run locally

```bash
git clone https://github.com/fedezuc/PRISM.git PRISM          # third-party, not vendored (no licence file)
# put inputs.npy and labels.npy in Z24-dataset-processed/
pip install -r requirements.txt
python run_prism_z24.py --split setup
```

`--log_every N` controls how often batch progress is printed. Everything is written to
`results/<run name>/` (see "Pipeline and outputs").

## Dedicated 10000-point Mamba experiment

`train_mamba.py` reconstructs each original 60000-point recording and divides it into six
non-overlapping windows of 10000 points. This produces `(918, 27, 10000)` with the scenario label
preserved for every window. The default `temporal_holdout` split assigns windows 0-2 of every
`(scenario, setup)` recording to train, window 3 to validation, leaves window 4 unused as a temporal
gap, and uses window 5 for testing. This gives 459/153/153 windows, with all 17 classes and all 9
setups represented in every split. The splits still come from the same source recordings, but the
gap makes test less correlated with train than the easier `within_recording` 4/1/1 split. Use
`--split-mode setup_holdout` for the recommended recording-group/domain-generalisation
experiment. It uses setups 0-4 for training, 5-6 for validation, and 7-8 for testing for every
class. The identical setup partition prevents setup membership from becoming a shortcut for the
class label. `unseen_setup` retains the older 6/1/2 partition; `balanced` is retained only for
reproducibility and is not recommended because its class-dependent rotating setup assignment
confounds class with split/setup membership.

```bash
python train_mamba.py --split-mode temporal_holdout --no-augment --epochs 100 --patience 20 \
  --batch-size 8 --learning-rate 3e-4
```

Recommended strict run:

```bash
python train_mamba.py --split-mode setup_holdout --epochs 60 --patience 12 \
  --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-3 --dropout 0.3
```

Augmentation is applied on the fly to the training set only; validation and test signals are never
augmented. Add `--no-augment` for a no-augmentation ablation. Checkpoint selection and the primary
test metrics operate at recording level by averaging the six window probabilities; window metrics
are retained as secondary diagnostics. Outputs are saved under
`results/<run_name>/`, including `trainlog.txt`, best and
final weights, the exact `split_manifest.json`, recording/window predictions, learning curves,
confusion matrices, per-class/setup accuracy, ROC curves, and 2D/3D t-SNE plots from epoch 1 and the
final selected model. Open `train_mamba.ipynb` for an interactive local
workflow, or upload `time-series.ipynb` to Kaggle. The Kaggle notebook clones this repository and
downloads only `inputs.npy` and `labels.npy` from Hugging Face before training.

## Models (`--model`)

All models read the full 6000-step, 27-channel sequence (no resampling); see `models_seq.py`.

| `--model` | what it is | size |
|---|---|---|
| `prism` | multi-resolution symmetric CNN + global pooling (reference, not a sequence model) | 0.105M |
| `ms4n` | S4D state-space model following the MS4N description in arXiv:2605.27406: linear input projection, S4D FFT convolution, gated (GLU) channel mixing, LayerNorm, average pooling, MLP | 0.024M |
| `mamba` | Selective state-space model (Mamba / S6, Gu & Dao 2023-2024 - the architecture behind most 2024-2026 sequence SOTA); unlike S4D, `A, B, C, dt` depend on the input at every step. Implemented as a genuine step-by-step recurrence, not the authors' hardware-aware CUDA scan (no Linux/CUDA build assumed here) | ~0.1M |
| `gru`, `lstm` | learned strided-conv stem (`--stem_stride`, default 5 -> 1200 steps; this is a learned layer, not resampling) + 2-layer bidirectional RNN | 0.15M / 0.19M |
| `cnn_lstm` | two conv+pool blocks + bidirectional LSTM (the 1DCNN-LSTM baseline family) | 0.087M |
| `transformer` | patch embedding over all channels (patch 50, stride 25 -> 239 tokens) + 3-layer Transformer encoder | 0.20M |

`ms4n` and `mamba` are re-implementations from published equations (no official/practical-to-install code was
found for either on this stack), so treat them as "S4D-style" / "selective-scan-style", not as the authors' exact
model. `mamba`'s Python-loop scan gets disproportionately slower as its effective sequence length grows (see
`models_seq.py`'s docstring for measurements), so its default `--stem_stride` (25) is higher than `gru`/`lstm`'s (5)
-- lower it only if you have time to spare. `--hidden` and `--layers` change width and depth. Example, all models
on both splits:

```bash
for m in ms4n mamba gru lstm cnn_lstm transformer prism; do
  for s in segment setup; do
    python -u run_prism_z24.py --model $m --split $s --batch_size 32 --lr_schedule constant --epochs 60 --patience 15
  done
done
```

## Pipeline and outputs

1. **Data**: each original 60000-sample recording is already cut into 10 segments of 6000 samples
   (1530 = 17 scenarios x 9 setups x 10 segments), see "Splits" above.
2. **Normalisation** (`--norm robust`, default): per channel, subtract the median and divide by the median
   per-sample std, then clip to +-10. Statistics come from the training set only. The raw data has huge spikes
   (a global std is ~10x the typical per-sample std), so a plain global z-score (`--norm zscore`) shrinks the
   signal. Clipping is `--clip`.
3. **Augmentation** (training set only, on the fly, one transform per sample per epoch; `--no_augment` turns it
   off): Gaussian noise 40%, time reversal 30%, random crop 10%, time warp 10%, circular shift 10%.
   Warping and resizing change frequencies, and damage shows up as small frequency shifts, so keep `--aug_warp`
   small and compare against `--no_augment`.
4. **Training / selection**: the checkpoint with the lowest *validation* loss is used; test data is only reported.
5. **Report** (`metrics.json`, printed in the log): accuracy, balanced accuracy, precision / recall / F1 (macro and
   weighted, plus per class), ROC-AUC (one-vs-rest), top-3 accuracy, Cohen's kappa, MCC.
6. **Figures**: `training_curves.png` (train/val loss, accuracy, lr), `confusion_matrix.png` (counts and
   normalised), `roc_curve.png` (per class + macro/micro), `tsne_2d.png` (coloured by scenario and by setup) and
   `tsne_3d.png`, computed on the 128-d PRISM embeddings of the test set. Raw predictions, embeddings and history
   are saved as `.npz` / `.json` so figures can be redrawn.

Run folder name: `<model>_<split>_<lr_schedule>_<aug|noaug>_seed<seed>`.
Pass `--run-name NAME` to override the generated folder name.

For the native 6000-point hard baseline with conservative train-only augmentation:

```bash
python run_prism_z24.py --model mamba --split setup_holdout --epochs 30 --patience 10 \
  --batch_size 16 --lr 3e-4 --lr_schedule cosine --weight_decay 1e-4 --dropout 0.1 \
  --stem_stride 25 --aug_noise 0.03 --aug_warp 0.01 --aug_crop_min 0.9 --seed 42
```

Only batches inside the training loop pass through `augment`; validation and test data are never
augmented. Add `--no_augment` for a no-augmentation ablation on this pipeline.

## Run on Kaggle

`kaggle/z24_prism_kaggle.py` clones this repo and PRISM, downloads the data from Hugging Face and trains both
splits. Enable GPU and Internet; for a private repo add a `GITHUB_TOKEN` Kaggle secret. Edit the id in
`kaggle/kernel-metadata.json`, then:

```bash
kaggle kernels push -p kaggle
kaggle kernels status <username>/z24-prism
kaggle kernels output <username>/z24-prism -p kaggle_out
```

### QUGS independent Dataset A/B benchmark

`train_quatar_models.py` compares Mamba, 1D-CNN and BiLSTM for QUGS damage
localisation. With both datasets it runs a two-direction cross-dataset benchmark:

- Fold 1 trains/validates on Dataset A and tests every matching Dataset B recording.
- Fold 2 trains/validates on Dataset B and tests every matching Dataset A recording.
- If A and B contain different condition sets, the script automatically restricts
  both sides to their class intersection and prints every excluded class. Classes
  present on only one side cannot be evaluated as supervised test targets.
- The training dataset supplies 96 training windows and 24 validation windows
  per class, separated by an 8-window guard gap.
- Per-window, per-sensor z-scoring removes absolute offset/gain fingerprints
  while preserving temporal and spectral shape. `--normalization train_global`
  is available as an ablation.
- The report includes correlated window-level metrics and recording-level
  metrics obtained by averaging all window probabilities from each test file.
- `summary_by_model.csv` reports mean and standard deviation across both directions.

Add Dataset A and Dataset B as Kaggle inputs, enable a GPU, then run:

```python
!git clone https://github.com/TuanHungGiang/time-series-project.git
%cd time-series-project
!python train_quatar_models.py --direction both --normalization per_window \
  --epochs 15 --patience 4 --batch-size 64 --num-workers 2
```

The script auto-detects `zzzAU.TXT`/`zzzAD*.TXT` and
`zzzBU.TXT`/`zzzBD*.TXT` below `/kaggle/input`. If more than one copy exists,
pass explicit paths:

```python
!python train_quatar_models.py \
  --data-a "/kaggle/input/qugs/Quatar-Dataset A" \
  --data-b "/kaggle/input/qugs/Quatar-Dataset B" \
  --out-dir /kaggle/working/qugs_results \
  --cache-dir /kaggle/working/qugs_cache
```

Checkpoints, predictions, confusion matrices, `summary.csv`, and the exact
split manifest are written to `/kaggle/working/qugs_results` by default.

If only Dataset A is available, omit `--data-b` and select the explicit
diagnostic mode:

```python
!python train_quatar_models.py \
  --data-a "/kaggle/input/datasets/giangtuanhung/quatar-a/Dataset A" \
  --direction temporal_a --normalization per_window \
  --epochs 15 --patience 4 --batch-size 64 --num-workers 2
```

This runs two symmetric time-block folds (`early_to_late` and
`late_to_early`). Every class has 72 train, 24 validation, and 24 test windows,
with two 4-window guard gaps. The files, logs, and manifests clearly mark these
scores as same-recording diagnostics; they are useful for model iteration but
are not a substitute for an independent Dataset B test. With `--direction
auto` (the default), the script uses A<->B when B is found and otherwise falls
back to these two Dataset A folds.

## Train on a short snippet, evaluate on the rest (`--train_window`)

`--train_window N` splits every sample's time axis in two: **train** samples use only their first N steps
(then augmented), while **val/test** samples use their *remaining* (later) steps instead. `--split` still
decides which recordings are train/val/test, so this is not a new source of cross-sample leakage; it tests
whether training on a short augmented snippet generalises to a longer, later portion of *different*
recordings.

```bash
python run_prism_z24.py --model prism --split setup --train_window 1000   # train: first 1000 steps
                                                                            # val/test: last 5000 steps
```

Normalisation statistics are computed from the train slice only (the exact data the model sees during
training), then applied to both slices. Combine with `--seq_len` if you also want to shorten the recording
overall before splitting it in two (e.g. `--seq_len 3000 --train_window 1000` -> train sees steps 0-999,
val/test sees steps 1000-2999). Run names get a `_tw<N>` suffix when this is on.

## Shortening the sequence (`--seq_len`)

`--seq_len N` (0 = full 6000, the default) uses only N time steps per sample, via `--seq_mode`:
- `crop` (default): keep the first N points - same sample rate, shorter duration, signal not distorted.
- `downsample`: N points evenly spaced across the full recording - same duration, lower sample rate, which
  aliases any frequency content above the new Nyquist limit instead of just discarding it.

```bash
python run_prism_z24.py --model transformer --split setup --seq_len 1000 --seq_mode crop
```

`visualize_z24.ipynb` reconstructs and visualises the `(918, 27, 10000)` dataset, creates the strict
setup-based train/validation/test split, and demonstrates length-preserving train-only augmentation in both
the time and frequency domains. It does not crop or downsample the 10000-point samples.

## Multi-GPU (Kaggle T4 x2)

If more than one GPU is visible, `run_prism_z24.py` wraps the model in `torch.nn.DataParallel` automatically
(`--gpus N` limits how many). `--batch_size` is the total batch and is split across GPUs, so with two T4s use
`--batch_size 32` to keep 16 samples per GPU:

```bash
python run_prism_z24.py --split setup --batch_size 32
```

PRISM is tiny (0.1M parameters) and runs one small convolution per channel, so it is launch-bound rather than
compute-bound; expect a modest speed-up from the second GPU, not 2x.

## Notes

PRISM's default schedule halves the learning rate every epoch (`--lr_schedule halve`, the default). On this
dataset that leaves the model under-fitted (training loss stays close to ln 17, test accuracy near chance), so
treat the default-schedule numbers as a baseline only. Alternatives:

```bash
python run_prism_z24.py --split setup --batch_size 32 --lr_schedule constant --epochs 60 --patience 15
python run_prism_z24.py --split setup --batch_size 32 --lr_schedule cosine   --epochs 60 --patience 60
```

Ablation without augmentation: add `--no_augment`.
