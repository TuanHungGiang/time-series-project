# MDR-SHM: Kaggle runner

Leakage-controlled Z24 baseline and corruption benchmark for the
missingness-aware bridge SHM study.

## Kaggle data

Create one private Kaggle Dataset from the local `kaggle_z24_split` folder.
Upload only:

- `train/`
- `val/` (treated as healthy calibration)
- `test/`
- `metadata/`
- `dataset-metadata.json` and the dataset README if desired

Do not upload `_excluded/`. Do not upload the original RAR archives after the
split dataset has been uploaded. The selected upload is about **3.27 GiB**:
train 0.266 GiB, val 1.513 GiB, test 1.493 GiB, plus negligible metadata.
Although half of the damaged files live under `val/damage`, the runner forces
every `label=1` record into evaluation-only; damage is never used for fitting,
model selection, or threshold calibration.

The runner finds a CSV containing `record_id`, `role`, and `label`. Absolute
Windows paths in the manifest are resolved by matching their file names inside
`/kaggle/input`.

Hell Bridge is a separate external-validation dataset. Upload
`data_100Hz.h5` later as a second private Kaggle Dataset (2.344 GiB); it must
not be mixed into Z24 training.

## Run on Kaggle

Enable a GPU only for the future neural Model A. The current spectral B2
baseline runs on CPU.

```bash
!git clone -b codex/mdr-shm-kaggle https://github.com/TuanHungGiang/time-series-project.git
!pip install -q -r time-series-project/research/mdr_shm_kaggle/requirements.txt
!python time-series-project/research/mdr_shm_kaggle/run_z24_b2.py \
  --input-root /kaggle/input \
  --output-dir /kaggle/working/mdr_shm_b2
```

The script writes the frozen model, record-level clean scores, corruption
scores, and summaries under `/kaggle/working/mdr_shm_b2`.


## Model A v1: mask-aware healthy-only detector

After B2 has been reproduced, update the branch and run the quick Model A gate:

```bash
!git -C /kaggle/working/time-series-project pull
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/train_z24_model_a.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v1_quick \
  --mode quick
```

Quick mode evaluates clean data plus 30% contiguous missing time, 5 dB noise,
30% random sensor dropout, and 30% B2-evidence-targeted sensor dropout. It uses
one deterministic replicate for the initial gate.

Only after the quick result is accepted should the full protocol be run:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/train_z24_model_a.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v1_full \
  --mode full
```

Model selection uses healthy calibration consistency only. All 252 damage
records remain evaluation-only. Outputs include `metrics.csv`,
`record_scores.csv`, `risk_coverage.json`, `training_history.csv`, and
`model_a.pt`.


## Model A v1 decision and Model A v2

The neural spectral reconstruction v1 is retained for reproducibility but was
rejected at the quick gate (clean AUROC 0.4934). Do not run its full mode.

Model A v2 preserves the frozen B2 damage score and adds a healthy-calibrated
reliability layer based on leave-one-sensor-out score instability, sensor
coverage, and observed-data quality. Run:

```bash
!git -C /kaggle/working/time-series-project pull origin codex/mdr-shm-kaggle
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_model_a_v2.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v2_quick \
  --mode quick
```

The clean score must reproduce B2 exactly. The research question for v2 is
whether abstention improves accepted-subset performance and risk--coverage
under random and evidence-targeted sensor loss.


## Model A v3: corruption-specific selective reliability

V2 is retained as a sensor-dropout baseline. V3 adds healthy-calibrated
corruption diagnostics: high-frequency local outliers and spectral roughness,
while treating coverage and explicit missing-time quality as soft evidence.
Calibration contains only clean and mild conditions (missing 10%, noise 20 dB,
dropout 10%). Severe conditions are unseen stress tests.

```bash
!git -C /kaggle/working/time-series-project pull origin codex/mdr-shm-kaggle
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_model_a_v3.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v3_quick \
  --mode quick
```

V3 passed syntax and end-to-end smoke tests. Smoke metrics are intentionally not
used for research conclusions.


## Model A v3.1: paper-ready selective evaluation

V3.1 keeps the frozen B2 score and the healthy-only V3 reliability calibration.
It changes the evaluation protocol in four places:

- evidence-targeted dropout samples from the highest-evidence sensor pool, so
  the three replicate IDs are distinct but remain targeted;
- accepted cohorts that lose either class are explicitly marked invalid;
- stratified bootstrap confidence intervals and class-conditional coverage are
  reported;
- recall and specificity use exact Clopper--Pearson intervals, avoiding the
  degenerate `[1, 1]` bootstrap interval produced by very small healthy sets;
- balanced AURC and reliability-to-error AUROC are added.

Run the quick gate first:

```bash
!git -C /kaggle/working/time-series-project pull origin codex/mdr-shm-kaggle
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_model_a_v31.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v31_quick \
  --mode quick \
  --bootstrap-samples 300
```

Quick mode uses one replicate for ordinary corruptions and three replicates for
`targeted_0.30`, so the repaired targeted protocol is tested before the full
run.

If the quick run ends with `MODEL_A_V31_PASS`, run the full protocol:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_model_a_v31.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/model_a_v31_full \
  --mode full \
  --bootstrap-samples 1000
```

The new outputs are `bootstrap_ci.csv` and `risk_coverage.csv`, in addition to
the V3 artifacts. An `accepted_status=invalid_missing_class` row is a fail-safe
abstention result, not an accepted-subset performance improvement. Treat rows
with `accepted_low_class_support=True` as exploratory because fewer than five
records remain in the minority class.


## Paper tables, baselines, ablations, and figures

After the V3.1 full run, reuse its record-level outputs; the model does not need
to be run again. The analysis compares every rejection rule at exactly the same
realized coverage and includes confidence-margin, coverage-only, quality-only,
jackknife-only, and repeated random-rejection baselines.

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/analyze_z24_v31_paper.py \
  --input-dir /kaggle/working/model_a_v31_full \
  --output-dir /kaggle/working/model_a_v31_paper \
  --random-repeats 200
```

The output directory contains five CSV tables and four publication-ready PNG
figures. In particular, use `table4_proposed_vs_random.csv` to test whether the
proposed reliability ranking beats chance at matched coverage, and
`table5_matched_aurc.csv` to compare overall selective risk. The component
ablation is in `table3_ablation.csv`; lower balanced AURC is better.

Download everything from Kaggle as one archive:

```bash
!cd /kaggle/working && zip -qr model_a_v31_paper.zip model_a_v31_paper
```


## Classical and deep detector baselines

This experiment isolates detector quality from the selective-reliability
experiment. Every detector is fitted on healthy train records only, calibrated
on healthy calibration records only, and evaluated on the identical clean and
corrupted test records. Deep inputs are record-level median log-spectra; sensor
subset augmentation is applied only after the record split, so no window or
record leakage is possible.

Enable a Kaggle GPU and run the quick gate:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_detector_baselines.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/z24_detector_baselines_quick \
  --mode quick \
  --epochs 12 \
  --seeds 0
```

The runner compares B2, PCA-SPE, one-class SVM, Isolation Forest, CNN
autoencoder, masked/denoising CNN autoencoder, and Deep SVDD. After
`DETECTOR_BASELINES_PASS`, run the paper protocol:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_z24_detector_baselines.py \
  --input-root /kaggle/input/datasets/giangtunhng/z24-v2/kaggle_z24_split \
  --output-dir /kaggle/working/z24_detector_baselines_full \
  --mode full \
  --epochs 60 \
  --seeds 0,1,2
```

Generate the detector comparison tables and figures:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/analyze_z24_detector_baselines.py \
  --input-dir /kaggle/working/z24_detector_baselines_full \
  --output-dir /kaggle/working/z24_detector_paper
!cd /kaggle/working && zip -qr z24_detector_results.zip \
  z24_detector_baselines_full z24_detector_paper
```


## Final Z24 statistics and framework evidence

Once both `z24_detector_baselines_full` and `model_a_v31_paper` are available
in the same Kaggle session, run the paired record-level hierarchical bootstrap,
separate training-seed from corruption-replicate variability, and merge the
detector and matched-coverage reliability evidence:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/analyze_z24_final_statistics.py \
  --detector-dir /kaggle/working/z24_detector_baselines_full \
  --reliability-dir /kaggle/working/model_a_v31_paper \
  --output-dir /kaggle/working/z24_final_statistics \
  --bootstrap-samples 2000 \
  --coverage 0.70
```

The bootstrap resamples healthy and damage record IDs separately and retains
all corruption replicates for each sampled record. It therefore does not treat
attack replicates or deep-model seeds as new bridge records.


## HellBridge healthy-adapted external validation

Upload `data_100Hz.h5` as a separate private Kaggle Dataset. The external
protocol freezes the B2/OCSVM algorithm and hyperparameters but re-estimates a
regime-specific healthy reference from HellBridge UDS recordings. Damage labels
are evaluation-only. This is healthy-only domain adaptation, not zero-shot
transfer of Z24 modal anchors.

Quick gate using UDS and DS1:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_hellbridge_b2_external.py \
  --h5 /kaggle/input/datasets/giangtunhng/hellbridge/data_100Hz.h5 \
  --output-dir /kaggle/working/hellbridge_b2_quick \
  --mode quick \
  --bootstrap-samples 300
```

Full DS1-DS8 validation:

```bash
!python /kaggle/working/time-series-project/research/mdr_shm_kaggle/run_hellbridge_b2_external.py \
  --h5 /kaggle/input/datasets/giangtunhng/hellbridge/data_100Hz.h5 \
  --output-dir /kaggle/working/hellbridge_b2_full \
  --mode full \
  --bootstrap-samples 2000
!cd /kaggle/working && zip -qr hellbridge_b2_full.zip hellbridge_b2_full
```


