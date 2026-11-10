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
