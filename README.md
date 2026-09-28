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
| `setup` | setups 0-5 / 6 / 7-8 | strict: test setups are never seen in training |

Sample index = `scenario*90 + setup*10 + segment`.

## Run locally

```bash
git clone https://github.com/fedezuc/PRISM.git PRISM          # third-party, not vendored (no licence file)
# put inputs.npy and labels.npy in Z24-dataset-processed/
pip install -r requirements.txt
python run_prism_z24.py --split setup
```

Results go to `results/` (`.json` metrics, `_confusion.npy`, `.pt` best checkpoint).
`--log_every N` controls how often batch progress is printed.

## Run on Kaggle

`kaggle/z24_prism_kaggle.py` clones this repo and PRISM, downloads the data from Hugging Face and trains both
splits. Enable GPU and Internet; for a private repo add a `GITHUB_TOKEN` Kaggle secret. Edit the id in
`kaggle/kernel-metadata.json`, then:

```bash
kaggle kernels push -p kaggle
kaggle kernels status <username>/z24-prism
kaggle kernels output <username>/z24-prism -p kaggle_out
```

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

PRISM's default schedule halves the learning rate every epoch. On this dataset that leaves the model
under-fitted (training loss stays close to ln 17), so treat the default-schedule numbers as a baseline only.
