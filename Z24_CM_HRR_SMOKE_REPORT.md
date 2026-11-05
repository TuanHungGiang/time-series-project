# Z24 CM-HRR preprocessing smoke benchmark

## Protocol

- Data split: the exported `kaggle_z24_split` manifest.
- Fit: 12 S01/train healthy records only.
- Calibration: three S01/val healthy records only.
- Evaluation: 132 locked-test records.
- Damage and S08 used for fitting: no.
- Record decision threshold: maximum S01/val record-level q95 anomaly score.
- CM-HRR reference fallback: `setup+excitation -> excitation -> global`, with
  all reference statistics fitted from S01/train only.
- Smoke setting: autoencoder limited to 10 epochs with patience 3. The other
  detectors are deterministic under seed 42.

This is an exploratory preprocessing ablation, not a final paper result. Since
the locked test has now been inspected for this preprocessing choice, a later
claim of generalisation must be confirmed on a new untouched split/dataset.

## Locked-test results

| Preprocessing | Detector | AUROC | Balanced accuracy | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| Raw log-PSD | OCSVM | 0.3823 | 0.5000 | 0.0000 | 1.0000 |
| CM-HRR | OCSVM | 0.5979 | 0.5556 | 0.2000 | 1.0000 |
| Raw log-PSD | Robust PCA | 0.6151 | 0.5873 | 0.2973 | 1.0000 |
| CM-HRR | Robust PCA | 0.5714 | 0.4722 | 0.6054 | 0.5000 |
| Raw log-PSD | Isolation Forest | 0.4603 | 0.5000 | 0.0000 | 1.0000 |
| CM-HRR | Isolation Forest | 0.5331 | 0.5278 | 0.1053 | 1.0000 |
| Raw log-PSD | Small AE | 0.5185 | 0.5357 | 0.3822 | 0.8333 |
| CM-HRR | Small AE | 0.6085 | 0.5000 | 0.4941 | 0.6667 |

## Preprocessing deltas

| Detector | Delta AUROC | Delta balanced accuracy | Delta F1 |
|---|---:|---:|---:|
| OCSVM | +0.2156 | +0.0556 | +0.2000 |
| Robust PCA | -0.0437 | -0.1151 | +0.3081 |
| Isolation Forest | +0.0728 | +0.0278 | +0.1053 |
| Small AE | +0.0899 | -0.0357 | +0.1120 |

## Decision

CM-HRR passes the AUROC improvement criterion for three of four detectors, but
it **fails the universal-improvement criterion** because Robust PCA AUROC and
balanced accuracy decline. It must therefore remain a candidate ablation, not
the selected preprocessing of the proposed framework.

Two likely causes need to be isolated next:

1. Residual-only input removes absolute spectral shape that PCA previously
   used. A pre-registered dual-channel input (absolute log-PSD plus CM-HRR)
   should be tested without discarding the original representation.
2. Three healthy calibration records make the maximum-q95 threshold unstable.
   Threshold-free AUROC improves more consistently than balanced accuracy, so
   calibration must be evaluated separately from representation quality.

Do not tune the next variant on these 132 test labels and then report the same
test as locked. Use a development split for preprocessing selection and retain
another dataset (for example Hell Bridge) or a newly reserved Z24 state group
for final confirmation.

