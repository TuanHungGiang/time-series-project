# Z24 preprocessing result snapshot — 2026-11-09

This file freezes the result before proposed-method development continues.

## Leakage protocol

- 24 whole healthy train records; no damage in training.
- Six healthy validation records calibrate thresholds.
- Damage validation selects preprocessing; the selected configuration alone is
  evaluated on test.
- No record is split across partitions.
- Modal anchors and every fitted detector statistic use healthy training only.

## Corrected multi-detector selection

PCA was corrected to retain fewer components than input dimensions. Continuous
modal centroid then ranked first across OCSVM, PCA-3, Isolation Forest and AE:

| preprocessing | mean validation AUROC | mean rank | improved detectors |
|---|---:|---:|---:|
| modal centroid | 0.6940 | 1.6 | 4/4 |
| modal peak | 0.6889 | 2.0 | 4/4 |
| domain CM-HRR | 0.6457 | 3.0 | 4/4 |
| ACF | 0.6427 | 3.5 | 4/4 |
| logPSD | 0.4371 | 4.9 | control |

## Final interaction experiment

Validation selected continuous modal centroid with four modes, window 4096,
stride 2048, 0.50-Hz minimum anchor separation and 0.45-Hz tracking radius.
It improved all four detectors and achieved mean validation AUROC 0.8069.

One-shot test consensus achieved AUROC 0.7791, balanced accuracy 0.7183,
recall 0.4365, specificity 1.0000 and F1 0.6077.

## Unresolved limitation

The representation result is publishable as a preliminary finding, but the
preprocessing is not physically locked. Grouped-OOF anchor ranges were 0.0122,
0.0122, 0.6958 and 6.2378 Hz. Modes 3--4 therefore require reference-aware
matching or consensus anchor selection. The healthy-train edge-hit rate was
0%, so boundary clipping is not the cause.

No later proposed-method result may silently replace or reinterpret this
snapshot. Any anchor correction must be reported as a new ablation.

