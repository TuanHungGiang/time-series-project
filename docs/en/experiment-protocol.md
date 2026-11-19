# Experiment protocol

[Tiếng Việt](../vi/experiment-protocol.md)

The proposed method is `clmm_time_frequency`. Compare it to TimesNet on the same model seed and masks. TimesNet is fitted on complete healthy data and frozen before adapter training. The seven variants successively test the native backbone, fixed periods, observed-only residuals, channel completion, cross-sensor prediction, observed-key attention, and spectral training loss.

With cross-prediction, partition the visible sensors into four groups. Temporarily hide each group and predict it from the other visible sensors. Score the prediction against actual measurements at the hidden group. The attention variant excludes hidden channels as keys and values. The final variant applies Smooth L1 on time-domain samples and adds 0.20 times Smooth L1 on non-DC log FFT power of the probed channels during training. All cross-prediction variants use the same time-domain residual for evaluation.

Use Z24 record-median scores and HellBridge window scores. For each method on each bridge, calibrate Q90 on clean healthy validation and hold that threshold across rates. Use the same mask identities for every variant. Report AUC, AP, balanced accuracy, recall, FPR, precision, F1, and TP/FP. The metric at FPR ≤ 5% is read from the test ROC. It is distinct from recall at the validation threshold.

The first run uses one seed and five masks per nonzero loss level. It measures paired effects on the existing cohorts. On Z24, one false positive is 1/6 of the healthy test set. The splits differ by evaluation unit across datasets. See the individual dataset READMEs.

Method references: [TimesNet](https://arxiv.org/abs/2210.02186), [multi-point random masking](https://doi.org/10.1016/j.eswa.2023.121506), and [frequency-distribution loss for SHM reconstruction](https://www.sciencedirect.com/science/article/pii/S0888327025008283).
