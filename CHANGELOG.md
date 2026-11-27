# Changelog

## Unreleased

- Added manuscript figures from saved CLMM scores and optional one-seed waveform reconstruction.

- Added a one-seed diagnostic for training and held-out healthy loss on both bridges.

- Added per-epoch training loss logs and Kaggle diagnostic figures for the matched baselines.

- Added a three-seed baseline pilot using the same 1,024-point inputs, nested sensor masks, and alarm calibration as CLMM whitening.

- Added the resumable ten-seed experiment for paired CLMM whitening on both bridges.

- Restored the selected baseline roster in the archived reporting table after removing USAD.

- Added a one-seed Z24 and HellBridge transfer check for the unchanged residual-whitening head.

- Added a HellBridge masked residual-whitening head, diagonal comparator, and protocol-labeled baseline comparison.

- Added a HellBridge within-window spectral consensus variant with paired low-FPR comparisons.

- Added a HellBridge low-FPR spectral scoring pilot with bounded adjustments and paired alarm diagnostics.

- Fixed initialization of the native TimesNet detector in the HellBridge spectral and gain-preserving pilots by explicitly disabling layer probes.

- Added the HellBridge gain-preserving spectral head with a clean-input gate and direct reuse of historical spectral scores.

- Added the 1024-point HellBridge spectral reconstruction pilot, six paired score heads, and healthy rank threshold comparisons.

- Added an optional per-sensor healthy normalization head and paired comparison with pooled reconstruction MSE on both bridges.

- Added a focused Z24 01-FVT audit that separates sensor selection, masked input, and adapter effects on reconstruction error.

- Added a two-bridge Q90 threshold audit over targeted and nested random masks, with healthy validation score and normalization diagnostics.

- Added a one-seed Kaggle comparison of adapted reconstruction, marginal gain, and conditional gain residuals on Z24 and HellBridge with separate healthy threshold calibration.

- Added a Kaggle scoring trial for graph Laplacian energy and gain variance at clean and cumulative 30/50% sensor loss on Z24 and HellBridge.

- Added layer and score diagnostics for the original CLMM gain head under the benchmark masks.

- Added a healthy-median centered gain-head ablation and a Kaggle launcher for the targeted masks.

- Display and export CNN-AE metrics as mean ± sample standard deviation across seeds.

- Added the optional one-dimensional CNN autoencoder baseline and a one-cell Kaggle runner using the targeted protocol.

- Added a one-cell Kaggle comparison of the post-block gain head and native TimesNet under the targeted baseline protocol.

- Save targeted baseline checkpoints before scoring and export reproducible figures, plot data and per-seed Kaggle reports.

- Added a resumable two-bridge baseline sweep for single and cumulative sensor removal, with per-seed metrics and scientific figures.

- Added a paired USAD transfer trial for the CLMM residual adapter and gain-based score on both bridges.

- Added a matched scoring-head trial using sensor-level reconstruction gains from the residual adapter, with paired controls and healthy-only gain references.

- Added a paired on/off audit for a learned residual adapter after TimesBlock 2, including class-wise score changes and shifts before and after LayerNorm.

- Added a center-only latent ablation and paired comparisons for the two-bridge TimesNet experiment.

- Added a two-bridge CLMM–TimesNet experiment for exact-mask latent normalization and observed spectral pair evidence, with paired ablations at 10/30/50% sensor loss.

- Added per-layer healthy–damage separation metrics to the native TimesNet diagnostic.

- Added native-signal exploratory analysis and sensor-pair plots for Z24 and HellBridge.

- Added a paired native-resolution ablation of sensor dropout and availability embedding on both bridges.

- Added a native 1024-point TimesNet diagnostic with no resampling for both bridges.

- Added a two-bridge TimesNet layer diagnostic and failure-mode figure.

- Added a controlled 128/256/512-point HellBridge resolution ablation under clean and 50% sensor-loss conditions.

- Added visual diagnostics for paired TimesNet and mask-aware TimesNet runs on Z24 and HellBridge.

- Added the installable CLMM–TimesNet pipeline and the two-bridge configuration.
- Added dataset provenance, a single-cell Kaggle notebook, and an experiment-first output layout.
- Registered the seven-variant clean and nested sensor-loss ablation. Real-data results are pending.
- Archived the earlier root-level experiments and Kaggle wrappers. The active CLMM–TimesNet package, notebook, and manuscript remain in the research layout.
