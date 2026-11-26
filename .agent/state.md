# Current research state

The active model is CLMM–TimesNet with observed-key attention and time–frequency training loss. The target is healthy-only bridge damage detection with clean input and nested random sensor loss of 10%, 30%, and 50%. The experiment uses seed 20270905 on both bridges.

The one-seed code and planned configuration are registered in `experiments/exp001_clmm_timesnet/`. Comparative real-data results remain pending. The older USAD experiments provide historical evidence under distinct score and calibration protocols.

The previous root-level workflows are pinned on `archive/pre-clmm-cleanup`. The manuscript baseline runners are referenced in `docs/en/legacy-workflows.md`.

The separate score-calibration diagnostic now renders two-bridge sensor-loss figures with waveform examples, all-mask metric ranges, score overlap, damage-state recall, and Q90 trade-offs. Run the score-calibration command with mask-aware training, then pass its output to the visualization module.

The HellBridge resolution diagnostic compares 128, 256, and 512 points per 1024-sample source window with identical splits and masks. Results are reported overall and by damage state.

The small TimesNet layer diagnostic reuses one healthy-fit checkpoint per group across native, period-anchored, uniform-period, and untrained cross-prediction scoring. It records activation shifts and renders failure-mode figures for Z24 and HellBridge.

The native-resolution TimesNet diagnostic preserves all 1024 detrended points and logs activation drift and FFT bin changes under sensor loss on both bridges.

The native 1024-point diagnostic now includes a three-branch ablation that isolates sensor dropout and learnable availability embedding under a common observed-channel loss and matched test masks.

The native-signal dataset EDA profiles Z24 by excitation and HellBridge by recording state. It exports waveform, Welch spectrum, surviving sensor-pair visuals, and record-level descriptive statistics without fitting a detector.

The native 1024-point runner now offers an optional fixed activation probe. It reports per-layer AUC and low-FPR recall changes without changing the original TimesNet model.

The CLMM mask-and-pair experiment adds a feature block after TimesBlock 2. It uses exact-mask healthy latent references and observed spectral ratios, with one anomaly score and separate ablations. The runner evaluates both bridges with clean and nested 10/30/50% sensor loss and reports clean-fixed and masked healthy calibration.

The latent ablation separates the effect of an exact-mask healthy center from the effect of an exact-mask scale. It keeps the clean scale for the center-only variant and exports paired comparisons against both latent variants and native TimesNet.

The post-TimesBlock 2 audit compares a healthy-trained residual adapter against the unchanged TimesNet checkpoint on both bridges. It measures changes before and after LayerNorm, reconstruction error by class, paired ranking flips, and the clean identity condition.

The post-block scoring trial keeps the fitted adapter and compares its original reconstruction score with a single score that also reads sensor-level reconstruction gains. Healthy reference masks fit the gain distribution. The existing score direction is shared by all three branches and the clean score is exactly preserved.

The USAD transfer trial uses the EURECOM encoder and two decoders with alternating training. It keeps the CLMM residual adapter on the first encoding and compares native scoring, adapter-only scoring, and the gain-based head under common masks on both bridges. All 1,024 native samples are scored in contiguous 128-point patches by default.

The targeted baseline benchmark reuses the seven source baselines from `codex/mdr-shm-kaggle` on both bridges. It reports fixed clean Q90 calibration, physical single-channel removals, and a cumulative sensor-removal order shared across methods and seeds. The order is estimated from held-out healthy PCA-SPE sensitivity and the curve stops with one channel per group.

The targeted runner saves inference-ready checkpoint weights, fit normalization, calibration and a score round-trip probe per group. It also exports raw and preprocessed examples, split and channel inventories, model geometry, window traces, score overlap inputs, ROC/PR points and confusion counts. The Kaggle cell partitions storage by seed and merges ten completed report shards under a common protocol.

The targeted gain-head runner ports the post-TimesBlock-2 residual adapter and gain head to the existing baseline split, input shapes and single/cumulative sensor masks. It shares a TimesNet backbone and frozen clean calibration across native, adapter-only and gain-head branches. Stored baseline weights and sensor orders may be reused.

The optional CNN-AE 1D baseline ports the official Keras convolutional architecture to multichannel PyTorch tensors. It retains Adam, MSE training and reconstruction MAE scoring. The targeted runner supplies the existing splits, corruption and healthy calibration.

The CNN-AE Kaggle cell checks coverage for each requested seed and displays mean ± sample SD for all reported metrics.

The centered gain-head ablation subtracts the healthy aggregate gain median under the exact evaluation mask and compares against the original gain head, the residual adapter and native TimesNet. It supports reusing the fitted backbone and adapter from the previous CLMM benchmark.
