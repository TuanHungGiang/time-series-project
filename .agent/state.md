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

The original gain-head diagnostic reports the pre-centering CLMM model alone at clean and the nearest cumulative 10/30/50% sensor loss. It logs matched-healthy separation at each layer, FFT changes and adapter activation shifts before and after LayerNorm.

The Laplacian score trial compares the original CLMM head, adapter reconstruction, clipped gain, gain variance, and graph energy on the same Z24 and HellBridge test cohorts. It uses seed 20270901 and the nearest cumulative masks at 30% and 50% loss. The graph uses absolute sensor correlations from healthy training data. Exact-mask healthy reference windows center the gain and normalize variance and graph energy. Real-data results for the new head are pending.

The conditional gain pilot compares mask-normalized adapted reconstruction, reconstruction with marginal gain, and reconstruction with gain residualized against the reconstruction error. It uses seed 20270901 on Z24 and HellBridge at clean and the nearest cumulative masks to 30% and 50%. The two gain terms use the same signed gain features and regularized covariance scoring. Head fitting uses healthy reference windows. HellBridge separates 12 of its 20 healthy development windows for head fitting and eight for threshold calibration. The test cohorts and TimesNet checkpoints follow the previous targeted benchmark.

The threshold-mask audit holds the adapted reconstruction score and test cohorts fixed while comparing clean-frozen and mask-specific healthy Q90 thresholds. It uses seed 20270901, one cumulative PCA-sensitivity order and five nested random mask orders at matched 30/50% removal counts per group on both bridges. It also exports raw reconstruction errors, healthy reference scaling, and validation ranks for Z24 record z24_s08_fvt_setup01.

The Z24 01-FVT audit traces the healthy validation record z24_s08_fvt_setup01 under the two prior targeted masks. Full-input reconstruction on all and retained channels isolates score composition. Paired masked reconstruction with the same checkpoint and the adapter off or on isolates the additional effects of input masking and adaptation. The audit exports raw errors and normalized lower-tail scores on validation and uses the unchanged healthy-window reference.

The per-sensor scoring pilot adds a normalize-before-pooling reconstruction head to the threshold-mask audit. It fits healthy log-MSE medians and robust scales independently for observed sensors at each exact mask. Both heads share forward passes, the clean-validation score direction, evaluation cohorts and Q90 policies. The pilot exports paired differences, per-sensor references and a score/rank trace of the healthy Z24 01-FVT record.

The HellBridge-specific spectral pilot returns to the 1024-point P2_NM_Z split with 50 healthy training, 10 healthy reference and 20 healthy calibration windows. It compares native TimesNet, the post-block adapter, the original gain head, two spectrum-only controls, and a combined time/spectral reconstruction head on the historical five nested masks. Clean and exact-mask calibration policies use 5% and 10% nominal healthy rank targets. Shared healthy source recordings limit the interpretation of calibration guarantees.

The HellBridge gain-preserving pilot adds the same observed-spectrum reconstruction term to the original adapter-and-gain score under sensor loss. The new term is gated to zero for complete input. The runner can reuse scores from the native-resolution spectral pilot and compares the complete method with the original gain head and a spectral variant with the same gating. Threshold comparisons use separate healthy validation windows with both clean and matched-mask calibration.

The HellBridge spectral runner now explicitly disables the optional TimesNet layer probe during detector initialization. A synthetic PyTorch smoke test exercised detector initialization, adapter fitting, and spectral scoring with clean and masked windows.

The HellBridge low-FPR pilot compares the original gain head and unbounded spectral addition with signed and positive-only bounded spectral adjustments. Both bounded scores use a fixed 0.25 cap on the original CLMM scale and reuse the stored 1024-point model scores when available. The runner exports paired alarm changes, per-state recall and standardized partial AUC up to 5% FPR. Test ROC thresholds are diagnostic, while operating thresholds are set on healthy validation.

The HellBridge within-window consensus pilot computes band-distribution differences separately in the three Welch segments of each native 1024-point window. It takes each retained sensor's median segment discrepancy before pooling sensors. The comparison holds the reference split, masks, reconstructions and bounded additive score fixed to test whether persistent segment evidence improves recall at low FPR.

The HellBridge residual-whitening pilot replaces pooled reconstruction MSE with healthy cross-sensor-whitened residual energy while retaining the previous gain term. It estimates one regularized residual correlation matrix per mask on 50 masked healthy fit windows. The 10 separate reference windows scale scores and 20 healthy validation windows calibrate the threshold. The diagonal comparator tests whether sensor correlations improve over per-sensor scaling. An archived baseline table records provenance and protocol for each row. The pilot compares paired native 1024-point scores and reports historical baselines with their original protocols. The first Kaggle GPU seed improved HellBridge diagnostic recall at 5% FPR from 0.4016 to 0.4738 under 50% loss, with five nested masks.

The two-bridge whitening pilot holds the head, gain weight, residual shrinkage and native 1024-point backbone configuration fixed. It fits separate parameters on each bridge’s healthy train split. Z24 chooses the shared score tail from labeled clean validation and takes the median score over windows within each record. HellBridge uses the existing upper-tail and healthy calibration protocol. Full paired and archived baseline tables preserve their distinct protocols.

The active comparison uses the retained baseline roster. It reuses archived scores for PCA-SPE, LSTM-AE, TranAD, TimesNet, ModernTCN, ScatterAD, and CNN-AE 1D. Historical results keep their original protocols. USAD is excluded from the reporting roster.

The whitening seed sweep uses the packaged two-bridge pilot with the original hyperparameters and mask IDs 300–304. The ten-seed roster begins at 20270905. Each seed exports paired per-mask measurements for both threshold policies. Results are averaged over masks before aggregation over training seeds. Paired percentile intervals resample seeds on the fixed test split and mask set. Portable completed-seed checkpoints can be merged when their dataset fingerprints and protocol match. Multi-seed real-data results are pending.

The matched three-seed baseline pilot reuses the native whitening split, 1,024-point preprocessing, and masks 300–304 on both bridges. It reports the six neural baselines and a deterministic PCA-SPE run with mask-matched and clean-frozen calibration. Results are pending.

The Kaggle baseline launcher logs training loss at each epoch and saves per-seed training curves. Archived notebook logs support plotting the recorded epochs only.

The one-seed baseline diagnostic repeats the matched protocol and evaluates each training objective on held-out healthy reference windows after every epoch. The alarm calibration and test partitions remain separate.
