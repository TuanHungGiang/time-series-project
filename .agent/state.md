# Current research state

The active model is CLMM–TimesNet with observed-key attention and time–frequency training loss. The target is healthy-only bridge damage detection with clean input and nested random sensor loss of 10%, 30%, and 50%. The experiment uses seed 20270905 on both bridges.

The one-seed code and planned configuration are registered in `experiments/exp001_clmm_timesnet/`. Comparative real-data results remain pending. The older USAD experiments provide historical evidence under distinct score and calibration protocols.

The previous root-level workflows are pinned on `archive/pre-clmm-cleanup`. The manuscript baseline runners are referenced in `docs/en/legacy-workflows.md`.

The separate score-calibration diagnostic now renders two-bridge sensor-loss figures with waveform examples, all-mask metric ranges, score overlap, damage-state recall, and Q90 trade-offs. Run the score-calibration command with mask-aware training, then pass its output to the visualization module.
