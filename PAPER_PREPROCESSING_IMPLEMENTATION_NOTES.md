# Elsevier paper-preprocessing implementation audit

This document records what was reproduced, what had to be adapted for the Z24
progressive-damage-test files, and the dead ends found during implementation.
It is intended to prevent equation reproductions from being described as exact
author implementations in a paper.

## Shared evaluation protocol

- Training: state 01, setups 1--6, 12 healthy MAT records.
- Healthy calibration and early stopping: state 01, setups 7--9, 6 records.
- Damage validation: states 04, 05, 06, 10, 14, 16 and 17.
- Locked test: state 08 healthy plus damage states 03, 07, 09, 11, 12, 13
  and 15.
- Splitting is performed before windowing. No window from one MAT record can
  occur in more than one split.
- Damage validation is not used in gradient fitting, scaling or threshold
  estimation. It is an opened model-development set and must be reported as
  such. The locked test is evaluated only after the pipeline is fixed.

## MDFA preprocessing and feature fusion

Source: Mousavi et al., *Information Fusion* 126 (2026) 103534,
DOI `10.1016/j.inffus.2025.103534`.

Reproduced components:

- linear least-squares detrending;
- Welch power-band estimation and a fourth-order Butterworth low-pass filter;
- separate detrended-raw, time, frequency and three-level db4 wavelet streams;
- RMS, crest factor and Shannon entropy;
- spectral centroid, spread and kurtosis plus normalized log-Welch shape;
- db4 energy ratios;
- one encoder per domain, latent fusion and multi-domain reconstruction loss;
- healthy-only training and reconstruction-error damage index.

Adaptation required by Z24:

- The paper has a fixed sensor array. Z24 PDT records contain 15--17 roving
  vertical sensors, so a fixed all-sensor fusion tensor cannot be constructed
  without invented padding. The implementation treats each sensor-window as a
  sample and aggregates sensor-window errors at record level.
- The paper's 200 Hz cutoff is bridge-specific. Here the cutoff is re-estimated
  only from healthy training data using its stated 90%-power Welch rule and is
  constrained to the measured Z24 band.

No public author repository was found. The runner is a workflow/equation
reproduction, not author code.

## Time-Vertex sensor selection

Source: Niresi et al., *Reliability Engineering & System Safety* 270 (2026)
112153, DOI `10.1016/j.ress.2025.112153`.

Reproduced components:

- statistical and spectral descriptors for every sensor;
- feature-space clustering to reduce redundant sensors;
- graph-similarity adjacency and central representative selection;
- healthy-only autoencoder on the selected sensor set;
- threshold equal to healthy-validation mean plus two standard deviations.

Adaptation required by Z24:

- A separate healthy graph is learned for every `AVT/FVT x setup` because the
  Z24 PDT does not contain one fixed sensor graph.
- Sensor selection is stored by physical channel name, not array index.
- A real schema mismatch was discovered during smoke testing: S01 FVT setup 09
  has 17 vertical channels, while S09 FVT setup 09 has 15; `99V` and `512V`
  are absent. Zero padding would introduce a test-state shortcut. Instead, the
  implementation substitutes the next available sensor in the healthy-learned
  graph-centrality ranking and writes every substitution to
  `sensor_selection_audit.json`.

Dead end:

- The first author's publication page exposes a `Code` link, but it resolves
  to `EPFL-IMOS/PEGNN_SS`. Inspection showed that repository belongs to the
  different paper *Physics-Enhanced Graph Neural Networks For Soft Sensing in
  Industrial Internet of Things*. It was therefore rejected and is not cited
  as TVML author code.

## EdgeConvFormer and Tail-p

Source: Liu et al., *Reliability Engineering & System Safety* 252 (2024)
110465, DOI `10.1016/j.ress.2024.110465`.

The existing `run_z24_graph_transformer_tailp.py` reproduces moving-window
preprocessing, dynamic nearest-neighbour graph construction, Transformer
temporal encoding and empirical Tail-p scoring. Its Z24 graph is inferred from
spectral similarity because a common physical node topology does not exist
across the roving setups. No public author repository was found.

## Smoke-test status

Both new runners completed one-epoch end-to-end tests over all 288 required
records using the local manifest. Those smoke metrics are implementation tests,
not scientific results, and are excluded from version control. Full Kaggle
runs must use the default feature resolution and a meaningful epoch budget.

