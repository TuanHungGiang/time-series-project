# Z24 healthy/damage overlap methods

## Fixed split and leakage boundary

- Train: 24 healthy records, comprising 12 S01 pre-lifting source records and
  12 S08 post-lifting target records.
- Validation: six healthy and 126 damage records.
- Locked test: six healthy and 126 damage records.
- Modal anchors and robust scaling use S01 healthy train only.
- S08 healthy train is used only for unsupervised domain alignment/mapping.
- No damage record contributes to feature fitting, gradient fitting,
  alignment, mapping, OCSVM fitting, or threshold estimation.
- S01--S07 are treated as the pre-lifting source domain; S08 and later states
  are treated as the post-lifting target domain. This follows Z24 chronology.

## Method 1: natural-frequency AE + Orthogonal Procrustes

Paper: Nogueira, Figueiredo and da Silva, *Mechanical Systems and Signal
Processing* 261 (2026) 115013, DOI `10.1016/j.ymssp.2026.115013`.

Reproduced paper operations:

1. train the source autoencoder on healthy natural-frequency observations;
2. encode healthy source and target observations;
3. solve `min_R ||Z_t R - Z_s||_F^2`, subject to `R^T R = I`;
4. decode aligned target latent vectors with the frozen source decoder;
5. use reconstruction MSE as the anomaly score.

The paper-reported optimizer settings available in the article are retained:
Adam, learning rate 0.003, maximum 600 epochs, batch size 16, dropout 0.2 and
MSE loss. The Z24 PDT adapter estimates fixed modal anchors from S01 train and
tracks local peaks per non-overlapping record window.

Two-epoch local smoke test (reduced window count; not a final experiment):

- locked-test balanced accuracy: 0.7500;
- AUROC: 0.7341;
- damage recall: 0.6667;
- specificity: 0.8333;
- F1: 0.7962.

## Method 2: target-to-source mapping + one-class SVM

Paper: Heravi et al., *Mechanical Systems and Signal Processing* 242 (2026)
113673, DOI `10.1016/j.ymssp.2025.113673`.

Reproduced paper operations:

1. learn a neural mapping from target healthy to source healthy features;
2. fit the one-class SVM only on source healthy features;
3. map target-domain observations before applying the source detector;
4. use the native OCSVM decision boundary without damage calibration.

The paper does not release author code or specify how unpaired source and
target observations are paired. The adapter therefore uses nearest-source
modal observations, computed using healthy train only. This is an explicit
reproduction gap, not an author-code claim.

Two-epoch local smoke test:

- locked-test balanced accuracy: 0.4365;
- AUROC: 0.3942;
- damage recall: 0.2063;
- specificity: 0.6667;
- F1: 0.3377.

The mapper result is retained as a negative result. It must not be tuned using
the locked test.

## Causal deconfounding blocker

Chen and Liu's MSSP 2026 causal deconfounding method requires synchronized
environmental observations (especially temperature) and a causal graph linking
environment, configuration, structural condition and measurements. The
current manifest contains setup/excitation metadata but no temperature or
environment observation per MAT record. Applying the method now would invent
an unobserved confounder, so it is not implemented in this suite.
