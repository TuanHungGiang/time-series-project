# Z24 supervised VCOS baseline result

## Protocol

- Method: original VCOS/MsFSVE equation reproduction, without proposed terms.
- Train: setups 1--5, 20 healthy and 140 damage records.
- Validation: setups 6--7, 8 healthy and 56 damage records.
- Test: setups 8--9, 8 healthy and 56 damage records.
- State 02 excluded; states 01 and 08 treated as healthy.
- Delta grid: 0.1 to 1.0 in increments of 0.1.
- Delta selected on validation balanced accuracy, macro-F1 and MCC.
- Final selector and classifier refitted on train plus validation.
- Locked-test metrics were inspected during development, so these results are a
  frozen comparative development baseline rather than pristine external proof.

## Validation selection

| classifier | delta | validation features | balanced accuracy | macro-F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CART | 0.9 | 13 | 0.5893 | 0.5990 | 0.2026 |
| Kernel NB | 0.9 | 13 | 0.5625 | 0.5817 | 0.3333 |
| KNN-3 | 1.0 | 3 | 0.5000 | 0.4667 | 0.0000 |
| Linear SVM | 1.0 | 3 | 0.5000 | 0.4667 | 0.0000 |

## Locked-test result

| classifier | delta | final features | balanced accuracy | macro-F1 | MCC | damage recall | specificity | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 1.0 | 3 | 0.5000 | 0.4667 | 0.0000 | 1.0000 | 0.0000 | 0.2991 | 0.8262 |
| Kernel NB | 0.9 | 12 | 0.4911 | 0.4622 | -0.0476 | 0.9821 | 0.0000 | 0.7478 | 0.9572 |
| CART | 0.9 | 12 | 0.4821 | 0.4828 | -0.0325 | 0.8393 | 0.1250 | 0.4821 | 0.8711 |
| KNN-3 | 1.0 | 3 | 0.4375 | 0.4336 | -0.1325 | 0.8750 | 0.0000 | 0.4230 | 0.8581 |

## Empirical gap exposed by the baseline

1. The original fuzzy consistency and FRCE objective are dominated by the
   87.5% damage majority. Delta values 0.1--0.7 produced the same 27-feature
   reduct, showing that coarse consistency did not distinguish the useful
   operating region.
2. Three classifiers produced zero healthy specificity; CART reached only
   0.125. High damage recall is therefore not evidence of useful detection.
3. Kernel NB AUROC (0.7478) and AUPRC (0.9572) show that its continuous score
   retains ranking information even though the empirical-prior decision rule
   labels every healthy test record as damage.
4. The result motivates class-balanced fuzzy-rough accuracy and setup-stable
   significance. It must not be repaired by test-tuned thresholds.

This result is the fixed original-VCOS baseline for the subsequent B-VCOS,
GS-VCOS and full GSB-VCOS ablation.

