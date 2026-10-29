# Validation results (`data/val.csv`, n = 120)

Delivered model: **baseline**. Thresholds were frozen on train CV before this run.
Brackets: 95% bootstrap CI (2,000 row resamples).

## F1

| Model | lexical_load | syntactic_complexity | conceptual_abstraction | macro F1 | micro F1 |
|---|---|---|---|---|---|
| **baseline** | 0.684 [0.56, 0.79] | 0.842 [0.74, 0.92] | 0.889 [0.81, 0.95] | 0.805 [0.75, 0.86] | 0.807 |
| baseline @ 0.5 | 0.684 | 0.842 | 0.889 | 0.805 | 0.807 |
| all-positive floor | 0.545 [0.45, 0.63] | 0.500 [0.40, 0.59] | 0.528 [0.43, 0.61] | 0.524 [0.47, 0.58] | 0.525 |

## Threshold-free and calibration metrics

| Model | Metric | lexical_load | syntactic_complexity | conceptual_abstraction |
|---|---|---|---|---|
| baseline | roc_auc | 0.812 | 0.885 | 0.906 |
| baseline | avg_precision | 0.781 | 0.853 | 0.881 |
| baseline | brier | 0.155 | 0.093 | 0.083 |
| baseline | subset acc / Hamming | 0.658 / 0.125 | | |

## Shortcut check: `syntactic_complexity` F1 within each template

If the model only learned the template, it cannot separate labels *within* one.

| Model | Template | n | positives | F1 |
|---|---|---|---|---|
| baseline | simple | 84 | 8 | 0.000 |
| baseline | complex | 36 | 32 | 0.941 |
