# Dhad readability: multi-label difficulty classifier

Given a passage, returns three probabilities in [0, 1]: `lexical_load`, `syntactic_complexity`,
`conceptual_abstraction`. The brief is in [TASK.md](TASK.md).

## Results on `val.csv` (one run, after the freeze)

| Model | lexical | syntactic | abstraction | **macro F1** |
|---|---|---|---|---|
| TF-IDF + LogReg (delivered) | 0.684 [0.56, 0.79] | 0.842 [0.74, 0.92] | 0.889 [0.81, 0.95] | **0.805** [0.75, 0.86] |
| Always "present" (floor) | 0.545 | 0.500 | 0.528 | 0.524 |

- Brackets are 95% bootstrap CIs.
- Train CV macro F1 is 0.828, and 0.741 on unseen words.
- Full metrics, including AUC, average precision and Brier score:
  [`results/val_report.md`](results/val_report.md).

**Encoder (exploratory).** DeBERTa-v3-base was trained after the freeze and never run on val.
On unseen words it reaches CV macro F1 0.772 against the baseline's 0.741, so it would have won
under the pre-set rule. Details are in [docs/encoder.md](docs/encoder.md).

## Install, train, evaluate

```bash
uv sync                                          # Python ≥ 3.12; core deps only
uv run dhad-train --model baseline               # train.csv → checkpoints/baseline + results/baseline
uv run dhad-evaluate --runs checkpoints/baseline # val.csv → results/val_report.md
uv run pytest                                    # unit tests
```

- Scoring: `dhad.predict.predict_proba(texts, "checkpoints/baseline")` returns an `(n, 3)` array.
- Encoder: run `uv sync --extra neural`, then `uv run dhad-train --model encoder`.

## Design choices

1. **Test on unseen vocabulary, not only unseen rows.**
   - In train, each label depends on a few slot words (technical terms, abstract or concrete
     topics) or on the template ([analysis](results/train_analysis.md)). Random folds share every
     one of those words, so they reward memorisation.
   - `splits.heldout_vocab_folds` holds out groups of words, balanced by word type.
   - The baseline drops from 0.83 to 0.74 macro F1 on those folds. That gap is the case for a
     pretrained encoder: `microsoft/deberta-v3-base` (MIT), with `ModernBERT-base` (Apache-2.0) as
     the fallback.
2. **`val.csv` is touched once.**
   - C and the per-label thresholds are chosen on train out-of-fold predictions. A threshold moves
     away from 0.5 only if that gains more than 0.01 F1.
   - The choice is frozen in [`results/selection.json`](results/selection.json) and tagged
     `pre-val` before `dhad-evaluate`, the only reader of val, runs.
   - On 120 rows F1 varies by about ±0.06, so every val number comes with a CI.
3. **Independent, calibrated probabilities.**
   - The labels are uncorrelated in train (|phi| < 0.05), so each one gets its own logistic output
     (BCE for the encoder) rather than a joint model.
   - There is no class reweighting or label smoothing, so the probabilities stay calibrated. Brier
     and AUC are reported next to F1.

## Limitations

- **Syntactic complexity is learned from the template.** Within a template its F1 is 0.00
  (simple template, 8 positives) and 0.94 (complex template).
- **Noisy labels.** The labels are synthetic and noisy, which caps the reachable F1; the ceilings
  are in [the analysis](results/train_analysis.md).
- **Small val set.** On 120 rows, differences under about 0.05 F1 are not meaningful.

## How I worked

- Rules for the coding agents are in [AGENTS.md](AGENTS.md).
- Parallel agents did the research and reviewed the plan before any code was written. I then
  built the full set of deliverables on the baseline first, and trained the encoder on a personal
  RTX 5090 after that.
- The process is in [docs/how-i-worked.md](docs/how-i-worked.md), and the next steps, including
  transfer to Arabic, are in [docs/next-steps.md](docs/next-steps.md).
