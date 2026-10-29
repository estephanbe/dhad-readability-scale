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

## Behavioural tests: what did each model actually learn?

F1 cannot tell you *why* a model is right. [`scripts/behavioural_tests.py`](scripts/behavioural_tests.py)
builds 200 minimal pairs from the training templates and changes **one factor** at a time. It
uses words seen in train and words that **never occur in train** (this is checked). Each cell is
the mean change in predicted probability (val is not touched):

| Edit | Model | lexical | syntactic | abstraction |
|---|---|---|---|---|
| Add a technical term, seen in train | baseline / encoder | **+0.39** / **+0.69** | +0.01 / −0.02 | +0.03 / +0.01 |
| Add a technical term, **unseen** (*isotope*, *jurisprudence*) | baseline / encoder | +0.10 / **+0.65** | +0.00 / −0.03 | +0.01 / +0.00 |
| Abstract topics, seen in train | baseline / encoder | −0.03 / +0.02 | −0.00 / −0.01 | **+0.64** / **+0.76** |
| Abstract topics, **unseen** (*morality*, *virtue*) | baseline / encoder | −0.00 / +0.00 | +0.01 / +0.00 | +0.22 / **+0.74** |
| Same words, complex template | baseline / encoder | +0.10 / +0.17 | **+0.89** / **+0.89** | +0.04 / −0.02 |

- **Both models respond to the right factor.** Each edit moves its own label and leaves the
  other two almost flat.
- **Only the encoder generalises.** With unseen words the baseline moves only +0.10 and +0.22,
  while the encoder keeps +0.65 and +0.74. This is the held-out-vocabulary result, shown one
  passage at a time.
- **A shortcut both models share:** the complex template alone raises `lexical_load`, by +0.10 and
  +0.17. It contains the word "Terminology".

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
  (simple template, 8 positives) and 0.94 (complex template). The template also leaks into
  `lexical_load`, as the behavioural tests show.
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
