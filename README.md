# Dhad readability: multi-label difficulty classifier

Given a passage, returns three probabilities in [0, 1]: `lexical_load`, `syntactic_complexity`,
`conceptual_abstraction`. The brief is in [TASK.md](TASK.md).

## Results on `val.csv` (one run, after the freeze)

| Model | lexical | syntactic | abstraction | **macro F1** |
|---|---|---|---|---|
| TF-IDF + LogReg (delivered) | 0.684 [0.56, 0.79] | 0.842 [0.74, 0.92] | 0.889 [0.81, 0.95] | **0.805** [0.75, 0.86] |
| Always "present" (floor) | 0.545 | 0.500 | 0.528 | 0.524 |

- Brackets are 95% bootstrap CIs.
- Train CV macro F1: **0.828** (standard CV) and **0.741** when the words are unseen (held-out
  vocabulary CV).
- The full report, with ROC-AUC, average precision, Brier, subset accuracy and Hamming loss, is
  in [`results/val_report.md`](results/val_report.md).

## Install, train, evaluate

```bash
uv sync                                          # Python ≥ 3.12; core deps only
uv run dhad-train --model baseline               # train.csv → checkpoints/baseline + results/baseline
uv run dhad-evaluate --runs checkpoints/baseline # val.csv → results/val_report.md
uv run pytest                                    # unit tests
```

- Scoring from Python: `dhad.predict.predict_proba(texts, "checkpoints/baseline")` returns an
  `(n, 3)` array.
- Encoder: `uv sync --extra neural`, then `uv run dhad-train --model encoder` (DeBERTa-v3-base on
  CUDA/MPS/CPU).

## Design choices

1. **Test on unseen vocabulary, not only on unseen rows.**
   - Every label in train depends on a few slot words: technical terms, abstract vs concrete
     topics, and the template ([analysis](results/train_analysis.md)). Every one of those words
     appears in every random fold, so standard CV only measures memorisation.
   - `splits.heldout_vocab_folds` holds out groups of words, balanced by word type. The model is
     then tested on rows containing words it never saw.
   - The baseline drops from 0.83 to 0.74 macro F1 there, and lexical load from 0.72 to 0.59.
     That gap is the real risk for a readability scorer, and it is the case for a pretrained
     encoder: `microsoft/deberta-v3-base` (MIT), fallback `ModernBERT-base` (Apache-2.0), with the
     licences checked on Hugging Face.
2. **`val.csv` is touched once.**
   - Hyperparameters (C) and per-label thresholds are chosen on train out-of-fold predictions.
     Thresholds must beat 0.5 by more than 0.01 F1 to move away from it; none did.
   - The choice is frozen in [`results/selection.json`](results/selection.json) and committed under
     the git tag `pre-val` before `dhad-evaluate`, the only code that reads val, runs. The val
     result commit comes after it.
   - With 120 rows, F1 has a CI of about ±0.06, so every number is reported with one.
3. **Independent probabilistic heads, kept calibrated.**
   - The labels are uncorrelated in train (|phi| < 0.05), so the model is one logistic output per
     label (one-vs-rest; BCE for the encoder) rather than a joint label model.
   - No class reweighting or label smoothing: the deliverable is a probability, so Brier and AUC
     are reported next to F1.

## Limitations, found by the checks above

- **Syntactic complexity is learned as "which template".** Within each template, its F1 is 0.00
  on the simple template (8 positives) and 0.94 on the complex one. The labels inside a template
  look like noise, and the model cannot see real sentence structure.
- **Label noise.** The labels are synthetic and noisy, so near-perfect scores are not reachable;
  the per-label ceilings are in [`results/train_analysis.md`](results/train_analysis.md).
- **Small validation set.** With 120 rows, differences under about 0.05 F1 between models are
  not meaningful.

## How I worked

- Rules for the coding agents are in [AGENTS.md](AGENTS.md). The main one is that val is never
  used for any decision.
- Parallel research agents covered models, evaluation, the stack and the data. Their findings
  are in [docs/plan/research.md](docs/plan/research.md).
- The [plan](docs/plan/implementation-plan.md) was reviewed by three further agents (licences,
  data scientist, grader) before any code was committed. I accepted most of their points and
  rejected some for time (§8). One example: the plan was reordered so that the full set of
  deliverables is built on the baseline first, with the GPU encoder after it.
- Progress was tracked in [docs/progress.md](docs/progress.md), with small conventional commits.
- The encoder is trained locally on a personal RTX 5090 workstation (no paid compute).

## Next steps, and transfer to Arabic

Swap in an Arabic encoder (CAMeLBERT-MSA, AraBERTv2) and turn the held-out-vocabulary CV into a
held-out-root CV. With more time: human-labelled ordinal data, frequency/concreteness features,
shortcut-perturbation tests. Details in [docs/next-steps.md](docs/next-steps.md).
