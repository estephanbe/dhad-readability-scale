# Dhad readability: multi-label difficulty classifier

Given a passage, returns three probabilities in [0, 1]: `lexical_load`, `syntactic_complexity`,
`conceptual_abstraction`. The brief is in [TASK.md](TASK.md).

## Results on `val.csv` (one run, after the freeze)

| Model | lexical | syntactic | abstraction | **macro F1** |
|---|---|---|---|---|
| TF-IDF + LogReg (delivered) | 0.684 [0.56, 0.79] | 0.842 [0.74, 0.92] | 0.889 [0.81, 0.95] | **0.805** [0.75, 0.86] |
| Always "present" (floor) | 0.545 | 0.500 | 0.528 | 0.524 |

- Brackets are 95% bootstrap CIs.
- Train CV macro F1 is 0.828, and 0.741 on held-out vocabulary, i.e. passages whose topic or
  technical words were never seen in training (design choice 1).
- Full metrics: [`results/val_report.md`](results/val_report.md).
- **Encoder (exploratory):** DeBERTa-v3-base was trained after the freeze and never run on val.
  On held-out vocabulary its train CV macro F1 is 0.772. That clears the pre-set selection margin
  on train CV ([docs/encoder.md](docs/encoder.md)), but it has no val score.

## Behavioural tests: what did each model actually learn?

F1 cannot show *why* a model is right. [`scripts/behavioural_tests.py`](scripts/behavioural_tests.py)
changes **one factor** in 200 minimal pairs built from the training templates. It uses words seen in
train and words that **never occur in train**, which is checked. The table shows the mean change in
the targeted label's probability. Val is not touched.

| Edit (one factor) | Target | Baseline | Encoder |
|---|---|---|---|
| Add a technical term, seen in train | lexical | +0.39 | +0.69 |
| Add a technical term, **unseen** (*isotope*, *jurisprudence*) | lexical | +0.10 | **+0.65** |
| Abstract topics, seen in train | abstraction | +0.64 | +0.76 |
| Abstract topics, **unseen** (*morality*, *virtue*) | abstraction | +0.22 | **+0.74** |
| Same words, complex template | syntactic | +0.89 | +0.89 |

- **Both models respond to the right factor.** Off-target shifts are at most 0.04, with one
  exception (see the last point).
- **Only the encoder generalises to new words.** This is the held-out-vocabulary result, seen
  one passage at a time.
- **A shared shortcut:** the complex template alone raises `lexical_load` (+0.10 baseline, +0.17
  encoder). That template contains the word "Terminology".

Full results: [`results/behavioural_tests.json`](results/behavioural_tests.json). To rerun:
`uv run python scripts/behavioural_tests.py --runs checkpoints/baseline [checkpoints/encoder]`.
The encoder column needs the `neural` extra and a trained encoder checkpoint.

## Install, train, evaluate

```bash
uv sync                                          # Python ≥ 3.12; core deps only
uv run dhad-train --model baseline               # train.csv → checkpoints/baseline + results/baseline
uv run dhad-evaluate --runs checkpoints/baseline # val.csv → results/val_report.md
uv run pytest                                    # unit tests
```

- Scoring from Python: `dhad.predict.predict_proba(texts, "checkpoints/baseline")`.
- Encoder: run `uv sync --extra neural`, then `uv run dhad-train --model encoder`.

## Design choices

1. **Test on unseen vocabulary, not only unseen rows.** Each label depends on a few slot words
   or on the template ([analysis](results/train_analysis.md)). Random folds share all of those
   words. `splits.heldout_vocab_folds` holds out groups of words instead, and the baseline drops
   from 0.83 to 0.74. That gap is the case for a pretrained encoder: `microsoft/deberta-v3-base`
   (MIT), with `ModernBERT-base` (Apache-2.0) as the fallback. It finished after the freeze, so
   the baseline ships.
2. **`val.csv` is touched once.** C and the thresholds are chosen on train out-of-fold
   predictions; a threshold leaves 0.5 only if that gains more than 0.01 F1. The choice is frozen
   in [`results/selection.json`](results/selection.json) and tagged `pre-val` before
   `dhad-evaluate` runs. Every val number comes with a CI, because 120 rows give about ±0.06.
3. **Independent, calibrated probabilities.** The labels are uncorrelated (|phi| < 0.05), so
   each gets its own logistic output (BCE for the encoder). There is no reweighting or label
   smoothing to distort the probabilities, and Brier and AUC are reported next to F1.

## Limitations

- **Syntactic complexity is learned from the template.** Its F1 within the simple template
  (8 positives) is 0.00, and within the complex template 0.94. The template also leaks into
  lexical load.
- **Synthetic, noisy labels** cap the reachable F1 ([ceilings](results/train_analysis.md)).
- **Small val set:** with 120 rows, gaps under about 0.05 F1 are noise.

## How I worked

Rules for the coding agents are in [AGENTS.md](AGENTS.md). Parallel agents did the research and
reviewed the plan before any code was written. I built the deliverables on the baseline first,
then trained the encoder on a personal RTX 5090. More in [docs/how-i-worked.md](docs/how-i-worked.md);
next steps, including transfer to Arabic, in [docs/next-steps.md](docs/next-steps.md).
