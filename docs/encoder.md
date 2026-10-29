# Encoder: DeBERTa-v3-base (exploratory)

The encoder was trained **after** model selection was frozen (tag `pre-val`). By design it was
never evaluated on `val.csv`, so it cannot replace the delivered baseline. All numbers below are
train cross-validation, from [`results/encoder/cv_metrics.json`](../results/encoder/cv_metrics.json).

| Train CV F1 | Model | lexical | syntactic | abstraction | macro |
|---|---|---|---|---|---|
| Standard | Baseline | 0.720 | 0.912 | 0.853 | **0.828** |
| Standard | DeBERTa | 0.685 | 0.912 | 0.852 | 0.816 |
| Held-out vocabulary | Baseline | 0.588 | 0.905 | 0.728 | 0.741 |
| Held-out vocabulary | DeBERTa | 0.559 | 0.905 | 0.853 | **0.772** |

## What the numbers show

- **On familiar words** the two models are tied. Standard CV favours the baseline by 0.01, which
  is within noise.
- **On unseen words the encoder wins**, by +0.031 macro F1. All of the gain is on abstraction
  (0.85 vs 0.73): the pretrained model knows that an unseen noun is abstract.
- **Lexical load does not improve** (0.56 vs 0.59). Fine-tuning on 480 rows does not recover
  word-rarity knowledge well; a word-frequency feature is the obvious next step.
- **Selection rule:** DeBERTa is 0.031 ahead on held-out-vocabulary CV, which clears the margin
  set before any results (> 0.02). It would therefore have been selected had it finished before
  the freeze.
- **Undertrained:** CV picked 10 epochs, the cap, and macro F1 was still rising at each epoch (see
  `candidate_scores_macro_f1`). A longer schedule or a higher learning rate is likely to help.

## Setup

- `microsoft/deberta-v3-base`, a three-logit multi-label head trained with BCE, and a plain
  PyTorch loop.
- Settings: AdamW lr 2e-5, weight decay 0.01, 10% warmup, batch size 16, max_len 96, fp32, seed 42.
- Thresholds tuned out-of-fold: 0.4 / 0.5 / 0.4.
- **Bug found and fixed:** transformers 5 loads DeBERTa-v3 in its stored fp16 dtype, and
  training with AdamW on fp16 weights diverged to NaN from epoch 1. The model is now loaded with
  `dtype=torch.float32` (`edcb880`).
- **Run record:** the run took 116 s on an RTX 5090. `results/encoder/config.json` records a stale
  `git_sha` (`0ed5061-dirty`), because the code was copied to the GPU host without its `.git`.
  The code that ran is commit `edcb880`.
- **Second bug, found by the behavioural tests:** `dhad-train` wrote its run `config.json` into
  the checkpoint directory, which overwrote the Hugging Face model `config.json`, so the encoder
  checkpoint could not be reloaded.
  - The run file is now `run_config.json`.
  - For the existing checkpoint, the architecture config was regenerated from the backbone and
    the training arguments (`num_labels=3`, multi-label, label names). The weights are unchanged.
  - The baseline was unaffected (its weights are `model.joblib`).

## Behavioural tests

See the README section and [`results/behavioural_tests.json`](../results/behavioural_tests.json).
- On unseen technical terms and abstract nouns, the encoder's probability shift is +0.65 / +0.74,
  against the baseline's +0.10 / +0.22.

## Known issues (found in final QC, not fixed in code)

- **The epoch schedule does not match between CV and refit.** CV trains each fold once with a
  10-epoch learning-rate schedule and scores it after every epoch. The refit then uses a schedule
  that decays over the *chosen* number of epochs. Had CV chosen fewer than 10 epochs, the refit
  would have used a different schedule from the one that was scored. This run chose 10, the cap,
  so the two match here; the cap also means the search was cut off rather than converged. Fix:
  keep the full-length schedule and stop early at the chosen epoch.
- **Reports are keyed by model kind.** `dhad-evaluate` and `scripts/behavioural_tests.py` name
  results by model kind, so two runs of the same kind would overwrite each other in the report.
