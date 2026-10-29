# Implementation plan

The plan for delivering [`TASK.md`](../../TASK.md). It builds on the [research notes](research.md).
Every rule in [`AGENTS.md`](../../AGENTS.md) takes precedence over this plan.

## 1. Deliverables, mapped one-to-one to the brief

| Brief asks for | Delivered as |
|---|---|
| Training code that reads `train.csv` and writes a checkpoint | `uv run dhad-train --model {baseline,encoder}` → `checkpoints/<run>/` |
| Model outputs three probabilities in [0, 1] | `dhad.predict.predict_proba(texts) -> np.ndarray (n, 3)`, plus a `dhad-predict "text"` CLI |
| Evaluation on `val.csv`: per-dimension and macro F1, "and whatever else matters" | `uv run dhad-evaluate` → `results/val_report.md` (table) + `results/val_metrics.json` |
| README of one page at most: install, train, eval, 2–3 non-trivial design choices | `README.md` |
| Local or free compute only, no paid APIs | Mac for development; a remote RTX 5090 workstation for the encoder (§5) |
| No regex rules written from val | No rules anywhere produce labels. Val is read only by `dhad-evaluate`. |

Nothing beyond this list is in scope: no W&B, no data cleaning, no hyperparameter sweeps.

## 2. Guiding decisions

These become the README's design choices.

1. **A pretrained encoder, tested against unseen vocabulary.**
   - In train, every label depends on a handful of slot words, and every slot word appears in every
     fold. A bag-of-words model can therefore look perfect and still fail on a passage with a new
     technical or abstract word.
   - Two models are trained: a TF-IDF + LR baseline and a fine-tuned `microsoft/deberta-v3-base`.
   - Both are compared under two cross-validation schemes:
     - **standard:** multi-label stratified 5-fold;
     - **held-out vocabulary:** the slot words are split into 5 groups. Fold *k* tests on the rows
       that contain group *k* words and trains only on rows that contain none of them.
   - The second scheme is the honest proxy for real readability scoring.
2. **One multi-label head trained with BCE; thresholds tuned out-of-fold.**
   - The labels are independent (|phi| < 0.05), so a single 3-logit sigmoid head is enough. No
     classifier chains or label-powerset models.
   - Each label's threshold is tuned on out-of-fold train predictions, with a rule fixed in
     advance: keep 0.5 unless tuning gains more than 0.01 F1.
3. **Val is touched exactly once.**
   - All selection uses train cross-validation.
   - The model choice and the thresholds are frozen in a commit before `dhad-evaluate` runs. The
     evaluation records that commit's SHA, so a reviewer can check the order.
   - Reported metrics carry bootstrap 95% CIs, because 120 rows give roughly ±0.05 F1 of noise.

**Model selection rule** (written down before any results exist):

- The delivered model is the one with the higher **macro F1 under held-out-vocabulary CV**.
- If the two are within 0.02, the simpler model (the baseline) wins.
- Both models, plus dummy baselines, still appear in the val table for transparency, with the
  delivered model clearly marked.

## 3. Repository layout

```
pyproject.toml            uv project; ruff/pytest config; entry points
uv.lock                   committed lockfile
README.md                 ≤1 page (deliverable)
TASK.md AGENTS.md CLAUDE.md
data/train.csv val.csv    unchanged
docs/plan/                these notes
src/dhad/
  __init__.py
  config.py               LABELS, frozen dataclasses: BaselineConfig, EncoderConfig, CVConfig
  data.py                 load_split(): schema checks, returns texts + (n, 3) int matrix
  splits.py               stratified_folds(), heldout_vocab_folds(), slot_vocabulary()
  metrics.py              compute_metrics(), tune_thresholds(), bootstrap_ci()
  models/baseline.py      TfidfLogReg: fit / predict_proba / save / load
  models/encoder.py       EncoderClassifier: plain PyTorch loop around HF model; save / load
  predict.py              load_model(run_dir) + predict_proba(); dhad-predict CLI
  train.py                dhad-train CLI: CV → thresholds → refit on all train → checkpoint
  evaluate.py             dhad-evaluate CLI: val metrics, CIs, report table
  utils.py                set_seed(), get_device(), write_manifest()
tests/                    fast unit tests; nothing is downloaded
results/                  committed small artefacts (JSON, CSV, MD)
checkpoints/              gitignored weights
```

### Dependencies

- **Core** (installed by `uv sync`, light enough for the Mac): scikit-learn, numpy, pandas,
  iterative-stratification, joblib.
- **`neural` extra** (`uv sync --extra neural`, only on the GPU host): torch ≥ 2.7, transformers,
  sentencepiece.
  - Torch is kept off the Mac because the Mac is short on disk.
  - Neural tests use `pytest.importorskip("torch")`, so the suite still passes on the Mac.
- **dev group:** ruff, pytest.
- **Python:** ≥ 3.12.

## 4. Component specifications

### `data.py`
- `load_split(path) -> Split(ids, texts, y)`.
- Fails loudly if columns are missing, labels are not in {0, 1}, or there are NaNs or duplicate ids.
- Nothing else. No cleaning (out of scope).

### `splits.py`
- `stratified_folds(y, n_splits=5, n_repeats, seed)`: wraps `RepeatedMultilabelStratifiedKFold`.
- `slot_vocabulary(texts)`: the candidate held-out words.
  - Derived from **train** document frequencies, for example tokens in 2%–15% of rows, which
    excludes template boilerplate and very common words.
  - The list is written to `results/heldout_vocab.json` so it can be inspected.
  - It is used only to build evaluation splits, never to produce labels or features.
- `heldout_vocab_folds(texts, vocab, n_splits=5, seed)`:
  - randomly partitions `vocab` into groups;
  - test fold = rows containing a word from the group;
  - train fold = rows containing no word from the group.
  - Rows that fit neither are dropped for that fold, and fold sizes are logged.

### `metrics.py`
- `compute_metrics(y, proba, thresholds) -> dict`:
  - per label: P/R/F1, ROC-AUC, AP, Brier, predicted vs true prevalence;
  - overall: macro and micro F1, subset accuracy, Hamming loss.
- `tune_thresholds(y, oof_proba, grid=0.20..0.70 step 0.05, min_gain=0.01)`.
- `bootstrap_ci(y, proba, thresholds, n=2000, seed=0)`:
  - row-level resampling;
  - skips a draw when a label has only one class (AUC is undefined there).
- `paired_bootstrap_diff(...)`: the macro-F1 difference between two models.

### `models/baseline.py`
- `FeatureUnion(word 1–2-gram TF-IDF, char_wb 2–5-gram TF-IDF)` →
  `OneVsRestClassifier(LogisticRegression(C, class_weight="balanced", max_iter=2000))`.
- C is chosen from {0.1, 0.3, 1, 3, 10} by standard-CV macro F1, inside train only.
- Saved with joblib.

### `models/encoder.py`
- `AutoModelForSequenceClassification(num_labels=3, problem_type="multi_label_classification")`.
- A **plain PyTorch training loop** (about 60 lines, fully commented), not `Trainer`:
  - no accelerate dependency;
  - nothing affected by transformers 5.x API changes;
  - every step can be read.
- Settings:
  - AdamW, lr 2e-5, weight decay 0.01, 10% linear warmup then linear decay
  - batch size 16, max_len 96, bf16 autocast on CUDA
  - BCE with label smoothing 0.05
- Epoch count:
  - CV runs train for 10 epochs and log OOF macro F1 after each epoch;
  - the final refit uses the epoch with the best mean CV score. No early stopping on held-out data.
- The delivered encoder is a **3-seed ensemble** (mean of probabilities), saved with
  `save_pretrained` per seed.
- `get_device()`: cuda → mps → cpu. The same code runs anywhere.

### `train.py` (`dhad-train`)

`--model {baseline,encoder} --cv {standard,heldout,both} --seeds 13 42 7 --out checkpoints/<run>`

1. Run CV and collect OOF probabilities.
2. Tune thresholds on the standard-CV OOF predictions.
3. Refit on all 480 rows.
4. Save weights + `thresholds.json` + `config.json` + `manifest.json` (git SHA, dirty flag,
   library versions, GPU name, SHA-256 of train.csv).
5. Copy the small CV outputs (`cv_metrics.json`, `oof.csv`) into `results/<run>/`.

### `evaluate.py` (`dhad-evaluate`)

`--runs checkpoints/baseline checkpoints/encoder --delivered encoder --out results/`

- The only code that reads `val.csv`.
- Adds the dummy baselines (prior, all-positive, stratified), fitted on train.
- Writes `val_metrics.json`, `val_predictions.csv` and `val_report.md`:
  - a main table of per-label F1, macro F1 and micro F1 with 95% CIs, at the tuned thresholds
    and at 0.5;
  - a secondary table of AUC, AP, Brier, subset accuracy and Hamming loss;
  - a slice: `syntactic_complexity` F1 per template, as a check for shortcut learning.
- Records the commit SHA the evaluation ran at.

### Tests (`tests/`, under 10 s on the Mac)
- `test_data.py`: schema, shapes, the dtype of the label matrix, rejection of bad labels.
- `test_metrics.py`:
  - hand-computed F1 on a toy example;
  - the threshold rule keeps 0.5 when the gain is below `min_gain`;
  - bootstrap CIs contain the point estimate.
- `test_splits.py`:
  - stratified folds partition the indices;
  - held-out-vocab test folds contain no held-out word in their train rows.
- `test_baseline.py`: fit/predict on 20 synthetic rows; output shape (n, 3) with values in [0, 1].
- `test_encoder.py` (skipped without torch):
  - a tiny randomly initialised config (no download), one training step;
  - logits shape (batch, 3); sigmoid output in [0, 1];
  - save/load round trip.

## 5. Compute: development on the Mac, encoder runs on the remote GPU

The encoder trains on a personal workstation with an **NVIDIA RTX 5090** (32 GB, Blackwell
sm_120), reached through a generic SSH alias, `gpu-host`. The baseline runs anywhere.

**Prerequisites on the GPU host**
- `nvidia-smi` works (the driver is loaded).
- The driver is ≥ 580, so the default PyPI torch (CUDA 13.x) runs. If it isn't, pin a cu128
  index for Linux in `pyproject.toml` (see research §4).
- Check with `torch.cuda.get_arch_list()`, which should contain `sm_120`.

**Loop**

```bash
# Mac: commit + push (the post-commit hook pushes)
ssh gpu-host
git clone <repo> dhad && cd dhad          # first time; afterwards: git pull
uv sync --extra neural
uv run pytest                              # full suite including encoder tests
tmux new -s dhad                           # survive SSH drops
uv run dhad-train --model encoder --cv both --out checkpoints/encoder
# Mac: pull back only the small artefacts, then commit them
rsync -av gpu-host:dhad/results/ results/
```

- Model weights and the HF cache stay on the GPU host (`checkpoints/`, `~/.cache/huggingface`).
  Nothing large crosses back.
- The final `dhad-evaluate` runs on the GPU host, where the checkpoints live.
- Its `results/` come back with rsync and are committed from the Mac.
- The baseline trains and evaluates on either machine.
- **Fallback if the GPU is unavailable:** run the encoder on the host's 32 CPU cores with 1 seed
  and the standard CV only, and say so in the README.

## 6. Execution order and commits

There is a 2-hour budget. Commit at least every 15 minutes, using conventional prefixes.

| # | Step | ~min | Commit |
|---|---|---|---|
| 0 | Initial commit of brief, data, agent rules, these docs | 5 | `chore: initial project files` / `docs: research notes and implementation plan` |
| 1 | `pyproject.toml`, src layout, ruff/pytest config, `.gitignore` additions, `uv.lock` | 10 | `chore: scaffold uv project` |
| 2 | `data.py`, `metrics.py`, `splits.py` + tests | 25 | `feat: data loading, metrics and CV splits` |
| 3 | Baseline + `dhad-train` (baseline path); run standard + held-out CV on the Mac | 20 | `feat: tf-idf baseline and training CLI`, `results: baseline CV` |
| 4 | Encoder module + tests; GPU-host setup; run CV (both schemes, 3 seeds) and final refit | 30 | `feat: deberta-v3 multi-label encoder`, `results: encoder CV` |
| 5 | Apply the selection rule; freeze thresholds and the delivered model in `config.json` | 5 | `chore: freeze model selection before val` |
| 6 | `dhad-evaluate` + `dhad-predict`; **single** val run | 10 | `feat: evaluation and prediction CLIs`, `results: val evaluation` |
| 7 | README (≤1 page): install / train / eval / 3 design choices / results table / limitations | 15 | `docs: README` |
| 8 | QA: `ruff check`, `ruff format --check`, `pytest`; fresh clone + `uv sync` + reproduce baseline | 10 | `chore: final checks` |

If time runs short, cut in this order:
1. the 3-repeat baseline CV;
2. the encoder seeds, 3 → 1;
3. the held-out-vocab CV for the encoder.

Never cut the single-use val discipline, the tests or the README.

## 7. Standards and documentation

- **Code:**
  - type hints on every public function;
  - Google-style docstrings explaining *why* and not only *what*;
  - no unexplained magic numbers (they live in config dataclasses);
  - `ruff` lint + format clean.
- **Honesty:**
  - every number in the README comes from a committed `results/` file produced by a real run;
  - failures and regressions are reported, not tuned away;
  - if the encoder does not beat the baseline, the README says so and ships the baseline.
- **Reproducibility:**
  - `uv.lock`;
  - seeds and deterministic flags;
  - a run manifest per checkpoint;
  - the README's commands reproduce every result from a fresh clone.
- **Limitations the README must state:**
  - synthetic, noisy labels (estimated ceiling about 0.78 / 0.91 / 0.86 F1);
  - 120-row val, so CIs are wide;
  - template shortcut learning for `syntactic_complexity`;
  - held-out-vocab CV only approximates real unseen text.

## 8. Revisions after review (2026-10-29, 13:55)

Three review agents read this plan: a model and licence check, a data scientist / readability
expert, and a grader's-eye review. Accepted changes, which override the sections above:

- **Models confirmed and licences verified** on the Hugging Face API:
  - `microsoft/deberta-v3-base` (MIT) is the primary encoder;
  - `answerdotai/ModernBERT-base` (Apache-2.0) is the fallback.
  - `deberta-v3-large` was rejected: it is unstable on 480 rows.
- **Reordered.** The full set of deliverables is built on the baseline first: CV, frozen
  thresholds, val, README. The encoder then gets a fixed time box. If it is not done in time,
  the baseline ships and the README says so.
- **Held-out-vocab CV:**
  - each word group is balanced by word type, using the label each word is most associated with
    in train;
  - results are reported as the mean of the per-fold metrics, because a row can sit in several
    test folds;
  - it is used for **model selection only**. Thresholds come from standard CV.
- **Probabilities left undistorted:** no `class_weight="balanced"` and no label smoothing,
  because the brief asks for probabilities and Brier score is reported.
- **Floor per label:** the all-positive F1 floor is computed per label, not as one figure.
- **A train-only analysis script** (`scripts/analyze_train.py`) is committed, so the figures in
  research.md can be reproduced.
- **Cut:**
  - the encoder seed ensemble (1 seed);
  - the paired bootstrap, cleanlab, the prior/stratified dummies (all-positive is kept);
  - the repeated baseline CV (one 5-fold pass, C in {1, 3, 10});
  - deterministic-algorithm flags (seed + git SHA + data hash are kept);
  - the `dhad-predict` CLI (`predict_proba` covers the brief);
  - the CPU fallback;
  - the fresh-clone check (ruff + pytest are kept).
- **Additions:**
  - a `pre-val` git tag on the freeze commit;
  - a short README "How I worked" section;
  - a note on transferring this to Arabic;
  - a "more time" list.
- **Considered, not adopted (time):** a `wordfreq` / concreteness feature baseline. It is listed
  under "more time".
