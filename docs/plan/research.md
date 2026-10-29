# Research notes

Findings that inform the [implementation plan](implementation-plan.md).
Gathered on 2026-10-29 by four research agents working in parallel
(modelling, evaluation, engineering stack, training-data analysis).
`data/val.csv` was not analysed: only its row count and label totals were looked at.

## 1. Training data (`data/train.csv` only)

- 480 rows, two rigid templates:
  - **Simple** (318 rows): "We discuss X and Y. The text uses words like a, b, c, d, e. ..."
  - **Complex** (162 rows): "Although The analysis of X and Y unfolds across layered claims ...",
    in 3 clause orders, which make no difference to the labels.
- The variable content is a small set of slot words:
  - **Topic nouns:** 20 in total. 10 are concrete (cookie, river, ...) and 10 abstract
    (justice, entropy, ...). Both topics in a row always come from the same group.
  - **Word list:** 20 simple words and 10 technical terms (mitochondria, epistemological, ...).
    No slot word appears only once.
- **Each label is driven by one slot:**

  | Label | Driver | P(label=1 \| driver) vs P(label=1 \| not) |
  |---|---|---|
  | syntactic_complexity | complex template | 0.93 vs 0.05 |
  | conceptual_abstraction | abstract topics | 0.92 vs 0.10 |
  | lexical_load | at least one technical term | ≥0.90 vs 0.17 |

- **Label noise** is roughly 6–15% per label. It is random, and no other feature explains it.
  The best rule on these slots, measured in-sample, gives F1 of about 0.78 / 0.91 / 0.86
  (lexical / syntactic / abstraction). That is a rough ceiling for any model.
- **The labels are independent** of each other (|phi| < 0.05), and all 8 combinations occur.
- **No duplicate or conflicting texts.**

**What this means for the model**

- A regularised bag-of-words model should come close to the ceiling on rows that look like train.
- A more expressive model can only gain where words are **unseen**. A pretrained encoder
  "knows" that an unseen word is technical or abstract; TF-IDF does not.
- Random K-fold makes generalisation look better than it is, because every slot word appears
  in every fold. Testing on unseen words needs a CV that **holds out vocabulary**.
- Watch for shortcuts. "Terminology", "dense" and "careful rereading" mark the complex template
  and could become proxies for `lexical_load`.

## 2. Modelling

- **Baseline:** TF-IDF (word 1–2-grams + char_wb 2–5-grams, `sublinear_tf`) feeding
  one-vs-rest `LogisticRegression`. It trains in seconds and is a fair control.
- **Main model:** fine-tune `microsoft/deberta-v3-base` with a single 3-logit head,
  using `problem_type="multi_label_classification"` (BCEWithLogitsLoss) and a sigmoid at inference.
  - It is more sample-efficient than ModernBERT on NLU (arXiv 2504.08716).
  - `answerdotai/ModernBERT-base` is the fallback.
- **Rejected:**
  - Frozen mean-pooled sentence embeddings + LR: the template is about 80% of the tokens and
    drowns out the slot words.
  - SetFit: its advantage is at 8–64 examples per class, and it handles multi-label awkwardly.
  - MiniLM: too little lexical knowledge.
- **Hyperparameters:**
  - learning rate 2e-5, warmup 10% then linear decay, weight decay 0.01
  - batch size 16, max_len 96 (52 words is about 75 tokens)
  - number of epochs chosen by CV (8–10 at most)
  - bf16 on the GPU (DeBERTa-v3 overflows in fp16)
  - 3 seeds
- **Label noise:** light label smoothing (0.05), no training past the CV-chosen epoch count,
  and thresholds tuned out-of-fold.
- **Licences:** DeBERTa-v3 is MIT and ModernBERT is Apache-2.0. Confirm both on the model
  cards before citing them in the README.

## 3. Evaluation and quality assurance

- **CV:** `iterative-stratification` 0.1.9 (`MultilabelStratifiedKFold`) is maintained.
  scikit-multilearn's last release was in 2018, so it is not used.
  - Use 5 folds. Repeat 3× for the cheap baseline; run 1 pass per seed for the encoder.
- **Thresholds:** for each label, pick the threshold that maximises F1 on out-of-fold (OOF)
  predictions, from a coarse grid of 0.20–0.70 in steps of 0.05.
  - Fixed rule: keep 0.5 unless the tuned threshold improves OOF F1 by more than 0.01.
- **Metrics** (scikit-learn 1.9.1):
  - per-label precision, recall and F1; macro and micro F1
  - ROC-AUC and average precision (threshold-free)
  - Brier score (calibration)
  - subset accuracy and Hamming loss
  - predicted vs true prevalence
- **Uncertainty:** 95% CIs from a row-level bootstrap on val (2,000 resamples), plus a paired
  bootstrap for model differences. With 120 rows, F1 moves about ±0.05 by chance.
- **Reference baselines:**
  - `DummyClassifier(strategy="prior")`
  - all-positive: F1 = 2p/(1+p) ≈ 0.53, which is the real floor for F1
  - stratified random
- **Error analysis** (train OOF only): read the predictions that are confidently wrong.
  Optionally flag suspect labels with cleanlab, but do not relabel anything.
- **Reproducibility:**
  - seeds; `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG`
  - a run manifest recording the config, git SHA + dirty flag, library versions, the GPU,
    and data SHA-256 hashes

## 4. Engineering stack

- **Current versions** (PyPI, 2026-10-29): torch 2.15.0, transformers 5.21.1,
  scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, ruff 0.17.3, pytest 9.2.0.
  - numpy 2.5 requires Python ≥ 3.12.
  - transformers is on 5.x, so check APIs against the current docs.
- **PyTorch on the RTX 5090** (Blackwell, sm_120):
  - needs torch ≥ 2.7 built against CUDA ≥ 12.8
  - plain PyPI torch for Linux is now built on CUDA 13.x, which needs NVIDIA driver ≥ 580
  - with a recent driver, no custom index is needed, and macOS gets its normal wheel from PyPI
  - check with `torch.cuda.get_arch_list()`, which should contain `sm_120`
- **Layout:** `src/` package, built with `uv_build`.
  - `[project.scripts]` entry points for train and evaluate
  - `[dependency-groups] dev` for ruff/pytest
  - commit `uv.lock`
- **CLI:** argparse plus a frozen `@dataclass` config. No typer and no YAML, so no extra
  dependencies.
- **Code quality:** ruff lint + format, Google-style docstrings, type hints, and fast pytest
  tests that download nothing (the model test uses a tiny randomly initialised config).
- **Commits:** conventional commit prefixes, one logical change per commit, and no checkpoints.
