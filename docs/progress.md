# Progress

The backlog behind the [implementation plan](plan/implementation-plan.md), run as four short
sprints within the 2-hour budget. Each story has story points (SP) for its relative effort.
A story counts as done only when its acceptance criteria are met and the work is committed.

**Overall: 29 / 31 SP**

```
[███████████████████░] 94%
```

| Sprint | Goal | Progress |
|---|---|---|
| 1. Discovery | Understand the brief and data, agree a plan | `[██████████]` 7/7 SP |
| 2. Baseline deliverables | Working train + CV pipeline on the baseline | `[██████████]` 10/10 SP |
| 3. Encoder | Fine-tuned DeBERTa-v3 compared under the same CV (exploratory, after the freeze) | `[███████░░░]` 4/6 SP |
| 4. Ship | Freeze, single val run, README, QA | `[██████████]` 8/8 SP |

Status: ✅ done · 🔄 in progress · ⬜ to do

## Sprint 1: Discovery

| # | Story | SP | Status | Acceptance criteria |
|---|---|---|---|---|
| 1.1 | As the reviewer, I want the modelling, evaluation and stack choices researched | 3 | ✅ | `docs/plan/research.md` committed |
| 1.2 | As the developer, I want an implementation plan mapped to the brief | 2 | ✅ | `docs/plan/implementation-plan.md` committed |
| 1.3 | As the candidate, I want the plan reviewed (models/licences, data scientist, grader) | 2 | ✅ | Revisions recorded in plan §8 |

## Sprint 2: Baseline deliverables

| # | Story | SP | Status | Acceptance criteria |
|---|---|---|---|---|
| 2.1 | As the developer, I want a uv project skeleton | 1 | ✅ | `pyproject.toml`, `uv.lock`, src layout, ruff config |
| 2.2 | As the developer, I want validated data loading, CV splits and metrics | 3 | ✅ | `data.py`, `splits.py`, `metrics.py` |
| 2.3 | As the reviewer, I want `dhad-train` to read train.csv and write a checkpoint | 3 | ✅ | Baseline run: standard CV macro F1 0.828, held-out-vocab CV 0.741 (`results/baseline/`) |
| 2.4 | As the developer, I want fast unit tests | 2 | ✅ | 20 passed, 1 skipped (encoder test needs torch) |
| 2.5 | As the reviewer, I want the train-data figures in research.md to be reproducible | 1 | ✅ | `scripts/analyze_train.py` → `results/train_analysis.md` |

## Sprint 3: Encoder

| # | Story | SP | Status | Acceptance criteria |
|---|---|---|---|---|
| 3.1 | As the developer, I want a DeBERTa-v3 multi-label fine-tuning module | 3 | ✅ | `models/encoder.py` written (not yet run) |
| 3.2 | As the developer, I want the GPU environment ready | 1 | ✅ | torch 2.15.0 and transformers 5.21.1 on the RTX 5090 host |
| 3.3 | As the reviewer, I want the encoder compared with the baseline under both CV schemes | 2 | 🔄 | `results/encoder/cv_metrics.json` committed; hard stop 15:05 |

## Sprint 4: Ship

| # | Story | SP | Status | Acceptance criteria |
|---|---|---|---|---|
| 4.1 | As the reviewer, I want model selection frozen before val is opened | 1 | ✅ | `results/selection.json` committed and tagged `pre-val` |
| 4.2 | As the reviewer, I want an evaluation table on val.csv | 3 | ✅ | Baseline macro F1 0.805 [0.75, 0.86] against an all-positive floor of 0.524 |
| 4.3 | As the reviewer, I want a README of one page at most | 3 | ✅ | `README.md` committed |
| 4.4 | As the reviewer, I want clean, checked code | 1 | ✅ | All pass |

## Log

- **13:33** Task materials committed.
- **13:45** Research notes and plan committed.
- **13:55** Plan revised after three reviews: baseline-first ordering, single-seed encoder.
- **13:54–14:00** Baseline pipeline committed. On unseen words it drops from 0.83 to 0.74
  macro F1 (lexical load 0.72 → 0.59), which motivates the encoder.
- **14:04** GPU host dependencies installed; verification interrupted.
- **14:08** Change of plan: the GPU work was stopped, and the complete submission is shipped
  on the baseline first.
- **14:10** Model selection frozen (tag `pre-val`). Single val run: macro F1 0.805.
- **14:11** README, tests (parallel agent) and train analysis script (parallel agent) committed.
- **14:16** Encoder CV started on the GPU host, with a hard stop at 15:05. Any result is
  exploratory, because selection was frozen before it.
