# Miqyas al Dhad Readability Scale - Homework

Homework as part of the application procedure for Miqyas Al Dhad contractor by Bisharah Estephan. Expected time investment: 2 hours.

## Task

You receive a CSV dataset of short English passages. Each row has a `text` field and three **binary** difficulty-related flags (0/1):

| Field | Meaning (1 = present) |
|-------|------------------------|
| `lexical_load` | Uncommon or technical vocabulary for a general reader |
| `syntactic_complexity` | Long or heavily embedded sentence structure |
| `conceptual_abstraction` | Abstract ideas with little concrete anchoring |

Train a model that, given a passage, outputs **three probabilities** in `[0, 1]` (one per dimension). This is **multi-label classification** (not multi-class): dimensions can co-occur.

### Deliverables (2-hour budget)

A git repo that contains

1. **Runnable training code** (script or small package) that reads `train.csv` and writes a checkpoint or saves weights.
2. **Evaluation on `val.csv`** with a short table: per-dimension and macro-averaged **F1** (and whatever else you think matters).
3. **README** (≤1 page): how to install deps, train, and eval; plus **2–3 non-trivial design choices** you made (e.g. backbone, hyper parameters, quality assurance) and why.

### Constraints

- Use **local or free Colab** compute; no paid API training or inference.
- Do **not** hand-engineer regex rules on the validation set to produce labels.

### Out of scope

- No need for data cleaning. The data is synthetically generated and not very high quality. The goal is to gauge thought process and way of working, not to get a high quality model.

- Integration with an experiment tracking framework like W&B, as long as it is sufficiently clear how quality assurance choices would be made in a real project. 

### What we look for

- Ways of working (usage of git, coding agents and ML stack)
- Motivated technical choices


