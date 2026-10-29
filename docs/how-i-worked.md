# How I worked

- **Agent rules first.** [AGENTS.md](../AGENTS.md) sets hard constraints for the coding agents:
  - never use `val.csv` for any decision;
  - every number must come from a real run;
  - commit at least every 15 minutes, and never rewrite history.
- **Research in parallel.** Four agents covered modelling, evaluation, the engineering stack, and
  the training data (train only). Their findings are in [plan/research.md](plan/research.md).
- **Plan, then review, then code.** The [implementation plan](plan/implementation-plan.md) was
  reviewed by three more agents: a licence check, a data scientist, and a grader's view.
  - Accepted, in §8: build the deliverables on the baseline first; balance the held-out word
    groups; drop class reweighting; cut the seed ensemble.
  - Not adopted, for time: a word-frequency feature baseline.
- **Parallel implementation where files did not overlap.** The unit tests (`tests/`) and the
  train-analysis script (`scripts/`) were written by separate agents while I built the
  evaluation and the README. Agent output was checked before committing:
  - The test agent's `conftest.py` path workaround was replaced with a `pythonpath` setting in
    `pyproject.toml`.
  - The analysis agent reported that research.md had two figures swapped. Checking showed the
    agent had misread the file, so research.md was left unchanged.
- **Freeze before val.** Model and thresholds were committed and tagged `pre-val`, then val was
  run exactly once.
- **Encoder last, time-boxed.** It was trained on a personal RTX 5090 after the freeze
  ([encoder.md](encoder.md)). A NaN bug was diagnosed by inspecting the logits' dtype and fixed.
- **Tracking.** Progress is tracked as sprints with story points in [progress.md](progress.md).
  Commits use conventional prefixes and carry `Co-Authored-By` trailers.
