# AGENTS.md

Rules for any coding agent working in this repository. They apply at all times
and take precedence over any plan or task instruction. If a plan asks for
something these rules forbid, stop and ask Bisharah.

This repo is a take-home assessment; the brief is in [`TASK.md`](TASK.md).
The implementation plan is provided separately. Do not write planning notes or
working documents into this repo unless asked.

## Assessment constraints (from the brief)

- Local compute only (or free Colab). No paid APIs for training or inference,
  and the delivered model must never call an LLM API.
- Never hand-craft rules or regexes from `data/val.csv` to produce labels.
- Never train on, tune on, or select models or thresholds using `data/val.csv`.
  It is the held-out evaluation set.

## Honesty

- Every metric, table or claim in the repo must come from an actual run.
  Never invent, estimate or round up a number.
- Report failures, regressions and shortcuts plainly. Do not hide them.
- Do not claim anything was tested or verified unless it was.

## Git

- Commit at least every 15 minutes while work is in progress, even unfinished
  work. A `post-commit` hook pushes each commit; if a push fails
  (`.git/push.log`), tell Bisharah right away.
- Create tags as annotated tags (`git tag -a NAME -m "..."`). The hook pushes
  annotated tags with the next commit; lightweight tags are never pushed.
- Never rewrite history: no `commit --amend` on pushed commits, no rebase,
  no `reset` of pushed commits, no force-push, no deleting branches or tags.
- Do not change the git remote, the hooks, repo visibility or git config.
- Never commit secrets, tokens, credentials or personal data.
- Never commit large files: model weights, checkpoints, caches, virtualenvs.
  Keep them in gitignored paths (`checkpoints/`, `.venv/`).

## Files and system

- Work only inside this repository. Do not read, write or delete anything outside it.
- Never modify `data/train.csv`, `data/val.csv` or `TASK.md`.
- No destructive commands: no `rm -rf` outside generated folders, no
  `git clean -fdx`, no `git reset --hard`, no `git checkout -- .` over
  uncommitted work. If something seems to need one, ask first.
- Install dependencies into the project environment only (`uv`, `.venv/`).
  No global installs, no `sudo`, no system configuration changes.
- Network use is limited to installing packages and downloading open model
  weights. Do not upload data or code anywhere.

## Outward actions

- Never send anything on Bisharah's behalf: no emails, messages, GitHub issues,
  PR comments or posts. Pushing to this repo's own remote via the hook is the
  only outward action allowed.

## When unsure

Stop and ask. A short question costs less than an action that is hard to undo.
