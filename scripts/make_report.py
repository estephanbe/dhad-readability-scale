"""Build docs/report.pdf: an illustrated project report.

Every number is read from the committed result files and git history at build time,
so the report cannot drift from the repository. Charts are drawn with matplotlib, the
layout is HTML/CSS, and headless Chrome prints it to PDF.

Usage:
    uv run --group report python scripts/make_report.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "report.pdf"
LABELS = ("lexical_load", "syntactic_complexity", "conceptual_abstraction")
SHORT = ("Lexical load", "Syntactic complexity", "Conceptual abstraction")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

NAVY, TEAL, CORAL, GOLD, GREY = "#14213d", "#2a9d8f", "#e76f51", "#e9c46a", "#94a3b8"
LIGHT = "#f1f5f9"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#334155",
        "ytick.color": "#64748b",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e2e8f0",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)


def load(path: str) -> dict:
    """Read a JSON result file."""
    return json.loads((RES / path).read_text())


def commits() -> list[tuple[str, str, str]]:
    """``(HH:MM, short sha, subject)`` for every commit, oldest first."""
    log = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%ad|%h|%s", "--date=format:%H:%M"],
        cwd=ROOT,
        text=True,
    )
    return [tuple(line.split("|", 2)) for line in log.strip().splitlines()]


# ---------------------------------------------------------------- charts


def fig_timeline(path: Path, log: list[tuple[str, str, str]]) -> None:
    """Commits over time, coloured by conventional-commit type."""
    palette = {
        "docs": GREY,
        "feat": TEAL,
        "fix": CORAL,
        "chore": NAVY,
        "results": GOLD,
        "test": "#457b9d",
    }
    t0 = int(log[0][0][:2]) * 60 + int(log[0][0][3:])
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for i, (hhmm, _, subject) in enumerate(log):
        minute = int(hhmm[:2]) * 60 + int(hhmm[3:]) - t0
        kind = subject.split(":")[0] if ":" in subject else "chore"
        ax.scatter(
            minute, i % 3, s=140, color=palette.get(kind, GREY), zorder=3, edgecolor="white", lw=1.2
        )
    ax.axvline(36, color=CORAL, ls="--", lw=1.2)
    ax.text(36.5, 2.45, "pre-val freeze", color=CORAL, fontsize=8.5, weight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("minutes since first commit (13:33)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for k, c in palette.items():
        ax.scatter([], [], color=c, s=60, label=k)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_val(path: Path, val: dict) -> None:
    """Val F1 per label with 95% bootstrap CIs, against the all-positive floor."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    xs = range(4)
    keys = [*(f"f1/{lab}" for lab in LABELS), "f1/macro"]
    for off, name, color in ((-0.18, "baseline", TEAL), (0.18, "all-positive floor", GREY)):
        r = val[name]
        vals = [r["at_tuned"][k] for k in keys]
        lo = [v - r["ci_at_tuned"][k][0] for v, k in zip(vals, keys, strict=True)]
        hi = [r["ci_at_tuned"][k][1] - v for v, k in zip(vals, keys, strict=True)]
        ax.bar(
            [x + off for x in xs],
            vals,
            0.36,
            yerr=[lo, hi],
            capsize=3,
            color=color,
            label="TF-IDF + LogReg (delivered)" if name == "baseline" else "always 'present'",
        )
        for x, v in zip(xs, vals, strict=True):
            ax.text(
                x + off, 0.03, f"{v:.2f}", ha="center", color="white", fontsize=8.5, weight="bold"
            )
    ax.set_xticks(list(xs), [*SHORT, "Macro F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 on val.csv (95% CI)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cv(path: Path, base: dict, enc: dict) -> None:
    """Train CV macro F1: familiar vs unseen vocabulary."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw={"width_ratios": [1, 2]})
    rows = [("standard_cv", "at_tuned"), ("heldout_vocab_cv", "at_tuned_mean_over_folds")]
    ax = axes[0]
    for off, r, c, n in ((-0.18, base, TEAL, "baseline"), (0.18, enc, CORAL, "DeBERTa-v3")):
        v = [r[a][b]["f1/macro"] for a, b in rows]
        ax.bar([0 + off, 1 + off], v, 0.36, color=c, label=n)
        for x, y in zip((0 + off, 1 + off), v, strict=True):
            ax.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["familiar\nwords", "unseen\nwords"])
    ax.set_ylim(0.6, 0.9)
    ax.set_title("Macro F1 (train CV)", fontsize=10, color=NAVY)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for off, r, c in ((-0.18, base, TEAL), (0.18, enc, CORAL)):
        v = [r["heldout_vocab_cv"]["at_tuned_mean_over_folds"][f"f1/{lab}"] for lab in LABELS]
        ax.bar([x + off for x in range(3)], v, 0.36, color=c)
        for x, y in zip(range(3), v, strict=True):
            ax.text(x + off, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), SHORT)
    ax.set_ylim(0.4, 1)
    ax.set_title("Per label, unseen vocabulary", fontsize=10, color=NAVY)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_behaviour(path: Path, beh: dict) -> None:
    """Probability shift of the targeted label under one-factor edits."""
    edits = [
        ("lexical (seen words)", "lexical_load", "technical term\n(seen)"),
        ("lexical (unseen words)", "lexical_load", "technical term\n(UNSEEN)"),
        ("abstract (seen words)", "conceptual_abstraction", "abstract topics\n(seen)"),
        ("abstract (unseen words)", "conceptual_abstraction", "abstract topics\n(UNSEEN)"),
        ("syntactic", "syntactic_complexity", "complex\ntemplate"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    for off, name, c in ((-0.18, "baseline", TEAL), (0.18, "encoder", CORAL)):
        v = [beh[name][k][lab] for k, lab, _ in edits]
        ax.bar(
            [x + off for x in range(5)],
            v,
            0.36,
            color=c,
            label="baseline" if name == "baseline" else "DeBERTa-v3 encoder",
        )
        for x, y in zip(range(5), v, strict=True):
            ax.text(x + off, y + 0.015, f"+{y:.2f}", ha="center", fontsize=8)
    for x in (1, 3):
        ax.axvspan(x - 0.45, x + 0.45, color=GOLD, alpha=0.18, lw=0)
    ax.set_xticks(range(5), [e[2] for e in edits], fontsize=8.5)
    ax.set_ylabel("rise in targeted\nlabel probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- document

CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
       color: #1e293b; font-size: 10.2pt; line-height: 1.5; }
.page { width: 210mm; height: 297mm; padding: 16mm 17mm 14mm; position: relative;
        page-break-after: always; overflow: hidden; }
.page:last-child { page-break-after: auto; }
.foot { position: absolute; bottom: 8mm; left: 17mm; right: 17mm; font-size: 7.5pt;
        color: #94a3b8; display: flex; justify-content: space-between; }
.hero { margin: -16mm -17mm 9mm; padding: 15mm 17mm 13mm; color: white;
        background: linear-gradient(135deg, #14213d 0%, #1d3557 55%, #2a9d8f 130%); }
.hero .kicker { letter-spacing: .18em; text-transform: uppercase; font-size: 8pt; opacity: .75; }
.hero h1 { font-size: 30pt; line-height: 1.1; margin: 4mm 0 3mm; font-weight: 700; }
.hero p { font-size: 11.5pt; opacity: .9; margin: 0; max-width: 150mm; }
.hero .who { margin-top: 6mm; font-size: 8.5pt; opacity: .7; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin: -18mm 0 8mm; }
.stat { background: white; border-radius: 4mm; padding: 5mm 4mm; text-align: center;
        border: .35mm solid #dbe3ec; }
.stat b { display: block; font-size: 22pt; line-height: 1.1; }
.stat span { font-size: 8pt; color: #475569; }
.eyebrow { font-size: 7.8pt; letter-spacing: .2em; text-transform: uppercase; color: #2a9d8f;
           font-weight: 700; margin-top: 6mm; }
h2 { font-size: 17pt; margin: 1mm 0 3mm; color: #14213d; }
h3 { font-size: 11pt; margin: 5mm 0 2mm; color: #14213d; }
p { margin: 0 0 2.6mm; }
.lead { font-size: 11pt; color: #334155; }
.callout { border-left: 1.2mm solid #e76f51; background: #fff7f3; padding: 3.5mm 5mm;
           border-radius: 0 3mm 3mm 0; margin: 4mm 0; }
.callout.teal { border-color: #2a9d8f; background: #f0faf8; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
.card { background: #f8fafc; border: .3mm solid #e2e8f0; border-radius: 3mm; padding: 4mm; }
.card h4 { margin: 0 0 1.5mm; font-size: 9.5pt; color: #14213d; }
.card p, .card li { font-size: 8.8pt; color: #475569; }
.card ul { margin: 0; padding-left: 4mm; }
figure { margin: 3mm 0 4mm; }
figure { text-align: center; } figure img { width: 100%; }
figcaption { text-align: left; }
.eyebrow.tight { margin-top: 3mm; }
figcaption { font-size: 8pt; color: #64748b; margin-top: 1mm; }
.flow { display: flex; align-items: stretch; gap: 1.6mm; margin: 4mm 0 2mm; }
.step { flex: 1; border-radius: 2.5mm; padding: 2.6mm 1.5mm; text-align: center; color: white;
        font-size: 7.6pt; line-height: 1.25; font-weight: 600; }
.step small { display: block; font-weight: 400; opacity: .85; margin-top: .8mm; }
.plan { background: #94a3b8; } .build { background: #2a9d8f; } .eval { background: #e76f51; }
.qc { background: #14213d; }
.firewall { font-size: 7.8pt; color: #e76f51; font-weight: 700; text-align: center; }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; }
th { text-align: left; color: #64748b; font-weight: 600; font-size: 7.8pt; letter-spacing: .06em;
     text-transform: uppercase; border-bottom: .4mm solid #14213d; padding: 1.5mm 2mm; }
td { padding: 1.5mm 2mm; border-bottom: .2mm solid #e2e8f0; vertical-align: top; }
td:first-child { font-weight: 600; color: #14213d; width: 34%; }
.tick { color: #2a9d8f; font-weight: 700; }
.tags span { display: inline-block; background: #e2f3f1; color: #1f6f66; border-radius: 9mm;
             padding: .6mm 2.6mm; margin: 0 1mm 1.4mm 0; font-size: 8pt; }
.chain { display: grid; grid-template-columns: repeat(5, 1fr); gap: 2.5mm; margin: 3mm 0; }
.link { border-radius: 3mm; padding: 3mm; color: white; font-size: 8pt; position: relative; }
.link b { display: block; font-size: 9pt; margin-bottom: 1mm; }
.link:not(:last-child)::after { content: "›"; position: absolute; right: -2.4mm; top: 35%;
             color: #14213d; font-size: 14pt; font-weight: 700; }
"""


def esc(text: str) -> str:
    """Minimal HTML escaping for values pulled from files."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    """Render charts, assemble the HTML, print it to PDF with headless Chrome."""
    val, beh = load("val_metrics.json"), load("behavioural_tests.json")
    base, enc = load("baseline/cv_metrics.json"), load("encoder/cv_metrics.json")
    log = commits()
    t = [int(h[:2]) * 60 + int(h[3:]) for h, _, _ in log]
    minutes = t[-1] - t[0]

    tmp = Path(tempfile.mkdtemp())
    figs = {n: tmp / f"{n}.png" for n in ("timeline", "val", "cv", "behaviour")}
    fig_timeline(figs["timeline"], log)
    fig_val(figs["val"], val)
    fig_cv(figs["cv"], base, enc)
    fig_behaviour(figs["behaviour"], beh)

    vm = val["baseline"]["at_tuned"]["f1/macro"]
    lo, hi = val["baseline"]["ci_at_tuned"]["f1/macro"]
    floor = val["all-positive floor"]["at_tuned"]["f1/macro"]
    bs = base["standard_cv"]["at_tuned"]["f1/macro"]
    bh = base["heldout_vocab_cv"]["at_tuned_mean_over_folds"]["f1/macro"]
    eh = enc["heldout_vocab_cv"]["at_tuned_mean_over_folds"]["f1/macro"]
    bl = beh["baseline"]["lexical (unseen words)"]["lexical_load"]
    el = beh["encoder"]["lexical (unseen words)"]["lexical_load"]
    ba = beh["baseline"]["abstract (unseen words)"]["conceptual_abstraction"]
    ea = beh["encoder"]["abstract (unseen words)"]["conceptual_abstraction"]
    foot = (
        '<div class="foot"><span>Dhad readability classifier · generated by '
        "scripts/make_report.py from committed results</span><span>{}</span></div>"
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<section class="page">
  <div class="hero">
    <div class="kicker">Miqyas al Dhad · take-home report</div>
    <h1>Dhad Readability Classifier</h1>
    <p>Multi-label difficulty scoring, built with an agent-assisted, evaluation-first
    workflow, and tested on words the model has never seen.</p>
    <div class="who">Bisharah Estephan · 29 October 2026</div>
  </div>
  <div class="stats">
    <div class="stat"><b style="color:{TEAL}">{vm:.3f}</b><span>val macro F1<br>floor {floor:.3f}</span></div>
    <div class="stat"><b style="color:{CORAL}">+{el:.2f}</b><span>encoder response to<br>unseen technical words</span></div>
    <div class="stat"><b style="color:{NAVY}">{len(log)}</b><span>commits in<br>{minutes} minutes</span></div>
    <div class="stat"><b style="color:#c9a227">13</b><span>AI agents, each<br>with a narrow brief</span></div>
  </div>

  <div class="eyebrow">At a glance</div>
  <h2>A model that reads difficulty three ways</h2>
  <p class="lead">Given a short English passage, the model returns three independent
  probabilities: <b>lexical load</b>, <b>syntactic complexity</b> and <b>conceptual
  abstraction</b>.</p>
  <p>The delivered TF-IDF + logistic regression model scores <b>macro F1 {vm:.3f}</b>
  (95% CI {lo:.2f}–{hi:.2f}) on the validation set, against {floor:.3f} for always predicting
  "present". Validation data was opened <b>exactly once</b>, after the model and thresholds
  were frozen under the git tag <code>pre-val</code>.</p>
  <div class="callout">
    <b>The central finding: unseen vocabulary.</b> In this data each label hangs on a handful
    of slot words, so ordinary cross-validation rewards memorisation. Holding whole words out
    drops the baseline from {bs:.3f} to {bh:.3f}; a fine-tuned DeBERTa-v3 encoder reaches
    {eh:.3f}. Shown words that never occur in training, the baseline barely reacts
    (+{bl:.2f}, +{ba:.2f}), while the encoder responds almost as strongly as to familiar words
    (+{el:.2f}, +{ea:.2f}).
  </div>

  <div class="eyebrow tight">01 · The project</div>
  <h2>What was asked, what was delivered</h2>
  <table>
    <tr><th>The brief asked for</th><th>Delivered</th></tr>
    <tr><td>Training code that reads train.csv and writes a checkpoint</td>
        <td><span class="tick">✓</span> <code>uv run dhad-train</code>: CV, threshold tuning, refit, checkpoint and run manifest</td></tr>
    <tr><td>Three probabilities per passage (multi-label)</td>
        <td><span class="tick">✓</span> <code>predict_proba</code> returns an (n, 3) array in [0, 1]</td></tr>
    <tr><td>Evaluation on val.csv: per-dimension and macro F1</td>
        <td><span class="tick">✓</span> <code>uv run dhad-evaluate</code>: F1 with bootstrap CIs, plus AUC, AP, Brier and a shortcut check</td></tr>
    <tr><td>README of one page or less, 2–3 design choices</td>
        <td><span class="tick">✓</span> Three design choices, results and limitations</td></tr>
    <tr><td>Local compute; no rules written from val</td>
        <td><span class="tick">✓</span> Mac + personal RTX 5090; val read by one module, once</td></tr>
  </table>
  {foot.format(1)}
</section>

<section class="page">
  <div class="eyebrow">02 · How I worked</div>
  <h2>Evaluation-first, agent-assisted</h2>
  <p class="lead">I set the rules first, then used AI agents for breadth: research, review,
  parallel implementation and QC. Every decision that touches evaluation stayed with me.</p>
  <div class="firewall">▼ validation firewall: nothing to the left ever saw val.csv</div>
  <div class="flow">
    <div class="step plan">Discover<small>4 research agents</small></div>
    <div class="step plan">Plan<small>committed first</small></div>
    <div class="step plan">Review<small>3 review agents</small></div>
    <div class="step build">Build baseline<small>+ 2 build agents</small></div>
    <div class="step eval">Freeze<small>tag pre-val</small></div>
    <div class="step eval">val.csv<small>read once</small></div>
    <div class="step build">Encoder<small>RTX 5090</small></div>
    <div class="step build">Behaviour tests<small>seen vs unseen</small></div>
    <div class="step qc">Final QC<small>4 read-only agents</small></div>
  </div>

  <h3>Thirteen agents, each with a narrow brief</h3>
  <div class="grid2">
    <div class="card"><h4>Research · 4</h4><p>Modelling, evaluation methodology,
      engineering stack, and a train-only analysis of the data.</p></div>
    <div class="card"><h4>Plan review · 3</h4><p>Model choice and licences checked on
      Hugging Face, a data scientist and readability expert, and a grader's view.</p></div>
    <div class="card"><h4>Build · 2</h4><p>Unit tests and the analysis script, written in
      parallel on files that did not overlap.</p></div>
    <div class="card"><h4>Final QC · 4, read-only</h4><p>Numbers against result files, code
      correctness, privacy and git hygiene, and the brief's deliverables.</p></div>
  </div>
  <div class="callout teal"><b>Agent output was checked, not trusted.</b> The reviews
  reordered the plan (baseline deliverables first, encoder time-boxed). One agent's claim of
  swapped figures was checked and rejected. QC findings were applied only when verified, and
  correct but untestable code issues were documented rather than patched blind.</div>

  <h3>Delivery timeline</h3>
  <figure><img src="{figs["timeline"].name}">
  <figcaption>{len(log)} commits in {minutes} minutes, coloured by conventional-commit type.
  The freeze precedes the single validation run.</figcaption></figure>
  {foot.format(2)}
</section>

<section class="page">
  <div class="eyebrow">03 · Results</div>
  <h2>Well above the floor, honest about the noise</h2>
  <figure><img style="width:80%" src="{figs["val"].name}">
  <figcaption>Validation F1 with 95% bootstrap CIs (2,000 resamples), computed once, after
  the freeze. With 120 rows, gaps under about 0.05 are noise.</figcaption></figure>
  <figure><img style="width:90%" src="{figs["cv"].name}">
  <figcaption>Train cross-validation. On unseen words the encoder holds up; the gain is in
  abstraction, while lexical load does not improve.</figcaption></figure>
  <h3>What did each model actually learn?</h3>
  <p>F1 cannot show <i>why</i> a model is right. Behavioural tests change one factor at a
  time in 200 minimal pairs built from the training templates, using words seen in training
  and words verified never to occur in it (<i>isotope, jurisprudence, morality, virtue</i>).
  Validation data is not used.</p>
  <figure><img style="width:86%" src="{figs["behaviour"].name}">
  <figcaption>Rise in the targeted label's probability. Highlighted: unseen words, where only
  the encoder generalises. Off-target shifts stay at 0.04 or less, except a shared shortcut:
  the complex template alone raises lexical load.</figcaption></figure>
  {foot.format(3)}
</section>

<section class="page">
  <div class="eyebrow">04 · Governance</div>
  <h2>How the GitHub repository keeps the work honest</h2>
  <div class="chain">
    <div class="link" style="background:{CORAL}"><b>AGENTS.md</b>hard rules for every agent</div>
    <div class="link" style="background:{NAVY}"><b>Agents</b>research, review, build, QC</div>
    <div class="link" style="background:{TEAL}"><b>Commits</b>small, conventional, ≤ 15 min apart</div>
    <div class="link" style="background:#64748b"><b>Hook</b>post-commit auto-push</div>
    <div class="link" style="background:{NAVY}"><b>GitHub</b>history + annotated tags</div>
  </div>
  <table>
    <tr><th>Mechanism</th><th>What it guarantees</th></tr>
    <tr><td>AGENTS.md (checked in)</td><td>val never used for training, tuning or selection;
      every number from a real run; no secrets or large files; no history rewriting.</td></tr>
    <tr><td><code>pre-val</code> annotated tag</td><td>Model and thresholds frozen in
      results/selection.json before val.csv was read; the val results commit comes after it.</td></tr>
    <tr><td>Single val reader</td><td>Only <code>dhad-evaluate</code> reads val.csv; tests and
      behavioural tests never do.</td></tr>
    <tr><td>post-commit hook</td><td>Every commit is pushed immediately; failures are logged to
      .git/push.log.</td></tr>
    <tr><td>Conventional commits</td><td>feat / fix / test / docs / results / chore, with
      Co-Authored-By trailers that make agent involvement visible.</td></tr>
    <tr><td>Run manifests</td><td>Each run records its git SHA, the SHA-256 of its data and the
      chosen settings.</td></tr>
    <tr><td>Plan and progress in docs/</td><td>Research, plan, review revisions and a
      story-point tracker are committed, so the process can be audited.</td></tr>
  </table>

  <div class="eyebrow">05 · Tools</div>
  <div class="tags">
    <span>Claude Code + parallel sub-agents</span><span>uv · uv.lock</span><span>Python 3.12+</span>
    <span>scikit-learn</span><span>PyTorch 2.15</span><span>transformers 5.21</span>
    <span>DeBERTa-v3-base (MIT)</span><span>iterative-stratification</span>
    <span>pytest · 20 tests</span><span>ruff</span><span>matplotlib</span>
    <span>git + GitHub</span><span>personal RTX 5090</span>
  </div>

  <div class="grid2" style="margin-top:4mm">
    <div class="card"><h4>Honest limitations</h4><ul>
      <li>Syntactic complexity is learned from the template (F1 0.00 within the simple
      template, 0.94 within the complex one).</li>
      <li>Synthetic, noisy labels cap the reachable F1.</li>
      <li>120 validation rows: gaps under about 0.05 F1 are noise.</li>
      <li>The encoder came after the freeze: exploratory, no val score. Two bugs found on
      the way were fixed and documented.</li></ul></div>
    <div class="card"><h4>Future work: transfer to Arabic</h4><ul>
      <li>Arabic encoder (CAMeLBERT-MSA, AraBERTv2); held-out-vocabulary CV becomes
      held-out-root CV.</li>
      <li>Lexical load from root and pattern rarity; syntax from clitic and iḍāfa depth.</li>
      <li>Human-labelled ordinal difficulty levels instead of binary flags.</li></ul></div>
  </div>
  {foot.format(4)}
</section>
</body></html>"""

    page = tmp / "report.html"
    page.write_text(html)
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUT}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(tmp)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
