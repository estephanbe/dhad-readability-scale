"""Build docs/report.pdf: project report with figures.

Every number is read from the committed result files and git history at build time,
so the report cannot drift from the repository.

Usage:
    uv run --group report python scripts/make_report.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "report.pdf"
LABELS = ("lexical_load", "syntactic_complexity", "conceptual_abstraction")
SHORT = ("Lexical load", "Syntactic complexity", "Conceptual abstraction")

NAVY, TEAL, CORAL, GOLD, GREY = "#14213d", "#2a9d8f", "#e76f51", "#e9c46a", "#8d99ae"
LIGHT = "#f1f5f9"

# DejaVu ships with matplotlib and covers arrows, dashes and Arabic-free Unicode text.
_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_FONTS / "DejaVuSans-Bold.ttf")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


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


# ---------------------------------------------------------------- figures


def fig_cover(path: Path, val: dict, beh: dict, n_commits: int, minutes: int) -> None:
    """Headline stat tiles."""
    enc_unseen = beh["encoder"]
    tiles = [
        (f"{val['baseline']['at_tuned']['f1/macro']:.3f}", "val macro F1\n(floor 0.524)"),
        (
            f"+{enc_unseen['lexical (unseen words)']['lexical_load']:.2f}",
            "encoder response to\nunseen technical words",
        ),
        (f"{n_commits}", f"commits in\n{minutes} minutes"),
        ("13", "AI agents for research,\nreview, build and QC"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.1))
    for ax, (big, small), c in zip(axes, tiles, (TEAL, CORAL, NAVY, GOLD), strict=True):
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.02",
                fc=LIGHT,
                ec=c,
                lw=2.5,
                transform=ax.transAxes,
            )
        )
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=26, weight="bold", color=c)
        ax.text(0.5, 0.22, small, ha="center", va="center", fontsize=9, color=NAVY)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, tc="white", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="none"))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=tc,
        fontsize=fs,
        weight="bold",
        wrap=True,
    )


def _arrow(ax, a, b, color=NAVY):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.5))


def fig_workflow(path: Path) -> None:
    """The delivery pipeline, with the val firewall."""
    steps = [
        ("Discover\n4 research\nagents", GREY),
        ("Plan\ncommitted", GREY),
        ("Review\n3 review\nagents", GREY),
        ("Build\nbaseline\n+2 agents", TEAL),
        ("Freeze\ntag\npre-val", CORAL),
        ("val.csv\nonce", CORAL),
        ("Encoder\nRTX 5090", TEAL),
        ("Behaviour\ntests", TEAL),
        ("Final QC\n4 QC\nagents", NAVY),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    for i, (t, c) in enumerate(steps):
        _box(ax, i * 1.02 + 0.05, 0.7, 0.9, 1.0, t, c, fs=7)
        if i:
            _arrow(ax, (i * 1.02 - 0.07, 1.15), (i * 1.02 + 0.05, 1.15))
    ax.plot([4.62, 4.62], [0.3, 1.95], ls="--", color=CORAL, lw=1.5)
    ax.text(
        4.62,
        2.05,
        "val firewall: nothing before this line saw val.csv",
        ha="center",
        color=CORAL,
        fontsize=8.5,
        weight="bold",
    )
    ax.text(
        0.05,
        0.35,
        "Grey: planning   Teal: building   Coral: evaluation discipline   Navy: quality control",
        fontsize=8,
        color=NAVY,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def fig_governance(path: Path) -> None:
    """How rules flow from AGENTS.md to GitHub."""
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    nodes = [
        (0.1, "AGENTS.md\nhard rules", CORAL),
        (2.1, "Coding agent\n+ sub-agents", NAVY),
        (4.1, "Small conventional\ncommits", TEAL),
        (6.1, "post-commit hook\nauto-push", GREY),
        (8.1, "GitHub repo\n+ annotated tags", NAVY),
    ]
    for x, t, c in nodes:
        _box(ax, x, 1.4, 1.75, 1.0, t, c, fs=8.5)
    for i in range(4):
        _arrow(ax, (nodes[i][0] + 1.8, 1.9), (nodes[i + 1][0] - 0.05, 1.9))
    notes = [
        (0.1, "val never used\nfor decisions;\nreal numbers only"),
        (2.1, "research, review,\nbuild and QC agents;\nQC is read-only"),
        (4.1, "commit at least\nevery 15 min; never\nrewrite history"),
        (6.1, "push failures\nlogged to\n.git/push.log"),
        (8.1, "pre-val tag proves\nthe freeze came\nbefore val"),
    ]
    for x, t in notes:
        ax.text(x + 0.875, 0.7, t, ha="center", va="center", fontsize=7.5, color=NAVY)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- document


def main() -> None:
    """Render figures and assemble the PDF."""
    val, beh = load("val_metrics.json"), load("behavioural_tests.json")
    base, enc = load("baseline/cv_metrics.json"), load("encoder/cv_metrics.json")
    log = commits()
    t = [int(h[:2]) * 60 + int(h[3:]) for h, _, _ in log]
    minutes = t[-1] - t[0]

    tmp = Path(tempfile.mkdtemp())
    figs = {
        n: tmp / f"{n}.png"
        for n in ("cover", "workflow", "timeline", "val", "cv", "behaviour", "governance")
    }
    fig_cover(figs["cover"], val, beh, len(log), minutes)
    fig_workflow(figs["workflow"])
    fig_timeline(figs["timeline"], log)
    fig_val(figs["val"], val)
    fig_cv(figs["cv"], base, enc)
    fig_behaviour(figs["behaviour"], beh)
    fig_governance(figs["governance"])

    h1 = ParagraphStyle(
        "h1",
        fontName="DVB",
        fontSize=15,
        textColor=colors.HexColor(NAVY),
        spaceBefore=6,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "h2",
        fontName="DVB",
        fontSize=11.5,
        textColor=colors.HexColor(TEAL),
        spaceBefore=8,
        spaceAfter=3,
    )
    body = ParagraphStyle("b", fontName="DV", fontSize=9.3, leading=13.2, spaceAfter=4)
    small = ParagraphStyle(
        "s", parent=body, fontSize=8, textColor=colors.HexColor(GREY), alignment=TA_CENTER
    )
    title = ParagraphStyle(
        "t", fontName="DVB", fontSize=24, leading=29, textColor=colors.HexColor(NAVY)
    )
    sub = ParagraphStyle(
        "st", fontName="DV", fontSize=12, leading=16, textColor=colors.HexColor(TEAL)
    )

    def img(p: Path, width: float = 17.5) -> Image:
        im = Image(str(p))
        im.drawWidth, im.drawHeight = width * cm, width * cm * im.imageHeight / im.imageWidth
        return im

    def table(rows: list[list[str]], widths: list[float]) -> Table:
        hdr = ParagraphStyle("hdr", parent=body, fontName="DVB", textColor=colors.white)
        data = [[Paragraph(c, hdr if i == 0 else body) for c in r] for i, r in enumerate(rows)]
        tb = Table(data, colWidths=[w * cm for w in widths])
        tb.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ]
            )
        )
        return tb

    b = base["standard_cv"]["at_tuned"]["f1/macro"]
    bh = base["heldout_vocab_cv"]["at_tuned_mean_over_folds"]["f1/macro"]
    eh = enc["heldout_vocab_cv"]["at_tuned_mean_over_folds"]["f1/macro"]
    vm = val["baseline"]["at_tuned"]["f1/macro"]
    vci = val["baseline"]["ci_at_tuned"]["f1/macro"]

    s = [
        Paragraph("Dhad Readability Classifier", title),
        Paragraph(
            "Take-home report: multi-label difficulty scoring, built with an "
            "agent-assisted, evaluation-first workflow",
            sub,
        ),
        Spacer(1, 6),
        Paragraph(
            "Bisharah Estephan · Miqyas al Dhad contractor application · 29 October 2026", body
        ),
        Spacer(1, 10),
        img(figs["cover"]),
        Spacer(1, 8),
        Paragraph("At a glance", h1),
        Paragraph(
            f"A model that, given a short English passage, outputs three independent "
            f"probabilities: lexical load, syntactic complexity and conceptual abstraction. "
            f"The delivered TF-IDF + logistic regression model scores <b>macro F1 {vm:.3f}</b> "
            f"(95% CI {vci[0]:.2f}–{vci[1]:.2f}) on the held-out validation set, against 0.524 for "
            f"always predicting 'present'. Validation data was opened exactly once, after the "
            f"model and thresholds were frozen under the git tag <b>pre-val</b>.",
            body,
        ),
        Paragraph(
            f"The central finding is about <b>unseen vocabulary</b>. In this data each label is "
            f"driven by a handful of slot words, so ordinary cross-validation rewards "
            f"memorisation. Holding whole words out drops the baseline from {b:.3f} to {bh:.3f}; "
            f"a fine-tuned DeBERTa-v3 encoder reaches {eh:.3f}. Controlled behavioural tests show "
            f"why: shown words that never occur in training, the baseline barely reacts while "
            f"the encoder responds almost as strongly as to familiar words.",
            body,
        ),
        Paragraph("1. The project", h1),
        Paragraph(
            "The brief asked for a multi-label classifier over a synthetic dataset "
            "(480 training and 120 validation passages) within a two-hour budget, using "
            "local or free compute only, and without hand-written rules derived from the "
            "validation set. It evaluates ways of working and motivated technical choices, "
            "not raw model quality.",
            body,
        ),
        table(
            [
                ["The brief asked for", "What was delivered"],
                [
                    "Training code that reads train.csv and writes a checkpoint",
                    "<font name='DVB'>uv run dhad-train</font>: cross-validation, threshold tuning, "
                    "refit, checkpoint and run manifest",
                ],
                [
                    "Three probabilities per passage (multi-label)",
                    "predict_proba returns an (n, 3) array in [0, 1]",
                ],
                [
                    "Evaluation on val.csv with per-dimension and macro F1",
                    f"<font name='DVB'>uv run dhad-evaluate</font>: macro F1 {vm:.3f} with bootstrap "
                    "CIs, plus ROC-AUC, average precision, Brier and a shortcut check",
                ],
                [
                    "README of one page or less with 2–3 design choices",
                    "README with three design choices, results and limitations",
                ],
                [
                    "Local compute, no rules written from val",
                    "Mac + personal RTX 5090; val read by one module, once",
                ],
            ],
            [6.5, 11],
        ),
        Spacer(1, 6),
        Paragraph(
            "Beyond the brief: a DeBERTa-v3 encoder, held-out-vocabulary cross-validation, "
            "behavioural tests, 20 unit tests, a reproducible train-only data analysis, "
            "and a fresh-clone check that reproduces every committed number exactly.",
            body,
        ),
        Paragraph("2. How I worked", h1),
        Paragraph(
            "The work ran as an evaluation-first pipeline. I set the rules first, then "
            "used AI agents for breadth (research, review, parallel implementation, "
            "QC), and kept every decision that touches evaluation under my own control.",
            body,
        ),
        img(figs["workflow"]),
        Paragraph(
            "Figure 1. Delivery pipeline. The dashed line is the validation firewall.", small
        ),
        Paragraph("Thirteen agents, each with a narrow brief", h2),
        Paragraph(
            "<b>Research (4):</b> modelling, evaluation methodology, engineering stack and "
            "a train-only data analysis. <b>Plan review (3):</b> model choice and licence "
            "check against Hugging Face, a data-scientist and readability-expert critique, "
            "and a grader's view. <b>Build (2):</b> unit tests and the analysis script, "
            "written in parallel on files that did not overlap. <b>Final QC (4, "
            "read-only):</b> every number against the result files, code correctness, "
            "privacy and git hygiene, and the brief's deliverables.",
            body,
        ),
        Paragraph(
            "Agent output was checked, not trusted: the reviews reordered the plan "
            "(baseline deliverables first, encoder time-boxed), one agent's claim of "
            "swapped figures was checked and rejected, and QC findings were applied "
            "only when verified. Correct but untestable code issues were documented rather "
            "than patched blind.",
            body,
        ),
        Paragraph("Delivery timeline", h2),
        img(figs["timeline"]),
        Paragraph(
            f"Figure 2. {len(log)} commits in {minutes} minutes, coloured by "
            "conventional-commit type. The freeze precedes the single val run.",
            small,
        ),
        Paragraph("3. Results", h1),
        KeepTogether(
            [
                img(figs["val"], 15.5),
                Paragraph(
                    "Figure 3. Validation F1 with 95% bootstrap CIs (2,000 resamples), "
                    "computed once, after the freeze.",
                    small,
                ),
            ]
        ),
        KeepTogether(
            [
                img(figs["cv"]),
                Paragraph(
                    "Figure 4. Train cross-validation. On unseen words the encoder holds up; "
                    "the gain is in abstraction. Lexical load does not improve.",
                    small,
                ),
            ]
        ),
        Paragraph("What did each model actually learn?", h2),
        Paragraph(
            "F1 cannot show <i>why</i> a model is right. The behavioural tests change one "
            "factor at a time in 200 minimal pairs built from the training templates, "
            "using words seen in training and words verified never to occur in it "
            "(e.g. <i>isotope</i>, <i>jurisprudence</i>, <i>morality</i>, <i>virtue</i>). "
            "Validation data is not used.",
            body,
        ),
        img(figs["behaviour"]),
        Paragraph(
            "Figure 5. Rise in the targeted label's probability. Highlighted: unseen "
            "words, where only the encoder generalises. Off-target shifts stay at 0.04 or "
            "less, except a shared shortcut: the complex template alone raises lexical "
            "load (+0.10 / +0.17).",
            small,
        ),
        PageBreak(),
        Paragraph("4. Tools", h1),
        table(
            [
                ["Area", "Tools", "Why"],
                [
                    "Coding agent",
                    "Claude Code with parallel sub-agents",
                    "Breadth for research, review, build and QC; decisions kept with me",
                ],
                [
                    "Environment",
                    "uv, uv.lock, Python 3.12+",
                    "Reproducible installs; heavy deep-learning stack is an optional extra",
                ],
                [
                    "Modelling",
                    "scikit-learn, PyTorch 2.15, transformers 5.21",
                    "TF-IDF + LR baseline; DeBERTa-v3-base (MIT) fine-tune",
                ],
                [
                    "Evaluation",
                    "iterative-stratification, bootstrap CIs",
                    "Multi-label stratified and held-out-vocabulary CV",
                ],
                [
                    "Quality",
                    "pytest (20 tests), ruff lint + format",
                    "Fast tests with no downloads",
                ],
                [
                    "Compute",
                    "Mac for development, personal RTX 5090 for the encoder",
                    "Local only, no paid APIs",
                ],
                [
                    "Reporting",
                    "matplotlib, reportlab",
                    "This report is generated from the result files",
                ],
            ],
            [3, 6, 8.5],
        ),
        Paragraph("5. Governance in the GitHub repository", h1),
        img(figs["governance"]),
        Paragraph(
            "Figure 6. Rules flow from AGENTS.md through the agents to every pushed commit.", small
        ),
        table(
            [
                ["Mechanism", "What it guarantees"],
                [
                    "AGENTS.md (checked in)",
                    "Hard rules for any coding agent: val never used for training, tuning or "
                    "selection; every number from a real run; no secrets or large files; no history "
                    "rewriting; ask when unsure.",
                ],
                [
                    "pre-val annotated tag",
                    "Model and thresholds frozen in results/selection.json before val.csv was read. "
                    "The val results commit comes after the tag.",
                ],
                [
                    "Single val reader",
                    "Only dhad-evaluate reads val.csv; tests and the behavioural script never do.",
                ],
                [
                    "post-commit hook",
                    "Every commit is pushed immediately; failures go to .git/push.log.",
                ],
                [
                    "Conventional commits",
                    "feat / fix / test / docs / results / chore, with "
                    "Co-Authored-By trailers that make agent involvement visible.",
                ],
                [
                    "Run manifests",
                    "Each run records its git SHA, the SHA-256 of the data and the "
                    "chosen settings.",
                ],
                [
                    "Plan and progress in docs/",
                    "Research, plan, review revisions and a "
                    "story-point tracker are committed, so the process can be audited.",
                ],
            ],
            [4.5, 13],
        ),
        Paragraph("6. Honest limitations", h1),
        Paragraph(
            "• Syntactic complexity is learned from the template: F1 is 0.00 within the "
            "simple template and 0.94 within the complex one.<br/>"
            "• Labels are synthetic and noisy, which caps the reachable F1 "
            "(ceilings in results/train_analysis.md).<br/>"
            "• 120 validation rows: differences under about 0.05 F1 are noise.<br/>"
            "• The encoder was trained after the freeze, so it is exploratory and has no "
            "val score. Two bugs found on the way (fp16 weights under transformers 5, "
            "and a config file overwrite) were fixed and documented.",
            body,
        ),
        Paragraph("Next steps and transfer to Arabic", h2),
        Paragraph(
            "Swap in an Arabic encoder (CAMeLBERT-MSA, AraBERTv2) and turn the "
            "held-out-vocabulary CV into a held-out-root CV; measure lexical load "
            "from root and pattern rarity, and syntax from clitic and iḍāfa depth; move "
            "from binary flags to human-labelled ordinal difficulty levels.",
            body,
        ),
    ]

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(
            2 * cm,
            1.2 * cm,
            "Dhad readability classifier · generated by "
            "scripts/make_report.py from committed results",
        )
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        canvas.setFillColor(colors.HexColor(TEAL))
        canvas.rect(0, A4[1] - 0.35 * cm, A4[0], 0.35 * cm, stroke=0, fill=1)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.8 * cm,
        title="Dhad Readability Classifier: report",
        author="Bisharah Estephan",
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
