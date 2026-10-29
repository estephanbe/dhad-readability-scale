"""``dhad-evaluate``: the only code that reads val.csv.

Scores one or more trained runs on the held-out validation set using the thresholds
frozen at training time, and writes a metrics JSON, per-row predictions and a Markdown
report. Nothing is tuned here.

Example:
    uv run dhad-evaluate --runs checkpoints/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dhad import LABELS
from dhad.data import load_split
from dhad.metrics import bootstrap_ci, compute_metrics
from dhad.predict import load_model
from dhad.utils import file_sha256, git_sha


def _fmt(value: float, ci: tuple[float, float] | None = None) -> str:
    return f"{value:.3f}" if ci is None else f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def report_table(results: dict[str, dict], delivered: str) -> str:
    """Markdown report: F1 with 95% CIs, threshold-free metrics, and a shortcut check."""
    lines = [
        "# Validation results (`data/val.csv`, n = {n})",
        "",
        f"Delivered model: **{delivered}**. Thresholds were frozen on train CV before this run.",
        "Brackets: 95% bootstrap CI (2,000 row resamples).",
        "",
        "## F1",
        "",
        "| Model | " + " | ".join(LABELS) + " | macro F1 | micro F1 |",
        "|---|" + "---|" * (len(LABELS) + 2),
    ]
    n = None
    for name, r in results.items():
        n = r["n"]
        m, ci = r["at_tuned"], r["ci_at_tuned"]
        cells = [_fmt(m[f"f1/{lab}"], ci[f"f1/{lab}"]) for lab in LABELS]
        cells += [_fmt(m["f1/macro"], ci["f1/macro"]), _fmt(m["f1/micro"])]
        label = f"**{name}**" if name == delivered else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if "at_0.5" in r:
            m5 = r["at_0.5"]
            cells = [_fmt(m5[f"f1/{lab}"]) for lab in LABELS]
            cells += [_fmt(m5["f1/macro"]), _fmt(m5["f1/micro"])]
            lines.append(f"| {name} @ 0.5 | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Threshold-free and calibration metrics",
        "",
        "| Model | Metric | " + " | ".join(LABELS) + " |",
        "|---|---|" + "---|" * len(LABELS),
    ]
    for name, r in results.items():
        if name == "all-positive floor":
            continue
        for metric in ("roc_auc", "avg_precision", "brier"):
            vals = [_fmt(r["at_tuned"][f"{metric}/{lab}"]) for lab in LABELS]
            lines.append(f"| {name} | {metric} | " + " | ".join(vals) + " |")
        m = r["at_tuned"]
        lines.append(
            f"| {name} | subset acc / Hamming | {m['subset_accuracy']:.3f} / "
            f"{m['hamming_loss']:.3f} | | |"
        )
    lines += [
        "",
        "## Shortcut check: `syntactic_complexity` F1 within each template",
        "",
        "If the model only learned the template, it cannot separate labels *within* one.",
        "",
        "| Model | Template | n | positives | F1 |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        for template, s in r.get("syntactic_by_template", {}).items():
            lines.append(f"| {name} | {template} | {s['n']} | {s['positives']} | {s['f1']:.3f} |")
    return "\n".join(lines).replace("{n}", str(n)) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="dhad-train output dirs")
    p.add_argument("--val", type=Path, default=Path("data/val.csv"))
    p.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    delivered = json.loads(args.selection.read_text())["delivered"]
    val = load_split(args.val)
    y = val.y
    # Template membership for the shortcut check (the complex template opens with "Although").
    # Reporting slice only: it never produces labels, features or any modelling decision.
    complex_template = np.array([t.startswith("Although") for t in val.texts])

    results: dict[str, dict] = {}
    predictions = pd.DataFrame({"id": val.ids})
    for run in args.runs:
        model, config = load_model(run)
        name = config["kind"]
        proba = model.predict_proba(val.texts)
        thresholds = config["thresholds"]
        pred = (proba >= np.asarray(thresholds)).astype(int)
        by_template = {}
        for tname, mask in (("simple", ~complex_template), ("complex", complex_template)):
            sub = compute_metrics(y[mask], proba[mask], thresholds)
            by_template[tname] = {
                "n": int(mask.sum()),
                "positives": int(y[mask, 1].sum()),
                "f1": sub["f1/syntactic_complexity"],
            }
        results[name] = {
            "n": len(y),
            "thresholds": thresholds,
            "at_tuned": compute_metrics(y, proba, thresholds),
            "at_0.5": compute_metrics(y, proba),
            "ci_at_tuned": bootstrap_ci(y, proba, thresholds, n_boot=args.n_boot),
            "syntactic_by_template": by_template,
        }
        for j, label in enumerate(LABELS):
            predictions[f"{name}_p_{label}"] = proba[:, j].round(4)
            predictions[f"{name}_pred_{label}"] = pred[:, j]

    # Reference floor: always predicting "present" gives F1 = 2p / (1 + p) per label.
    ones = np.ones_like(y, dtype=float)
    results["all-positive floor"] = {
        "n": len(y),
        "at_tuned": compute_metrics(y, ones),
        "ci_at_tuned": bootstrap_ci(y, ones, (0.5, 0.5, 0.5), n_boot=args.n_boot),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"git_sha": git_sha(), "val_sha256": file_sha256(args.val), "delivered": delivered}
    (args.out / "val_metrics.json").write_text(json.dumps({"meta": meta, **results}, indent=2))
    predictions.insert(1, "text", val.texts)
    predictions.to_csv(args.out / "val_predictions.csv", index=False)
    (args.out / "val_report.md").write_text(report_table(results, delivered))
    print((args.out / "val_report.md").read_text())


if __name__ == "__main__":
    main()
