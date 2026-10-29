"""``dhad-train``: cross-validate on train.csv, tune thresholds, refit, save a checkpoint.

Only ``train.csv`` is read. Every choice made here (hyperparameter, thresholds) is based
on out-of-fold predictions on the training set.

Example:
    uv run dhad-train --model baseline
    uv run dhad-train --model encoder        # needs `uv sync --extra neural`
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dhad import LABELS
from dhad.data import load_split
from dhad.metrics import compute_metrics, tune_thresholds
from dhad.splits import Fold, heldout_vocab_folds, slot_vocabulary, stratified_folds, word_types
from dhad.utils import file_sha256, git_sha, set_seed

Predictions = list[tuple[np.ndarray, np.ndarray]]
"""Per fold: ``(test_indices, probabilities)``."""


def cv_predict(
    kind: str, args: argparse.Namespace, texts: list[str], y: np.ndarray, folds: list[Fold]
) -> dict[str, Predictions]:
    """Out-of-fold predictions for every hyperparameter candidate.

    Baseline candidates are values of ``C``. Encoder candidates are epoch counts: one
    training run per fold is scored after every epoch, which avoids retraining per
    candidate.

    Returns:
        ``{candidate_name: [(test_idx, proba), ...]}``.
    """
    preds: dict[str, Predictions] = defaultdict(list)
    for k, (tr, te) in enumerate(folds):
        train_texts, test_texts = [texts[i] for i in tr], [texts[i] for i in te]
        if kind == "baseline":
            from dhad.models.baseline import TfidfLogReg

            for c in args.c_grid:
                model = TfidfLogReg(C=c).fit(train_texts, y[tr])
                preds[f"C={c}"].append((te, model.predict_proba(test_texts)))
        else:
            from dhad.models.encoder import EncoderClassifier

            per_epoch = EncoderClassifier(encoder_config(args)).fit(
                train_texts, y[tr], eval_texts=test_texts
            )
            for e, proba in enumerate(per_epoch, start=1):
                preds[f"epochs={e}"].append((te, proba))
        print(f"  fold {k + 1}/{len(folds)} done", flush=True)
    return preds


def encoder_config(args: argparse.Namespace, epochs: int | None = None) -> Any:
    """Build the encoder config from CLI arguments."""
    from dhad.models.encoder import EncoderConfig

    return EncoderConfig(
        backbone=args.backbone, epochs=epochs or args.epochs, lr=args.lr, seed=args.seed
    )


def build_model(kind: str, args: argparse.Namespace, candidate: str) -> Any:
    """Instantiate an unfitted model for a candidate name such as ``C=1`` or ``epochs=4``."""
    value = candidate.split("=")[1]
    if kind == "baseline":
        from dhad.models.baseline import TfidfLogReg

        return TfidfLogReg(C=float(value))
    from dhad.models.encoder import EncoderClassifier

    return EncoderClassifier(encoder_config(args, epochs=int(value)))


def pool(preds: Predictions, n: int) -> np.ndarray:
    """Assemble out-of-fold predictions from a partition into one ``(n, 3)`` array."""
    oof = np.full((n, len(LABELS)), np.nan)
    for te, proba in preds:
        oof[te] = proba
    return oof


def mean_fold_metrics(
    preds: Predictions, y: np.ndarray, thresholds: list[float]
) -> dict[str, float]:
    """Average per-fold metrics (for folds that overlap, where pooling is undefined)."""
    per_fold = [compute_metrics(y[te], proba, thresholds) for te, proba in preds]
    keys = set.intersection(*(set(m) for m in per_fold))
    return {k: float(np.mean([m[k] for m in per_fold])) for k in sorted(keys)}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["baseline", "encoder"], required=True)
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--out", type=Path, help="checkpoint dir (default checkpoints/<model>)")
    p.add_argument("--results", type=Path, help="small artefacts (default results/<model>)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--c-grid", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    p.add_argument("--backbone", default="microsoft/deberta-v3-base")
    p.add_argument("--epochs", type=int, default=10, help="max epochs searched by CV")
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args(argv)
    out = args.out or Path("checkpoints") / args.model
    results = args.results or Path("results") / args.model
    results.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    started = time.time()

    train = load_split(args.train)
    texts, y = train.texts, train.y

    # 1. Hyperparameter choice on standard multi-label stratified CV (macro F1 at 0.5).
    print("standard CV", flush=True)
    std = cv_predict(args.model, args, texts, y, stratified_folds(y, args.folds, seed=args.seed))
    scores = {
        cand: compute_metrics(y, pool(preds, len(y)))["f1/macro"] for cand, preds in std.items()
    }
    best = max(scores, key=lambda c: scores[c])
    oof = pool(std[best], len(y))

    # 2. Per-label thresholds on the same out-of-fold predictions.
    thresholds = tune_thresholds(y, oof)

    # 3. Generalisation to unseen words: held-out-vocabulary CV with the chosen candidate.
    print("held-out-vocabulary CV", flush=True)
    vocab = slot_vocabulary(texts)
    vocab_folds = heldout_vocab_folds(texts, y, vocab, args.folds, seed=args.seed)
    if args.model == "encoder":
        args.epochs = int(best.split("=")[1])
    held = cv_predict(args.model, args, texts, y, vocab_folds)[best]

    cv_report = {
        "candidate_scores_macro_f1": scores,
        "chosen": best,
        "thresholds": dict(zip(LABELS, thresholds, strict=True)),
        "standard_cv": {
            "at_0.5": compute_metrics(y, oof),
            "at_tuned": compute_metrics(y, oof, thresholds),
        },
        "heldout_vocab_cv": {
            "fold_test_sizes": [len(te) for te, _ in held],
            "at_tuned_mean_over_folds": mean_fold_metrics(held, y, thresholds),
        },
    }

    # 4. Refit on all training rows and save.
    print(f"refit with {best}", flush=True)
    model = build_model(args.model, args, best)
    model.fit(texts, y)
    model.save(out)
    config = {
        "kind": args.model,
        "chosen": best,
        "thresholds": thresholds,
        "labels": list(LABELS),
        "seed": args.seed,
        "backbone": args.backbone if args.model == "encoder" else None,
        "git_sha": git_sha(),
        "train_sha256": file_sha256(args.train),
        "train_seconds": round(time.time() - started, 1),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (results / "config.json").write_text(json.dumps(config, indent=2))
    (results / "cv_metrics.json").write_text(json.dumps(cv_report, indent=2))
    (results / "heldout_vocab.json").write_text(
        json.dumps(word_types(texts, y, vocab), indent=2, sort_keys=True)
    )
    oof_df = pd.DataFrame(oof, columns=[f"p_{n}" for n in LABELS])
    oof_df.insert(0, "id", train.ids)
    oof_df.to_csv(results / "oof.csv", index=False, float_format="%.4f")

    m, h = cv_report["standard_cv"]["at_tuned"], cv_report["heldout_vocab_cv"]
    print(f"chosen {best}; thresholds {thresholds}")
    print(f"standard CV macro F1 {m['f1/macro']:.3f}")
    print(f"held-out-vocab CV macro F1 {h['at_tuned_mean_over_folds']['f1/macro']:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
