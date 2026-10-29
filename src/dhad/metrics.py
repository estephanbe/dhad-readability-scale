"""Metrics, threshold tuning and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dhad import LABELS

THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.20, 0.701, 0.05), 2)
"""Coarse grid: a fine grid would overfit about 170 positives per label."""


def binarize(proba: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Apply per-label thresholds (``proba >= threshold`` is positive)."""
    return (proba >= np.asarray(thresholds)[None, :]).astype(int)


def compute_metrics(
    y: np.ndarray, proba: np.ndarray, thresholds: Sequence[float] = (0.5, 0.5, 0.5)
) -> dict[str, float]:
    """Compute the full metric set for one model on one split.

    Threshold-dependent metrics (F1, precision, recall, subset accuracy, Hamming loss)
    are complemented by threshold-free ones (ROC-AUC, average precision) and a
    calibration metric (Brier), because the deliverable is a *probability* per label.

    Args:
        y: True labels, shape ``(n, 3)``.
        proba: Predicted probabilities, shape ``(n, 3)``.
        thresholds: Per-label decision thresholds.

    Returns:
        Flat ``{metric_name: value}`` dictionary, e.g. ``"f1/lexical_load"``.
    """
    pred = binarize(proba, thresholds)
    out: dict[str, float] = {
        "f1/macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1/micro": f1_score(y, pred, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(y, pred),
        "hamming_loss": hamming_loss(y, pred),
    }
    p, r, f, _ = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    for j, name in enumerate(LABELS):
        out[f"f1/{name}"] = f[j]
        out[f"precision/{name}"] = p[j]
        out[f"recall/{name}"] = r[j]
        out[f"brier/{name}"] = brier_score_loss(y[:, j], proba[:, j])
        out[f"prevalence_pred/{name}"] = pred[:, j].mean()
        out[f"prevalence_true/{name}"] = y[:, j].mean()
        if len(np.unique(y[:, j])) == 2:  # AUC is undefined for a single-class column.
            out[f"roc_auc/{name}"] = roc_auc_score(y[:, j], proba[:, j])
            out[f"avg_precision/{name}"] = average_precision_score(y[:, j], proba[:, j])
    return {k: float(v) for k, v in out.items()}


def tune_thresholds(
    y: np.ndarray, oof_proba: np.ndarray, min_gain: float = 0.01
) -> list[float]:
    """Choose a per-label threshold on out-of-fold predictions.

    The rule is fixed in advance to limit overfitting: keep 0.5 unless the best grid
    threshold beats it by more than ``min_gain`` F1 on the out-of-fold predictions.

    Args:
        y: True training labels, shape ``(n, 3)``.
        oof_proba: Out-of-fold probabilities for the same rows.
        min_gain: Minimum F1 improvement required to move away from 0.5.

    Returns:
        One threshold per label.
    """
    thresholds = []
    for j in range(y.shape[1]):
        scores = {
            float(t): f1_score(y[:, j], oof_proba[:, j] >= t, zero_division=0)
            for t in THRESHOLD_GRID
        }
        best = max(scores, key=lambda t: scores[t])
        thresholds.append(best if scores[best] - scores[0.5] > min_gain else 0.5)
    return thresholds


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    thresholds: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for every metric.

    Whole rows are resampled, which keeps the correlation between labels intact.
    Resamples where a metric is undefined (e.g. AUC with one class) simply do not
    contribute to that metric.

    Args:
        y: True labels.
        proba: Predicted probabilities.
        thresholds: Per-label thresholds.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.
        alpha: Two-sided significance level (0.05 gives a 95% interval).

    Returns:
        ``{metric_name: (low, high)}``.
    """
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for k, v in compute_metrics(y[idx], proba[idx], thresholds).items():
            samples.setdefault(k, []).append(v)
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {k: tuple(float(x) for x in np.percentile(v, q)) for k, v in samples.items()}
