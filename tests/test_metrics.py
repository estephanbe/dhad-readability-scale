import numpy as np
import pytest

from dhad import LABELS
from dhad.metrics import THRESHOLD_GRID, binarize, bootstrap_ci, compute_metrics, tune_thresholds

Y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])


def test_binarize_is_inclusive():
    proba = np.array([[0.5, 0.49, 0.7]])
    assert binarize(proba, [0.5, 0.5, 0.7]).tolist() == [[1, 0, 1]]


def test_threshold_grid_contains_half():
    assert 0.5 in THRESHOLD_GRID.tolist()


def test_perfect_predictions():
    proba = Y * 0.8 + 0.1
    m = compute_metrics(Y, proba)
    assert m["f1/macro"] == pytest.approx(1.0)
    assert m["subset_accuracy"] == pytest.approx(1.0)
    assert m["hamming_loss"] == pytest.approx(0.0)
    for name in LABELS:
        assert m[f"f1/{name}"] == pytest.approx(1.0)
        assert m[f"roc_auc/{name}"] == pytest.approx(1.0)


def test_all_positive_predictor_f1():
    proba = np.full(Y.shape, 0.9)
    m = compute_metrics(Y, proba)
    for j, name in enumerate(LABELS):
        p = Y[:, j].mean()
        assert m[f"f1/{name}"] == pytest.approx(2 * p / (1 + p))
        assert m[f"recall/{name}"] == pytest.approx(1.0)
        assert m[f"precision/{name}"] == pytest.approx(p)


def test_auc_absent_for_single_class_column():
    y = Y.copy()
    y[:, 1] = 0
    m = compute_metrics(y, np.full(y.shape, 0.3))
    assert f"roc_auc/{LABELS[1]}" not in m
    assert f"avg_precision/{LABELS[1]}" not in m
    assert f"roc_auc/{LABELS[0]}" in m


def test_tune_thresholds_keeps_half_when_no_gain():
    proba = np.where(Y == 1, 0.8, 0.2)  # separable around 0.5: F1 is already 1.0
    assert tune_thresholds(Y, proba) == [0.5, 0.5, 0.5]


def test_tune_thresholds_moves_when_clearly_better():
    proba = np.where(Y == 1, 0.8, 0.2).astype(float)
    proba[:, 0] = np.where(Y[:, 0] == 1, 0.35, 0.1)  # nothing reaches 0.5 for label 0
    thr = tune_thresholds(Y, proba)
    assert thr[0] != 0.5
    assert 0.1 < thr[0] <= 0.35
    assert thr[1:] == [0.5, 0.5]
    assert compute_metrics(Y, proba, thr)["f1/macro"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (60, 3))
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, y.shape), 0, 1)
    thr = [0.5, 0.5, 0.5]
    point = compute_metrics(y, proba, thr)["f1/macro"]
    low, high = bootstrap_ci(y, proba, thr, n_boot=200)["f1/macro"]
    assert low <= point <= high
