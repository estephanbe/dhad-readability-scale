"""TF-IDF + logistic regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


class TfidfLogReg:
    """One logistic regression per label on word and character n-gram TF-IDF.

    Word n-grams capture the slot words and template phrases; character n-grams
    (within word boundaries) give a little morphological generalisation to unseen
    words, e.g. ``-ology`` or ``-ism``. The labels are independent in train
    (|phi| < 0.05), so one-vs-rest loses nothing against a joint model.

    Class weights are left uniform on purpose: re-weighting would shift the predicted
    probabilities away from calibrated values, and the deliverable is a probability.

    Args:
        C: Inverse L2 regularisation strength.
    """

    kind = "baseline"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000))),
            ]
        )

    def fit(self, texts: list[str], y: np.ndarray) -> Self:
        """Fit on passages and an ``(n, 3)`` label matrix."""
        self.pipeline.fit(texts, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probabilities, shape ``(n, 3)``."""
        return self.pipeline.predict_proba(texts)

    def save(self, directory: Path) -> None:
        """Write the fitted pipeline to ``directory/model.joblib``."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> TfidfLogReg:
        """Load a pipeline written by :meth:`save` (only load trusted files)."""
        return joblib.load(directory / "model.joblib")
