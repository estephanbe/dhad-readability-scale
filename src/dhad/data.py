"""Loading and validating the CSV splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dhad import LABELS


@dataclass(frozen=True)
class Split:
    """One dataset split.

    Attributes:
        ids: Row identifiers, shape ``(n,)``.
        texts: Passages, length ``n``.
        y: Binary label matrix, shape ``(n, 3)``, columns ordered as :data:`dhad.LABELS`.
    """

    ids: np.ndarray
    texts: list[str]
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)


def load_split(path: str | Path) -> Split:
    """Read a split and fail loudly on schema problems.

    The brief puts data cleaning out of scope, so this only *validates*: it never
    alters rows. A silent schema problem would invalidate every downstream metric.

    Args:
        path: CSV with columns ``id``, ``text`` and the three label columns.

    Returns:
        The validated split.

    Raises:
        ValueError: If columns are missing, values are null, ids repeat, or a label
            is not 0/1.
    """
    df = pd.read_csv(path)
    missing = {"id", "text", *LABELS} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df[["id", "text", *LABELS]].isna().any().any():
        raise ValueError(f"{path}: contains null values")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids")
    y = df[list(LABELS)].to_numpy()
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f"{path}: labels must be 0/1")
    return Split(ids=df["id"].to_numpy(), texts=df["text"].astype(str).tolist(), y=y.astype(int))
