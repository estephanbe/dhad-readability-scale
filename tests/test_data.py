from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dhad import LABELS
from dhad.data import load_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["a short text", "another one", "third passage"],
            LABELS[0]: [0, 1, 0],
            LABELS[1]: [1, 1, 0],
            LABELS[2]: [0, 0, 1],
        }
    )


def _write(tmp_path, df):
    path = tmp_path / "split.csv"
    df.to_csv(path, index=False)
    return path


def test_valid_csv_loads(tmp_path):
    split = load_split(_write(tmp_path, _frame()))
    assert len(split) == 3
    assert split.ids.tolist() == [1, 2, 3]
    assert split.texts == ["a short text", "another one", "third passage"]
    assert split.y.shape == (3, 3)
    assert np.issubdtype(split.y.dtype, np.integer)
    assert split.y.tolist() == [[0, 1, 0], [1, 1, 0], [0, 0, 1]]


def test_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        load_split(_write(tmp_path, _frame().drop(columns=[LABELS[1]])))


def test_null_value_raises(tmp_path):
    df = _frame()
    df.loc[1, "text"] = None
    with pytest.raises(ValueError, match="null"):
        load_split(_write(tmp_path, df))


def test_duplicate_id_raises(tmp_path):
    df = _frame()
    df.loc[2, "id"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        load_split(_write(tmp_path, df))


def test_label_outside_01_raises(tmp_path):
    df = _frame()
    df.loc[0, LABELS[2]] = 2
    with pytest.raises(ValueError, match="0/1"):
        load_split(_write(tmp_path, df))


@pytest.mark.skipif(not (DATA_DIR / "train.csv").exists(), reason="train.csv not present")
def test_train_csv_smoke():
    split = load_split(DATA_DIR / "train.csv")
    assert split.y.shape == (480, 3)
    assert len(split.texts) == 480
