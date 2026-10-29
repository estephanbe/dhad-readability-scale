import numpy as np

from dhad.models.baseline import TfidfLogReg


def _data():
    rng = np.random.default_rng(0)
    vocab = ["epistemology", "ontology", "cat", "dog", "notwithstanding", "whereas", "run", "sun"]
    texts, y = [], []
    for _ in range(24):
        words = rng.choice(vocab, size=4).tolist()
        texts.append(" ".join(words))
        y.append(
            [
                int(any(w in ("epistemology", "ontology") for w in words)),
                int(any(w in ("notwithstanding", "whereas") for w in words)),
                int(len(set(words)) > 3),
            ]
        )
    y = np.array(y)
    y[0], y[1] = [1, 1, 1], [0, 0, 0]  # guarantee both classes per label
    return texts, y


def test_fit_predict_shape_and_range():
    texts, y = _data()
    proba = TfidfLogReg().fit(texts, y).predict_proba(texts)
    assert proba.shape == (len(texts), 3)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_save_load_roundtrip(tmp_path):
    texts, y = _data()
    model = TfidfLogReg(C=2.0).fit(texts, y)
    model.save(tmp_path / "m")
    loaded = TfidfLogReg.load(tmp_path / "m")
    assert loaded.C == 2.0
    np.testing.assert_array_equal(model.predict_proba(texts), loaded.predict_proba(texts))
