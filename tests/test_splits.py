import numpy as np

from dhad.splits import heldout_vocab_folds, slot_vocabulary, stratified_folds, tokenize

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
WORDS += ["india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"]


def _corpus(n=80, seed=0):
    rng = np.random.default_rng(seed)
    texts = []
    for _ in range(n):
        picked = rng.choice(WORDS, size=2, replace=False)
        texts.append(f"the passage mentions {picked[0]} and {picked[1]}")
    y = rng.integers(0, 2, (n, 3))
    return texts, y


def test_tokenize():
    assert tokenize("The cat, the CAT! 42 dogs") == {"the", "cat", "dogs"}


def test_stratified_folds_partition():
    _, y = _corpus()
    folds = stratified_folds(y, n_splits=5, n_repeats=1, seed=0)
    assert len(folds) == 5
    counts = np.zeros(len(y), dtype=int)
    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert len(train) + len(test) == len(y)
        counts[test] += 1
    assert (counts == 1).all()


def test_slot_vocabulary_bounds():
    texts = ["common rare"] + ["common mid"] * 3 + ["common"] * 6  # 10 rows
    # doc freq: common=10, mid=3, rare=1
    assert slot_vocabulary(texts, min_doc_frac=0.2, max_doc_frac=0.5) == ["mid"]
    assert slot_vocabulary(texts, min_doc_frac=0.1, max_doc_frac=0.3) == ["mid", "rare"]
    assert slot_vocabulary(texts, min_doc_frac=0.0, max_doc_frac=1.0) == ["common", "mid", "rare"]


def test_heldout_vocab_folds():
    texts, y = _corpus()
    vocab = slot_vocabulary(texts, min_doc_frac=0.02, max_doc_frac=0.5)
    assert set(vocab) <= set(WORDS)
    folds = heldout_vocab_folds(texts, y, vocab, n_splits=4, seed=0)
    assert len(folds) == 4
    tokens = [tokenize(t) for t in texts]
    for train, test in folds:
        assert len(test) > 0
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(len(texts)))
        held = set().union(*(tokens[i] for i in test)) & set(vocab)
        held -= set().union(*(tokens[i] for i in train))
        assert held, "fold holds out no word"
        for i in train:
            assert not tokens[i] & held
