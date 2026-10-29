"""Cross-validation splits on the training set.

Two schemes are provided:

* :func:`stratified_folds` - standard multi-label stratified K-fold. Every slot word
  (topic noun, technical term) appears in every fold, so this measures performance on
  *familiar* vocabulary.
* :func:`heldout_vocab_folds` - a group of mid-frequency words is held out per fold:
  test rows contain at least one held-out word, training rows contain none of them.
  This measures generalisation to *unseen* vocabulary, which is what a readability
  scorer faces on real text.

Both are used only to evaluate and select models on ``train.csv``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold

Fold = tuple[np.ndarray, np.ndarray]
"""A ``(train_indices, test_indices)`` pair."""

_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> set[str]:
    """Lower-cased alphabetic word types in ``text``."""
    return set(_TOKEN.findall(text.lower()))


def stratified_folds(
    y: np.ndarray, n_splits: int = 5, n_repeats: int = 1, seed: int = 0
) -> list[Fold]:
    """Multi-label stratified (repeated) K-fold.

    Args:
        y: Label matrix, shape ``(n, n_labels)``.
        n_splits: Folds per repeat.
        n_repeats: Number of repeats with different shuffles.
        seed: Random state.

    Returns:
        ``n_splits * n_repeats`` folds.
    """
    cv = RepeatedMultilabelStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    return list(cv.split(np.zeros(len(y)), y))


def slot_vocabulary(
    texts: Sequence[str], min_doc_frac: float = 0.02, max_doc_frac: float = 0.15
) -> list[str]:
    """Words frequent enough to matter but too rare to be template boilerplate.

    Template words occur in most rows of a template and very common filler words in
    20-30% of rows; content words that carry the labels (topics, technical terms)
    sit in between. The bounds are chosen from *train* document frequencies only and
    the result is used solely to build evaluation splits, never as a feature or a label.

    Args:
        texts: Training passages.
        min_doc_frac: Minimum fraction of rows a word must appear in.
        max_doc_frac: Maximum fraction of rows a word may appear in.

    Returns:
        Sorted list of candidate held-out words.
    """
    df = Counter(word for text in texts for word in tokenize(text))
    lo, hi = min_doc_frac * len(texts), max_doc_frac * len(texts)
    return sorted(w for w, c in df.items() if lo <= c <= hi)


def word_types(texts: Sequence[str], y: np.ndarray, vocab: Sequence[str]) -> dict[str, str]:
    """Assign each word the label it moves most, e.g. ``"+lexical_load"``.

    For every word, compare each label's rate in rows containing the word with its base
    rate; the type is the label (and direction) with the largest absolute shift. Used
    only to balance the held-out word groups so each fold holds out every kind of word.

    Args:
        texts: Training passages.
        y: Training labels, shape ``(n, n_labels)``.
        vocab: Words to type.

    Returns:
        ``{word: "<sign><label index>"}``.
    """
    tokens = [tokenize(t) for t in texts]
    base = y.mean(axis=0)
    types = {}
    for word in vocab:
        has = np.array([word in tok for tok in tokens])
        shift = y[has].mean(axis=0) - base
        j = int(np.argmax(np.abs(shift)))
        types[word] = f"{'+' if shift[j] > 0 else '-'}{j}"
    return types


def heldout_vocab_folds(
    texts: Sequence[str],
    y: np.ndarray,
    vocab: Sequence[str],
    n_splits: int = 5,
    seed: int = 0,
) -> list[Fold]:
    """K-fold that holds out a disjoint group of words per fold.

    Words are grouped by :func:`word_types`, shuffled within each type and dealt
    round-robin into ``n_splits`` groups, so every group holds out a mix of e.g.
    technical, abstract and concrete words. For fold ``k`` the test set is every row
    containing a word from group ``k``; the training set is every other row, so the model
    never sees those words during training.

    A row can contain words from several groups and therefore appear in several test
    folds: aggregate these folds by averaging per-fold metrics, not by pooling
    out-of-fold predictions.

    Args:
        texts: Training passages.
        y: Training labels (only used to balance word groups).
        vocab: Words to hold out, typically from :func:`slot_vocabulary`.
        n_splits: Number of word groups / folds.
        seed: Seed for the shuffle.

    Returns:
        ``(train_indices, test_indices)`` per fold.
    """
    rng = np.random.default_rng(seed)
    types = word_types(texts, y, vocab)
    ordered: list[str] = []
    for t in sorted(set(types.values())):
        ordered += rng.permutation([w for w in vocab if types[w] == t]).tolist()
    groups = [set(ordered[k::n_splits]) for k in range(n_splits)]
    tokens = [tokenize(t) for t in texts]
    folds = []
    for held in groups:
        is_test = np.array([bool(tok & held) for tok in tokens])
        folds.append((np.flatnonzero(~is_test), np.flatnonzero(is_test)))
    return folds
