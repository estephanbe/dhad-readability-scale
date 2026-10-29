"""Behavioural tests: does each label respond to the right cause?

Builds controlled minimal pairs from the two *training* templates and measures how each
predicted probability moves when exactly one factor changes (CheckList-style directional
tests, Ribeiro et al. 2020):

* lexical    - swap one plain word in the word list for a technical term
* abstract   - swap both concrete topic nouns for abstract ones
* syntactic  - keep every slot word, move them from the simple to the complex template

Ideal behaviour: only the targeted label rises; the other two stay flat. Each edit is run
twice, with words seen in train and with words that never occur in train, which tests
generalisation to unseen vocabulary.

No labels are read and val.csv is never touched; the templates and seen words come from
train.csv, the unseen words are checked to be absent from it.

Usage:
    uv run python scripts/behavioural_tests.py --runs checkpoints/baseline [checkpoints/encoder]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dhad import LABELS
from dhad.data import load_split
from dhad.predict import load_model
from dhad.splits import tokenize

SIMPLE = (
    "We discuss {x} and {y}. The text uses words like {words}. "
    "Readers can follow the main idea quickly."
)
COMPLEX = (
    "Although The analysis of {x} and {y} unfolds across layered claims, which many readers "
    "find dense, the passage notes that Dependencies between clauses require careful "
    "rereading, while Terminology such as {words} appears throughout. Furthermore, "
    "Dependencies between clauses require careful rereading, given prior context, remains "
    "open to interpretation."
)
# Slot words as they occur in train.csv (see results/train_analysis.md).
PLAIN = ["water", "school", "happy", "tree", "friend", "house", "play", "walk", "book", "dog"]
CONCRETE = ["cookie", "apple", "river", "pencil", "garden", "bridge", "kitchen", "hammer"]
SEEN = {
    "technical": ["mitochondria", "epistemological", "coefficient", "paradigm", "hypothesis"],
    "abstract": ["justice", "entropy", "freedom", "causality", "obligation"],
}
# Never occur in train.csv (asserted below).
UNSEEN = {
    "technical": ["photosynthesis", "isotope", "jurisprudence", "algorithm", "neurotransmitter"],
    "abstract": ["morality", "ambiguity", "consciousness", "virtue", "democracy"],
}


def make_pairs(vocab: dict[str, list[str]], n: int, seed: int) -> dict[str, list[tuple[str, str]]]:
    """Minimal pairs ``(base, edited)`` per edit type, with random slot fillers."""
    rng = np.random.default_rng(seed)
    pairs: dict[str, list[tuple[str, str]]] = {"lexical": [], "abstract": [], "syntactic": []}
    for _ in range(n):
        x, y = rng.choice(CONCRETE, 2)
        words = list(rng.choice(PLAIN, 5))
        template = SIMPLE if rng.random() < 0.5 else COMPLEX
        base = template.format(x=x, y=y, words=", ".join(words))
        tech = [*words[:-1], rng.choice(vocab["technical"])]
        pairs["lexical"].append((base, template.format(x=x, y=y, words=", ".join(tech))))
        ax, ay = rng.choice(vocab["abstract"], 2)
        pairs["abstract"].append((base, template.format(x=ax, y=ay, words=", ".join(words))))
        simple = SIMPLE.format(x=x, y=y, words=", ".join(words))
        pairs["syntactic"].append((simple, COMPLEX.format(x=x, y=y, words=", ".join(words))))
    return pairs


def main() -> None:
    """Run every edit through every model and write a delta table."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("results/behavioural_tests.json"))
    args = p.parse_args()

    train_vocab = set().union(*(tokenize(t) for t in load_split("data/train.csv").texts))
    leaked = {w for ws in UNSEEN.values() for w in ws} & train_vocab
    assert not leaked, f"'unseen' words occur in train: {leaked}"

    results: dict[str, dict] = {}
    for run in args.runs:
        model, config = load_model(run)
        name = config["kind"]
        results[name] = {}
        for vocab_name, vocab in (("seen", SEEN), ("unseen", UNSEEN)):
            for edit, pairs in make_pairs(vocab, args.n, seed=0).items():
                if edit == "syntactic" and vocab_name == "unseen":
                    continue  # No new words involved; identical to the seen case.
                base = model.predict_proba([b for b, _ in pairs])
                edited = model.predict_proba([e for _, e in pairs])
                delta = (edited - base).mean(axis=0)
                key = f"{edit} ({vocab_name} words)" if edit != "syntactic" else edit
                results[name][key] = dict(zip(LABELS, delta.round(3).tolist(), strict=True))

    args.out.write_text(json.dumps(results, indent=2))
    for name, rows in results.items():
        print(f"\n{name}: mean change in probability (edited - base), n={args.n} pairs")
        print("| edit | " + " | ".join(LABELS) + " |")
        print("|---|" + "---|" * len(LABELS))
        for key, d in rows.items():
            print(f"| {key} | " + " | ".join(f"{d[k]:+.2f}" for k in LABELS) + " |")


if __name__ == "__main__":
    main()
