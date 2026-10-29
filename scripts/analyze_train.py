"""Descriptive analysis of the training data, reproducing docs/plan/research.md section 1.

Reads ``data/train.csv`` only. The path is hard-coded on purpose and there is no
option to point the script at another file: ``data/val.csv`` is the held-out
evaluation set and must never be analysed.

The regexes below parse the two fixed sentence templates so that the slot words
(topic nouns and word-list words) can be counted. They are used for descriptive
analysis of train only, never to produce labels or features for the model. The
groupings of slot words are derived from the train labels, not from hand-typed
word lists.

Run from the repository root::

    uv run python scripts/analyze_train.py

Output: ``results/train_analysis.md``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
OUT_PATH = ROOT / "results" / "train_analysis.md"

LABELS = ["lexical_load", "syntactic_complexity", "conceptual_abstraction"]

# Topic slot: "We discuss X and Y." (simple) or "The analysis of X and Y unfolds" (complex).
TOPIC_RE = re.compile(r"(?:We discuss|The analysis of) (\w+) and (\w+)")
# Word-list slot: "words like a, b, c." (simple) or "Terminology such as a, b appears" (complex).
LIST_RE = re.compile(r"(?:words like|Terminology such as) ([\w, ]+?)(?:\.| appears)")


def template_of(text: str) -> str:
    """Return the template name of a training text.

    Args:
        text: Raw text from train.csv.

    Returns:
        ``"simple"``, ``"complex"`` or ``"other"``.
    """
    if text.startswith("We discuss"):
        return "simple"
    if text.startswith("Although"):
        return "complex"
    return "other"


def parse_slots(text: str) -> tuple[list[str], list[str]]:
    """Extract topic nouns and word-list words from a text.

    Args:
        text: Raw text from train.csv.

    Returns:
        Tuple ``(topics, list_words)``. ``list_words`` keeps repeats, in order.
    """
    topic_match = TOPIC_RE.search(text)
    list_match = LIST_RE.search(text)
    topics = list(topic_match.groups()) if topic_match else []
    words = [w.strip() for w in list_match.group(1).split(",")] if list_match else []
    return topics, [w for w in words if w]


def skeleton(text: str) -> str:
    """Mask the slot words, leaving the fixed template wording.

    Args:
        text: Raw text from train.csv.

    Returns:
        The text with topic and word-list slots replaced by placeholders.
    """
    text = TOPIC_RE.sub(lambda m: m.group(0).split(" ")[0] + " … X and Y", text)
    return LIST_RE.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + "LIST" + " …", text)


def phi(a: pd.Series, b: pd.Series) -> float:
    """Phi coefficient between two binary series (Pearson r on 0/1 values).

    Args:
        a: Binary series.
        b: Binary series.

    Returns:
        Phi coefficient.
    """
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def word_rates(df: pd.DataFrame, column: str, vocab: Iterable[str]) -> pd.DataFrame:
    """Per-word label rates when the word is present vs absent in a slot.

    Args:
        df: Training frame with a list-valued slot column and the label columns.
        column: Name of the slot column (``"topics"`` or ``"list_words"``).
        vocab: Words to report.

    Returns:
        One row per word with its row count and, per label, the rate when present
        and the rate when absent.
    """
    rows = []
    for word in sorted(vocab):
        present = df[column].apply(lambda ws, w=word: w in ws)
        row: dict[str, object] = {"word": word, "n_rows": int(present.sum())}
        for label in LABELS:
            row[f"{label}|present"] = df.loc[present, label].mean()
            row[f"{label}|absent"] = df.loc[~present, label].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("word")


def majority_rule(group: pd.Series, y: pd.Series) -> tuple[float, float, dict[object, int]]:
    """Fit and score, in-sample, a majority-vote rule on a grouping.

    Each group value predicts the majority label of the training rows in that group.

    Args:
        group: Group value per row.
        y: Binary label per row.

    Returns:
        Tuple ``(accuracy, f1, rule)`` where ``rule`` maps group value to prediction.
    """
    rule = {g: int(y[group == g].mean() > 0.5) for g in sorted(group.unique())}
    pred = group.map(rule)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return float((pred == y).mean()), f1, rule


def largest_gap_split(rates: pd.Series) -> set[str]:
    """Split words into two groups at the largest gap between sorted rates.

    Args:
        rates: Per-word rate, indexed by word.

    Returns:
        The words above the largest gap.
    """
    ordered = rates.sort_values()
    cut = int(np.argmax(np.diff(ordered.to_numpy())))
    return set(ordered.index[cut + 1 :])


def conditional_table(group: pd.Series, y: pd.Series, name: str) -> str:
    """Markdown table of P(label=1 | group value).

    Args:
        group: Group value per row.
        y: Binary label per row.
        name: Column header for the group value.

    Returns:
        Markdown table.
    """
    lines = [f"| {name} | n | P({y.name}=1) |", "|---|---:|---:|"]
    for g in sorted(group.unique()):
        mask = group == g
        lines.append(f"| {g} | {int(mask.sum())} | {y[mask].mean():.2f} |")
    return "\n".join(lines)


def fmt_words(words: Iterable[str]) -> str:
    """Format a word collection for markdown.

    Args:
        words: Words.

    Returns:
        Comma-separated, sorted, backticked words.
    """
    return ", ".join(f"`{w}`" for w in sorted(words))


def main() -> None:
    """Run the analysis on train.csv and write results/train_analysis.md."""
    df = pd.read_csv(TRAIN_PATH)
    df["template"] = df["text"].map(template_of)
    slots = df["text"].map(parse_slots)
    df["topics"] = slots.map(lambda s: s[0])
    df["list_words"] = slots.map(lambda s: s[1])
    df["complex"] = (df["template"] == "complex").astype(int)
    out: list[str] = [
        "# Training-data analysis (`data/train.csv` only)",
        "",
        "Generated by `scripts/analyze_train.py`. All figures are in-sample on train.",
        "Regexes here only describe train; they never produce labels or model features.",
        "",
    ]

    # 1. Templates.
    n_bad_parse = int(((df["topics"].map(len) != 2) | (df["list_words"].map(len) == 0)).sum())
    complex_skel = df.loc[df["template"] == "complex", "text"].map(skeleton)
    out += [
        "## 1. Templates",
        "",
        f"- Rows: {len(df)}",
        *[f"- {t}: {n}" for t, n in df["template"].value_counts().items()],
        f"- Rows where the slot regexes failed: {n_bad_parse}",
        f"- Distinct clause orders in the complex template: {complex_skel.nunique()}",
        "",
    ]
    skel_ids = {s: i + 1 for i, s in enumerate(complex_skel.value_counts().index)}
    out += [
        "| complex clause order | n | " + " | ".join(LABELS) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, i in skel_ids.items():
        sub = df.loc[complex_skel[complex_skel == s].index]
        rates = " | ".join(f"{sub[label].mean():.2f}" for label in LABELS)
        out.append(f"| order {i} | {len(sub)} | {rates} |")
    out.append("")

    # 2. Prevalence and phi.
    base = {label: float(df[label].mean()) for label in LABELS}
    combos = df.groupby(LABELS).size()
    out += ["## 2. Label prevalence and correlation", "", "| label | P(label=1) |", "|---|---:|"]
    out += [f"| {label} | {base[label]:.3f} |" for label in LABELS]
    out += ["", "| pair | phi |", "|---|---:|"]
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1 :]:
            out.append(f"| {a} / {b} | {phi(df[a], df[b]):+.3f} |")
    out += ["", f"- Label combinations present: {len(combos)} of 8 (min count {combos.min()})", ""]

    # 3. Slot words and per-word rates.
    topic_counts = Counter(w for ws in df["topics"] for w in ws)
    list_counts = Counter(w for ws in df["list_words"] for w in ws)
    topic_rates = word_rates(df, "topics", topic_counts)
    list_rates = word_rates(df, "list_words", list_counts)

    # Data-driven groupings; no word lists are typed by hand.
    # Topics -> conceptual_abstraction: a topic joins the "abstract" group when the
    # abstraction rate among rows containing it exceeds the overall base rate. Each row
    # carries two topics from one group, so per-topic rates separate cleanly.
    abstract_topics = set(
        topic_rates.index[topic_rates["conceptual_abstraction|present"] > base[LABELS[2]]]
    )
    # List words -> lexical_load: the base-rate rule does not work here. A row holds
    # five or six list words, so a common word's lexical rate sits near the base rate
    # and lands above it by chance (reported below). Instead, sort the per-word lexical
    # rates and split at the largest gap between neighbours (1-D two-cluster split);
    # the words above the gap form the "technical" group.
    lex_present = list_rates["lexical_load|present"]
    technical_words = largest_gap_split(lex_present)
    naive_technical = set(list_rates.index[lex_present > base[LABELS[0]]])
    other_topics = set(topic_counts) - abstract_topics
    other_words = set(list_counts) - technical_words

    df["n_abstract_topics"] = df["topics"].map(lambda ws: sum(w in abstract_topics for w in ws))
    # Count distinct technical words per row (repeats such as "water, water" occur).
    df["n_technical"] = df["list_words"].map(lambda ws: len(set(ws) & technical_words))
    mixed_topics = int((df["n_abstract_topics"] == 1).sum())

    out += [
        "## 3. Slot words",
        "",
        f"- Topic nouns: {len(topic_counts)} distinct; min occurrences "
        f"{min(topic_counts.values())}",
        f"- List words: {len(list_counts)} distinct; min occurrences {min(list_counts.values())}",
        f"- Slot words seen in only one row: "
        f"{sum(n == 1 for n in (topic_rates['n_rows'].tolist() + list_rates['n_rows'].tolist()))}",
        "",
        "Grouping rules (data-driven): a topic is abstract when P(abstraction=1 | topic "
        "present) is above base rate; a list word is technical when P(lexical=1 | word "
        "present) lies above the largest gap in the sorted per-word rates.",
        "",
        f"- Topics with abstraction rate above base ({len(abstract_topics)}): "
        f"{fmt_words(abstract_topics)}",
        f"- Other topics ({len(other_topics)}): {fmt_words(other_topics)}",
        f"- Rows whose two topics fall in different groups: {mixed_topics}",
        f"- Technical list words, above the largest gap ({len(technical_words)}): "
        f"{fmt_words(technical_words)}",
        f"- Other list words ({len(other_words)}): {fmt_words(other_words)}",
        f"- List words split at the largest gap in sorted lexical rates, between "
        f"{lex_present[list(other_words)].max():.2f} and "
        f"{lex_present[list(technical_words)].min():.2f}. The base-rate rule would also "
        f"admit {fmt_words(naive_technical - technical_words)}, by chance.",
        "",
    ]

    def rate_table(rates: pd.DataFrame, title: str) -> list[str]:
        head = "| word | rows | " + " | ".join(f"{lab[:4]} pres / abs" for lab in LABELS) + " |"
        lines = [f"**{title}** (label rate when word present / absent)", "", head]
        lines.append("|---|---:|---:|---:|---:|")
        for word, r in rates.iterrows():
            cells = " | ".join(
                f"{r[f'{lab}|present']:.2f} / {r[f'{lab}|absent']:.2f}" for lab in LABELS
            )
            lines.append(f"| {word} | {int(r['n_rows'])} | {cells} |")
        return [*lines, ""]

    out += rate_table(topic_rates, "Topic nouns")
    out += rate_table(list_rates, "List words")

    # 4. Conditional rates on the driver slots.
    df["template_name"] = df["template"]
    df["has_technical"] = (df["n_technical"] > 0).astype(int)
    df["any_abstract"] = (df["n_abstract_topics"] > 0).astype(int)
    out += [
        "## 4. Conditional rates",
        "",
        conditional_table(df["template_name"], df["syntactic_complexity"], "template"),
        "",
        conditional_table(
            df["n_abstract_topics"], df["conceptual_abstraction"], "abstract topics in row"
        ),
        "",
        conditional_table(df["n_technical"], df["lexical_load"], "distinct technical words"),
        "",
        conditional_table(df["has_technical"], df["lexical_load"], "any technical word"),
        "",
    ]

    # 5. In-sample ceiling.
    drivers = {
        "lexical_load": ("n_technical", "distinct technical words"),
        "syntactic_complexity": ("template_name", "template"),
        "conceptual_abstraction": ("n_abstract_topics", "abstract topics"),
    }
    out += [
        "## 5. Rough in-sample ceiling (approximate)",
        "",
        "Majority vote per value of the label's driver, fitted and scored on the same train "
        "rows. In-sample and optimistic about the groupings, which were themselves chosen "
        "from train labels; a rough ceiling, not a validation result.",
        "",
        "| label | driver | rule | accuracy | F1 | error rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for label, (col, desc) in drivers.items():
        acc, f1, rule = majority_rule(df[col], df[label])
        rule_txt = ", ".join(f"{k}→{v}" for k, v in rule.items())
        out.append(f"| {label} | {desc} | {rule_txt} | {acc:.3f} | {f1:.3f} | {1 - acc:.3f} |")
    out.append("")

    # 6. Duplicates.
    dup_texts = int(df["text"].duplicated().sum())
    conflicts = int((df.groupby("text")[LABELS].nunique() > 1).any(axis=1).sum())
    out += [
        "## 6. Duplicates",
        "",
        f"- Duplicate texts (rows beyond the first copy): {dup_texts}",
        f"- Texts with conflicting labels across copies: {conflicts}",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
