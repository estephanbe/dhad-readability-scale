"""Multi-label readability classifier for the Miqyas al Dhad take-home task."""

LABELS: tuple[str, ...] = ("lexical_load", "syntactic_complexity", "conceptual_abstraction")
"""The three binary difficulty dimensions, in the column order used everywhere."""
