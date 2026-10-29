"""Load a trained checkpoint and score passages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_model(run_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load the model and its run config from a ``dhad-train`` output directory.

    The heavy encoder stack is imported only when needed, so baseline checkpoints load
    without torch installed.

    Returns:
        ``(model, config)`` where ``config["thresholds"]`` holds the per-label thresholds.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    if config["kind"] == "baseline":
        from dhad.models.baseline import TfidfLogReg

        return TfidfLogReg.load(run_dir), config
    from dhad.models.encoder import EncoderClassifier

    return EncoderClassifier.load(run_dir), config


def predict_proba(texts: list[str], run_dir: str | Path) -> np.ndarray:
    """Three probabilities in [0, 1] per passage, columns ordered as :data:`dhad.LABELS`."""
    model, _ = load_model(run_dir)
    return model.predict_proba(texts)
