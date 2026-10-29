"""Reproducibility helpers."""

from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds third-party code using the global RNG.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> Any:
    """Best available torch device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_sha() -> str:
    """Current commit, suffixed with ``-dirty`` if the tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, recorded so a run can be tied to the exact data it used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
