"""Fine-tuned transformer encoder with a three-logit multi-label head.

A plain PyTorch loop is used instead of ``transformers.Trainer``: the loop is short,
has no extra dependency (``accelerate``), and makes the training procedure explicit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dhad import LABELS
from dhad.utils import get_device, set_seed


@dataclass(frozen=True)
class EncoderConfig:
    """Fine-tuning hyperparameters.

    Defaults follow common practice for base-size encoders on small datasets
    (lr 2e-5, 10% warmup, weight decay 0.01). ``max_len=96`` covers the longest
    training passage (52 words, about 75 tokens) without truncation.
    """

    backbone: str = "microsoft/deberta-v3-base"
    epochs: int = 10
    lr: float = 2e-5
    batch_size: int = 16
    max_len: int = 96
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    seed: int = 42
    # Off by default: the first GPU run with bf16 autocast produced NaN probabilities.
    # fp32 costs little here (480 short passages).
    bf16: bool = False


class EncoderClassifier:
    """Fine-tunes a pretrained encoder for three independent binary labels.

    The model is ``AutoModelForSequenceClassification`` with three outputs, trained with
    binary cross-entropy (one sigmoid per label), because the labels can co-occur.

    Args:
        config: Hyperparameters.
    """

    kind = "encoder"

    def __init__(self, config: EncoderConfig = EncoderConfig()) -> None:  # noqa: B008
        self.config = config
        self.device = get_device()
        self.tokenizer = None
        self.model = None

    def _build(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.backbone,
            num_labels=len(LABELS),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)},
        ).to(self.device)

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def _autocast(self) -> torch.autocast:
        # Optional bf16 on CUDA (never fp16: DeBERTa-v3 is known to overflow in fp16).
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.bf16 and self.device.type == "cuda",
        )

    def fit(
        self, texts: list[str], y: np.ndarray, eval_texts: list[str] | None = None
    ) -> list[np.ndarray]:
        """Train from the pretrained checkpoint.

        Args:
            texts: Training passages.
            y: Training labels, shape ``(n, 3)``.
            eval_texts: Optional passages to score after every epoch. This lets
                cross-validation choose the number of epochs from a single run.

        Returns:
            One ``(len(eval_texts), 3)`` probability array per epoch (empty list if
            ``eval_texts`` is None).
        """
        cfg = self.config
        set_seed(cfg.seed)
        self._build()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        steps_per_epoch = math.ceil(len(texts) / cfg.batch_size)
        total = steps_per_epoch * cfg.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_frac * total), total)
        targets = torch.tensor(y, dtype=torch.float32)
        rng = np.random.default_rng(cfg.seed)
        per_epoch = []
        for _ in range(cfg.epochs):
            self.model.train()
            order = rng.permutation(len(texts))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                batch = self._encode([texts[i] for i in idx])
                with self._autocast():
                    logits = self.model(**batch).logits
                loss = binary_cross_entropy_with_logits(
                    logits.float(), targets[idx].to(self.device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if eval_texts is not None:
                per_epoch.append(self.predict_proba(eval_texts))
        return per_epoch

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return sigmoid probabilities, shape ``(n, 3)``."""
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            with self._autocast():
                logits = self.model(**self._encode(texts[start : start + batch_size])).logits
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out)

    def save(self, directory: Path) -> None:
        """Save weights, tokenizer and hyperparameters to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "encoder_config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, directory: Path) -> EncoderClassifier:
        """Load a model written by :meth:`save`."""
        config = EncoderConfig(**json.loads((directory / "encoder_config.json").read_text()))
        obj = cls(config)
        obj.tokenizer = AutoTokenizer.from_pretrained(directory)
        obj.model = AutoModelForSequenceClassification.from_pretrained(directory).to(obj.device)
        return obj
