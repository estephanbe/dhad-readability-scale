import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from dhad.models.encoder import EncoderConfig


def test_encoder_config_defaults():
    cfg = EncoderConfig()
    assert cfg.max_len == 96
    assert cfg.backbone == "microsoft/deberta-v3-base"
