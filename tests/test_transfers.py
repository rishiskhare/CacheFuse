# SPDX-License-Identifier: Apache-2.0
"""Tests for the direct CacheTransfer builder."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from cachefuse.config import Settings
from cachefuse.transfers import build_transfer


def test_build_transfer_rejects_mismatched_kv_shape():
    cfg = Settings(device="cpu")
    sender = MagicMock()
    sender.config = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
    )
    receiver = MagicMock()
    receiver.config = SimpleNamespace(
        num_key_value_heads=4,
        head_dim=4,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
    )
    with pytest.raises(ValueError, match="identical KV geometry"):
        build_transfer(cfg, sender, receiver, tok=MagicMock())


def test_build_transfer_copies_prefill_cache(monkeypatch):
    cfg = Settings(device="cpu")
    shared = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
    )
    sender = MagicMock()
    sender.config = shared
    receiver = MagicMock()
    receiver.config = shared
    past = object()
    moved = object()
    monkeypatch.setattr("cachefuse.transfers.core.prefill", lambda model, ids: past)
    monkeypatch.setattr(
        "cachefuse.transfers.core.move_cache", lambda cache, device: moved
    )
    transfer = build_transfer(cfg, sender, receiver, tok=MagicMock())
    ids = torch.zeros(1, 3, dtype=torch.long)
    assert transfer(ids) is moved
