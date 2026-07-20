# SPDX-License-Identifier: Apache-2.0
"""Tests for model loading and DynamicCache helpers (mocked HF, CPU tensors)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from cachefuse import core
from cachefuse.core import load_models, make_cache, move_cache, prefill


def test_make_cache_and_move_cache_roundtrip():
    keys = [torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4)]
    values = [torch.full((1, 2, 3, 4), 2.0), torch.full((1, 2, 3, 4), 3.0)]
    cache = make_cache(keys, values, "cpu")
    assert len(cache.layers) == 2
    assert cache.layers[0].keys.dtype == torch.bfloat16
    moved = move_cache(cache, "cpu")
    assert moved is cache
    assert moved.layers[0].device == torch.device("cpu")


def test_prefill_requests_cache_and_returns_past():
    past = object()
    model = MagicMock()
    model.return_value = SimpleNamespace(past_key_values=past)
    ids = torch.zeros(1, 4, dtype=torch.long)
    assert prefill(model, ids) is past
    kwargs = model.call_args.kwargs
    assert kwargs["input_ids"] is ids
    assert kwargs["use_cache"] is True
    assert kwargs["logits_to_keep"] == 1


def test_load_models_same_device_and_matching_shape(monkeypatch):
    cfg = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
    )
    tok = object()
    sender = MagicMock()
    sender.config = cfg
    sender.to.return_value = sender
    sender.eval.return_value = sender
    receiver = MagicMock()
    receiver.config = cfg
    receiver.to.return_value = receiver
    receiver.eval.return_value = receiver

    monkeypatch.setattr(core.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(
        core.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **k: sender if name == "S" else receiver,
    )
    monkeypatch.delenv("TRUST_REMOTE_CODE", raising=False)

    out_tok, out_s, out_r = load_models("S", "R", "cpu")
    assert out_tok is tok
    assert out_s is sender
    assert out_r is receiver
    sender.to.assert_called_with("cpu")
    receiver.to.assert_called_with("cpu")


def test_load_models_split_devices_and_mismatch_warns(monkeypatch, caplog):
    sender_cfg = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
    )
    receiver_cfg = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        num_hidden_layers=2,
    )
    sender = MagicMock()
    sender.config = sender_cfg
    sender.to.return_value = sender
    sender.eval.return_value = sender
    receiver = MagicMock()
    receiver.config = receiver_cfg
    receiver.to.return_value = receiver
    receiver.eval.return_value = receiver

    monkeypatch.setattr(core.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        core.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **k: sender if name == "S" else receiver,
    )
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")

    with caplog.at_level("WARNING", logger="cachefuse.core"):
        load_models("S", "R", "cpu", sender_device="cpu")
    assert "KV shape differs" in caplog.text
    sender.to.assert_called_with("cpu")
