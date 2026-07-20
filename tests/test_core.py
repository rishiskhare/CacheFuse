# SPDX-License-Identifier: Apache-2.0
"""Tests for KV-geometry helpers (config-only; no weights, no GPU).

These functions read config via ``getattr``, so a ``SimpleNamespace`` standing in
for a ``PretrainedConfig`` exercises both the explicit and fallback branches.
"""

from types import SimpleNamespace

from cachefuse.core import head_dim, num_kv_heads, same_kv_shape


def cfg(**kw):
    base = dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=36,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_head_dim_explicit_wins():
    assert head_dim(cfg(head_dim=128)) == 128


def test_head_dim_fallback_divides_hidden_by_heads():
    # No explicit head_dim -> hidden_size // num_attention_heads.
    assert head_dim(cfg(hidden_size=4096, num_attention_heads=32)) == 128


def test_num_kv_heads_explicit_gqa():
    assert num_kv_heads(cfg(num_key_value_heads=8)) == 8


def test_num_kv_heads_fallback_to_attention_heads():
    # No GQA field -> equals the attention-head count.
    assert num_kv_heads(cfg(num_attention_heads=32)) == 32


def test_same_kv_shape_identical():
    a = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=36)
    b = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=36)
    assert same_kv_shape(a, b) is True


def test_same_kv_shape_differs_on_kv_heads():
    a = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=36)
    b = cfg(num_key_value_heads=4, head_dim=128, num_hidden_layers=36)
    assert same_kv_shape(a, b) is False


def test_same_kv_shape_differs_on_head_dim():
    a = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=36)
    b = cfg(num_key_value_heads=8, head_dim=64, num_hidden_layers=36)
    assert same_kv_shape(a, b) is False


def test_same_kv_shape_differs_on_depth():
    a = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=36)
    b = cfg(num_key_value_heads=8, head_dim=128, num_hidden_layers=28)
    assert same_kv_shape(a, b) is False
