# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint round-trip and record aggregation (no models, no GPU)."""

import json

import pytest

from cachefuse.config import Settings
from cachefuse.evaluate import (
    _aggregate,
    _checkpoint_config,
    _load_checkpoint,
    _open_checkpoint,
)


def test_checkpoint_config_fingerprint():
    cfg = Settings(benchmark="openbookqa", thetas=(0.5,))
    assert _checkpoint_config(cfg) == {
        "sender": cfg.sender,
        "receiver": cfg.receiver,
        "benchmark": "openbookqa",
        "split": "test",
        "num_examples": None,
        "thetas": [0.5],
        "score_t2t": False,
    }


def test_open_checkpoint_none_when_unset():
    assert _open_checkpoint(Settings()) is None


def test_open_and_load_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "run.jsonl")
    cfg = Settings(checkpoint=path)
    ck = _open_checkpoint(cfg)
    assert ck is not None
    rec = {"i": 0, "ok": True, "correct": {"receiver": 1}}
    ck.write(json.dumps(rec) + "\n")
    ck.flush()
    ck.close()
    assert _load_checkpoint(path, cfg) == {0: rec}


def test_load_checkpoint_header_only_is_empty(tmp_path):
    path = str(tmp_path / "run.jsonl")
    cfg = Settings(checkpoint=path)
    _open_checkpoint(cfg).close()  # writes just the config header
    assert _load_checkpoint(path, cfg) == {}


def test_load_checkpoint_fingerprint_mismatch_raises(tmp_path):
    path = str(tmp_path / "run.jsonl")
    _open_checkpoint(Settings(checkpoint=path, benchmark="arc_challenge")).close()
    with pytest.raises(ValueError, match="different run config"):
        _load_checkpoint(path, Settings(checkpoint=path, benchmark="openbookqa"))


def test_aggregate_scores_only_successes_and_lists_failures():
    cache_treatments = ["receiver", "replace", "sharer", "t2t"]
    records = {
        0: {
            "i": 0,
            "ok": True,
            "correct": {"receiver": 1, "replace": 0, "sharer": 1, "t2t": 1},
        },
        1: {
            "i": 1,
            "ok": True,
            "correct": {"receiver": 0, "replace": 1, "sharer": 1, "t2t": 0},
        },
        2: {"i": 2, "ok": False, "error": "CUDA out of memory"},
    }
    result = _aggregate(records, cache_treatments, n=3)

    assert result.n == 3
    assert result.n_scored == 2
    assert result.failed == [2]
    # Accuracy averages over the 2 scored questions, not all 3.
    assert result.accuracy["receiver"] == pytest.approx(0.5)
    assert result.accuracy["replace"] == pytest.approx(0.5)
    assert result.accuracy["sharer"] == pytest.approx(1.0)
    assert result.accuracy["t2t"] == pytest.approx(0.5)


def test_aggregate_all_failed_yields_empty_accuracy():
    records = {0: {"i": 0, "ok": False, "error": "boom"}}
    result = _aggregate(records, ["receiver"], n=1)
    assert result.n_scored == 0
    assert result.failed == [0]
    assert result.accuracy == {}
