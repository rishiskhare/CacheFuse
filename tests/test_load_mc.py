# SPDX-License-Identifier: Apache-2.0
"""Tests for `load_mc` with mocked Hugging Face datasets (no network)."""

import pytest

from cachefuse.benchmarks import MCQuestion, load_mc


def test_load_mc_unknown_benchmark_rejected():
    with pytest.raises(ValueError, match="bench must be one of"):
        load_mc("not_a_bench")


def test_load_mc_arc_challenge_maps_answer_key(monkeypatch):
    rows = [
        {
            "question": "Q1?",
            "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        }
    ]

    def fake_load_dataset(path, config, split):
        assert path == "allenai/ai2_arc"
        assert config == "ARC-Challenge"
        assert split == "train"
        return rows

    monkeypatch.setattr("cachefuse.benchmarks.load_dataset", fake_load_dataset)
    out = load_mc("arc_challenge", split="train")
    assert out == [MCQuestion(question="Q1?", options=["a", "b", "c"], gold=1)]


def test_load_mc_openbookqa_uses_question_stem(monkeypatch):
    rows = [
        {
            "question_stem": "Stem?",
            "choices": {"text": ["x", "y"], "label": ["A", "B"]},
            "answerKey": "A",
        }
    ]
    monkeypatch.setattr(
        "cachefuse.benchmarks.load_dataset",
        lambda path, config, split: rows,
    )
    out = load_mc("openbookqa")
    assert out == [MCQuestion(question="Stem?", options=["x", "y"], gold=0)]


def test_load_mc_mmlu_redux_skips_out_of_range_and_rejects_non_test(monkeypatch):
    with pytest.raises(ValueError, match="only supports the test split"):
        load_mc("mmlu_redux", split="train")

    monkeypatch.setattr(
        "cachefuse.benchmarks.get_dataset_config_names",
        lambda name: ["subj"],
    )

    def fake_load_dataset(path, subject, split):
        assert path == "edinburgh-dawg/mmlu-redux-2.0"
        assert subject == "subj"
        assert split == "test"
        return [
            {"question": "ok?", "choices": ["a", "b"], "answer": 1},
            {"question": "bad?", "choices": ["a", "b"], "answer": 9},
        ]

    monkeypatch.setattr("cachefuse.benchmarks.load_dataset", fake_load_dataset)
    out = load_mc("mmlu_redux")
    assert out == [MCQuestion(question="ok?", options=["a", "b"], gold=1)]


def test_load_mc_mmlu_pro_skips_out_of_range_and_rejects_non_test(monkeypatch):
    with pytest.raises(ValueError, match="only supports the test split"):
        load_mc("mmlu_pro", split="validation")

    def fake_load_dataset(path, split):
        assert path == "TIGER-Lab/MMLU-Pro"
        assert split == "test"
        return [
            {"question": "ok?", "options": ["a", "b", "c"], "answer_index": 2},
            {"question": "bad?", "options": ["a"], "answer_index": -1},
        ]

    monkeypatch.setattr("cachefuse.benchmarks.load_dataset", fake_load_dataset)
    out = load_mc("mmlu_pro")
    assert out == [MCQuestion(question="ok?", options=["a", "b", "c"], gold=2)]
