# SPDX-License-Identifier: Apache-2.0
"""Tests for Settings validation and CLI parsing (no models, no GPU)."""

import pytest

from cachefuse.config import BENCHMARKS, Settings


def test_defaults_are_valid():
    cfg = Settings()
    assert cfg.benchmark in BENCHMARKS
    assert cfg.thetas == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_unknown_benchmark_rejected():
    with pytest.raises(ValueError, match="benchmark must be one of"):
        Settings(benchmark="not_a_benchmark")


def test_from_cli_parses_split():
    assert Settings.from_cli(["--split", "train"]).split == "train"


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="split must be one of"):
        Settings(split="dev")


def test_from_cli_parses_num_examples():
    assert Settings.from_cli(["--num-examples", "500"]).num_examples == 500


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_num_examples_rejected(bad):
    with pytest.raises(ValueError, match="num_examples must be"):
        Settings(num_examples=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_t2t_max_new_tokens_rejected(bad):
    with pytest.raises(ValueError, match="t2t_max_new_tokens must be > 0"):
        Settings(t2t_max_new_tokens=bad)


def test_empty_thetas_rejected():
    with pytest.raises(ValueError, match="thetas must be non-empty"):
        Settings(thetas=())


@pytest.mark.parametrize("bad_theta", [-0.1, 1.1, 2.0])
def test_out_of_range_theta_rejected(bad_theta):
    with pytest.raises(ValueError, match="each theta must be in"):
        Settings(thetas=(0.5, bad_theta))


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_boundary_thetas_allowed(edge):
    assert edge in Settings(thetas=(edge,)).thetas


def test_negative_seed_rejected():
    with pytest.raises(ValueError, match="seed must be >= 0"):
        Settings(seed=-1)


def test_seed_none_and_zero_allowed():
    assert Settings(seed=None).seed is None
    assert Settings(seed=0).seed == 0


@pytest.mark.parametrize("bad", ["", "retry", "FAIL", "raise"])
def test_invalid_on_error_rejected(bad):
    with pytest.raises(ValueError, match="on_error must be"):
        Settings(on_error=bad)


@pytest.mark.parametrize("good", ["fail", "skip"])
def test_valid_on_error_allowed(good):
    assert Settings(on_error=good).on_error == good


def test_cross_gpu_sender_device_allowed():
    cfg = Settings(device="cuda:0", sender_device="cuda:1")
    assert cfg.sender_device == "cuda:1"


def test_sender_device_same_as_device_allowed():
    cfg = Settings(device="cuda:0", sender_device="cuda:0")
    assert cfg.sender_device == "cuda:0"


def test_from_cli_defaults_match_settings():
    assert Settings.from_cli([]) == Settings()


def test_from_cli_parses_robustness_flags():
    cfg = Settings.from_cli(
        ["--seed", "7", "--on-error", "skip", "--checkpoint", "/tmp/run.jsonl"]
    )
    assert cfg.seed == 7
    assert cfg.on_error == "skip"
    assert cfg.checkpoint == "/tmp/run.jsonl"


def test_from_cli_robustness_defaults():
    cfg = Settings.from_cli([])
    assert cfg.seed == 42
    assert cfg.on_error == "skip"
    assert cfg.checkpoint is None


def test_score_t2t_defaults_off_and_flag_enables():
    assert Settings().score_t2t is False
    assert Settings.from_cli([]).score_t2t is False
    assert Settings.from_cli(["--score-t2t"]).score_t2t is True


def test_from_cli_parses_thetas_and_overrides():
    cfg = Settings.from_cli(["--benchmark", "openbookqa", "--theta", "0.1,0.9"])
    assert cfg.benchmark == "openbookqa"
    assert cfg.thetas == (0.1, 0.9)


def test_from_cli_invalid_theta_rejected_by_validation():
    with pytest.raises(ValueError, match="each theta must be in"):
        Settings.from_cli(["--theta", "0.5,3.0"])
