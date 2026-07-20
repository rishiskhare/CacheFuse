# SPDX-License-Identifier: Apache-2.0
"""Run settings for the CacheFuse evaluation.

`Settings.from_cli` parses the command-line flags (``python -m cachefuse.evaluate
--help``); every knob has a sensible default, so ``Settings()`` is ready to run.
"""

# Standard
import argparse
from dataclasses import dataclass

BENCHMARKS: tuple[str, ...] = ("arc_challenge", "openbookqa", "mmlu_redux", "mmlu_pro")


@dataclass(frozen=True)
class Settings:
    """A full evaluation run: the model pair and the eval set.

    Defaults describe a Qwen3-8B -> Qwen3-4B pair (same KV geometry and depth). Both
    models share one `device` unless `sender_device` places the sender on a second GPU.
    """

    sender: str = "Qwen/Qwen3-8B"  # big "sharer" whose cache is reused
    receiver: str = "Qwen/Qwen3-4B"  # model that answers
    device: str = (
        "cuda:0"  # receiver's GPU (also the sender's unless sender_device set)
    )
    # If set, load the sender on a separate GPU (its prefilled cache is moved to
    # `device` for fusion). Lets a pair that won't fit one card span two GPUs.
    sender_device: str | None = None
    benchmark: str = "arc_challenge"
    split: str = "test"
    num_examples: int | None = None
    thetas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    t2t_max_new_tokens: int = 256  # cap on the sharer's t2t hint (same as C2C)
    out: str | None = None  # if set, write the run's results JSON to this path
    # Also score the t2t baseline as a full-split accuracy treatment. Each question
    # then costs a hint generation, so those runs are much slower.
    score_t2t: bool = False
    seed: int | None = 42  # transformers.set_seed for reproducible runs (None disables)
    # Per-question error policy: "skip" (default) logs the failure, records it, frees
    # CUDA memory, and continues so a single bad question (e.g. an OOM on an over-long
    # prompt) does not abort the whole run; "fail" re-raises.
    on_error: str = "skip"
    # If set, a JSONL checkpoint written one line per question as the run proceeds. A
    # rerun with the same path resumes (skips finished questions) and retries failures.
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validate the run settings, raising ValueError on any out-of-range value."""
        if self.benchmark not in BENCHMARKS:
            raise ValueError(
                f"benchmark must be one of {BENCHMARKS}, got {self.benchmark!r}"
            )
        if self.split not in ("train", "validation", "test"):
            raise ValueError(
                "split must be one of ('train', 'validation', 'test'), "
                f"got {self.split!r}"
            )
        if self.num_examples is not None and self.num_examples <= 0:
            raise ValueError(
                f"num_examples must be > 0 or None, got {self.num_examples}"
            )
        if self.t2t_max_new_tokens <= 0:
            raise ValueError(
                f"t2t_max_new_tokens must be > 0, got {self.t2t_max_new_tokens}"
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0 or None, got {self.seed}")
        if self.on_error not in ("fail", "skip"):
            raise ValueError(
                f"on_error must be 'fail' or 'skip', got {self.on_error!r}"
            )
        if not self.thetas:
            raise ValueError("thetas must be non-empty")
        for theta in self.thetas:
            if not 0.0 <= theta <= 1.0:
                raise ValueError(f"each theta must be in [0, 1], got {theta}")

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "Settings":
        """Build a validated `Settings` from command-line flags (or `argv`)."""
        p = argparse.ArgumentParser(
            prog="python -m cachefuse.evaluate",
            description=(
                "Evaluate training-free cross-model KV-cache reuse by a "
                "direct cache copy."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("--sender", default="Qwen/Qwen3-8B", help="big sharer model")
        p.add_argument("--receiver", default="Qwen/Qwen3-4B", help="answerer model")
        p.add_argument(
            "--device",
            default="cuda:0",
            help="receiver's GPU (also the sender's unless --sender-device is set)",
        )
        p.add_argument(
            "--sender-device",
            default=None,
            help="place the sender on a separate GPU, e.g. cuda:1, to fit a pair that "
            "won't share one card",
        )
        p.add_argument("--benchmark", choices=BENCHMARKS, default="arc_challenge")
        p.add_argument(
            "--split",
            choices=("train", "validation", "test"),
            default="test",
            help="dataset split to evaluate",
        )
        p.add_argument(
            "--num-examples",
            type=int,
            default=None,
            help="evaluate only the first N examples (default: entire split)",
        )
        p.add_argument(
            "--theta",
            default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
            help="comma-separated fusion thetas in [0,1]",
        )
        p.add_argument(
            "--t2t-max-new-tokens",
            type=int,
            default=256,
            help="max tokens the sharer generates for the text-to-text baseline hint",
        )
        p.add_argument(
            "--out",
            default=None,
            help="path to write the run's results JSON (default: none)",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=42,
            help="random seed (transformers.set_seed) for reproducibility",
        )
        p.add_argument(
            "--on-error",
            choices=("fail", "skip"),
            default="skip",
            help="per-question error policy: fail (raise) or skip (log and continue)",
        )
        p.add_argument(
            "--score-t2t",
            action="store_true",
            help="score the t2t baseline as a full-split accuracy treatment "
            "(per-question generation; slower)",
        )
        p.add_argument(
            "--checkpoint",
            default=None,
            help="JSONL checkpoint path; rerun with the same path to resume and retry "
            "failed questions",
        )
        a = p.parse_args(argv)
        return cls(
            sender=a.sender,
            receiver=a.receiver,
            device=a.device,
            sender_device=a.sender_device,
            benchmark=a.benchmark,
            split=a.split,
            num_examples=a.num_examples,
            thetas=tuple(float(x) for x in a.theta.split(",")),
            t2t_max_new_tokens=a.t2t_max_new_tokens,
            out=a.out,
            score_t2t=a.score_t2t,
            seed=a.seed,
            on_error=a.on_error,
            checkpoint=a.checkpoint,
        )
