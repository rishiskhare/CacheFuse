# SPDX-License-Identifier: Apache-2.0
"""Multiple-choice benchmark loading and prompting for the CacheFuse evaluation.

`load_mc` returns a uniform list of `MCQuestion`; `build_prompt` renders the prompt
scored by the option-letter argmax in `cachefuse.evaluate`.
"""

# Standard
from dataclasses import dataclass

# Third Party
from datasets import get_dataset_config_names, load_dataset

# First Party
from cachefuse.config import BENCHMARKS

# Option labels A..J (up to 10 options, for MMLU-Pro).
LETTERS = list("ABCDEFGHIJ")

# Benchmarks sharing the {question, choices:{text,label}, answerKey} schema; they differ
# only in the question field name (ARC "question" vs OpenBookQA "question_stem").
# Value: (hf_path, config, question_field).
_CHOICES: dict[str, tuple[str, str, str]] = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "question"),
    "openbookqa": ("allenai/openbookqa", "main", "question_stem"),
}


@dataclass(frozen=True)
class MCQuestion:
    """A multiple-choice question: the stem, option texts, and 0-based gold index."""

    question: str
    options: list[str]
    gold: int


def load_mc(bench: str, split: str = "test") -> list[MCQuestion]:
    """Load the requested split for `bench` as a uniform `MCQuestion` list.

    Each benchmark's exact fields:
    - ``arc_challenge`` / ``openbookqa``: stem in ``question``/``question_stem``,
      ``choices`` = ``{"text": [...], "label": [...]}``, and ``answerKey`` (the gold
      option's label); the gold index is ``answerKey``'s position in the label list.
    - ``mmlu_redux`` (``edinburgh-dawg/mmlu-redux-2.0``, 57 subjects): ``question``,
      ``choices`` (option strings), integer ``answer`` index; subjects concatenated.
    - ``mmlu_pro`` (``TIGER-Lab/MMLU-Pro``, ``test``): ``question``, ``options`` (up to
      10 strings), integer ``answer_index``; categories concatenated.

    For the two integer-indexed sets, rows with an out-of-range answer are skipped.
    """
    if bench not in BENCHMARKS:
        raise ValueError(f"bench must be one of {BENCHMARKS}, got {bench!r}")
    if bench in _CHOICES:
        path, config, qfield = _CHOICES[bench]
        out: list[MCQuestion] = []
        for ex in load_dataset(path, config, split=split):
            choices = ex["choices"]
            out.append(
                MCQuestion(
                    question=ex[qfield],
                    options=list(choices["text"]),
                    gold=list(choices["label"]).index(ex["answerKey"]),
                )
            )
        return out
    if bench == "mmlu_redux":
        if split != "test":
            raise ValueError("mmlu_redux only supports the test split")
        out = []
        for subject in get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0"):
            for ex in load_dataset(
                "edinburgh-dawg/mmlu-redux-2.0", subject, split="test"
            ):
                options, gold = list(ex["choices"]), ex["answer"]
                if 0 <= gold < len(options):
                    out.append(MCQuestion(ex["question"], options, gold))
        return out
    # mmlu_pro
    if split != "test":
        raise ValueError("mmlu_pro only supports the test split")
    out = []
    for ex in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        options, gold = list(ex["options"]), ex["answer_index"]
        if 0 <= gold < len(options):
            out.append(MCQuestion(ex["question"], options, gold))
    return out


def render_question(item: MCQuestion) -> str:
    """Render the question stem and lettered options (no trailing ``Answer:``).

    Shared by the scored prompt (`build_prompt`) and the text-to-text sharer instruction
    (`cachefuse.evaluate`), so receiver and sharer see identical question text.
    """
    lines = [f"Question: {item.question}"]
    for i, opt in enumerate(item.options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def build_prompt(item: MCQuestion) -> str:
    """Render a C2C-style multiple-choice prompt ending in ``Answer:`` for `item`."""
    return render_question(item) + "\nAnswer:"


def build_t2t_prompt(item: MCQuestion, hint: str) -> str:
    """Prepend the sharer's `hint` as ``Background:`` above the scored prompt.

    The prompt still ends in ``Answer:``, so the same option-letter argmax scores it.
    """
    return f"Background: {hint}\n{build_prompt(item)}"
