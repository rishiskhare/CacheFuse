# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt rendering (no dataset download)."""

from cachefuse.benchmarks import (
    MCQuestion,
    build_prompt,
    build_t2t_prompt,
    render_question,
)


def q(text):
    return MCQuestion(question=text, options=["x", "y"], gold=0)


def test_render_question_lists_lettered_options():
    item = MCQuestion(question="What is 2+2?", options=["3", "4", "5"], gold=1)
    assert render_question(item) == "Question: What is 2+2?\nA. 3\nB. 4\nC. 5"


def test_build_prompt_ends_with_answer_marker():
    # Scoring reads the next-token argmax right after "Answer:"; the marker is load-
    # bearing, so pin it.
    prompt = build_prompt(q("anything"))
    assert prompt.endswith("\nAnswer:")


def test_build_t2t_prompt_prepends_background_and_keeps_answer_marker():
    out = build_t2t_prompt(q("anything"), hint="some context")
    assert out.startswith("Background: some context\n")
    assert out.endswith("\nAnswer:")
