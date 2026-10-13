"""Unit tests for IFBench programmatic scoring."""

from __future__ import annotations

import pytest

from _evaluation import (
    InputExample,
    compute_scores,
    evaluate_instruction_following_loose,
    evaluate_instruction_following_strict,
)


# Mirrors the IFBench_test.jsonl shape: unused keys are null.
_NULL_KWARGS = {
    "N": None,
    "capital_frequency": None,
    "keyword": None,
    "keywords": None,
    "keyword1": None,
    "keyword2": None,
    "keyword3": None,
    "keyword4": None,
    "keyword5": None,
}


def _existence_example(*, prompt: str = "Say hello.") -> InputExample:
    kwargs = dict(_NULL_KWARGS)
    kwargs["keywords"] = ["zebra"]
    return InputExample(
        key="0",
        instruction_id_list=["keywords:existence"],
        prompt=prompt,
        kwargs=[kwargs],
    )


def test_none_kwargs_do_not_crash_strict():
    """Dataset rows ship many null fields; build_description must ignore them."""
    example = _existence_example()
    output = evaluate_instruction_following_strict(example, {example.prompt: "zebra"})
    assert output.follow_all_instructions is True


def test_none_kwargs_do_not_crash_loose():
    example = _existence_example()
    output = evaluate_instruction_following_loose(example, {example.prompt: "zebra"})
    assert output.follow_all_instructions is True


def test_empty_response_fails():
    example = _existence_example()
    output = evaluate_instruction_following_strict(example, {example.prompt: ""})
    assert output.follow_all_instructions is False
    assert output.follow_instruction_list == [False]


def test_missing_response_fails():
    example = _existence_example()
    output = evaluate_instruction_following_strict(example, {})
    assert output.follow_all_instructions is False
    assert output.response == ""


def test_known_good_completion_passes():
    example = _existence_example()
    output = evaluate_instruction_following_strict(
        example, {example.prompt: "The zebra is here."}
    )
    assert output.follow_all_instructions is True
    assert output.follow_instruction_list == [True]


def test_known_bad_completion_fails():
    example = _existence_example()
    output = evaluate_instruction_following_strict(
        example, {example.prompt: "The giraffe is here."}
    )
    assert output.follow_all_instructions is False


def test_compute_scores_prompt_and_instruction_level():
    example = _existence_example()
    passed = evaluate_instruction_following_strict(
        example, {example.prompt: "zebra"}
    )
    failed = evaluate_instruction_following_strict(
        example, {example.prompt: "nope"}
    )
    scores = compute_scores([passed, failed])
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["prompt_level_accuracy"] == pytest.approx(0.5)
    assert scores["instruction_level_accuracy"] == pytest.approx(0.5)
    assert scores["n_evaluated"] == pytest.approx(2.0)
