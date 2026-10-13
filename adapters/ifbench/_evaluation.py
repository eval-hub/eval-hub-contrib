"""IFBench evaluation helpers.

Adapted from allenai/IFBench evaluation_lib.py (Apache 2.0).
https://github.com/allenai/IFBench
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ifbench import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int | str
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, Any]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename: str | Path) -> list[InputExample]:
    """Read IFBench prompts from jsonl."""
    inputs: list[InputExample] = []
    with open(input_jsonl_filename, encoding="utf-8") as handle:
        for line in handle:
            example = json.loads(line)
            inputs.append(
                InputExample(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
            )
    return inputs


def _empty_output(inp: InputExample) -> OutputExample:
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response="",
        follow_all_instructions=False,
        follow_instruction_list=[False] * len(inp.instruction_id_list),
    )


def _get_response_for_prompt(inp: InputExample, prompt_to_response: dict[str, str]) -> str | None:
    if inp.prompt in prompt_to_response:
        return prompt_to_response[inp.prompt]
    return prompt_to_response.get(inp.prompt.strip())


def evaluate_instruction_following_strict(
    inp: InputExample,
    prompt_to_response: dict[str, str],
) -> OutputExample:
    """Evaluate whether all constraints are satisfied (strict matching)."""
    response = _get_response_for_prompt(inp, prompt_to_response)
    if response is None:
        return _empty_output(inp)

    is_following_list: list[bool] = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {
            key: value
            for key, value in inp.kwargs[index].items()
            if value is not None
        }
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response and response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def evaluate_instruction_following_loose(
    inp: InputExample,
    prompt_to_response: dict[str, str],
) -> OutputExample:
    """Evaluate constraint satisfaction with loose response normalization."""
    response = _get_response_for_prompt(inp, prompt_to_response)
    if response is None:
        return _empty_output(inp)

    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    is_following_list: list[bool] = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {
            key: value
            for key, value in inp.kwargs[index].items()
            if value is not None
        }
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for candidate in all_responses:
            if candidate.strip() and instruction.check_following(candidate):
                is_following = True
                break
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def compute_scores(outputs: list[OutputExample]) -> dict[str, float]:
    """Compute prompt-level and instruction-level accuracy from evaluation outputs."""
    if not outputs:
        return {
            "accuracy": 0.0,
            "prompt_level_accuracy": 0.0,
            "instruction_level_accuracy": 0.0,
            "n_evaluated": 0.0,
        }

    prompt_total = len(outputs)
    prompt_correct = sum(1 for example in outputs if example.follow_all_instructions)
    instruction_total = sum(len(example.instruction_id_list) for example in outputs)
    instruction_correct = sum(
        sum(example.follow_instruction_list) for example in outputs
    )

    prompt_accuracy = prompt_correct / prompt_total
    instruction_accuracy = (
        instruction_correct / instruction_total if instruction_total > 0 else 0.0
    )
    return {
        "accuracy": prompt_accuracy,
        "prompt_level_accuracy": prompt_accuracy,
        "instruction_level_accuracy": instruction_accuracy,
        "n_evaluated": float(prompt_total),
    }
