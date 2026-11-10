"""Dataset records -> DeepEval test cases."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn

from .benchmarks import BENCHMARKS, BenchmarkSpec

logger = logging.getLogger(__name__)

Record = Mapping[str, Any]


def coerce_list(value: Any) -> list[str]:
    """Parse a string-encoded or native list into list[str]."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


def _missing_columns(record: Record, spec: BenchmarkSpec) -> list[str]:
    return [c for c in spec.required_columns if record.get(c) is None]


def _parse_tool_calls(raw: Any, field: str, *, with_output: bool) -> Any:
    """Parse a list of ``{name, input_parameters[, output]}`` dicts into ToolCalls.

    A value that cannot be parsed is passed through unchanged, with a warning, so
    DeepEval reports the problem instead of the adapter hiding it.
    """
    try:
        calls = json.loads(raw) if isinstance(raw, str) else raw
        return [
            ToolCall(
                name=call.get("name", ""),
                input_parameters=call.get("input_parameters", {}),
                **({"output": call.get("output")} if with_output else {}),
            )
            for call in calls
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s parsing failed, passing raw: %s", field, exc)
        return raw


def build_single_turn_test_cases(
    records: Sequence[Record], benchmark_id: str
) -> list[LLMTestCase]:
    """Convert raw records into DeepEval LLMTestCase objects.

    Records missing a required column are skipped with a warning; if none
    remain the job is misconfigured and this raises ``ValueError``.
    """
    spec = BENCHMARKS[benchmark_id]
    test_cases: list[LLMTestCase] = []

    for i, rec in enumerate(records):
        missing = _missing_columns(rec, spec)
        if missing:
            logger.warning("Skipping record %d: missing columns %s", i, missing)
            continue

        kwargs: dict[str, Any] = {
            "input": str(rec["input"]),
            "actual_output": str(rec["actual_output"]),
        }
        if rec.get("expected_output") is not None:
            kwargs["expected_output"] = str(rec["expected_output"])
        for field in ("retrieval_context", "context"):
            if rec.get(field) is not None:
                kwargs[field] = coerce_list(rec[field])
        if rec.get("tools_called") is not None:
            kwargs["tools_called"] = _parse_tool_calls(
                rec["tools_called"], "tools_called", with_output=True
            )
        if rec.get("expected_tools") is not None:
            kwargs["expected_tools"] = _parse_tool_calls(
                rec["expected_tools"], "expected_tools", with_output=False
            )

        test_cases.append(LLMTestCase(**kwargs))

    if not test_cases:
        raise ValueError(f"No valid test cases built from {len(records)} records")

    logger.info("Built %d single-turn test cases for %s", len(test_cases), benchmark_id)
    return test_cases


def build_conversational_test_cases(
    records: Sequence[Record], benchmark_id: str
) -> list[ConversationalTestCase]:
    """Convert raw records into DeepEval ConversationalTestCase objects."""
    spec = BENCHMARKS[benchmark_id]
    test_cases: list[ConversationalTestCase] = []

    for i, rec in enumerate(records):
        missing = _missing_columns(rec, spec)
        if missing:
            logger.warning("Skipping record %d: missing columns %s", i, missing)
            continue

        raw_turns = rec["turns"]
        if isinstance(raw_turns, str):
            raw_turns = json.loads(raw_turns)

        kwargs: dict[str, Any] = {
            "turns": [Turn(role=t["role"], content=t["content"]) for t in raw_turns]
        }
        for field in ("chatbot_role", "scenario", "expected_outcome"):
            if rec.get(field):
                kwargs[field] = str(rec[field])

        test_cases.append(ConversationalTestCase(**kwargs))

    if not test_cases:
        raise ValueError(
            f"No valid conversational test cases built from {len(records)} records"
        )

    logger.info(
        "Built %d conversational test cases for %s", len(test_cases), benchmark_id
    )
    return test_cases


def build_test_cases(
    records: Sequence[Record], spec: BenchmarkSpec
) -> list[LLMTestCase] | list[ConversationalTestCase]:
    """Build the test cases matching ``spec.kind``."""
    if spec.is_conversational:
        return build_conversational_test_cases(records, spec.id)
    return build_single_turn_test_cases(records, spec.id)
