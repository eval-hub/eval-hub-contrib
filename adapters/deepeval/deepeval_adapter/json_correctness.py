"""Direct JSON validity and JSON Schema validation (no judge model, no metric)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import jsonschema
from evalhub.adapter import CapabilityEvalEntry, EvaluationResult, SafetyEvalEntry

logger = logging.getLogger(__name__)

CardEntries = list[CapabilityEvalEntry | SafetyEvalEntry]


def _load_schema(params: Mapping[str, Any]) -> tuple[Any, bool]:
    """Return ``(schema, supplied)`` from ``parameters.json_schema``.

    "Supplied" is tested with ``is not None``, not truthiness: ``false`` is a
    valid JSON Schema that rejects every instance, and ``{}`` / ``true`` accept
    every instance. A schema string that is not JSON falls back to a
    validity-only run, but a schema that parses and is malformed is a
    configuration error and fails the job rather than scoring every record 0.
    """
    schema = params.get("json_schema")
    if isinstance(schema, str):
        if not schema.strip():
            return None, False
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            logger.warning(
                "parameters.json_schema could not be parsed; falling back to validity-only"
            )
            return None, False

    if schema is None:
        return None, False

    if not isinstance(schema, (dict, bool)):
        # A ValueError, not a TypeError: callers treat it as a configuration error.
        raise ValueError(  # noqa: TRY004
            "parameters.json_schema is not a valid JSON Schema: "
            f"expected an object or boolean, got {type(schema).__name__}"
        )
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ValueError(
            f"parameters.json_schema is not a valid JSON Schema: {exc.message}"
        ) from exc
    return schema, True


def _score_record(output: Any, schema: Any, has_schema: bool) -> tuple[float, str]:
    """Return ``(score, reason)`` for one ``actual_output`` value."""
    try:
        if output is None:
            raise ValueError("actual_output is null")
        parsed = json.loads(output if isinstance(output, str) else str(output))
    except (json.JSONDecodeError, ValueError) as exc:
        return 0.0, f"Invalid JSON: {exc}"

    if not has_schema:
        return 1.0, "Valid JSON"
    try:
        jsonschema.validate(
            instance=parsed,
            schema=schema,
            format_checker=jsonschema.FormatChecker(),
        )
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"Schema violation: {exc}"
    return 1.0, "Valid JSON conforming to schema"


def run_json_correctness(
    records: Sequence[Mapping[str, Any]], params: Mapping[str, Any]
) -> tuple[list[EvaluationResult], CardEntries]:
    """Validate JSON output correctness without a DeepEval metric or LLM judge.

    Uses json.loads() for validity and jsonschema.validate() (if a schema is
    provided) for schema conformance, including ``format`` keywords.
    """
    schema, has_schema = _load_schema(params)

    results: list[EvaluationResult] = []
    scores: list[float] = []

    for i, rec in enumerate(records):
        score, reason = _score_record(rec.get("actual_output"), schema, has_schema)
        scores.append(score)
        results.append(
            EvaluationResult(
                metric_name=f"case_{i}.json_valid",
                metric_value=score,
                metric_type="float",
                num_samples=1,
                metadata={"success": score >= 1.0, "reason": reason},
            )
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    results.append(
        EvaluationResult(
            metric_name="json_correctness_score",
            metric_value=round(mean_score, 6),
            metric_type="float",
            num_samples=len(scores),
        )
    )
    results.append(
        EvaluationResult(
            metric_name="schema_valid",
            metric_value=1 if mean_score >= 1.0 else 0,
            metric_type="int",
            num_samples=len(scores),
        )
    )

    card_entries: CardEntries = [
        CapabilityEvalEntry(
            ability="format",
            benchmark="deepeval/json-correctness",
            metric="json_correctness_score",
            zero_shot=round(mean_score, 4),
        )
    ]
    return results, card_entries
