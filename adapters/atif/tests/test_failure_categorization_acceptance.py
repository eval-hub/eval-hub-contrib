"""Repeatable offline acceptance test for the ATIF failure taxonomy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from main import ATIFAdapter


CORPUS_DIR = Path(__file__).parent / "fixtures" / "failure_corpus"
JOB_SPEC_PATH = Path(__file__).resolve().parent.parent / "meta" / "job.json"
CRITERIA = {"criteria": [{"name": "failure detection", "weight": 1.0}]}


def _load_cases() -> list[dict[str, Any]]:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    cases: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        loaded = adapter._load_trajectories([CORPUS_DIR / case["file"]])
        assert len(loaded) == 1
        cases.append(loaded[0])
    return cases


def test_failure_categorization_acceptance_corpus() -> None:
    cases = _load_cases()
    assert len(cases) >= 8
    adapter = ATIFAdapter(job_spec_path=JOB_SPEC_PATH)
    categorization_requests: list[dict[str, Any]] = []

    async def judge_call(payload: dict[str, Any]) -> str:
        categorization_requests.append(payload)
        return json.dumps(
            {
                "category": "none",
                "confidence": 0.5,
                "rationale": "fixed response for request contract validation",
            }
        )

    adapter._judge_call = judge_call  # type: ignore[method-assign]
    for trajectory in cases:
        result = asyncio.run(
            adapter._categorize_failure(trajectory["steps"][0], 0.2, CRITERIA)
        )
        assert result["category"] == "none"
        assert result["categorization_status"] == "categorized"

    assert len(categorization_requests) == len(cases)
    expected_categories = {
        "tool_selection_failure",
        "context_loss",
        "policy_boundary_violation",
        "reasoning_failure",
        "none",
    }
    for trajectory, payload in zip(cases, categorization_requests, strict=True):
        assert payload["criteria"] == CRITERIA
        assert payload["step"] == trajectory["steps"][0]
        assert payload["score"] == 0.2
        assert set(payload["allowed_categories"]) == expected_categories
        assert payload["request"].startswith("Classify this low-scoring agent step.")
        assert "Return only JSON with category, confidence, and rationale" in payload[
            "request"
        ]
