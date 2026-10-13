"""Built-in benchmark rubrics for ATIF reference scoring.

The registry contains scoring instructions only.  It does not contain the
benchmark datasets and it never executes a benchmark verifier.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class UnknownBenchmarkError(ValueError):
    """Raised when benchmark mode names an unregistered benchmark."""


class RubricRegistry:
    """Small, deterministic registry of benchmark-specific scoring rubrics."""

    VERSION = 1

    _RUBRICS: dict[str, dict[str, Any]] = {
        "swe-bench-lite": {
            "description": "Assess software-engineering agent trajectories.",
            "criteria": [
                {
                    "name": "task_correctness",
                    "description": "The proposed change addresses the requested issue.",
                    "weight": 1.0,
                },
                {
                    "name": "patch_quality",
                    "description": "The change is focused, maintainable, and consistent with the repository.",
                    "weight": 1.0,
                },
                {
                    "name": "regression_safety",
                    "description": "The agent checks or preserves behavior outside the requested change.",
                    "weight": 1.0,
                },
            ],
        },
        "terminal-bench": {
            "description": "Assess terminal-use agent trajectories.",
            "criteria": [
                {
                    "name": "task_completion",
                    "description": "The agent completes the requested terminal task.",
                    "weight": 1.0,
                },
                {
                    "name": "tool_effectiveness",
                    "description": "The agent uses commands and tools appropriately to make progress.",
                    "weight": 1.0,
                },
                {
                    "name": "environment_safety",
                    "description": "The agent avoids unnecessary destructive or unsafe operations.",
                    "weight": 1.0,
                },
            ],
        },
        "humaneval": {
            "description": "Assess code-generation agent trajectories.",
            "criteria": [
                {
                    "name": "functional_correctness",
                    "description": "The produced solution satisfies the programming task.",
                    "weight": 1.0,
                },
                {
                    "name": "reasoning_quality",
                    "description": "The agent's reasoning and implementation choices are technically sound.",
                    "weight": 1.0,
                },
                {
                    "name": "code_quality",
                    "description": "The resulting code is clear, concise, and appropriate for the task.",
                    "weight": 1.0,
                },
            ],
        },
    }

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return benchmark names in stable order for errors and documentation."""
        return tuple(cls._RUBRICS)

    @classmethod
    def get(cls, benchmark_name: str) -> dict[str, Any]:
        """Return a copy of a registered rubric with its registry identity."""
        normalized = benchmark_name.strip().lower()
        rubric = cls._RUBRICS.get(normalized)
        if rubric is None:
            supported = ", ".join(cls.names())
            raise UnknownBenchmarkError(
                f"unknown benchmark {benchmark_name!r}; supported benchmarks: {supported}"
            )
        cls._validate(normalized, rubric)
        return {
            "rubric": normalized,
            "registry_version": cls.VERSION,
            **deepcopy(rubric),
        }

    @staticmethod
    def _validate(name: str, rubric: dict[str, Any]) -> None:
        criteria = rubric.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f"benchmark rubric {name!r} must contain criteria")
        seen: set[str] = set()
        for criterion in criteria:
            if not isinstance(criterion, dict) or not isinstance(
                criterion.get("name"), str
            ):
                raise ValueError(f"benchmark rubric {name!r} has an invalid criterion")
            criterion_name = criterion["name"]
            if criterion_name in seen:
                raise ValueError(
                    f"benchmark rubric {name!r} has duplicate criterion {criterion_name!r}"
                )
            seen.add(criterion_name)
            weight = criterion.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(
                    f"benchmark rubric {name!r} criterion {criterion_name!r} "
                    "weight must be numeric"
                )
            if weight <= 0:
                raise ValueError(
                    f"benchmark rubric {name!r} criterion {criterion_name!r} "
                    "weight must be positive"
                )
