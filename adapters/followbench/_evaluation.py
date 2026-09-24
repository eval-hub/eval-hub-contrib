"""FollowBench data loading, parsing, checks, and metric helpers."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


CATEGORIES = (
    "content",
    "example",
    "format",
    "mixed",
    "situation",
    "style",
)


@dataclass(frozen=True)
class FollowBenchExample:
    example_id: int
    category: str
    source: str
    level: int
    instruction: str
    target: str


def load_examples(data_dir: str | Path) -> list[FollowBenchExample]:
    """Load all pinned FollowBench JSON data files."""
    root = Path(data_dir)
    examples: list[FollowBenchExample] = []

    for category in CATEGORIES:
        path = root / f"{category}_constraints.json"

        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)

        for record in records:
            examples.append(
                FollowBenchExample(
                    example_id=int(record["example_id"]),
                    category=category,
                    source=str(record["source"]),
                    level=int(record["level"]),
                    instruction=str(record["instruction"]),
                    target=str(record.get("target", "")),
                )
            )

    return examples


def parse_judge_result(response: str, level: int) -> tuple[int, float]:
    """Parse a judge response into hard and soft satisfaction values."""
    if level < 1:
        raise ValueError("judge level must be at least 1")

    text = response.strip().strip("`").strip()

    if level == 1:
        value = text.upper()

        if "YES" in value:
            return 1, 1.0

        if "NO" in value:
            return 0, 0.0

        raise ValueError("judge response must contain YES or NO")

    start = text.find("[")
    end = text.rfind("]")

    if start < 0 or end <= start:
        raise ValueError("judge response does not contain a result list")

    parsed = ast.literal_eval(text[start : end + 1])

    if not isinstance(parsed, list) or len(parsed) != level:
        raise ValueError("judge result has the wrong number of elements")

    values = [str(item).upper() for item in parsed]
    allowed = {"YES", "NO", "PARTIAL", "MAYBE", "UNKNOWN", "N/A"}

    if any(value not in allowed for value in values):
        raise ValueError("judge result contains an invalid value")

    satisfied = sum(value == "YES" for value in values)

    return int(satisfied == level), satisfied / level


@dataclass(frozen=True)
class ScoredConstraint:
    example_id: int
    level: int
    hard_satisfied: bool
    soft_satisfied: float


def compute_metrics(results: list[ScoredConstraint]) -> dict[str, float]:
    """Compute normalized HSR, normalized SSR, and CSL on a 0–5 scale."""
    if not results:
        return {
            "hsr": 0.0,
            "ssr": 0.0,
            "csl": 0.0,
            "n_evaluated": 0.0,
        }

    hsr = sum(result.hard_satisfied for result in results) / len(results)
    ssr = sum(result.soft_satisfied for result in results) / len(results)

    grouped: dict[int, list[ScoredConstraint]] = {}

    for result in results:
        grouped.setdefault(result.example_id, []).append(result)

    consistency_total = 0

    for group in grouped.values():
        levels = {result.level: result for result in group}
        consecutive_levels = 0

        for level in range(1, 6):
            result = levels.get(level)

            if result is None or not result.hard_satisfied:
                break

            consecutive_levels += 1

        consistency_total += consecutive_levels

    csl = consistency_total / len(grouped)

    return {
        "hsr": hsr,
        "ssr": ssr,
        "csl": csl,
        "n_evaluated": float(len(results)),
    }