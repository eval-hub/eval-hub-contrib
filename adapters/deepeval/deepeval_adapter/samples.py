"""Per-sample CI artifacts (``parameters.save_sample_results``).

Two files are exported next to the results summary: ``sample_results.jsonl``
with one ``{sample_id, prediction, metric}`` row per test case, and
``manifest.json`` naming the benchmark's aggregate metric and its estimand.
Raw inputs and conversation turns are never written.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from evalhub.adapter import EvaluationResult
from evalhub.adapter.mlflow import MlflowArtifact

SAMPLE_FILE = "sample_results.jsonl"
MANIFEST_FILE = "manifest.json"
SAMPLE_FILES = (SAMPLE_FILE, MANIFEST_FILE)


def _check_score(score: Any, index: int) -> float:
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise ValueError(f"Invalid per-sample score for test result {index}")
    return score


def native_scores(raw_results: Any) -> list[float]:
    """Read one score per test case from DeepEval's own results.

    Call this before aggregation: ``extract_results`` replaces a missing score
    with 0.0, which would hide a failed metric.
    """
    scores: list[float] = []
    for index, case in enumerate(raw_results.test_results):
        if len(case.metrics_data) != 1:
            raise ValueError("Per-sample export requires exactly one metric per test case")
        scores.append(_check_score(case.metrics_data[0].score, index))
    return scores


def direct_scores(results: Sequence[EvaluationResult]) -> list[float]:
    """Read one score per record from a direct-validation benchmark's results."""
    return [
        _check_score(r.metric_value, index)
        for index, r in enumerate(x for x in results if x.metric_name.startswith("case_"))
    ]


def build_artifacts(scores: Sequence[float], metric: str) -> list[MlflowArtifact]:
    """Build the sample and manifest artifacts for ``metric``."""
    rows = "".join(
        json.dumps(
            {"sample_id": str(index), "prediction": score, "metric": metric},
            allow_nan=False,
        )
        + "\n"
        for index, score in enumerate(scores)
    )
    manifest = json.dumps({"primary_score": {"metric": metric}, "estimand": "mean"})
    return [
        MlflowArtifact(SAMPLE_FILE, rows.encode(), "application/x-ndjson"),
        MlflowArtifact(MANIFEST_FILE, manifest.encode(), "application/json"),
    ]
