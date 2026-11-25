"""Extra aggregate metrics reported next to a benchmark's primary score."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from evalhub.adapter import EvaluationResult

# A case counts as passing, and an issue as detected, relative to this score.
PASS_SCORE = 0.5

ExtraMetrics = Callable[[Sequence[float], float], list[EvaluationResult]]


def faithfulness_counts(
    scores: Sequence[float], mean_score: float
) -> list[EvaluationResult]:
    """Report how many cases were scored and how many reached ``PASS_SCORE``."""
    return [
        EvaluationResult(
            metric_name="cases_count", metric_value=len(scores), metric_type="int"
        ),
        EvaluationResult(
            metric_name="passing_cases_count",
            metric_value=sum(1 for s in scores if s >= PASS_SCORE),
            metric_type="int",
        ),
    ]


def hallucination_flag(
    scores: Sequence[float], mean_score: float
) -> list[EvaluationResult]:
    """Flag a run whose mean score is below ``PASS_SCORE``.

    deepeval >= 4.2 scores these metrics higher-is-better (1.0 means no issue).
    """
    return [
        EvaluationResult(
            metric_name="hallucination_detected",
            metric_value=1 if mean_score < PASS_SCORE else 0,
            metric_type="int",
        )
    ]


def pii_flag(scores: Sequence[float], mean_score: float) -> list[EvaluationResult]:
    """Flag a run whose mean score is below ``PASS_SCORE`` (1.0 means no PII)."""
    return [
        EvaluationResult(
            metric_name="pii_detected",
            metric_value=1 if mean_score < PASS_SCORE else 0,
            metric_type="int",
        )
    ]
