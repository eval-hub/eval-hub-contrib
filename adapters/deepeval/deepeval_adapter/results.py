"""DeepEval output -> eval-hub results and EvalCard entries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evalhub.adapter import CapabilityEvalEntry, EvaluationResult, SafetyEvalEntry

from .benchmarks import find_benchmark

CardEntries = list[CapabilityEvalEntry | SafetyEvalEntry]


def extract_results(
    eval_results: Any, benchmark_id: str
) -> tuple[list[EvaluationResult], CardEntries]:
    """Map DeepEval output to (EvaluationResult list, card entry list).

    Emits one result per case and metric, the benchmark's primary aggregate
    (the mean score) and any supplementary aggregates its spec declares. An
    unregistered ``benchmark_id`` still yields a ``<id>_score`` aggregate.
    """
    spec = find_benchmark(benchmark_id)
    primary_key = spec.primary_metric if spec else f"{benchmark_id}_score"

    results: list[EvaluationResult] = []
    scores: list[float] = []

    for i, test_result in enumerate(eval_results.test_results):
        for md in test_result.metrics_data:
            score = md.score if md.score is not None else 0.0
            scores.append(score)
            results.append(
                EvaluationResult(
                    metric_name=f"case_{i}.{md.name}",
                    metric_value=round(score, 6),
                    metric_type="float",
                    num_samples=1,
                    metadata={"success": md.success, "reason": md.reason or ""},
                )
            )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    results.append(
        EvaluationResult(
            metric_name=primary_key,
            metric_value=round(mean_score, 6),
            metric_type="float",
            num_samples=len(scores),
        )
    )
    if spec and spec.extra_metrics:
        results.extend(spec.extra_metrics(scores, mean_score))

    entry_cls = SafetyEvalEntry if spec and spec.is_safety else CapabilityEvalEntry
    field = "feature" if entry_cls is SafetyEvalEntry else "ability"
    card_entries: CardEntries = [
        entry_cls(
            **{field: spec.ability if spec else benchmark_id},
            benchmark=f"deepeval/{benchmark_id}",
            metric=primary_key,
            zero_shot=round(mean_score, 4),
        )
    ]
    return results, card_entries


def compute_overall_score(
    results: Sequence[EvaluationResult], benchmark_id: str
) -> float | None:
    """Return the value of the benchmark's primary aggregate, if present."""
    spec = find_benchmark(benchmark_id)
    if spec is None:
        return None
    for r in results:
        if r.metric_name == spec.primary_metric:
            return r.metric_value
    return None
