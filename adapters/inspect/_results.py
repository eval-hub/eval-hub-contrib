"""Log parsing and result extraction for the Inspect AI adapter."""

import logging
from pathlib import Path
from typing import Any

from evalhub.adapter import CapabilityEvalEntry, EvaluationResult

from _benchmarks import DIMENSION_ABILITY_MAP, PETRI_PRIMARY_METRIC

logger = logging.getLogger(__name__)


def parse_log(log_file: Path) -> dict[str, Any]:
    import json
    with open(log_file) as f:
        data = json.load(f)

    status = data.get("status")
    if status not in ("success", "cancelled"):
        error_info = data.get("error", {}) or {}
        raise RuntimeError(
            f"inspect eval ended with status '{status}'. "
            f"Error: {error_info.get('message', 'unknown')}"
        )
    return data


def extract_results(
    eval_log: dict[str, Any],
    benchmark_id: str,
    mode: str,
) -> tuple[list[EvaluationResult], list[CapabilityEvalEntry], int]:
    results_section = eval_log.get("results") or {}
    scores_list: list[dict[str, Any]] = results_section.get("scores", [])

    samples = eval_log.get("samples") or []
    num_samples = len(samples) if samples else results_section.get("total_samples", 0)

    evaluation_results: list[EvaluationResult] = []
    capability_entries: list[CapabilityEvalEntry] = []

    for score_entry in scores_list:
        scorer  = score_entry.get("scorer") or score_entry.get("name") or "unknown"
        dim_name = score_entry.get("name") or scorer
        metrics: dict[str, Any] = score_entry.get("metrics", {})

        for metric_name, metric_data in metrics.items():
            value = metric_data.get("value") if isinstance(metric_data, dict) else metric_data
            if value is None:
                continue

            full_name = f"{dim_name}/{metric_name}"
            ability = DIMENSION_ABILITY_MAP.get(
                dim_name,
                "alignment_risk" if mode in ("petri", "bloom") else "reasoning",
            )

            evaluation_results.append(EvaluationResult(
                metric_name=full_name,
                metric_value=float(value),
                metric_type="float",
                num_samples=num_samples,
                metadata={"framework": "inspect-ai", "mode": mode, "scorer": scorer, "dimension": dim_name},
            ))
            capability_entries.append(CapabilityEvalEntry(
                ability=ability,
                benchmark=benchmark_id,
                metric=full_name,
                zero_shot=float(value),
            ))

    return evaluation_results, capability_entries, num_samples


def compute_overall_score(results: list[EvaluationResult], mode: str) -> float | None:
    if not results:
        return None

    if mode in ("petri", "bloom"):
        primary = next(
            (r for r in results if r.metric_name == f"{PETRI_PRIMARY_METRIC}/mean"),
            None,
        )
        if primary is not None:
            return round(float(primary.metric_value), 4)

    values = [
        float(r.metric_value)
        for r in results
        if isinstance(r.metric_value, (int, float)) and r.metric_value == r.metric_value
    ]
    return round(sum(values) / len(values), 4) if values else None
