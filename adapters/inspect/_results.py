"""Log parsing and result extraction for the Inspect AI adapter."""

import logging
import math
import os
from pathlib import Path
from typing import Any

from evalhub.adapter import CapabilityEvalEntry, EvaluationResult
from evalhub.models.api import PrimaryScore

from _benchmarks import DIMENSION_ABILITY_MAP, FIXED_PRIMARY_METRICS, fixed_primary_for

logger = logging.getLogger(__name__)

# Inspect Evals' DocVQA scorer reports the ANLS aggregate under the generic
# ``docvqa_scorer/accuracy`` key. Publish the benchmark's canonical metric name
# so collection configs can select it without changing the computed value.
BENCHMARK_METRIC_NAME_MAP: dict[tuple[str, str], str] = {
    ("inspect/docvqa", "docvqa_scorer/accuracy"): "anls",
}


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

    if os.environ.get("INSPECT_DEBUG_COPY_LOG"):
        debug_copy = Path("/tmp") / f"inspect_eval_log_{log_file.stem}.json"
        try:
            debug_copy.write_text(json.dumps(data, indent=2))
            debug_copy.chmod(0o600)
            logger.info(f"Inspect log copy written to {debug_copy}")
        except OSError as exc:
            logger.warning(f"Could not write debug log copy: {exc}")

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

    logger.info(f"extract_results | top-level keys={list(eval_log.keys())}")
    logger.info(f"extract_results | results keys={list(results_section.keys())}")
    logger.info(f"extract_results | scores count={len(scores_list)} | num_samples={num_samples}")
    for i, s in enumerate(scores_list):
        logger.info(f"extract_results | score[{i}]={s}")

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
            metric_name = BENCHMARK_METRIC_NAME_MAP.get(
                (benchmark_id, full_name), full_name
            )
            ability = DIMENSION_ABILITY_MAP.get(
                dim_name,
                "alignment_risk" if mode in ("petri", "bloom") else "reasoning",
            )

            metadata = {
                "framework": "inspect-ai",
                "mode": mode,
                "scorer": scorer,
                "dimension": dim_name,
            }
            if metric_name != full_name:
                metadata["native_metric_name"] = full_name

            evaluation_results.append(EvaluationResult(
                metric_name=metric_name,
                metric_value=float(value),
                metric_type="float",
                num_samples=num_samples,
                metadata=metadata,
            ))
            capability_entries.append(CapabilityEvalEntry(
                ability=ability,
                benchmark=benchmark_id,
                metric=metric_name,
                zero_shot=float(value),
            ))

    return evaluation_results, capability_entries, num_samples


# Dispersion metrics describe the spread of a score, not the score itself, so they must
# not be averaged into the overall score (accuracy 1.0 with stderr 0.0 is 1.0, not 0.5).
_DISPERSION_METRICS = frozenset({"stderr", "bootstrap_stderr", "std", "var"})


def _is_dispersion_metric(metric_name: str) -> bool:
    """Recognize dispersion metrics, including category-specific standard errors."""
    name = metric_name.rsplit("/", 1)[-1]
    return name in _DISPERSION_METRICS or name.endswith(("_stderr", "_sterr"))


def _score_metric_names(results: list[EvaluationResult]) -> list[str]:
    """Non-dispersion metric names with finite numeric values."""
    names: list[str] = []
    for r in results:
        if _is_dispersion_metric(r.metric_name):
            continue
        if not isinstance(r.metric_value, (int, float)) or isinstance(r.metric_value, bool):
            continue
        if not math.isfinite(float(r.metric_value)):
            continue
        names.append(r.metric_name)
    return names


def resolve_primary_score(
    results: list[EvaluationResult],
    *,
    mode: str,
    benchmark_id: str | None = None,
    job_primary: PrimaryScore | None = None,
) -> PrimaryScore | None:
    """Pick the primary metric key to report on the completed status event.

    Preference:
    1. Job/catalog ``primary_score.metric`` when it exactly matches an emitted score
    2. Known fixed benchmarks (HLE, BFCL, Petri/Bloom)
    3. Unique ``*/accuracy`` (or bare ``accuracy``) among non-dispersion scores
    4. Unique non-dispersion score metric

    EvalHub builds ``results.benchmarks[].test`` from this event field when the
    catalog primary key is absent from metrics (e.g. inspect/custom).
    """
    score_names = _score_metric_names(results)
    if not score_names:
        return None

    if job_primary is not None and job_primary.metric and job_primary.metric in score_names:
        return PrimaryScore(
            metric=job_primary.metric,
            lower_is_better=bool(job_primary.lower_is_better),
        )

    fixed = fixed_primary_for(benchmark_id, mode)
    if fixed is not None and fixed.metric in score_names:
        return PrimaryScore(
            metric=fixed.metric,
            lower_is_better=fixed.lower_is_better,
        )
    # HLE/BFCL catalog primaries must not fall back to a secondary score when
    # accuracy is missing (same policy as compute_overall_score).
    if benchmark_id is not None and benchmark_id in FIXED_PRIMARY_METRICS:
        logger.warning(
            "Fixed primary_score metric %s is missing | benchmark_id=%s | candidates=%s",
            FIXED_PRIMARY_METRICS[benchmark_id].metric,
            benchmark_id,
            score_names,
        )
        return None

    accuracy_like = [
        n for n in score_names
        if n == "accuracy" or n.endswith("/accuracy")
    ]
    # Preserve order, unique
    seen: set[str] = set()
    accuracy_unique: list[str] = []
    for n in accuracy_like:
        if n not in seen:
            seen.add(n)
            accuracy_unique.append(n)
    if len(accuracy_unique) == 1:
        return PrimaryScore(metric=accuracy_unique[0])

    unique_scores = list(dict.fromkeys(score_names))
    if len(unique_scores) == 1:
        return PrimaryScore(metric=unique_scores[0])

    logger.warning(
        "Could not resolve a unique primary_score metric | benchmark_id=%s | candidates=%s",
        benchmark_id,
        unique_scores,
    )
    return None


def compute_overall_score(
    results: list[EvaluationResult], mode: str, benchmark_id: str | None = None
) -> float | None:
    """Representative score for logging and ``JobResults.overall_score``.

    HLE and BFCL report their accuracy metric. Petri/Bloom report
    ``concerning/mean``. For other benchmarks this remains the mean of
    the score metrics, excluding dispersion metrics such as stderr. EvalHub selects the
    primary metric for pass/fail from the event/job ``primary_score.metric``.
    """
    if not results:
        return None

    def selected_score(metric_name: str) -> float | None:
        matches = [r for r in results if r.metric_name == metric_name]
        if len(matches) != 1 or not math.isfinite(float(matches[0].metric_value)):
            logger.warning("Representative metric %s is missing, ambiguous or non-finite", metric_name)
            return None
        return round(float(matches[0].metric_value), 4)

    fixed = fixed_primary_for(benchmark_id, mode)
    if fixed is not None:
        if mode in ("petri", "bloom"):
            primary = next(
                (r for r in results if r.metric_name == fixed.metric),
                None,
            )
            if primary is not None:
                return round(float(primary.metric_value), 4)
        else:
            return selected_score(fixed.metric)

    values = [
        float(r.metric_value)
        for r in results
        if isinstance(r.metric_value, (int, float))
        and r.metric_value == r.metric_value
        and not _is_dispersion_metric(r.metric_name)
    ]
    return round(sum(values) / len(values), 4) if values else None
