"""Parse lm-eval JSON output into EvalHub EvaluationResult objects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evalhub.adapter import EvaluationResult

logger = logging.getLogger(__name__)


def find_results_file(output_dir: Path) -> Path:
    """Locate the aggregated results JSON written by lm_eval.

    lm_eval writes to: <output_dir>/<model_name_slug>/results_<timestamp>.json
    Falls back to any results.json in the tree.
    """
    # Primary pattern: timestamped file
    candidates = sorted(
        output_dir.rglob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logger.debug("Found results file: %s", candidates[0])
        return candidates[0]

    # Fallback: plain results.json
    candidates = sorted(
        output_dir.rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        logger.debug("Found results file (fallback): %s", candidates[0])
        return candidates[0]

    available = [str(p) for p in output_dir.rglob("*") if p.is_file()]
    raise FileNotFoundError(
        f"No lm_eval results file found in {output_dir}.\nFiles present: {available}"
    )


def parse_results_file(results_file: Path) -> dict[str, Any]:
    """Load and return the lm_eval results JSON from disk."""
    with open(results_file) as f:
        return json.load(f)


#: lm-eval's filter name for results that were not post-processed by a filter.
_UNFILTERED = "none"


def extract_evaluation_results(
    raw: dict[str, Any], primary_task: str | None = None
) -> list[EvaluationResult]:
    """Convert an lm-eval results dict to a list of EvaluationResult objects.

    lm-eval metric keys use a "<name>,<filter>" format (e.g. "acc,none",
    "exact_match,strict-match"). The paired stderr entries are skipped (they are
    captured in confidence_interval).

    Names are emitted as ``<task>.<metric>`` for unfiltered results and
    ``<task>.<metric>,<filter>`` otherwise, so one task's results under several
    filters (GSM8K's ``strict-match`` and ``flexible-extract``) stay distinct.

    ``primary_task`` is the task whose results are the benchmark's own score.
    Its metrics are additionally emitted under the bare metric name (``acc_norm``),
    the form the LMEvalJob-based ``lm_evaluation_harness`` provider reports and
    that this provider's ``primary_score.metric`` declares. When a metric exists
    under several filters the unfiltered value wins, otherwise the first filter
    lm-eval reported (GSM8K: ``strict-match``).
    """
    out: list[EvaluationResult] = []
    results_section = raw.get("results", {})
    # bare metric name -> (filter, qualified name, result) for ``primary_task``
    primary: dict[str, tuple[str, str, EvaluationResult]] = {}

    for task_name, metrics in results_section.items():
        if not isinstance(metrics, dict):
            continue

        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            # lm-eval always uses "<metric>,<filter>" for scored keys.
            if "," not in key:
                continue
            metric_base, filter_name = key.rsplit(",", 1)
            # Skip stderr companion entries in both formats:
            #   three-part: "exact_match,none,stderr" → filter_name == "stderr"
            #   two-part:   "acc_stderr,none"         → metric_base ends with "_stderr"
            if filter_name == "stderr" or metric_base.endswith("_stderr"):
                continue

            # lm-eval uses two stderr key formats depending on task/version:
            #   three-part: "exact_match,none,stderr"  (metric,filter,stderr)
            #   two-part:   "acc_stderr,none"           (metric_stderr,filter)
            stderr = metrics.get(f"{metric_base},{filter_name},stderr")
            if stderr is None:
                stderr = metrics.get(f"{metric_base}_stderr,{filter_name}")

            confidence_interval: tuple[float, float] | None = None
            if isinstance(stderr, (int, float)):
                margin = 1.96 * float(stderr)
                confidence_interval = (float(val) - margin, float(val) + margin)

            # Strip the "|N" fewshot suffix lm-eval appends to task names.
            clean_task = task_name.split("|")[0]
            qualified = (
                metric_base
                if filter_name == _UNFILTERED
                else f"{metric_base},{filter_name}"
            )

            result = EvaluationResult(
                metric_name=f"{clean_task}.{qualified}",
                metric_value=float(val),
                metric_type="float",
                confidence_interval=confidence_interval,
                num_samples=_task_num_samples(raw, task_name),
                metadata={
                    "task": task_name,
                    "metric": metric_base,
                    "filter": filter_name,
                    "stderr": stderr,
                },
            )
            out.append(result)

            if primary_task is not None and clean_task == primary_task:
                kept = primary.get(metric_base)
                if kept is None or (
                    filter_name == _UNFILTERED and kept[0] != _UNFILTERED
                ):
                    primary[metric_base] = (filter_name, result.metric_name, result)

    # Bare-name aliases for the benchmark's own task. num_samples is left unset
    # so per-task sample totals are not counted twice.
    for metric_base, (filter_name, full_name, result) in primary.items():
        out.append(
            EvaluationResult(
                metric_name=metric_base,
                metric_value=result.metric_value,
                metric_type="float",
                confidence_interval=result.confidence_interval,
                num_samples=None,
                metadata={
                    **(result.metadata or {}),
                    "alias_of": full_name,
                },
            )
        )

    logger.info("Extracted %d metrics from %d tasks", len(out), len(results_section))
    return out


def effective_num_fewshot(raw: dict[str, Any], requested: Any = None) -> int | None:
    """Return the few-shot count the run actually used, or None if it is not one number.

    An explicit ``num_fewshot`` overrides every task. When it is omitted the
    harness keeps each task's built-in default (GSM8K runs 5-shot), which
    lm-eval reports per task under ``n-shot`` in the results JSON. None means
    the tasks disagree or lm-eval did not report it.
    """
    if requested is not None:
        return int(requested)
    n_shot = raw.get("n-shot")
    if not isinstance(n_shot, dict):
        return None
    counts = {
        int(v)
        for v in n_shot.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return counts.pop() if len(counts) == 1 else None


def _task_num_samples(raw: dict[str, Any], task_name: str) -> int | None:
    """Read the effective sample count for a task from the n-samples section."""
    entry = raw.get("n-samples", {}).get(task_name, {})
    if isinstance(entry, dict):
        return entry.get("effective") or entry.get("original")
    return None


def compute_overall_score(results: list[EvaluationResult]) -> float | None:
    """Return a representative score normalised to [0, 1].

    Prefers canonical accuracy metrics; falls back to the first numeric value.
    The EvalHub service selects the primary metric from the Evaluation Collection
    YAML — this value is only used for logging and MLflow tracking.
    """
    _PRIORITY_SUFFIXES = ("acc", "exact_match", "pass@1", "accuracy")
    for suffix in _PRIORITY_SUFFIXES:
        for r in results:
            if r.metric_name.endswith(f".{suffix}"):
                v = float(r.metric_value)
                return v / 100.0 if v > 1.0 else v
    # Fallback: first numeric result
    for r in results:
        v = r.metric_value
        if isinstance(v, (int, float)):
            v = float(v)
            return v / 100.0 if v > 1.0 else v
    return None
