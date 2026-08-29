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


def extract_evaluation_results(raw: dict[str, Any]) -> list[EvaluationResult]:
    """Convert an lm-eval results dict to a list of EvaluationResult objects.

    lm-eval metric keys use a "<name>,<filter>" format (e.g. "acc,none",
    "exact_match,strict-match"). We strip the filter suffix for the metric name
    and skip the paired stderr entries (they are captured in confidence_interval).
    """
    out: list[EvaluationResult] = []
    results_section = raw.get("results", {})

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

            out.append(
                EvaluationResult(
                    metric_name=f"{clean_task}.{metric_base}",
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
            )

    logger.info("Extracted %d metrics from %d tasks", len(out), len(results_section))
    return out


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
