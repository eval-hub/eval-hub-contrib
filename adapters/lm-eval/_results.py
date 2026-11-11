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

#: Filters preferred, in order, for the bare-name alias when a metric exists only
#: under several filters. ``strict-match`` checks that the answer has the exact
#: template of the few-shot examples ("#### 42"), so it measures format
#: compliance as much as capability: a chat model run zero-shot (no examples to
#: copy) scored 0.00 strict-match and 0.75 flexible-extract on GSM8K. Which filter
#: a task lists first is an arbitrary choice of its author (gsm8k lists
#: strict-match first, gsm8k_platinum_cot_llama and the bbh_cot_zeroshot tasks differ),
#: so "first" is not a usable rule. The pass/fail verdict therefore uses
#: ``flexible-extract`` (the last number in the answer); every filter stays
#: reported under its qualified name (``gsm8k.exact_match,strict-match``).
_PREFERRED_FILTERS = ("flexible-extract",)


def _alias_rank(filter_name: str) -> int:
    """Lower wins: unfiltered, then a preferred filter, then whatever came first."""
    if filter_name == _UNFILTERED:
        return 0
    if filter_name in _PREFERRED_FILTERS:
        return 1 + _PREFERRED_FILTERS.index(filter_name)
    return 1 + len(_PREFERRED_FILTERS)


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
    under several filters the unfiltered value wins, then ``flexible-extract``
    (see ``_PREFERRED_FILTERS``), otherwise the first filter lm-eval reported.
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
                if kept is None or _alias_rank(filter_name) < _alias_rank(kept[0]):
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


_PRIORITY_METRICS = ("acc", "exact_match", "pass@1", "accuracy")


def headline_result(results: list[EvaluationResult]) -> EvaluationResult | None:
    """The result that stands for the benchmark in the EvalCard and the logs.

    A canonical accuracy metric (``_PRIORITY_METRICS``) wins; among those, the
    benchmark's own bare-name alias comes before a per-task row, and the filter
    follows ``_alias_rank`` so the card agrees with the verdict (a zero-shot GSM8K
    run is flexible-extract 0.75 in both, not strict-match 0.00 in the card).
    Without a canonical metric, the first numeric result that is an alias, else
    the first numeric result.
    """
    best: tuple[tuple[int, int, int, int], EvaluationResult] | None = None
    for order, r in enumerate(results):
        if not isinstance(r.metric_value, (int, float)):
            continue
        meta = r.metadata or {}
        base = meta.get("metric") or r.metric_name.rsplit(".", 1)[-1].split(",")[0]
        priority = (
            _PRIORITY_METRICS.index(base)
            if base in _PRIORITY_METRICS
            else len(_PRIORITY_METRICS)
        )
        is_alias = "alias_of" in meta
        key = (
            priority,
            0 if is_alias else 1,
            _alias_rank(meta.get("filter", _UNFILTERED)),
            order,
        )
        if best is None or key < best[0]:
            best = (key, r)
    return best[1] if best else None


def headline_metric_name(results: list[EvaluationResult]) -> str:
    """Label of ``headline_result`` for the EvalCard (``acc`` if there is none)."""
    r = headline_result(results)
    if r is None:
        return "acc"
    if "alias_of" in (r.metadata or {}):
        return r.metric_name
    parts = r.metric_name.rsplit(".", 1)
    return parts[1] if len(parts) == 2 else "acc"


def compute_overall_score(results: list[EvaluationResult]) -> float | None:
    """Return a representative score normalised to [0, 1] (see ``headline_result``).

    The EvalHub service selects the primary metric from the Evaluation Collection
    YAML — this value is only used for the EvalCard, logging and MLflow tracking.
    """
    r = headline_result(results)
    if r is None:
        return None
    v = float(r.metric_value)
    return v / 100.0 if v > 1.0 else v
