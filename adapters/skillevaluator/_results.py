"""Map SkillEvaluator Tier 1 JSON reports onto EvalHub metrics.

Pure functions only (no I/O, no subprocess), so the report contract can be
pinned with golden fixtures. The report envelope shared by every Tier 1 command
is::

    overall_status      "passed" | "failed" | "incomplete"
    overall_passed      bool
    incomplete_scans    [str]       scanners that could not finish
    severity_counts     {critical, high, medium, low}
    results[]           per validator: status, summary, findings[], legacy{}

``quality-check`` adds ``quality_summary[]`` with a 0-100 ``overall_score``, a
letter grade and four weighted dimensions. ``publication_status`` is ignored:
it reads ``incomplete`` whenever Tier 2 or Tier 3 did not run, which is the
normal case for a Tier 1 job.

Three outcomes are kept apart on purpose:

* ``passed`` / ``failed``: the check completed; ``failed`` is a real verdict
  (score ``0.0``), not a missing value.
* ``incomplete``: the check could not finish (scanner missing, or the skill
  references files outside its own directory so analysis was partial). No
  score and no severity counts are produced: recording ``0.0`` would be
  indistinguishable from a verdict of "worst", and SkillEvaluator zeroes the
  counts of a partial scan, which would read as "no findings".

Skill content is attacker-controlled. Finding messages, matched lines and
suggestions are therefore never copied into metrics or logs; only counts and
the scanner's own rule names are exposed. The full report stays available as
the OCI artifact.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evalhub.adapter import EvaluationResult

OUTCOME_PASSED = "passed"
OUTCOME_FAILED = "failed"
OUTCOME_INCOMPLETE = "incomplete"
_OUTCOMES = frozenset({OUTCOME_PASSED, OUTCOME_FAILED, OUTCOME_INCOMPLETE})

QUALITY_DIMENSIONS = ("correctness", "discoverability", "reliability", "efficiency")
SEVERITIES = ("critical", "high", "medium", "low")

_MAX_REASONS = 5
_MAX_REASON_CHARS = 300
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
#: ``legacy`` message lists mix tool status lines with finding lines such as
#: ``[SECURITY-HIGH] Supply Chain: <matched skill text>``. Finding lines quote
#: attacker-controlled content and are never used as a failure reason.
_FINDING_LINE = re.compile(r"^\s*\[[A-Z_]+-[A-Z]+\]")


class ReportSchemaError(ValueError):
    """The report does not match the contract this adapter was written against."""


@dataclass
class Outcome:
    """Verdict of one Tier 1 command, read from the report envelope."""

    status: str
    incomplete_scans: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Mapped:
    """Everything the adapter needs from one report."""

    results: list[EvaluationResult]
    overall_score: float | None
    outcome: Outcome
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean(text: object) -> str:
    """Strip control characters and bound the length of text taken from a report."""
    return _CONTROL_CHARS.sub("", str(text))[:_MAX_REASON_CHARS]


def _require_dict(value: Any, what: str) -> dict[str, Any]:
    """Return ``value`` if it is an object, else fail loudly naming the report field."""
    if not isinstance(value, dict):
        raise ReportSchemaError(f"report field '{what}' is missing or not an object")
    return value


def classify(report: dict[str, Any]) -> Outcome:
    """Read the command verdict and, when incomplete, why."""
    status = report.get("overall_status")
    if status not in _OUTCOMES:
        raise ReportSchemaError(f"unexpected overall_status {_clean(status)!r}")
    scans = report.get("incomplete_scans") or []
    if not isinstance(scans, list):
        raise ReportSchemaError("incomplete_scans is not a list")
    reasons: list[str] = []
    if status == OUTCOME_INCOMPLETE:
        for result in report.get("results") or []:
            legacy = result.get("legacy") if isinstance(result, dict) else None
            if not isinstance(legacy, dict):
                continue
            for key in ("errors", "warnings"):
                for line in legacy.get(key) or []:
                    if isinstance(line, str) and not _FINDING_LINE.match(line):
                        reasons.append(_clean(line))
    return Outcome(
        status=status,
        incomplete_scans=[_clean(s) for s in scans],
        reasons=reasons[:_MAX_REASONS],
    )


def skill_digest(report: dict[str, Any]) -> str | None:
    """Upstream's content digest of the evaluated skill tree, for provenance."""
    for result in report.get("results") or []:
        target = result.get("publication_target") if isinstance(result, dict) else None
        digest = target.get("skill_digest") if isinstance(target, dict) else None
        if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return digest
    return None


def _severity_counts(report: dict[str, Any]) -> dict[str, int]:
    """Read the per-severity finding counts, rejecting anything but non-negative integers."""
    counts = _require_dict(report.get("severity_counts"), "severity_counts")
    out: dict[str, int] = {}
    for sev in SEVERITIES:
        value = counts.get(sev, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReportSchemaError(
                f"severity_counts.{sev} is not a non-negative integer"
            )
        out[sev] = value
    return out


def _rule_counts(report: dict[str, Any]) -> dict[str, int]:
    """Occurrences per scanner rule name (the tool's own vocabulary, not skill text)."""
    rules: Counter[str] = Counter()
    for result in report.get("results") or []:
        if not isinstance(result, dict):
            continue
        for finding in result.get("findings") or []:
            if isinstance(finding, dict) and finding.get("check_name"):
                rules[_clean(finding["check_name"])] += 1
    return dict(rules.most_common(20))


def _float_metric(name: str, value: float, **meta: Any) -> EvaluationResult:
    """Build a float metric rounded to six places, with optional metadata."""
    return EvaluationResult(
        metric_name=name,
        metric_value=round(float(value), 6),
        metric_type="float",
        metadata=meta,
    )


def _int_metric(name: str, value: int) -> EvaluationResult:
    """Build an integer metric."""
    return EvaluationResult(
        metric_name=name, metric_value=int(value), metric_type="int"
    )


def _bool_metric(name: str, value: bool) -> EvaluationResult:
    """Build a boolean metric."""
    return EvaluationResult(
        metric_name=name, metric_value=bool(value), metric_type="bool"
    )


def map_quality(report: dict[str, Any], *, min_score: int) -> Mapped:
    """``quality-check``: composite 0-100 score normalised to 0..1, plus dimensions."""
    outcome = classify(report)
    summaries = report.get("quality_summary")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise ReportSchemaError("quality_summary must hold exactly one skill")
    summary = _require_dict(summaries[0], "quality_summary[0]")

    raw = summary.get("overall_score")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not 0 <= raw <= 100
    ):
        raise ReportSchemaError(
            "quality_summary[0].overall_score is not a number in 0..100"
        )
    dims = _require_dict(summary.get("dimensions"), "quality_summary[0].dimensions")

    results = [
        _float_metric("quality_score", raw / 100.0),
        EvaluationResult(
            metric_name="quality_grade",
            metric_value=_clean(summary.get("grade", "")),
            metric_type="str",
        ),
        _bool_metric("meets_min_score", raw >= min_score),
    ]
    issues = 0
    for name in QUALITY_DIMENSIONS:
        dim = _require_dict(dims.get(name), f"dimensions.{name}")
        score = dim.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 100
        ):
            raise ReportSchemaError(
                f"dimensions.{name}.score is not a number in 0..100"
            )
        results.append(
            _float_metric(f"{name}_score", score / 100.0, weight=dim.get("weight"))
        )
        count = dim.get("issues_count", 0)
        issues += count if isinstance(count, int) and not isinstance(count, bool) else 0
    results.append(_int_metric("issue_count", issues))

    return Mapped(
        results=results,
        overall_score=round(raw / 100.0, 6),
        outcome=outcome,
        metadata={
            "skill_type": _clean(summary.get("skill_type", "")),
            "grade": _clean(summary.get("grade", "")),
            "min_score": min_score,
        },
    )


def map_findings(report: dict[str, Any]) -> Mapped:
    """``security-scan`` / ``pii-scan``: severity counts and a pass/fail gate score.

    The gate verdict is the tool's own (``overall_status``), so policy changes
    upstream are not re-implemented here. ``gate_score`` is ``1.0``
    for ``passed``, ``0.0`` for ``failed`` and absent for ``incomplete``.
    """
    outcome = classify(report)
    counts = _severity_counts(report)
    complete = outcome.status != OUTCOME_INCOMPLETE

    if not complete:
        # SkillEvaluator discards the scanner's findings when analysis is partial and reports all-zero
        # counts. Zeros next to an incomplete scan would read as "nothing found", so no counts are emitted.
        return Mapped(
            results=[_bool_metric("scan_complete", False)],
            overall_score=None,
            outcome=outcome,
            metadata={
                "finding_rules": {},
                "incomplete_scans": outcome.incomplete_scans,
            },
        )

    score = 1.0 if outcome.status == OUTCOME_PASSED else 0.0
    results = [_float_metric("gate_score", score)]
    results += [_int_metric(f"{sev}_count", counts[sev]) for sev in SEVERITIES]
    results.append(_int_metric("finding_count", sum(counts.values())))
    results.append(
        _int_metric("high_or_critical_count", counts["high"] + counts["critical"])
    )
    results.append(_bool_metric("scan_complete", True))
    results.append(_bool_metric("gate_passed", outcome.status == OUTCOME_PASSED))

    return Mapped(
        results=results,
        overall_score=score,
        outcome=outcome,
        metadata={
            "finding_rules": _rule_counts(report),
            "incomplete_scans": outcome.incomplete_scans,
        },
    )
