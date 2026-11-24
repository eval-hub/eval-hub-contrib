"""The platform verdict needs a threshold: every benchmark's primary score must have a pass bar.

The EvalHub server reads the bar from ``pass_criteria``. With none at job or provider level it omits
the benchmark's ``Test`` section, so the job has no pass/fail verdict. These tests pin the thresholds
in ``provider.yaml`` to what the mapper actually emits.
"""

from pathlib import Path

import pytest
import yaml

import _results as res

PROVIDER = Path(__file__).resolve().parent.parent / "provider.yaml"


@pytest.fixture(scope="module")
def benchmarks():
    """Return the provider benchmarks keyed by id."""
    doc = yaml.safe_load(PROVIDER.read_text())
    return {b["id"]: b for b in doc["benchmarks"]}


def _by_name(mapped):
    """Map each result's metric name to its value."""
    return {r.metric_name: r.metric_value for r in mapped.results}


def _platform_pass(bench, metrics):
    """Mirror the server rule: pass when the primary metric is present and ``>=`` the threshold."""
    primary = bench["primary_score"]
    assert primary["lower_is_better"] is False
    if primary["metric"] not in metrics:
        return None  # no score: the server emits no Test section
    return metrics[primary["metric"]] >= bench["pass_criteria"]["threshold"]


@pytest.mark.parametrize(
    "bench_id, threshold",
    [("skill-quality", 0.7), ("skill-security", 1.0), ("skill-pii", 1.0)],
)
def test_every_benchmark_declares_primary_score_and_threshold(
    benchmarks, bench_id, threshold
):
    """Each benchmark has a primary score and the documented pass bar."""
    bench = benchmarks[bench_id]
    assert bench["primary_score"]["metric"]
    assert bench["pass_criteria"]["threshold"] == threshold


def test_no_benchmark_lacks_a_threshold(benchmarks):
    """Guard against a new benchmark that declares a primary score but no bar."""
    missing = [
        i
        for i, b in benchmarks.items()
        if "primary_score" in b and "pass_criteria" not in b
    ]
    assert not missing


def test_primary_metrics_are_emitted_by_the_mapper(benchmarks, report):
    """The declared primary metric is a name the mapper really emits."""
    quality = _by_name(res.map_quality(report("quality-pass"), min_score=70))
    assert benchmarks["skill-quality"]["primary_score"]["metric"] in quality
    for bench_id, name in (
        ("skill-security", "security-pass"),
        ("skill-pii", "pii-pass"),
    ):
        emitted = _by_name(res.map_findings(report(name)))
        assert benchmarks[bench_id]["primary_score"]["metric"] in emitted


@pytest.mark.parametrize(
    "bench_id, name", [("skill-security", "security-pass"), ("skill-pii", "pii-pass")]
)
def test_completed_clean_scan_passes_the_platform_bar(
    benchmarks, report, bench_id, name
):
    """gate_score 1.0 meets the 1.0 threshold."""
    metrics = _by_name(res.map_findings(report(name)))
    assert _platform_pass(benchmarks[bench_id], metrics) is True


def test_completed_failing_scan_fails_the_platform_bar(benchmarks, report):
    """gate_score 0.0 is a real failure, not a missing score."""
    metrics = _by_name(res.map_findings(report("security-failed")))
    assert _platform_pass(benchmarks["skill-security"], metrics) is False


def test_incomplete_scan_gets_no_platform_verdict(benchmarks, report):
    """An incomplete scan emits no gate_score, so it is neither passed nor failed."""
    metrics = _by_name(res.map_findings(report("security-incomplete-partial")))
    assert _platform_pass(benchmarks["skill-security"], metrics) is None


def test_quality_provider_bar_is_fixed_and_does_not_follow_min_score(
    benchmarks, report
):
    """Document the caveat: min_score only moves ``meets_min_score``; use a job-level bar to align."""
    mapped = res.map_quality(report("quality-pass"), min_score=95)
    metrics = _by_name(mapped)
    assert metrics["quality_score"] == pytest.approx(0.94)
    assert metrics["meets_min_score"] is False
    assert _platform_pass(benchmarks["skill-quality"], metrics) is True
