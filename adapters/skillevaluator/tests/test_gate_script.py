"""Tests for ``examples/gate-results.sh``, the CI gate over ``evalhub eval run --format json`` output."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "gate-results.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None, reason="jq is not installed"
)


def _job(statuses, results):
    """Build a job document shaped like ``evalhub eval run --format json`` output."""
    return [
        {
            "status": {
                "state": "completed",
                "benchmarks": [
                    {
                        "id": b,
                        "status": s,
                        **({"error_message": {"message_code": code}} if code else {}),
                    }
                    for b, s, code in statuses
                ],
            },
            "results": {"benchmarks": [{"id": b, "metrics": m} for b, m in results]},
        }
    ]


def _gate(tmp_path, job, **env):
    """Run the gate script on ``job`` and return the completed process."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(job))
    return subprocess.run(
        [str(SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", **env},
    )


def test_all_gates_passing_exits_zero(tmp_path):
    """Completed benchmarks with passing gates pass."""
    job = _job(
        [("skill-quality", "completed", None), ("skill-security", "completed", None)],
        [
            ("skill-quality", {"quality_score": 0.9, "meets_min_score": True}),
            ("skill-security", {"gate_score": 1.0, "scan_complete": True}),
        ],
    )
    assert _gate(tmp_path, job).returncode == 0


def test_partially_failed_job_fails_the_gate(tmp_path):
    """One failed benchmark fails the gate even though the CLI would exit 0 for PARTIALLY_FAILED."""
    job = _job(
        [
            ("skill-quality", "completed", None),
            ("skill-security", "failed", "scan_incomplete"),
        ],
        [("skill-quality", {"quality_score": 0.9, "meets_min_score": True})],
    )
    run = _gate(tmp_path, job)
    assert run.returncode == 1
    assert "skill-security: benchmark failed (scan_incomplete)" in run.stderr


def test_failed_gate_and_low_quality_fail(tmp_path):
    """A zero ``gate_score`` and ``meets_min_score=false`` are each reported."""
    job = _job(
        [("skill-quality", "completed", None), ("skill-security", "completed", None)],
        [
            ("skill-quality", {"quality_score": 0.5, "meets_min_score": False}),
            ("skill-security", {"gate_score": 0.0, "scan_complete": True}),
        ],
    )
    run = _gate(tmp_path, job)
    assert run.returncode == 1
    assert "skill-security: gate failed" in run.stderr
    assert "skill-quality: quality below min_score" in run.stderr


def test_incomplete_scan_fails_unless_allowed(tmp_path):
    """``scan_complete=false`` fails by default and is only a warning with ALLOW_INCOMPLETE=1."""
    job = _job(
        [("skill-security", "completed", None)],
        [("skill-security", {"scan_complete": False})],
    )
    assert _gate(tmp_path, job).returncode == 1
    allowed = _gate(tmp_path, job, ALLOW_INCOMPLETE="1")
    assert allowed.returncode == 0
    assert "warning: skill-security: scan incomplete" in allowed.stderr


def test_missing_results_fail(tmp_path):
    """A job with no benchmark results never passes."""
    assert _gate(tmp_path, _job([], [])).returncode == 1


def test_empty_or_invalid_input_fails_closed(tmp_path):
    """Empty output (the CLI exited before printing JSON) or invalid JSON must never pass."""
    for content in ("", "not json", "42"):
        path = tmp_path / "result.json"
        path.write_text(content)
        run = subprocess.run(
            [str(SCRIPT), str(path)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        )
        assert run.returncode == 1, content
        assert "no job result" in run.stderr
