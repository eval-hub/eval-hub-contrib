"""Integration tests for the IBM CLEAR adapter.

Verifies adapter plumbing by mocking the module-level _run_clear_unified_pipeline
function and seeding the output directory with canned clear_results.json.

Unlike the other adapters, CLEAR imports its framework at module level
(not inside methods), so conftest.py injects mock modules into sys.modules
before this file is imported.
"""

import copy
import json
from pathlib import Path
from unittest.mock import create_autospec

import main
import pytest
from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import ClearAdapter, _normalize_clear_agent_entry

# Canned output analogous to CLEAR's clear_results.json structure.
# Schema: https://github.com/IBM/CLEAR (agentic pipeline output)
# The _extract_agentic_results method (Adapter for CLEAR) reads
# metadata.statistics.* and agents.*.agent_summary / issues_catalog.
CANNED_CLEAR_RESULTS = {
    "metadata": {
        "statistics": {
            "total_interactions_analyzed": 100,
            "total_issues_discovered": 25,
            "total_interactions_with_issues": 40,
            "total_interactions_no_issues": 60,
            "total_agents": 2,
        }
    },
    "agents": {
        "planner": {
            "agent_summary": {"avg_score": 0.85},
            "issues_catalog": {
                "incomplete_plan": {"count": 10},
                "wrong_tool": {"count": 5},
            },
        },
        "executor": {
            "agent_summary": {"avg_score": 0.90},
            "issues_catalog": {
                "timeout": {"count": 10},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Unit tests for _normalize_clear_agent_entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("experiment_source", ["job", "parameter", "missing"])
@pytest.mark.parametrize("oci_enabled", [False, True])
def test_mlflow_streams_framework_artifacts_independently_of_oci(
    tmp_path, experiment_source, oci_enabled
):
    from evalhub.adapter import JobResults
    from evalhub.adapter.mlflow import MlflowArtifact, MlflowFileArtifact
    from evalhub.models.api import OCICoordinates
    from unittest.mock import MagicMock

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job_path = meta_dir / "job.json"
    job_path.write_text(Path("meta/job.json").read_text())
    adapter = ClearAdapter(job_spec_path=str(job_path))
    config = copy.deepcopy(adapter.job_spec)
    config.experiment_name = "clear-stream" if experiment_source == "job" else None
    config.parameters.pop("mlflow_experiment_name", None)
    if experiment_source == "parameter":
        config.parameters["mlflow_experiment_name"] = "clear-stream"
    config.exports = MagicMock()
    config.exports.oci.coordinates = OCICoordinates(
        oci_host="quay.io", oci_repository="test/clear"
    )
    if not oci_enabled:
        config.exports = None
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    clear_file = output_dir / "clear_results.json"
    clear_file.write_text(json.dumps(CANNED_CLEAR_RESULTS))
    html_file = output_dir / "clear_results.html"
    html_file.write_text("<html>report</html>")
    results = JobResults(
        id=config.id, benchmark_id=config.benchmark_id, benchmark_index=0,
        model_name=config.model.name, results=[], num_examples_evaluated=0,
        duration_seconds=0,
    )
    callbacks = create_autospec(JobCallbacks)
    callbacks.mlflow = MagicMock()
    callbacks.mlflow.save.return_value = "clear-run"
    run_id = adapter._save_results_to_mlflow(
        callbacks, config, results, output_dir,
        adapter._metrics_summary_dict(config, [], None, 0), [html_file],
    )
    callbacks.create_oci_artifact.assert_not_called()
    assert not (tmp_path / "results").exists()
    if experiment_source == "missing":
        assert run_id is None
        callbacks.mlflow.save.assert_not_called()
        return
    assert run_id == "clear-run"
    assert callbacks.mlflow.save.call_args.args[1].experiment_name == "clear-stream"
    artifacts = callbacks.mlflow.save.call_args.kwargs["artifacts"]
    assert {artifact.path for artifact in artifacts} == {
        "clear_results.json", "metrics_summary.json", "clear_results.html"
    }
    for artifact in artifacts:
        if artifact.path == "metrics_summary.json":
            assert isinstance(artifact, MlflowArtifact)
            assert json.loads(artifact.content) == adapter._metrics_summary_dict(
                config, [], None, 0
            )
            continue
        assert isinstance(artifact, MlflowFileArtifact)
        assert artifact.local_path == output_dir / artifact.path
        assert artifact.local_path.is_file()
    assert not (output_dir / "metrics_summary.json").exists()
    assert json.loads(clear_file.read_text()) == CANNED_CLEAR_RESULTS


def test_normalize_clear_agent_entry_2x_reasoning_eval():
    """CLEAR 2.x: returns the reasoning_eval block when it contains agent_summary."""
    agent = {
        "reasoning_eval": {
            "agent_summary": {"avg_score": 0.75},
            "issues_catalog": {"wrong_tool": {"count": 3}},
        }
    }
    result = _normalize_clear_agent_entry(agent)
    assert result == agent["reasoning_eval"]
    assert result["agent_summary"]["avg_score"] == 0.75


def test_normalize_clear_agent_entry_2x_tools_eval_fallback():
    """CLEAR 2.x: falls back to tools_eval when reasoning_eval has no relevant keys."""
    agent = {
        "reasoning_eval": {"some_other_key": 1},
        "tools_eval": {
            "agent_summary": {"avg_score": 0.60},
            "issues": ["bad_call"],
        },
    }
    result = _normalize_clear_agent_entry(agent)
    assert result == agent["tools_eval"]


def test_normalize_clear_agent_entry_1x_flat_shape():
    """CLEAR 1.x: returns the agent dict itself when agent_summary is at top level."""
    agent = {
        "agent_summary": {"avg_score": 0.85},
        "issues_catalog": {"incomplete_plan": {"count": 10}},
    }
    result = _normalize_clear_agent_entry(agent)
    assert result is agent


def test_normalize_clear_agent_entry_empty_dict():
    """Returns empty dict when no recognisable keys are present."""
    assert _normalize_clear_agent_entry({}) == {}


def test_normalize_clear_agent_entry_non_dict():
    """Returns empty dict for non-dict input (e.g. None, string, list)."""
    assert _normalize_clear_agent_entry(None) == {}
    assert _normalize_clear_agent_entry("bad") == {}
    assert _normalize_clear_agent_entry([1, 2]) == {}


# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_clear_happy_path(monkeypatch, tmp_path):
    """Full run_benchmark_job with mocked pipeline and canned results."""
    adapter = ClearAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    # Create a fake data directory with a trace file
    # (run_benchmark_job validates *.json files exist in data_dir)
    data_dir = tmp_path / "traces"
    data_dir.mkdir()
    (data_dir / "trace1.json").write_text('{"trace": "data"}')

    # Control output location via results_dir/run_name parameters
    results_base = tmp_path / "results"

    # Override data_dir and output paths in job spec
    config = copy.deepcopy(adapter.job_spec)
    config.parameters["data_dir"] = str(data_dir)
    config.parameters["results_dir"] = str(results_base)
    config.parameters["run_name"] = "test_run"
    config.experiment_name = None  # skip MLflow save attempt

    # Patch _run_clear_unified_pipeline (module-level function, not a method)
    # to write canned results into output_dir instead of calling CLEAR.
    def fake_pipeline(agentic_config, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "clear_results.json").write_text(
            json.dumps(CANNED_CLEAR_RESULTS)
        )

    monkeypatch.setattr(main, "_run_clear_unified_pipeline", fake_pipeline)
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr("main.read_model_auth_key", {"api-key": "api-key:ref"}.get)
    monkeypatch.setattr("main.resolve_model_credentials", lambda: type("Credentials", (), {"api_key": "api-key:ref"})())

    results = adapter.run_benchmark_job(config, callbacks)

    # FrameworkAdapter contract
    assert results.id == config.id
    assert results.benchmark_id == config.benchmark_id
    assert results.benchmark_index == config.benchmark_index
    assert results.model_name == config.model.name
    assert results.duration_seconds > 0

    # Metrics extracted from canned data
    assert len(results.results) > 0
    metric_names = {r.metric_name for r in results.results}
    assert "total_interactions" in metric_names
    assert "total_issues" in metric_names
    assert "interactions_with_issues" in metric_names
    assert "total_agents" in metric_names
    assert "pct_interactions_with_issues" in metric_names
    assert "issues_per_interaction" in metric_names
    assert "agent.planner.avg_score" in metric_names
    assert "agent.executor.avg_score" in metric_names
    assert "average_score" in metric_names

    # Overall score is the average of agent scores: (0.85 + 0.90) / 2 = 0.875
    assert results.overall_score == (0.85 + 0.90) / 2
    assert results.num_examples_evaluated == 100

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases


@pytest.mark.integration
def test_artifacts_use_local_jobs_base_path(tmp_path):
    """OCI artifacts are saved under local_jobs_base_path/results, not hardcoded /tmp paths."""
    import shutil

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")

    adapter = ClearAdapter(job_spec_path=str(meta_dir / "job.json"))

    expected_base = adapter.local_jobs_base_path
    assert expected_base is not None, "local mode should provide local_jobs_base_path"
    assert expected_base == tmp_path
    expected_results_dir = expected_base / "results"

    fake_results = tmp_path / "clear_results.json"
    fake_results.write_text(json.dumps(CANNED_CLEAR_RESULTS))

    from unittest.mock import MagicMock
    from evalhub.models.api import OCICoordinates

    config = copy.deepcopy(adapter.job_spec)
    config.exports = MagicMock()
    config.exports.oci.coordinates = OCICoordinates(
        oci_host="quay.io", oci_repository="test/test",
    )

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    adapter._report_artifacts(
        config=config,
        callbacks=callbacks,
        json_results_path=fake_results,
        evaluation_results=[],
        overall_score=0.85,
        num_evaluated=100,
    )

    assert expected_results_dir.exists()
    assert (expected_results_dir / "clear_results.json").exists()
    assert (expected_results_dir / "metrics_summary.json").exists()
    assert "/tmp/clear_results" not in str(expected_results_dir)


def _native_interactions():
    return {"metadata": {"statistics": {"total_interactions_analyzed": 2}}, "agents": {
        "judge": {"reasoning_eval": {"no_issues": [
            {"trace_id": "trace-a", "input_output_pair": {"id": "a", "score": 1.0}},
            {"trace_id": "trace-b", "input_output_pair": {"id": "b", "score": 0.0}},
        ], "issues": []}}
    }}


def test_interaction_export_uses_native_scores_and_no_prompts():
    rows = main._interaction_samples(_native_interactions())
    assert [r["prediction"] for r in rows] == [1.0, 0.0]
    assert len({r["sample_id"] for r in rows}) == 2
    assert all(set(r) == {"sample_id", "trace_id", "prediction"} for r in rows)


def test_repeated_issue_occurrences_are_deduplicated():
    document = _native_interactions()
    block = document["agents"]["judge"]["reasoning_eval"]
    row = block["no_issues"].pop()
    block["issues"] = [{"occurrences": [row]}, {"occurrences": [copy.deepcopy(row)]}]
    assert len(main._interaction_samples(document)) == 2
    block["issues"][1]["occurrences"][0]["input_output_pair"]["score"] = 0.4
    with pytest.raises(ValueError, match="Conflicting"):
        main._interaction_samples(document)


@pytest.mark.parametrize("score", [None, True, "1", float("nan"), float("inf")])
def test_invalid_interaction_scores_are_rejected(score):
    document = _native_interactions()
    document["agents"]["judge"]["reasoning_eval"]["no_issues"][0]["input_output_pair"]["score"] = score
    with pytest.raises(ValueError, match="Invalid CLEAR interaction score"):
        main._interaction_samples(document)


def test_missing_interactions_are_not_silently_omitted():
    document = _native_interactions()
    document["agents"]["judge"]["reasoning_eval"]["no_issues"].pop()
    with pytest.raises(ValueError, match="count"):
        main._interaction_samples(document)


@pytest.mark.parametrize("enabled,failure", [
    (None, None), (False, None), (True, None), ("false", None),
    (True, "oci_exception"), (True, "oci_missing"),
    (True, "mlflow_exception"), (True, "mlflow_missing"),
])
def test_sample_persistence_full_adapter(tmp_path, monkeypatch, enabled, failure):
    from unittest.mock import MagicMock
    from evalhub.models.api import OCICoordinates

    meta = tmp_path / "meta"
    meta.mkdir()
    job = meta / "job.json"
    job.write_text(Path("meta/job.json").read_text())
    adapter = ClearAdapter(job_spec_path=str(job))
    config = copy.deepcopy(adapter.job_spec)
    data = tmp_path / "traces"
    data.mkdir()
    (data / "trace.json").write_text('{}')
    config.parameters.update(data_dir=str(data), results_dir=str(tmp_path), run_name="native")
    if enabled is not None:
        config.parameters["save_sample_results"] = enabled
    config.experiment_name = "clear-unit-export"
    config.exports = MagicMock()
    config.exports.oci.coordinates = OCICoordinates(oci_host="quay.io", oci_repository="test/results")
    callbacks = create_autospec(JobCallbacks)
    callbacks.mlflow = MagicMock()
    callbacks.mlflow.save.return_value = "unit-run"
    captured = {}

    def upload(spec):
        captured.update({p.name: p.read_bytes() for p in spec.files_path.iterdir()})
        return OCIArtifactResult(digest="sha256:unit", reference="test/results@sha256:unit")

    callbacks.create_oci_artifact.side_effect = upload
    if failure == "oci_exception":
        callbacks.create_oci_artifact.side_effect = RuntimeError("OCI upload denied")
    elif failure == "oci_missing":
        callbacks.create_oci_artifact.side_effect = None
        callbacks.create_oci_artifact.return_value = None
    elif failure == "mlflow_exception":
        callbacks.mlflow.save.side_effect = RuntimeError("MLflow upload denied")
    elif failure == "mlflow_missing":
        callbacks.mlflow.save.return_value = None

    def pipeline(params, output):
        document = _native_interactions()
        document["agents"]["judge"]["reasoning_eval"]["agent_summary"] = {"avg_score": 0.6}
        (output / "clear_results.json").write_text(json.dumps(document))

    monkeypatch.setattr(main, "_run_clear_unified_pipeline", pipeline)
    monkeypatch.setattr(main, "read_model_auth_key", {"api-key": "api-key:ref"}.get)
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: type("Credentials", (), {"api_key": "api-key:ref"})())
    if enabled == "false":
        with pytest.raises(ValueError, match="must be a boolean"):
            adapter.run_benchmark_job(config, callbacks)
        callbacks.create_oci_artifact.assert_not_called()
        return
    if failure:
        with pytest.raises(RuntimeError):
            adapter.run_benchmark_job(config, callbacks)
        event = callbacks.report_status.call_args.args[0]
        assert event.status.value == "failed" and event.error_message.message
        return
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.6
    metrics = {m.metric_name: m.metric_value for m in result.results}
    assert ("sample_results.jsonl" in captured) == (enabled is True)
    if enabled is True:
        assert metrics["interaction_average_score"] == 0.5
        mlflow_files = {a.path: a.local_path.read_bytes() for a in callbacks.mlflow.save.call_args.kwargs["artifacts"] if hasattr(a, "local_path")}
        assert mlflow_files["sample_results.jsonl"] == captured["sample_results.jsonl"]
        assert mlflow_files["manifest.json"] == captured["manifest.json"]
    else:
        assert "interaction_average_score" not in metrics


def test_export_matches_checked_in_clear_native_report():
    document = json.loads(Path("examples/output/local/clear_results.json").read_text())
    rows = main._interaction_samples(document)
    assert len(rows) == document["metadata"]["statistics"]["total_interactions_analyzed"] == 6


def test_interaction_and_agent_weighted_means_are_distinct(tmp_path):
    document = _native_interactions()
    first = document["agents"]["judge"]["reasoning_eval"]
    first["no_issues"][1]["input_output_pair"]["score"] = 1.0
    first["agent_summary"] = {"avg_score": 1.0}
    document["agents"]["other"] = {"reasoning_eval": {
        "agent_summary": {"avg_score": 0.0}, "no_issues": [
            {"trace_id": "trace-c", "input_output_pair": {"id": "c", "score": 0.0}}
        ], "issues": []}}
    document["metadata"]["statistics"]["total_interactions_analyzed"] = 3
    path = tmp_path / "clear_results.json"
    path.write_text(json.dumps(document))
    adapter = ClearAdapter(job_spec_path="meta/job.json")
    original = adapter._compute_overall_score(adapter._extract_agentic_results(str(path)))
    samples = main._interaction_samples(document)
    assert original == 0.5
    assert sum(row["prediction"] for row in samples) / len(samples) == pytest.approx(2 / 3)
