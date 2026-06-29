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
    output_dir = results_base / "test_run"

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
