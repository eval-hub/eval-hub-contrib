"""Integration tests for the LightEval adapter.

Verifies adapter plumbing with a single monkeypatch point -- _run_lighteval
returns parsed results inline, so no filesystem setup is needed.
"""

import pytest
from pathlib import Path
from unittest.mock import create_autospec

from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import LightEvalAdapter

# Canned output matching LightEval's results JSON structure.
# Based losely on https://github.com/huggingface/lighteval/blob/main/docs/source/saving-and-reading-results.mdx#general-configuration
CANNED_RESULTS = {
    "results": {
        "boolq": {
            "accuracy": 0.78,
            "accuracy_stderr": 0.02,
        }
    },
    "config_general": {
        "max_samples": 5,
    },
}


@pytest.mark.integration
def test_lighteval_happy_path(monkeypatch):
    """Full run_benchmark_job with mocked _run_lighteval returning canned results."""
    adapter = LightEvalAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    # Single patch -- _run_lighteval returns parsed results directly
    monkeypatch.setattr(
        adapter, "_run_lighteval",
        lambda **kwargs: CANNED_RESULTS,
    )

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0

    # Metrics extracted from canned data
    assert len(results.results) > 0
    assert any(r.metric_name == "boolq.accuracy" for r in results.results)
    boolq_acc = next(r for r in results.results if r.metric_name == "boolq.accuracy")
    assert boolq_acc.metric_value == 0.78
    assert boolq_acc.confidence_interval is not None

    # Overall score and example count
    assert results.overall_score == 0.78
    assert results.num_examples_evaluated == 5

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    assert JobPhase.PERSISTING_ARTIFACTS in phases


@pytest.mark.integration
def test_results_use_local_jobs_base_path(tmp_path):
    """Results are saved under local_jobs_base_path/results, not hardcoded /tmp paths."""
    import shutil

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")

    adapter = LightEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    expected_base = adapter.local_jobs_base_path
    assert expected_base is not None, "local mode should provide local_jobs_base_path"
    assert expected_base == tmp_path
    expected_results_dir = expected_base / "results"

    saved_files = adapter._save_detailed_results(
        job_id=adapter.job_spec.id,
        benchmark_id=adapter.job_spec.benchmark_id,
        model_name=adapter.job_spec.model.name,
        lighteval_results=CANNED_RESULTS,
        evaluation_results=[],
    )

    assert len(saved_files) > 0
    for f in saved_files:
        assert str(f).startswith(str(expected_results_dir)), (
            f"Result file {f} should be under {expected_results_dir}"
        )
        assert "/tmp/lighteval_results" not in str(f)
