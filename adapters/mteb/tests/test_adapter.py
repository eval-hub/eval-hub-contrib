"""Integration tests for the MTEB adapter.

Verifies adapter plumbing by mocking _run_mteb_subprocess and seeding the
output directory with canned task JSON results.
"""

import json
import subprocess

import pytest
from pathlib import Path
from unittest.mock import create_autospec

from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import MTEBAdapter

# Canned output matching MTEB's per-task result JSON structure.
# Based on: https://github.com/embeddings-benchmark/mteb/blob/main/tests/mock_mteb_cache/results/no_model_name_available/no_revision_available/STS12.json
CANNED_STS12 = {
    "task_name": "STS12",
    "scores": {
        "test": [{
            "main_score": 0.847,
            "cosine_spearman": 0.847,
            "hf_subset": "default",
            "languages": ["eng-Latn"],
        }]
    },
}


@pytest.mark.integration
def test_mteb_happy_path(monkeypatch):
    """Full run_benchmark_job with mocked subprocess and canned task results."""
    adapter = MTEBAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    # Patch _run_mteb_subprocess to seed the output directory with canned results.
    # run_benchmark_job creates output_dir via tempfile.mkdtemp,
    # passes it to _build_mteb_command which adds --output-folder <path>,
    # then calls _run_mteb_subprocess. We extract output_dir from the adapter.
    def fake_run(cmd, timeout):
        idx = cmd.index("--output-folder")
        output_dir = Path(cmd[idx + 1])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter, "_run_mteb_subprocess", fake_run)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0

    # Metrics extracted from canned data
    assert len(results.results) > 0
    assert any(r.metric_name == "STS12.test.main_score" for r in results.results)
    main_score = next(
        r for r in results.results if r.metric_name == "STS12.test.main_score"
    )
    assert main_score.metric_value == 0.847

    # Overall score
    assert results.overall_score == 0.847

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    assert JobPhase.PERSISTING_ARTIFACTS in phases
