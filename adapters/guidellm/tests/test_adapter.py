"""Integration tests for the GuideLLM adapter.

Verifies the adapter plumbing (instantiation, lifecycle callbacks, metric
extraction, result structure) WITHOUT invoking the real GuideLLM CLI.
The subprocess method is monkeypatched and results are seeded on disk.
"""

import json

import pytest
from unittest.mock import create_autospec

from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import GuideLLMAdapter

# Minimal canned output matching GuideLLM's benchmarks.json structure.
# Based loosely on: https://github.com/vllm-project/guidellm/blob/main/tests/unit/entrypoints/assets/benchmarks_stripped.yaml
CANNED_BENCHMARKS = {
    "benchmarks": [{
        "metrics": {
            "requests_per_second": {"successful": {"mean": 12.5}},
            "prompt_tokens_per_second": {"successful": {"mean": 850.0}},
            "output_tokens_per_second": {"successful": {"mean": 420.0}},
            "time_to_first_token_ms": {"successful": {"mean": 45.2}},
            "inter_token_latency_ms": {"successful": {"mean": 8.1}},
            "request_totals": {"successful": 20},
        }
    }]
}


@pytest.mark.integration
def test_guidellm_happy_path(monkeypatch):
    """Full run_benchmark_job with mocked subprocess and canned results."""
    adapter = GuideLLMAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    # Patch _run_guidellm to write canned data into adapter.results_dir.
    # At call time, run_benchmark_job has already set self.results_dir
    # before calling _run_guidellm.
    def fake_run_guidellm(cmd):
        (adapter.results_dir / "benchmarks.json").write_text(
            json.dumps(CANNED_BENCHMARKS)
        )

    monkeypatch.setattr(adapter, "_run_guidellm", fake_run_guidellm)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0

    # Metrics extracted from canned data
    assert len(results.results) > 0
    metric_names = {r.metric_name for r in results.results}
    assert "requests_per_second" in metric_names
    assert "prompt_tokens_per_second" in metric_names
    assert "output_tokens_per_second" in metric_names
    assert "mean_ttft_ms" in metric_names
    assert "mean_itl_ms" in metric_names

    # Overall score and example count
    assert results.overall_score == 12.5
    assert results.num_examples_evaluated == 20

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
