"""Integration tests for the RAGAS adapter.

Verifies adapter plumbing with a single monkeypatch point — _run_ragas
returns a mock result object, so no model endpoint is needed.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, create_autospec

from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import RagasAdapter


def _make_mock_ragas_result(metric_names, n_rows=5):
    """Create a mock ragas EvaluationResult with to_pandas()."""
    data = {name: [0.8 + i * 0.01 for i in range(n_rows)] for name in metric_names}
    df = pd.DataFrame(data)
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = df
    return mock_result


@pytest.mark.integration
def test_ragas_happy_path(monkeypatch, tmp_path):
    """Full run_benchmark_job with mocked _run_ragas returning canned results."""
    adapter = RagasAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake",
        reference="fake:latest",
    )

    metric_names = ["answer_relevancy", "context_precision", "faithfulness", "context_recall"]
    mock_result = _make_mock_ragas_result(metric_names)

    monkeypatch.setattr(adapter, "_run_ragas", lambda **kwargs: mock_result)

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input": "What is AI?", "response": "Artificial Intelligence", "retrieved_contexts": ["AI is..."], "reference": "AI stands for..."}\n'
        '{"user_input": "What is ML?", "response": "Machine Learning", "retrieved_contexts": ["ML is..."], "reference": "ML stands for..."}\n'
    )
    monkeypatch.setattr(
        "main._resolve_data_path",
        lambda config: dataset_file,
    )

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds >= 0

    assert len(results.results) == 4
    assert any(r.metric_name == "answer_relevancy" for r in results.results)
    assert any(r.metric_name == "faithfulness" for r in results.results)

    assert results.overall_score is not None
    assert results.num_examples_evaluated == 5

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
