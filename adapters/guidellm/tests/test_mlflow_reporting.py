"""Verify that GuideLLM reports its MLflow benchmark run ID to EvalHub."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import evalhub.adapter
from evalhub.adapter import DefaultCallbacks
import main


@pytest.mark.parametrize(
    ("experiment_name", "run_id"),
    [("guidellm-repro", "mlflow-run-123"), (None, None)],
)
def test_main_reports_mlflow_run_id_before_completion(
    monkeypatch, experiment_name, run_id
):
    job_spec = SimpleNamespace(
        id="job-123",
        benchmark_id="constant",
        experiment_name=experiment_name,
        model=SimpleNamespace(name="test-model"),
    )
    results = SimpleNamespace(
        id=job_spec.id,
        mlflow_run_id=None,
        overall_score=1.0,
        num_examples_evaluated=2,
        duration_seconds=0.1,
    )
    adapter = SimpleNamespace(job_spec=job_spec)
    adapter.run_benchmark_job = Mock(return_value=results)
    callbacks = SimpleNamespace()
    events = []

    def save(actual_results, actual_spec):
        assert actual_results is results
        assert actual_spec is job_spec
        events.append("save")
        return run_id

    def report_results(actual_results):
        assert actual_results is results
        assert actual_results.mlflow_run_id == run_id
        events.append("report")

    callbacks.mlflow = SimpleNamespace(save=save)
    callbacks.report_results = report_results
    monkeypatch.setattr(main, "GuideLLMAdapter", lambda **_: adapter)
    monkeypatch.setattr(
        DefaultCallbacks, "from_adapter", lambda *_: callbacks
    )
    monkeypatch.setattr(evalhub.adapter, "configure_telemetry", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    assert events == ["save", "report"]
    adapter.run_benchmark_job.assert_called_once_with(job_spec, callbacks)
