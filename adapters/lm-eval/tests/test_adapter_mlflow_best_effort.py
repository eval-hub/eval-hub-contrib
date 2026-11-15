"""MLflow export failures must not discard results (adapted from PR #244)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from evalhub.adapter import DefaultCallbacks, JobStatus
from evalhub.adapter import mlflow as sdk_mlflow

import main


@pytest.fixture
def export_setup(adapter, monkeypatch):
    adapter.job_spec.experiment_name = "experiment"

    def run(cmd, env, timeout):
        output_dir = Path(cmd[cmd.index("--output_path") + 1])
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"gsm8k": {"exact_match,none": 0.75}},
                    "n-samples": {"gsm8k": {"effective": 1}},
                }
            )
        )

    monkeypatch.setattr(adapter, "_run_lmeval", run)
    callbacks = DefaultCallbacks.from_adapter(adapter)
    monkeypatch.setattr(callbacks, "report_status", Mock())
    monkeypatch.setattr(callbacks, "report_results", Mock())
    monkeypatch.setattr(main, "LMEvalAdapter", Mock(return_value=adapter))
    monkeypatch.setattr(DefaultCallbacks, "from_adapter", Mock(return_value=callbacks))
    client = MagicMock()
    client.__enter__.return_value = client
    client.start_run.return_value.__enter__.return_value = "run-1"
    monkeypatch.setattr(sdk_mlflow, "MlflowClient", Mock(return_value=client))
    return adapter, callbacks, client


@pytest.mark.parametrize(
    "operation", ["get_or_create_experiment", "log_batch", "upload_artifact_file"]
)
def test_mlflow_failure_still_reports_results(export_setup, caplog, operation):
    _, callbacks, client = export_setup
    getattr(client, operation).side_effect = RuntimeError("token=SECRET")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    getattr(client, operation).assert_called_once()
    callbacks.report_results.assert_called_once()
    results = callbacks.report_results.call_args.args[0]
    assert results.overall_score == 0.75
    assert results.mlflow_run_id is None
    assert callbacks.mlflow._callbacks is callbacks
    assert all(
        call.args[0].status != JobStatus.FAILED
        for call in callbacks.report_status.call_args_list
    )
    warnings = [r for r in caplog.records if r.name == main.logger.name]
    assert any("continuing with evaluation results" in r.message for r in warnings)
    assert all("SECRET" not in r.message and not r.exc_info for r in warnings)


def test_success_retains_run_id_and_restores_callback(export_setup):
    _, callbacks, client = export_setup

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    callbacks.report_results.assert_called_once()
    assert callbacks.report_results.call_args.args[0].mlflow_run_id == "run-1"
    assert callbacks.mlflow._callbacks is callbacks
    client.log_batch.assert_called_once()
    client.upload_artifact_file.assert_called_once()


def test_no_experiment_skips_mlflow(export_setup):
    adapter, callbacks, client = export_setup
    adapter.job_spec.experiment_name = None

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    client.__enter__.assert_not_called()
    callbacks.report_results.assert_called_once()
    assert callbacks.mlflow._callbacks is callbacks


@pytest.mark.parametrize("failure", ["evaluation", "report_results"])
def test_non_mlflow_failures_remain_fatal(export_setup, failure, monkeypatch):
    adapter, callbacks, client = export_setup
    if failure == "evaluation":
        monkeypatch.setattr(
            adapter, "_run_lmeval", Mock(side_effect=RuntimeError("evaluation failed"))
        )
    else:
        callbacks.report_results.side_effect = RuntimeError("delivery failed")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    if failure == "evaluation":
        client.__enter__.assert_not_called()
        callbacks.report_results.assert_not_called()
        assert callbacks.report_status.call_args.args[0].status == JobStatus.FAILED
    assert callbacks.mlflow._callbacks is callbacks
