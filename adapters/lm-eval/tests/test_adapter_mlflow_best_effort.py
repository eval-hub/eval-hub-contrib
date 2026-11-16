"""MLflow export failures must not discard results (adapted from PR #244)."""

import json
import logging
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
    getattr(client, operation).side_effect = RuntimeError("MLflow service unavailable")

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
    # Verify the adapter's generic warning only. SDK diagnostic redaction belongs
    # in SDK tests; this assertion does not establish credential-free logging.
    warnings = [
        r
        for r in caplog.records
        if r.name == main.logger.name and r.levelno == logging.WARNING
    ]
    assert any(
        r.getMessage() == "MLflow export failed; continuing with evaluation results"
        for r in warnings
    )
    assert all(
        "MLflow service unavailable" not in r.getMessage()
        and not r.exc_info
        and not r.exc_text
        for r in warnings
    )


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


@pytest.mark.parametrize("fails", [False, True])
def test_sdk_logging_restored_after_export(export_setup, caplog, fails):
    _, _, client = export_setup
    if fails:
        client.log_batch.side_effect = RuntimeError("MLflow service unavailable")
    sdk_logger = logging.getLogger("evalhub.adapter.callbacks")
    original_filters = sdk_logger.filters[:]

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    assert sdk_logger.filters == original_filters
    try:
        raise RuntimeError("non-export diagnostic")
    except RuntimeError:
        sdk_logger.exception("Logging outside export is unchanged")
    assert caplog.records[-1].exc_info
    assert "non-export diagnostic" in caplog.text


def test_no_experiment_skips_mlflow(export_setup, monkeypatch):
    adapter, callbacks, client = export_setup
    adapter.job_spec.experiment_name = None
    collect_artifact = Mock()
    monkeypatch.setattr(main, "MlflowFileArtifact", collect_artifact)
    save_mlflow = Mock()
    monkeypatch.setattr(main, "_save_mlflow_best_effort", save_mlflow)

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    collect_artifact.assert_not_called()
    save_mlflow.assert_not_called()
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
