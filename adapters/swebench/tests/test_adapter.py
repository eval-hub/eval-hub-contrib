"""Unit tests for the SWE-bench adapter."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from main import JobStatus, OCICoordinates, SWEBenchAdapter

# -- Sample data -------------------------------------------------------------

SAMPLE_PREDICTIONS = {
    "astropy__astropy-7336": {
        "instance_id": "astropy__astropy-7336",
        "model_patch": "diff --git a/astropy/units/decorators.py b/astropy/units/decorators.py\n",
        "model_name_or_path": "test-model",
    },
    "django__django-11099": {
        "instance_id": "django__django-11099",
        "model_patch": "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n",
        "model_name_or_path": "test-model",
    },
}

# -- Fixtures ----------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "meta"


@pytest.fixture
def job_spec_path():
    return str(FIXTURE_DIR / "job.json")


@pytest.fixture
def adapter(job_spec_path):
    return SWEBenchAdapter(job_spec_path=job_spec_path)


@pytest.fixture
def predictions_json(tmp_path):
    """Sample predictions as a JSON file (dict format)."""
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(SAMPLE_PREDICTIONS))
    return str(path)


@pytest.fixture
def predictions_jsonl(tmp_path):
    """Sample predictions as a JSONL file (list format)."""
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(v) for v in SAMPLE_PREDICTIONS.values()))
    return str(path)


# -- Tests -------------------------------------------------------------------


class TestResultsProcessing:
    def test_all_resolved(self, adapter):
        results = {
            "a": {"completed": True, "resolved": True},
            "b": {"completed": True, "resolved": True},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 1.0
        assert summary["resolved"] == 2
        assert summary["errors"] == 0

    def test_none_resolved(self, adapter):
        results = {
            "a": {"completed": True, "resolved": False},
            "b": {"completed": True, "resolved": False},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 0.0

    def test_with_errors(self, adapter):
        results = {
            "a": {"completed": True, "resolved": True},
            "b": {"completed": False, "resolved": False},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 1.0
        assert summary["errors"] == 1

    def test_with_no_results(self, adapter):
        results = {}
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 0.0
        assert summary["total"] == 0


class TestPredictionsLoading:
    def test_load_json_format(self, adapter, predictions_json):
        """Load predictions from a JSON dict file."""
        preds = adapter._load_predictions(predictions_json, "", "test")
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds
        assert preds["astropy__astropy-7336"]["model_patch"].startswith("diff")

    def test_load_jsonl_format(self, adapter, predictions_jsonl):
        """Load predictions from a JSONL file."""
        preds = adapter._load_predictions(predictions_jsonl, "", "test")
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds

    def test_missing_file_raises(self, adapter):
        with pytest.raises(FileNotFoundError):
            adapter._load_predictions("/nonexistent/path.json", "", "test")

    def test_load_from_test_data_path(self, adapter, tmp_path):
        """Test loading predictions via explicit path (eval-hub init container pattern)."""
        test_data = tmp_path / "test_data"
        test_data.mkdir()
        (test_data / "predictions.jsonl").write_text(
            "\n".join(json.dumps(v) for v in SAMPLE_PREDICTIONS.values())
        )
        preds = adapter._load_predictions(
            str(test_data / "predictions.jsonl"), "", "test"
        )
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds


class TestEmptyResults:
    def test_empty_results(self, adapter):

        job_result = adapter._empty_results(adapter.job_spec, 1.0)

        assert job_result.id == "swebench-eval-001"
        assert job_result.benchmark_id == "swebench_verified"
        assert job_result.benchmark_index == 0
        assert job_result.model_name == "gold-patches"
        assert job_result.results[0].metric_name == "resolve_rate"
        assert job_result.results[0].metric_value == 0.0
        assert job_result.results[0].metric_type == "accuracy"
        assert job_result.overall_score == 0.0
        assert job_result.num_examples_evaluated == 0


@pytest.mark.parametrize("save_samples", [None, True, False])
def test_sample_artifacts_match_completed_instance_resolve_rate(
    adapter, tmp_path, monkeypatch, save_samples
):
    monkeypatch.setattr(adapter, "_get_results_dir", lambda config: tmp_path)
    config = adapter.job_spec.model_copy(deep=True)
    if save_samples is not None:
        config.parameters["save_sample_results"] = save_samples
    results = {
        "a": {"completed": True, "resolved": True},
        "b": {"completed": True, "resolved": False},
        "error": {"completed": False, "resolved": True},
    }
    _, score, summary = adapter._process_results(results, config)
    files = adapter._save_artifacts(results, summary, config)
    if save_samples is not True:
        assert {p.name for p in files} == {"summary.json", "results.json"}
        return
    rows = [
        json.loads(line)
        for line in (tmp_path / "sample_results.jsonl").read_text().splitlines()
    ]
    assert {
        row["sample_id"]: (row["instance_id"], row["resolve_rate"]) for row in rows
    } == {"a": ("a", 1.0), "b": ("b", 0.0)}
    assert sum(row["resolve_rate"] for row in rows) / len(rows) == score == 0.5
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["primary_score"]["metric"]
        == "resolve_rate"
    )


def test_child_runner_inherits_pod_service_account(adapter):
    import base64

    claims = {"sub": "system:serviceaccount:tenant:adapter-account"}
    payload = base64.b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    with patch("builtins.open", mock_open(read_data=f"header.{payload}.signature")):
        assert adapter._current_service_account() == "adapter-account"


def test_child_runner_uses_namespace_default_without_pod_token(adapter):
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert adapter._current_service_account() is None


@pytest.mark.parametrize("save_samples", [None, False])
def test_configured_oci_export_failure_preserves_evaluation(
    adapter, runner_runtime, caplog, save_samples
):
    from types import SimpleNamespace

    config, callbacks, runner, _ = runner_runtime
    if save_samples is not None:
        config.parameters["save_sample_results"] = save_samples
    coordinates = OCICoordinates(
        oci_host="quay.io", oci_repository="test/results", oci_tag="test"
    )
    config = config.model_copy(
        update={
            "exports": SimpleNamespace(oci=SimpleNamespace(coordinates=coordinates))
        }
    )
    callbacks.create_oci_artifact.side_effect = RuntimeError("upload failed")

    result = adapter.run_benchmark_job(config, callbacks)

    runner.assert_called_once()
    callbacks.create_oci_artifact.assert_called_once()
    assert result.overall_score == 0.5
    assert result.num_examples_evaluated == 2
    assert result.oci_artifact is None
    assert result.mlflow_run_id == "run-1"
    callbacks.mlflow.save.assert_called_once()
    assert "Failed to create OCI artifact" in caplog.text
    assert "upload failed" in caplog.text
    assert not any(
        call.args[0].status == JobStatus.FAILED
        for call in callbacks.report_status.call_args_list
    )


@pytest.fixture
def runner_runtime(adapter, tmp_path, monkeypatch):
    """Exercise orchestration and export without cluster Jobs or dataset downloads."""
    import sys
    from types import SimpleNamespace

    config = adapter.job_spec.model_copy(deep=True)
    config.parameters.pop("k8s_namespace", None)
    loader = MagicMock(
        return_value={"a": {"instance_id": "a"}, "b": {"instance_id": "b"}}
    )
    monkeypatch.setattr(adapter, "_load_predictions", loader)
    monkeypatch.setattr(adapter, "_get_results_dir", lambda config: tmp_path)
    monkeypatch.setattr(adapter, "_current_namespace", lambda: "pod-namespace")
    monkeypatch.setattr(
        adapter, "_current_service_account", lambda: "discovered-runner"
    )
    runner = MagicMock(
        return_value={
            "a": {"completed": True, "resolved": True},
            "b": {"completed": True, "resolved": False},
        }
    )
    monkeypatch.setitem(
        sys.modules, "k8s_runner", SimpleNamespace(run_instances_k8s=runner)
    )
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.utils",
        SimpleNamespace(
            load_swebench_dataset=lambda *args, **kwargs: [
                {"instance_id": "a"},
                {"instance_id": "b"},
            ]
        ),
    )
    callbacks = MagicMock()
    callbacks.mlflow.save.return_value = "run-1"
    return config, callbacks, runner, loader


@pytest.mark.parametrize("save_samples", [None, True, False])
def test_mlflow_export_includes_native_samples_and_manifest(
    adapter, runner_runtime, tmp_path, save_samples
):
    config, callbacks, runner, _ = runner_runtime
    if save_samples is not None:
        config.parameters["save_sample_results"] = save_samples
    result = adapter.run_benchmark_job(config, callbacks)
    runner.assert_called_once()
    assert result.mlflow_run_id == "run-1"
    artifacts = {a.path: a for a in callbacks.mlflow.save.call_args.kwargs["artifacts"]}
    expected = {"summary.json", "results.json"}
    if save_samples is True:
        expected |= {"sample_results.jsonl", "manifest.json"}
        for name in ["sample_results.jsonl", "manifest.json"]:
            assert artifacts[name].content == (tmp_path / name).read_bytes()
        assert json.loads(artifacts["manifest.json"].content) == {
            "primary_score": {"metric": "resolve_rate"},
            "estimand": "mean",
        }
    assert set(artifacts) == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_runner_inherits_pod_namespace_and_trusted_service_account(
    adapter, runner_runtime, explicit
):
    config, callbacks, runner, _ = runner_runtime
    if explicit:
        config.parameters.update(k8s_namespace="requested-namespace")
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    assert runner.call_args.kwargs["namespace"] == "pod-namespace"
    assert runner.call_args.kwargs["service_account"] == "discovered-runner"


@pytest.mark.parametrize("value", ["false", 0, "0", None, 1, "true"])
def test_invalid_sample_flag_fails_before_dataset_or_runner(
    adapter, runner_runtime, value
):
    config, callbacks, runner, loader = runner_runtime
    config.parameters["save_sample_results"] = value
    with pytest.raises(ValueError, match="must be a boolean"):
        adapter.run_benchmark_job(config, callbacks)
    loader.assert_not_called()
    runner.assert_not_called()
    callbacks.mlflow.save.assert_not_called()
    assert any(
        "must be a boolean" in str(call)
        for call in callbacks.report_status.call_args_list
    )


def test_invalid_sample_flag_cannot_write_artifacts(adapter, tmp_path, monkeypatch):
    config = adapter.job_spec.model_copy(deep=True)
    config.parameters["save_sample_results"] = "false"
    monkeypatch.setattr(adapter, "_get_results_dir", lambda config: tmp_path)
    with pytest.raises(ValueError, match="must be a boolean"):
        adapter._save_artifacts({}, {}, config)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("account", ["admin", "default", "", False])
def test_job_service_account_parameter_is_ignored(adapter, runner_runtime, account):
    config, callbacks, runner, _ = runner_runtime
    config.parameters["service_account"] = account
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    assert runner.call_args.kwargs["service_account"] == "discovered-runner"


@pytest.mark.parametrize(
    "payload",
    ["!", "a", "_w", "W10", "bnVsbA", "eyJzdWIiOjEyM30"],
)
def test_malformed_pod_token_uses_default_account(adapter, payload, caplog):
    with patch("builtins.open", mock_open(read_data=f"header.{payload}.signature")):
        assert adapter._current_service_account() is None
    assert "Could not determine service account" in caplog.text


def test_oci_rerun_exports_only_current_artifacts(adapter, tmp_path, monkeypatch):
    """Inspect files during the synchronous upload, including an opted-out rerun."""
    config = adapter.job_spec.model_copy(deep=True)
    monkeypatch.setattr(adapter, "_get_results_dir", lambda config: tmp_path)
    callbacks = MagicMock()
    coordinates = OCICoordinates(
        oci_host="quay.io", oci_repository="test/results", oci_tag="test"
    )
    uploaded = []
    staging_paths = []

    def capture_export(spec):
        root = Path(spec.files_path)
        staging_paths.append(root)
        uploaded.append({path.name: path.read_bytes() for path in root.iterdir()})

    callbacks.create_oci_artifact.side_effect = capture_export
    for enabled, instance in [(True, "old"), (False, "new")]:
        config.parameters["save_sample_results"] = enabled
        results = {instance: {"completed": True, "resolved": enabled}}
        _, _, summary = adapter._process_results(results, config)
        files = adapter._save_artifacts(results, summary, config)
        (tmp_path / "unrelated.txt").write_text("not a current artifact")
        adapter._persist_oci_artifact(files, coordinates, callbacks)

    assert set(uploaded[0]) == {
        "summary.json",
        "results.json",
        "sample_results.jsonl",
        "manifest.json",
    }
    assert json.loads(uploaded[0]["sample_results.jsonl"])["sample_id"] == "old"
    assert set(uploaded[1]) == {"summary.json", "results.json"}
    assert set(json.loads(uploaded[1]["results.json"])) == {"new"}
    assert not (tmp_path / "sample_results.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "unrelated.txt").exists()
    assert staging_paths[0] != staging_paths[1]
    assert all(not path.exists() for path in staging_paths)


@pytest.mark.parametrize("invalid", ["sample_flag", "benchmark"])
def test_entrypoint_failure_never_reports_completion(
    adapter, runner_runtime, monkeypatch, invalid
):
    """Run the real entry point through validation and its failure callback."""
    import main as adapter_module

    config, callbacks, runner, loader = runner_runtime
    if invalid == "sample_flag":
        config.parameters["save_sample_results"] = "false"
    else:
        config.benchmark_id = "unsupported"
    adapter.job_spec.parameters = config.parameters
    adapter.job_spec.benchmark_id = config.benchmark_id
    monkeypatch.setattr(adapter_module, "SWEBenchAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(adapter_module, "configure_telemetry", lambda: None)
    monkeypatch.setattr(
        adapter_module.DefaultCallbacks, "from_adapter", lambda adapter: callbacks
    )
    # Model report_results' side effect: it would emit COMPLETED if invoked.
    callbacks.report_results.side_effect = lambda results: callbacks.report_status(
        adapter_module.JobStatusUpdate(status=JobStatus.COMPLETED)
    )

    with pytest.raises(SystemExit) as exc:
        adapter_module.main()

    assert exc.value.code == 1
    callbacks.report_results.assert_not_called()
    statuses = [call.args[0].status for call in callbacks.report_status.call_args_list]
    assert statuses[-1] == JobStatus.FAILED
    assert JobStatus.COMPLETED not in statuses
    runner.assert_not_called()
    loader.assert_not_called()


@pytest.mark.parametrize("experiment", [None, "configured-experiment"])
@pytest.mark.parametrize("run_id", [None, ""])
def test_missing_mlflow_run_id_warns_only_for_configured_export(
    adapter, runner_runtime, caplog, experiment, run_id
):
    config, callbacks, _, _ = runner_runtime
    config.experiment_name = experiment
    callbacks.mlflow.save.return_value = run_id
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    assert result.mlflow_run_id is None
    assert ("no run ID was returned" in caplog.text) == bool(experiment)


@pytest.mark.parametrize("oci", [False, True])
@pytest.mark.parametrize("mlflow", [False, True])
def test_persistence_phase_is_reserved_for_configured_oci(
    adapter, runner_runtime, monkeypatch, oci, mlflow
):
    from types import SimpleNamespace

    from main import JobPhase

    config, callbacks, _, _ = runner_runtime
    config.experiment_name = "experiment" if mlflow else None
    if oci:
        config = config.model_copy(
            update={
                "exports": SimpleNamespace(
                    oci=SimpleNamespace(
                        coordinates=OCICoordinates(
                            oci_host="quay.io",
                            oci_repository="test/results",
                            oci_tag="test",
                        )
                    )
                )
            }
        )
    save_artifacts = adapter._save_artifacts

    def phases():
        return [call.args[0].phase for call in callbacks.report_status.call_args_list]

    def save_with_phase(*args):
        assert phases()[-1] == JobPhase.POST_PROCESSING
        assert JobPhase.PERSISTING_ARTIFACTS not in phases()
        return save_artifacts(*args)

    def check_export_phase(*args, **kwargs):
        assert phases().count(JobPhase.PERSISTING_ARTIFACTS) == int(oci)
        assert phases()[-1] == (
            JobPhase.PERSISTING_ARTIFACTS if oci else JobPhase.POST_PROCESSING
        )

    monkeypatch.setattr(adapter, "_save_artifacts", save_with_phase)
    callbacks.create_oci_artifact.side_effect = check_export_phase
    callbacks.mlflow.save.side_effect = check_export_phase
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    check_export_phase()
    assert callbacks.create_oci_artifact.call_count == int(oci)


@pytest.mark.parametrize("save_samples", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_no_matching_instances_cleans_owned_outputs_and_skips_exports(
    adapter, runner_runtime, tmp_path, stale, save_samples
):
    config, callbacks, runner, loader = runner_runtime
    config.parameters["save_sample_results"] = save_samples
    owned = ("sample_results.jsonl", "manifest.json", "summary.json", "results.json")
    if stale:
        for name in owned:
            (tmp_path / name).write_text("earlier run")
        (tmp_path / "unrelated.txt").write_text("keep")
    loader.return_value = {"not-in-dataset": {"instance_id": "not-in-dataset"}}
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.0
    assert result.num_examples_evaluated == 0
    assert result.oci_artifact is None
    assert result.mlflow_run_id is None
    assert (tmp_path / "sample_results.jsonl").exists() == save_samples
    assert (tmp_path / "manifest.json").exists() == save_samples
    if save_samples:
        assert (tmp_path / "sample_results.jsonl").read_text() == ""
    assert json.loads((tmp_path / "results.json").read_text()) == {}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total"] == summary["completed"] == summary["errors"] == 0
    assert {path.name for path in tmp_path.iterdir()} == (
        {"summary.json", "results.json"} | ({"sample_results.jsonl", "manifest.json"} if save_samples else set()) | ({"unrelated.txt"} if stale else set())
    )
    if stale:
        assert (tmp_path / "unrelated.txt").read_text() == "keep"
    runner.assert_not_called()
    callbacks.create_oci_artifact.assert_not_called()
    callbacks.mlflow.save.assert_not_called()


def test_no_completed_instances_writes_empty_samples(adapter, runner_runtime, tmp_path):
    config, callbacks, runner, _ = runner_runtime
    config.parameters["save_sample_results"] = True
    runner.return_value = {"a": {"completed": False, "resolved": False}}
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.0
    assert (tmp_path / "sample_results.jsonl").read_text() == ""
    assert (tmp_path / "manifest.json").exists()
    callbacks.mlflow.save.assert_called_once()


@pytest.mark.parametrize("destination", ["local", "oci", "mlflow"])
def test_opted_in_persistence_failure_reports_actual_error_and_exits(
    adapter, runner_runtime, monkeypatch, destination
):
    """Inspect the real SDK event payload and ensure main never reports completion."""
    import main as adapter_module
    from evalhub.adapter.models.job import JobSpecExports

    config, _, _, _ = runner_runtime
    config.parameters["save_sample_results"] = True
    config.experiment_name = "experiment" if destination == "mlflow" else None
    if destination == "oci":
        config.exports = JobSpecExports.model_validate(
            {
                "oci": {
                    "coordinates": {
                        "oci_host": "quay.io",
                        "oci_repository": "test/results",
                        "oci_tag": "test",
                    }
                }
            }
        )
    http = MagicMock()
    with patch.object(
        adapter_module.DefaultCallbacks, "_create_http_client", return_value=http
    ):
        callbacks = adapter_module.DefaultCallbacks(
            job_id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            provider_id=config.provider_id,
            sidecar_url="http://sidecar:8080",
        )
    error = f"{destination} permission denied for sample_results.jsonl"
    if destination == "local":
        write_text = Path.write_text

        def fail_sample_write(path, *args, **kwargs):
            if path.name == "sample_results.jsonl":
                raise OSError(error)
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_sample_write)
    elif destination == "oci":
        monkeypatch.setattr(
            callbacks, "create_oci_artifact", MagicMock(side_effect=RuntimeError(error))
        )
    else:
        # The companion SDK tests cover inclusion of this cause in its first FAILED event.
        monkeypatch.setattr(
            callbacks.mlflow, "save", MagicMock(side_effect=RuntimeError(error))
        )
    report_results = MagicMock()
    monkeypatch.setattr(callbacks, "report_results", report_results)
    monkeypatch.setattr(type(adapter), "job_spec", property(lambda self: config))
    monkeypatch.setattr(adapter_module, "SWEBenchAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(adapter_module, "configure_telemetry", lambda: None)
    monkeypatch.setattr(
        adapter_module.DefaultCallbacks, "from_adapter", lambda adapter: callbacks
    )

    with pytest.raises(SystemExit) as exc:
        adapter_module.main()

    assert exc.value.code == 1
    events = [
        call.kwargs["json"]["benchmark_status_event"]
        for call in http.post.call_args_list
    ]
    failed = [event for event in events if event["status"] == "failed"]
    assert len(failed) == 1
    assert error in failed[0]["error_message"]["message"]
    assert failed[0]["error_message"]["message_code"] == "evaluation_failed"
    assert "warning_message" not in failed[0]
    assert all(event["status"] != "completed" for event in events)
    report_results.assert_not_called()


@pytest.mark.parametrize("destination", ["oci", "mlflow"])
def test_opted_in_export_requires_artifact_reference(
    adapter, runner_runtime, destination
):
    from evalhub.adapter.models.job import JobSpecExports

    config, callbacks, _, _ = runner_runtime
    config.parameters["save_sample_results"] = True
    if destination == "oci":
        config.exports = JobSpecExports.model_validate(
            {
                "oci": {
                    "coordinates": {
                        "oci_host": "quay.io",
                        "oci_repository": "test/results",
                        "oci_tag": "test",
                    }
                }
            }
        )
        callbacks.create_oci_artifact.return_value = None
    else:
        config.experiment_name = "experiment"
        callbacks.mlflow.save.return_value = None
    with pytest.raises(RuntimeError, match="no (artifact reference|run ID)"):
        adapter.run_benchmark_job(config, callbacks)
    failed = [
        call.args[0]
        for call in callbacks.report_status.call_args_list
        if call.args[0].status == JobStatus.FAILED
    ]
    assert len(failed) == 1
    assert "no " in failed[0].error_message.message


def test_opted_in_oci_success_returns_reference_and_current_samples(
    adapter, runner_runtime
):
    from evalhub.adapter.models.job import JobSpecExports, OCIArtifactResult

    config, callbacks, _, _ = runner_runtime
    config.parameters["save_sample_results"] = True
    config.exports = JobSpecExports.model_validate(
        {
            "oci": {
                "coordinates": {
                    "oci_host": "quay.io",
                    "oci_repository": "test/results",
                    "oci_tag": "test",
                }
            }
        }
    )
    expected = OCIArtifactResult(
        digest="sha256:test", reference="quay.io/test/results@sha256:test"
    )

    def upload(spec):
        root = Path(spec.files_path)
        assert {path.name for path in root.iterdir()} == {
            "summary.json",
            "results.json",
            "sample_results.jsonl",
            "manifest.json",
        }
        rows = [
            json.loads(row)
            for row in (root / "sample_results.jsonl").read_text().splitlines()
        ]
        assert {row["sample_id"]: row["resolve_rate"] for row in rows} == {
            "a": 1.0,
            "b": 0.0,
        }
        return expected

    callbacks.create_oci_artifact.side_effect = upload
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    assert result.oci_artifact == expected
    assert not any(
        call.args[0].status == JobStatus.FAILED
        for call in callbacks.report_status.call_args_list
    )


def test_tokenless_adapter_discovers_own_pod_account(adapter, monkeypatch):
    """Existing kubeconfig supplies API access; caller parameters never select the account."""
    from types import SimpleNamespace
    monkeypatch.setenv("KUBECONFIG", "/trusted/kubeconfig")
    monkeypatch.setenv("HOSTNAME", "adapter-pod")
    monkeypatch.setattr(adapter.__class__, "_current_namespace", staticmethod(lambda: "tenant"))
    with patch("pathlib.Path.is_file", return_value=True), patch("builtins.open", side_effect=FileNotFoundError), patch("kubernetes.config.load_kube_config") as load, patch("kubernetes.client.CoreV1Api") as core:
        core.return_value.read_namespaced_pod.return_value = SimpleNamespace(spec=SimpleNamespace(service_account_name="evalhub-job"))
        assert adapter._current_service_account() == "evalhub-job"
        load.assert_called_once_with(config_file="/trusted/kubeconfig")
        core.return_value.read_namespaced_pod.assert_called_once_with("adapter-pod", "tenant")


@pytest.mark.parametrize("failure", ["forbidden", "missing_account"])
def test_tokenless_discovery_never_silently_changes_account(adapter, monkeypatch, failure):
    monkeypatch.setenv("KUBECONFIG", "/trusted/kubeconfig")
    with patch("pathlib.Path.is_file", return_value=True), patch("kubernetes.config.load_kube_config"), patch("kubernetes.client.CoreV1Api") as core:
        if failure == "forbidden":
            core.return_value.read_namespaced_pod.side_effect = RuntimeError("forbidden")
        else:
            core.return_value.read_namespaced_pod.return_value.spec.service_account_name = None
        with pytest.raises(RuntimeError, match="Could not discover adapter pod service account"):
            adapter._current_service_account()


def test_provider_defaults_match_sample_manifest(adapter, tmp_path):
    import yaml
    provider = yaml.safe_load((FIXTURE_DIR.parent / "provider.yaml").read_text())
    config = adapter.job_spec.model_copy(deep=True)
    config.parameters["save_sample_results"] = True
    with patch.object(adapter, "_get_results_dir", return_value=tmp_path):
        observations = {
            "rejected": {"completed": True, "resolved": False},
            "resolved": {"completed": True, "resolved": True},
            "timeout": {"completed": False, "resolved": False},
        }
        _, score, summary = adapter._process_results(observations, config)
        assert score == 0.5 and summary["completed"] == 2 and summary["errors"] == 1
        adapter._save_artifacts(observations, summary, config)
        rows = [json.loads(line) for line in (tmp_path / "sample_results.jsonl").read_text().splitlines()]
        assert {row["sample_id"]: row["resolve_rate"] for row in rows} == {"rejected": 0.0, "resolved": 1.0}
    metric = json.loads((tmp_path / "manifest.json").read_text())["primary_score"]["metric"]
    assert len(provider["benchmarks"]) == 3
    for benchmark in provider["benchmarks"]:
        assert benchmark["primary_score"] == {"metric": metric, "lower_is_better": False}
        assert benchmark["pass_criteria"]["threshold"] == 0.5


@pytest.mark.parametrize("hostname", [None, "developer-laptop"])
def test_local_kubeconfig_does_not_lookup_laptop_as_pod(adapter, monkeypatch, hostname):
    """A laptop uses namespace-default child identity without querying a fictitious pod."""
    monkeypatch.setenv("KUBECONFIG", "/local/kubeconfig")
    if hostname:
        monkeypatch.setenv("HOSTNAME", hostname)
    else:
        monkeypatch.delenv("HOSTNAME", raising=False)
    with patch("pathlib.Path.is_file", return_value=False), patch("kubernetes.config.load_kube_config") as load, patch("kubernetes.client.CoreV1Api") as core:
        assert adapter._current_service_account() is None
        load.assert_not_called()
        core.assert_not_called()
