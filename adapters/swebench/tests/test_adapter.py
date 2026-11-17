"""Unit tests for the SWE-bench adapter."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from main import OCICoordinates, SWEBenchAdapter

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
    if save_samples is False:
        assert {p.name for p in files} == {"summary.json", "results.json"}
        return
    rows = [
        json.loads(line)
        for line in (tmp_path / "sample_results.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert sum(row["resolve_rate"] for row in rows) / len(rows) == score == 0.5
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["primary_score"]["metric"]
        == "resolve_rate"
    )


def test_child_runner_service_account_can_be_isolated(adapter, monkeypatch):
    monkeypatch.setenv("SWE_BENCH_SERVICE_ACCOUNT", "swebench-ci-runner")
    assert adapter._current_service_account() == "swebench-ci-runner"


def test_configured_oci_export_failure_is_fatal(adapter, tmp_path):
    callbacks = MagicMock()
    callbacks.create_oci_artifact.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        adapter._persist_oci_artifact(
            [tmp_path / "sample_results.jsonl"],
            OCICoordinates(
                oci_host="quay.io", oci_repository="test/results", oci_tag="test"
            ),
            callbacks,
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
    if save_samples is not False:
        expected |= {"sample_results.jsonl", "manifest.json"}
        for name in ["sample_results.jsonl", "manifest.json"]:
            assert artifacts[name].content == (tmp_path / name).read_bytes()
        assert json.loads(artifacts["manifest.json"].content) == {
            "primary_score": {"metric": "resolve_rate"},
            "estimand": "mean",
        }
    assert set(artifacts) == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_runner_receives_requested_namespace_and_trusted_service_account(
    adapter, runner_runtime, explicit
):
    config, callbacks, runner, _ = runner_runtime
    if explicit:
        config.parameters.update(k8s_namespace="requested-namespace")
    result = adapter.run_benchmark_job(config, callbacks)
    assert result.overall_score == 0.5
    assert runner.call_args.kwargs["namespace"] == (
        "requested-namespace" if explicit else "pod-namespace"
    )
    assert runner.call_args.kwargs["service_account"] == "discovered-runner"


@pytest.mark.parametrize("value", ["false", 0, "0", None, 1, "true"])
def test_invalid_sample_flag_fails_before_dataset_or_runner(
    adapter, runner_runtime, value
):
    config, callbacks, runner, loader = runner_runtime
    config.parameters["save_sample_results"] = value
    result = adapter.run_benchmark_job(config, callbacks)
    loader.assert_not_called()
    runner.assert_not_called()
    callbacks.mlflow.save.assert_not_called()
    assert result.overall_score is None
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
def test_job_cannot_select_child_service_account(adapter, runner_runtime, account):
    config, callbacks, runner, loader = runner_runtime
    config.parameters["service_account"] = account
    result = adapter.run_benchmark_job(config, callbacks)
    loader.assert_not_called()
    runner.assert_not_called()
    callbacks.mlflow.save.assert_not_called()
    assert result.overall_score is None
    assert any(
        "parameters.service_account is not allowed" in str(call)
        for call in callbacks.report_status.call_args_list
    )
