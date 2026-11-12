"""Integration tests for the MTEB adapter.

Verifies adapter plumbing by mocking _run_mteb_subprocess and seeding the
output directory with canned task JSON results.
"""

import json
import subprocess

import pytest
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

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
def test_mteb_happy_path(tmp_path, monkeypatch):
    """Full run_benchmark_job with mocked subprocess and canned task results."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"].pop("batch_size")
    (meta_dir / "job.json").write_text(json.dumps(job))

    adapter = MTEBAdapter(job_spec_path=str(meta_dir / "job.json"))

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    # Patch _run_mteb_subprocess to seed the output directory with canned results.
    # run_benchmark_job creates output_dir via tempfile.mkdtemp,
    # passes it to _build_mteb_command which adds --output-folder <path>,
    # then calls _run_mteb_subprocess. We extract output_dir from the adapter.
    def fake_run(cmd, timeout):
        assert cmd[cmd.index("--batch-size") + 1] == "32"
        idx = cmd.index("--output-folder")
        output_dir = Path(cmd[idx + 1])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        (output_dir / "evalhub_sampling_info.json").write_text(
            json.dumps({"num_tasks_evaluated": 1})
        )
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
    # PERSISTING_ARTIFACTS only emitted when OCI exports are configured


def test_non_finite_metrics_are_omitted_and_raw_results_are_json_safe(
    tmp_path, monkeypatch
):
    adapter = MTEBAdapter(job_spec_path="meta/job.json")
    mteb_results = {
        "SciFact": {
            "scores": {
                "test": [
                    {
                        "main_score": float("nan"),
                        "ndcg_at_10": float("nan"),
                        "map_at_10": float("inf"),
                        "accuracy": 0.5,
                    },
                    {
                        "main_score": 0.75,
                        "ndcg_at_10": 0.8,
                        "map_at_10": float("-inf"),
                    },
                ]
            }
        }
    }

    results = adapter._extract_evaluation_results(mteb_results)
    metric_values = {result.metric_name: result.metric_value for result in results}
    assert metric_values == {
        "SciFact.test.accuracy": 0.5,
        "SciFact.test.main_score": 0.75,
        "SciFact.test.ndcg_at_10": 0.8,
    }
    assert adapter._compute_overall_score(results) == 0.75

    monkeypatch.setattr(
        MTEBAdapter,
        "local_jobs_base_path",
        property(lambda _self: tmp_path),
    )
    files = adapter._save_detailed_results(
        job_id="job-1",
        benchmark_id="mteb_retrieval",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        mteb_results=mteb_results,
        evaluation_results=results,
    )
    raw_results = json.loads(next(p for p in files if p.name == "mteb_results.json").read_text())
    assert raw_results["SciFact"]["scores"]["test"][0]["main_score"] is None
    assert raw_results["SciFact"]["scores"]["test"][1]["main_score"] == 0.75

    structured_results = json.loads(
        next(p for p in files if p.name == "results.json").read_text()
    )
    assert structured_results["overall_score"] == 0.75


@pytest.mark.integration
def test_oci_export_persists_artifacts(tmp_path, monkeypatch):
    """When exports.oci is configured, PERSISTING_ARTIFACTS is emitted and create_oci_artifact is called."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["exports"] = {
        "oci": {
            "coordinates": {
                "oci_host": "quay.io",
                "oci_repository": "test-org/test-repo",
                "oci_tag": "test-tag",
                "annotations": {},
            }
        }
    }
    (meta_dir / "job.json").write_text(json.dumps(job))

    adapter = MTEBAdapter(job_spec_path=str(meta_dir / "job.json"))

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )

    def fake_run(cmd, timeout):
        idx = cmd.index("--output-folder")
        output_dir = Path(cmd[idx + 1])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter, "_run_mteb_subprocess", fake_run)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # PERSISTING_ARTIFACTS phase was reported
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS in phases

    # create_oci_artifact was called with a directory containing result files
    call_args = callbacks.create_oci_artifact.call_args
    assert call_args is not None
    spec = call_args.args[0]
    assert spec.files_path.exists()
    assert spec.files_path.is_dir()

    # OCI artifact is attached to results
    assert results.oci_artifact is not None
    assert results.oci_artifact.digest == "sha256:fake"

    from evalhub.adapter.mlflow import MlflowFileArtifact
    import main

    assert {artifact.path for artifact in adapter.mlflow_artifacts} == {
        "mteb_results.json", "results.json", "summary.txt"
    }
    for artifact in adapter.mlflow_artifacts:
        assert isinstance(artifact, MlflowFileArtifact)
        assert artifact.local_path == spec.files_path / artifact.path
        assert artifact.local_path.is_file()
    callbacks.mlflow = MagicMock()
    callbacks.mlflow.save.return_value = "mteb-run"
    monkeypatch.setattr(main, "MTEBAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(main.DefaultCallbacks, "from_adapter", lambda adapter: callbacks)
    monkeypatch.setattr(main, "configure_telemetry", lambda: None)
    monkeypatch.setattr(adapter, "run_benchmark_job", lambda *args: results)
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    callbacks.mlflow.save.assert_called_once_with(
        results, adapter.job_spec, artifacts=adapter.mlflow_artifacts
    )
    assert callbacks.report_results.call_args.args[0].mlflow_run_id == "mteb-run"


@pytest.mark.integration
def test_results_use_local_jobs_base_path(tmp_path):
    """Results are saved under local_jobs_base_path/results, not hardcoded /tmp paths."""
    import shutil

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")

    adapter = MTEBAdapter(job_spec_path=str(meta_dir / "job.json"))

    expected_base = adapter.local_jobs_base_path
    assert expected_base is not None, "local mode should provide local_jobs_base_path"
    assert expected_base == tmp_path
    expected_results_dir = expected_base / "results"

    saved_files = adapter._save_detailed_results(
        job_id=adapter.job_spec.id,
        benchmark_id=adapter.job_spec.benchmark_id,
        model_name=adapter.job_spec.model.name,
        mteb_results=CANNED_STS12,
        evaluation_results=[],
    )

    assert len(saved_files) > 0
    for f in saved_files:
        assert str(f).startswith(str(expected_results_dir)), (
            f"Result file {f} should be under {expected_results_dir}"
        )
        assert "/tmp/mteb_results" not in str(f)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("num_examples", 0),
        ("num_examples", -1),
        ("num_examples", True),
        ("batch_size", 0),
        ("batch_size", 1.5),
        ("batch_size", "4"),
    ],
)
def test_sample_limit_and_batch_size_must_be_positive_integers(
    tmp_path, parameter, value
):
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"][parameter] = value
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = MTEBAdapter(job_spec_path=str(job_path))

    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        adapter._validate_config(adapter.job_spec)


def test_sample_limited_job_uses_runner_and_reports_effective_parameters(
    tmp_path, monkeypatch
):
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"].update({"num_examples": 2, "batch_size": 7})
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job_path = meta_dir / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = MTEBAdapter(job_spec_path=str(job_path))
    callbacks = create_autospec(JobCallbacks)

    def fake_run(cmd, timeout):
        assert Path(cmd[1]).name == "mteb_runner.py"
        runner_config = json.loads(Path(cmd[2]).read_text())
        assert runner_config["num_examples"] == 2
        assert runner_config["batch_size"] == 7

        output_dir = Path(runner_config["output_dir"])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        (output_dir / "evalhub_sampling_info.json").write_text(
            json.dumps(
                {
                    "num_examples_by_task": {"STS12": 2},
                    "total_examples_evaluated": 2,
                    "num_tasks_evaluated": 1,
                }
            )
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter, "_run_mteb_subprocess", fake_run)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # JobResults.num_examples_evaluated keeps its historical task-count meaning.
    assert results.num_examples_evaluated == 1
    assert results.additional_info["num_examples"] == 2
    assert results.additional_info["batch_size"] == 7
    assert results.additional_info["num_examples_by_task"] == {"STS12": 2}
    assert results.additional_info["total_examples_evaluated"] == 2


def test_task_type_jobs_use_runner_to_report_resolved_task_count(tmp_path, monkeypatch):
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"].pop("tasks")
    job["parameters"]["task_types"] = ["STS"]
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job_path = meta_dir / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = MTEBAdapter(job_spec_path=str(job_path))
    callbacks = create_autospec(JobCallbacks)

    def fake_run(cmd, timeout):
        assert Path(cmd[1]).name == "mteb_runner.py"
        runner_config = json.loads(Path(cmd[2]).read_text())
        assert runner_config["tasks"] == []
        assert runner_config["task_types"] == ["STS"]
        output_dir = Path(runner_config["output_dir"])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        (output_dir / "evalhub_sampling_info.json").write_text(
            json.dumps({"num_tasks_evaluated": 3})
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter, "_run_mteb_subprocess", fake_run)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.num_examples_evaluated == 3


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:0", "cuda:12"])
def test_device_override_uses_python_runner_for_full_dataset_jobs(
    tmp_path, monkeypatch, device
):
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"]["device"] = device
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job_path = meta_dir / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = MTEBAdapter(job_spec_path=str(job_path))
    callbacks = create_autospec(JobCallbacks)

    def fake_run(cmd, timeout):
        assert Path(cmd[1]).name == "mteb_runner.py"
        runner_config = json.loads(Path(cmd[2]).read_text())
        assert runner_config["num_examples"] is None
        assert runner_config["device"] == device
        output_dir = Path(runner_config["output_dir"])
        model_dir = output_dir / "model" / "no_revision"
        model_dir.mkdir(parents=True)
        (model_dir / "STS12.json").write_text(json.dumps(CANNED_STS12))
        (output_dir / "evalhub_sampling_info.json").write_text(
            json.dumps({"num_tasks_evaluated": 1})
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter, "_run_mteb_subprocess", fake_run)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.num_examples_evaluated == 1


@pytest.mark.parametrize("device", ["", "cuda:", "cuda:gpu", "cuda:-1", "gpu", 0])
def test_invalid_device_override_is_rejected(tmp_path, device):
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"]["device"] = device
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = MTEBAdapter(job_spec_path=str(job_path))

    with pytest.raises(ValueError, match="device must be one of"):
        adapter._validate_config(adapter.job_spec)
