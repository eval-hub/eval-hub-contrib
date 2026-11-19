"""Runtime output paths and streamed MLflow artifacts (adapted from PR #244)."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from evalhub.adapter import DefaultCallbacks, JobStatus
from evalhub.adapter.mlflow import MlflowFileArtifact
from evalhub.adapter.settings import AdapterSettings

import main


@pytest.mark.parametrize("mode", ["local", "k8s"])
@pytest.mark.parametrize("oci_enabled", [False, True])
def test_runtime_output_and_mlflow_artifacts(
    monkeypatch, tmp_path, mode, oci_enabled, mock_callbacks
):
    job_dir = tmp_path / "jobs/job-1/0/lm-eval/gsm8k"
    job_path = job_dir / "meta/job.json"
    job_path.parent.mkdir(parents=True)
    job = json.loads(Path("meta/job.json").read_text())
    job["experiment_name"] = "experiment"
    if oci_enabled:
        job["exports"] = {
            "oci": {
                "coordinates": {
                    "oci_host": "registry.example.com",
                    "oci_repository": "results",
                }
            }
        }
    job_path.write_text(json.dumps(job))
    settings = AdapterSettings(EVALHUB_MODE=mode, EVALHUB_JOB_SPEC_PATH=job_path)
    adapter = main.LMEvalAdapter(settings=settings)
    callbacks = DefaultCallbacks.from_adapter(adapter)
    monkeypatch.setattr(callbacks, "report_status", Mock())
    monkeypatch.setattr(
        callbacks, "create_oci_artifact", mock_callbacks.create_oci_artifact
    )
    output_dirs = []

    def run(cmd, env, timeout):
        output_dir = Path(cmd[cmd.index("--output_path") + 1])
        output_dirs.append(output_dir)
        nested = output_dir / "model"
        nested.mkdir()
        (nested / "results.json").write_text(
            json.dumps(
                {
                    "results": {"gsm8k": {"exact_match,none": 0.75}},
                    "n-samples": {"gsm8k": {"effective": 1}},
                }
            )
        )
        (nested / "samples.jsonl").write_text('{"doc_id": 0}\n')
        (nested / "data.bin").write_bytes(b"binary")
        (output_dir / "linked.json").symlink_to(nested / "results.json")
        (output_dir / "linked-dir").symlink_to(nested, target_is_directory=True)

    monkeypatch.setattr(adapter, "_run_lmeval", run)

    def save(results, config, *, artifacts):
        assert callbacks.mlflow._callbacks is None
        assert results.overall_score == 0.75
        assert config is adapter.job_spec
        files = {artifact.path: artifact for artifact in artifacts}
        assert set(files) == {
            "model/results.json",
            "model/samples.jsonl",
            "model/data.bin",
        }
        for path, artifact in files.items():
            assert isinstance(artifact, MlflowFileArtifact)
            assert artifact.local_path == output_dirs[0] / path
            assert artifact.local_path.is_file()
        assert files["model/results.json"].content_type == "application/json"
        assert files["model/data.bin"].content_type == "application/octet-stream"
        return "run-1"

    save_mock = Mock(side_effect=save)
    monkeypatch.setattr(callbacks.mlflow, "save", save_mock)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    save_mock.assert_called_once()
    assert results.mlflow_run_id == "run-1"
    assert callbacks.mlflow._callbacks is callbacks
    output_dir = output_dirs[0]
    if mode == "local":
        assert output_dir == job_dir / "output"
        assert (output_dir / "model/results.json").is_file()
    else:
        assert output_dir != job_dir / "output"
        assert not output_dir.parent.exists()
        assert not (job_dir / "output").exists()
    if oci_enabled:
        callbacks.create_oci_artifact.assert_called_once()
        assert callbacks.create_oci_artifact.call_args.args[0].files_path == output_dir
    else:
        callbacks.create_oci_artifact.assert_not_called()
    assert all(
        call.args[0].status != JobStatus.FAILED
        for call in callbacks.report_status.call_args_list
    )


@pytest.mark.parametrize("mode", ["local", "k8s", None])
def test_main_uses_runtime_settings(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("EVALHUB_MODE", raising=False)
    else:
        monkeypatch.setenv("EVALHUB_MODE", mode)
    monkeypatch.delenv("EVALHUB_JOB_SPEC_PATH", raising=False)
    adapter = Mock()
    constructor = Mock(return_value=adapter)
    callbacks = Mock()
    monkeypatch.setattr(main, "LMEvalAdapter", constructor)
    monkeypatch.setattr(DefaultCallbacks, "from_adapter", Mock(return_value=callbacks))

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    settings = constructor.call_args.kwargs["settings"]
    assert settings.mode == (mode or "k8s")
    assert settings.resolved_job_spec_path == Path(
        "meta/job.json" if mode == "local" else "/meta/job.json"
    )
    callbacks.report_results.assert_called_once_with(
        adapter.run_benchmark_job.return_value
    )
