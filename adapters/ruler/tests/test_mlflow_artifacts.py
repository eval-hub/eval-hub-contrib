"""Stream RULER exports before temporary results are cleaned up."""

import json
from unittest.mock import MagicMock

from evalhub.adapter import JobSpec, OCIArtifactResult
from evalhub.adapter.mlflow import MlflowFileArtifact

import main


def test_mlflow_saves_oci_files_before_cleanup(monkeypatch):
    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    config = JobSpec(
        id="ruler-stream", provider_id="ruler", benchmark_id="niah-single-noise",
        benchmark_index=0, model={"name": "test", "url": "http://model.invalid/v1"},
        callback_url="http://callback.invalid", experiment_name="ruler-stream",
        parameters={"context_lengths": [4096]},
        exports={"oci": {"coordinates": {
            "oci_host": "quay.io", "oci_repository": "test/ruler",
        }}},
    )
    monkeypatch.setattr(main.RulerAdapter, "local_jobs_base_path", property(lambda _: None))
    for method in ("_verify_tokenizer", "_prepare_task_assets", "_generate_task_data"):
        monkeypatch.setattr(adapter, method, lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_compute_dataset_hash", lambda *args: "test-hash")
    monkeypatch.setattr(adapter, "_make_api_client", lambda *args: None)
    monkeypatch.setattr(adapter, "_run_ruler_task", lambda **kwargs: [
        {"index": 0, "pred": "answer", "outputs": ["answer"]}
    ])
    monkeypatch.setattr(adapter, "_evaluate_predictions", lambda *args: [])
    monkeypatch.setattr(adapter, "_build_eval_card", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_build_env_card", lambda **kwargs: None)
    callbacks = MagicMock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest"
    )

    def save(results, job, *, artifacts):
        assert job is config
        root = callbacks.create_oci_artifact.call_args.args[0].files_path
        assert {artifact.path for artifact in artifacts} == {
            "summary.csv", "results.json", "samples.jsonl"
        }
        for artifact in artifacts:
            assert isinstance(artifact, MlflowFileArtifact)
            assert artifact.local_path == root / artifact.path
            assert artifact.local_path.is_file()
        sample = next(a for a in artifacts if a.path == "samples.jsonl")
        assert json.loads(sample.local_path.read_text())["pred"] == "answer"
        return "ruler-run"

    callbacks.mlflow.save.side_effect = save
    results = adapter.run_benchmark_job(config, callbacks)
    callbacks.mlflow.save.assert_called_once()
    assert results.mlflow_run_id == "ruler-run"
    assert not callbacks.create_oci_artifact.call_args.args[0].files_path.exists()
