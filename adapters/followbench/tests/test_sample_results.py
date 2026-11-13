"""Sample persistence keeps observations intact and propagates upload failures."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from evalhub.adapter import JobResults, JobSpec, OCIArtifactResult
from sample_results import persist_sample_results


def objects():
    config = JobSpec.model_validate(
        {
            "id": "sample-test",
            "provider_id": "test",
            "benchmark_id": "test",
            "benchmark_index": 0,
            "model": {"name": "test", "url": "http://localhost"},
            "experiment_name": "samples",
            "parameters": {},
            "callback_url": "http://localhost:8080",
            "exports": {
                "oci": {
                    "coordinates": {
                        "oci_host": "registry.example",
                        "oci_repository": "test/results",
                        "oci_tag": "test",
                    }
                }
            },
        }
    )
    result = JobResults(
        id=config.id,
        benchmark_id=config.benchmark_id,
        benchmark_index=0,
        model_name="test",
        results=[],
        num_examples_evaluated=2,
        duration_seconds=0.1,
        completed_at=datetime.now(UTC),
    )
    callbacks = MagicMock()
    callbacks.mlflow.save.return_value = "run-1"
    return config, result, callbacks


def test_same_native_bytes_reach_oci_and_mlflow():
    config, result, callbacks = objects()
    records = [{"sample_id": "a", "metric": 0.2}, {"sample_id": "b", "metric": 0.9}]
    uploaded = {}

    def upload(spec):
        uploaded["samples"] = (spec.files_path / "sample_results.jsonl").read_bytes()
        uploaded["manifest"] = json.loads(
            (spec.files_path / "manifest.json").read_text()
        )
        return OCIArtifactResult(digest="sha256:abc", reference="test@sha256:abc")

    callbacks.create_oci_artifact.side_effect = upload
    payload = persist_sample_results(config, callbacks, result, records, "metric")
    assert [json.loads(line) for line in payload.splitlines()] == records
    assert uploaded["samples"] == payload
    assert uploaded["manifest"]["primary_score"]["metric"] == "metric"
    artifacts = callbacks.mlflow.save.call_args.kwargs["artifacts"]
    assert artifacts[0].path == "sample_results.jsonl"
    assert artifacts[0].content == payload
    assert result.mlflow_run_id == "run-1"
    assert result.oci_artifact.digest == "sha256:abc"


def test_failed_upload_is_not_reported_as_success():
    config, result, callbacks = objects()
    callbacks.create_oci_artifact.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        persist_sample_results(
            config, callbacks, result, [{"sample_id": "a", "metric": 1}], "metric"
        )
    callbacks.mlflow.save.assert_not_called()


@pytest.mark.parametrize("records", [[], [{"sample_id": "a"}, {"sample_id": "a"}]])
def test_empty_or_duplicate_observations_are_rejected(records):
    config, result, callbacks = objects()
    with pytest.raises(ValueError, match="unique sample IDs"):
        persist_sample_results(config, callbacks, result, records, "metric")
    callbacks.create_oci_artifact.assert_not_called()


def test_missing_mlflow_run_id_is_fatal():
    config, result, callbacks = objects()
    config.exports = None
    callbacks.mlflow.save.return_value = None
    with pytest.raises(RuntimeError, match="did not confirm"):
        persist_sample_results(
            config, callbacks, result, [{"sample_id": "a", "metric": 1}], "metric"
        )


def test_oci_only_does_not_require_an_experiment():
    config, result, callbacks = objects()
    config.experiment_name = None
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:abc", reference="test@sha256:abc"
    )
    persist_sample_results(
        config, callbacks, result, [{"sample_id": "a", "metric": 1}], "metric"
    )
    callbacks.create_oci_artifact.assert_called_once()
    callbacks.mlflow.save.assert_not_called()


def test_missing_persistence_destination_is_rejected():
    config, result, callbacks = objects()
    config.experiment_name = None
    config.exports = None
    with pytest.raises(ValueError, match="OCI exports or MLflow persistence"):
        persist_sample_results(
            config, callbacks, result, [{"sample_id": "a", "metric": 1}], "metric"
        )
    callbacks.create_oci_artifact.assert_not_called()
    callbacks.mlflow.save.assert_not_called()
