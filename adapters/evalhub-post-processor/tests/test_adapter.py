import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from evalhub.adapter import JobSpec

import main
from post_processor.config import ci_config
from post_processor.operations import OPERATIONS, artifact_sources
from post_processor.transport import PostProcessorCallbacks, Sidecar


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    cal = tmp_path / "calibration"
    cal.mkdir()
    rows = []
    for index in [0, 1]:
        for sid, label, prediction in [("c1", 1, 0.8), ("c2", 0, 0.2), ("c3", 1, 0.7)]:
            rows.append(
                {
                    "example_id": sid,
                    "human_score": label,
                    "judge_score": prediction,
                    "benchmark_id": f"benchmark-{index}",
                    "provider_id": "test-provider",
                }
            )
    (cal / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setenv(
        "EVALHUB_POST_PROCESSOR_PVC_MOUNTS", json.dumps({"calibration-data": str(cal)})
    )
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.setenv("EVALHUB_MODE", "local")
    spec = json.loads((Path(__file__).resolve().parents[1] / "meta/job.json").read_text())
    spec["model"] = {"name": "evaluation-post-processor"}
    spec["parameters"]["operations"]["confidence_interval"]["results_data_ref"]["eval_job"][
        "id"
    ] = "source"
    spec["callback_url"] = "http://sidecar"
    job = {
        "resource": {"id": "source"},
        "collection": {"id": "collection-1"},
        "experiment": {"name": "evaluation"},
        "status": {"state": "completed", "benchmarks": []},
        "results": {"benchmarks": []},
    }
    for index in [1, 0]:
        benchmark = {
            "id": f"benchmark-{index}",
            "provider_id": "test-provider",
            "benchmark_index": index,
        }
        job["status"]["benchmarks"].append({**benchmark, "status": "completed"})
        job["results"]["benchmarks"].append(
            {
                **benchmark,
                "mlflow_run_id": f"run-{index}",
                "metrics": {"accuracy": 0.8},
                "test": {"primary_score": 0.8, "primary_score_metric": "accuracy"},
            }
        )
    result_data = {
        "primary_score": {"metric": "accuracy"},
        "samples": [
            {"sample_id": f"e{i}", "prediction": p} for i, p in enumerate([0.2, 0.4, 0.8, 0.9])
        ],
    }
    events = []
    gets = []
    registry = {}

    def handler(request):
        assert request.url.host == "sidecar"
        if request.method == "POST":
            assert request.url.path == "/api/v1/evaluations/jobs/post-processing-job-001/events"
            events.append(json.loads(request.content)["benchmark_status_event"])
            return httpx.Response(200, json={})
        gets.append(str(request.url))
        if request.url.path in registry:
            return httpx.Response(200, content=registry[request.url.path])
        if request.url.path == "/api/v1/evaluations/jobs/source":
            return httpx.Response(200, json=job)
        if request.url.path == "/api/2.0/mlflow/runs/get":
            run_id = request.url.params["run_id"]
            return httpx.Response(
                200,
                json={
                    "run": {
                        "info": {
                            "run_id": run_id,
                            "experiment_id": "1",
                            "artifact_uri": f"mlflow-artifacts:/1/{run_id}/artifacts",
                        }
                    }
                },
            )
        if request.url.path == "/api/2.0/mlflow/artifacts/list":
            return httpx.Response(200, json={"files": [{"path": "samples.json", "is_dir": False}]})
        if request.url.path.endswith("/samples.json"):
            return httpx.Response(200, json=result_data)
        raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(main, "Sidecar", lambda url: Sidecar(url, client=client))
    path = tmp_path / "job.json"

    def execute():
        path.write_text(json.dumps(spec))
        adapter = main.PostProcessorAdapter(job_spec_path=str(path))
        callbacks = PostProcessorCallbacks(
            adapter.job_spec, Sidecar("http://sidecar", client=client)
        )
        result = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(result)
        return result

    yield {
        "spec": spec,
        "job": job,
        "events": events,
        "gets": gets,
        "run": execute,
        "data": result_data,
        "cal": cal,
        "path": path,
        "client": client,
        "registry": registry,
    }
    client.close()


def test_completed_collection_downloads_each_benchmark_and_reports_real_ppi(runtime):
    runtime["spec"]["parameters"].pop("operation_order")
    result = runtime["run"]()
    assert result.results == []
    assert result.overall_score is None
    assert runtime["events"][0]["status"] == "running"
    completed = runtime["events"][-1]
    assert completed["status"] == "completed"
    assert completed["phase"] == "completed"
    assert "metrics" not in completed
    assert completed["benchmark_index"] == 0
    assert completed["started_at"] <= completed["completed_at"]
    assert completed["duration_seconds"] > 0
    output = completed["additional_info"]["confidence_interval"]["benchmarks"]
    assert [r["benchmark_index"] for r in output] == [1, 0]
    assert all(
        r["confidence_interval"]["lower"] < r["confidence_interval"]["upper"] for r in output
    )
    assert sum("/runs/get" in path for path in runtime["gets"]) == 2


@pytest.mark.parametrize("mlflow_fallback", [False, True])
def test_completed_job_oci_pipeline_and_mlflow_fallback(runtime, mlflow_fallback):
    payload = json.dumps(runtime["data"]).encode()
    blob_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/json",
                    "digest": blob_digest,
                    "size": len(payload),
                    "annotations": {"org.opencontainers.image.title": "samples.json"},
                }
            ],
        }
    ).encode()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    coordinates = {"oci_host": "quay.io", "oci_repository": "org/results", "oci_tag": "source"}
    exports = {"oci": {"coordinates": coordinates}}
    runtime["spec"]["exports"] = exports
    runtime["job"]["exports"] = exports
    if mlflow_fallback:
        # Aggregate-only MLflow data is insufficient; usable OCI samples exist.
        runtime["data"].clear()
        runtime["data"]["metrics"] = {"accuracy": 0.8}
    else:
        runtime["job"].pop("experiment")
    for benchmark in runtime["job"]["results"]["benchmarks"]:
        benchmark["artifacts"] = {"oci_digest": manifest_digest}
    runtime["registry"][f"/v2/org/results/manifests/{manifest_digest}"] = manifest
    runtime["registry"][f"/v2/org/results/blobs/{blob_digest}"] = payload
    result = runtime["run"]()
    assert len(result.additional_info["confidence_interval"]["benchmarks"]) == 2
    assert runtime["events"][-1]["status"] == "completed"
    assert sum("/manifests/" in path for path in runtime["gets"]) == 2
    assert any("/runs/get" in path for path in runtime["gets"]) == mlflow_fallback


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda j: j["status"].update(state="partially_failed"), "completed successfully"),
        (lambda j: j["status"]["benchmarks"][0].update(status="failed"), "Every source benchmark"),
        (lambda j: j["results"]["benchmarks"].pop(), "matching unique"),
        (
            lambda j: j["results"]["benchmarks"][0].update(
                metrics_schema=[{"name": "accuracy", "type": "categorical"}]
            ),
            "not numeric",
        ),
    ],
)
def test_invalid_source_fails_without_completed_event(runtime, mutation, message):
    mutation(runtime["job"])
    with pytest.raises(ValueError, match=message):
        runtime["run"]()
    assert runtime["events"][-1]["status"] == "failed"
    assert not any(e["status"] == "completed" for e in runtime["events"])


def test_repeated_benchmark_ids_are_distinguished_by_provider(runtime):
    for part in ("status", "results"):
        for benchmark in runtime["job"][part]["benchmarks"]:
            benchmark["id"] = "same-benchmark"
            benchmark["provider_id"] = f"provider-{benchmark['benchmark_index']}"
    calibration_path = runtime["cal"] / "labels.jsonl"
    calibration = [json.loads(line) for line in calibration_path.read_text().splitlines()]
    for row in calibration:
        row["provider_id"] = row["benchmark_id"].replace("benchmark-", "provider-")
        row["benchmark_id"] = "same-benchmark"
    calibration_path.write_text("\n".join(json.dumps(row) for row in calibration))
    result = runtime["run"]()
    rows = result.additional_info["confidence_interval"]["benchmarks"]
    assert [(r["id"], r["provider_id"]) for r in rows] == [
        ("same-benchmark", "provider-1"),
        ("same-benchmark", "provider-0"),
    ]
    assert [r["benchmark_index"] for r in rows] == [1, 0]


def test_operations_run_in_requested_order(runtime, monkeypatch):
    order = []

    def first(context, config):
        assert order == ["second"]
        order.append("first")
        return {"value": 1}

    def second(context, config):
        assert order == []
        order.append("second")
        return {"value": 2}

    monkeypatch.setitem(OPERATIONS, "first", first)
    monkeypatch.setitem(OPERATIONS, "second", second)
    runtime["spec"]["parameters"] = {
        "operations": {"first": {}, "second": {}},
        "operation_order": ["second", "first"],
    }
    assert runtime["run"]().additional_info == {"second": {"value": 2}, "first": {"value": 1}}
    assert order == ["second", "first"]


def test_operation_failure_stops_later_operations(runtime, monkeypatch):
    called = []

    def broken(context, config):
        raise ValueError("operation failed")

    monkeypatch.setitem(OPERATIONS, "broken", broken)
    monkeypatch.setitem(OPERATIONS, "later", lambda c, p: called.append(True))
    runtime["spec"]["parameters"] = {
        "operations": {"broken": {}, "later": {}},
        "operation_order": ["broken", "later"],
    }
    with pytest.raises(ValueError, match="operation failed"):
        runtime["run"]()
    assert not called
    assert runtime["events"][-1]["status"] == "failed"


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"operations": {}}, "nonempty"),
        ({"operations": {"unknown": {}}}, "Unknown"),
        (
            {"operations": {"a": {}}, "operation_order": ["unknown"]},
            "list each operation name exactly once",
        ),
        (
            {"operations": {"a": {}}, "operation_order": "a"},
            "array of operation names",
        ),
    ],
)
def test_bad_operations_fail_early(runtime, parameters, message):
    runtime["spec"]["parameters"] = parameters
    with pytest.raises(ValueError, match=message):
        runtime["run"]()
    assert not runtime["gets"]


def test_flat_server_parameters_compatibility(runtime):
    runtime["spec"]["parameters"] = runtime["spec"]["parameters"]["operations"][
        "confidence_interval"
    ]
    assert "confidence_interval" in runtime["run"]().additional_info


def test_event_delivery_failure_is_not_silently_successful():
    spec = JobSpec(
        id="post",
        provider_id="internal",
        benchmark_id="processor",
        benchmark_index=0,
        model={"url": "", "name": "unused"},
        parameters={},
        callback_url="http://sidecar",
    )
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        callbacks = PostProcessorCallbacks(spec, Sidecar("http://sidecar", client=client))
        from evalhub.adapter import JobStatus, JobStatusUpdate

        with pytest.raises(httpx.HTTPStatusError):
            callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING))


def test_alpha_and_thread_validation(runtime):
    config = copy.deepcopy(
        runtime["spec"]["parameters"]["operations"]["confidence_interval"]
    )
    config["significance_level"] = "0.7"
    assert ci_config(config)["significance_level"] == 0.7
    config["significance_level"] = True
    with pytest.raises(ValueError):
        ci_config(config)


def test_oci_reference_uses_benchmark_digest():
    digest = "sha256:" + "a" * 64
    source = artifact_sources(
        {
            "exports": {
                "oci": {
                    "coordinates": {
                        "oci_host": "quay.io",
                        "oci_repository": "org/job",
                        "oci_tag": "job",
                    }
                }
            }
        },
        {
            "artifacts": {
                "oci_reference": f"quay.io/org/benchmark:result@{digest}",
                "oci_digest": digest,
            }
        },
    )
    assert source[0]["oci"]["coordinates"]["oci_repository"] == "org/benchmark"
    assert source[0]["oci"]["digest"] == digest


def test_external_data_uses_manifest_primary_metric(runtime, tmp_path, monkeypatch):
    results = tmp_path / "external-results"
    results.mkdir()
    (results / "samples.json").write_text(json.dumps(runtime["data"]))
    # A single external source has no source benchmark identities.
    calibration = [
        {"sample_id": "c1", "label": 1, "prediction": 0.8},
        {"sample_id": "c2", "label": 0, "prediction": 0.3},
    ]
    (runtime["cal"] / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in calibration))
    monkeypatch.setenv(
        "EVALHUB_POST_PROCESSOR_PVC_MOUNTS",
        json.dumps({"calibration-data": str(runtime["cal"]), "results": str(results)}),
    )
    config = runtime["spec"]["parameters"]["operations"]["confidence_interval"]
    config["results_data_ref"] = {"pvc": {"claim_name": "results"}}
    config["calibration_data_ref"][0].pop("data_config")
    result = runtime["run"]()
    bounds = result.additional_info["confidence_interval"]["confidence_interval"]
    assert bounds["lower"] < bounds["upper"]
    assert not runtime["gets"]


@pytest.mark.parametrize("broken", [False, True])
def test_cli_exit_code_and_terminal_event(runtime, monkeypatch, broken):
    if broken:
        runtime["spec"]["parameters"] = {
            "operations": {"not-registered": {}}
        }
    runtime["path"].write_text(json.dumps(runtime["spec"]))
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(runtime["path"]))
    assert main.main() == int(broken)
    assert runtime["events"][-1]["status"] == ("failed" if broken else "completed")


@pytest.mark.parametrize("alpha", [0, 1, -0.5, float("nan"), float("inf"), None])
def test_invalid_alpha_is_rejected(runtime, alpha):
    config = runtime["spec"]["parameters"]["operations"]["confidence_interval"]
    config["significance_level"] = alpha
    with pytest.raises(ValueError, match="significance_level"):
        ci_config(config)


@pytest.mark.parametrize("threads", [0, 65, True, 1.5, "3"])
def test_invalid_thread_count_is_rejected(runtime, threads):
    config = runtime["spec"]["parameters"]["operations"]["confidence_interval"]
    config["num_parallel_threads"] = threads
    with pytest.raises(ValueError, match="num_parallel_threads"):
        ci_config(config)
