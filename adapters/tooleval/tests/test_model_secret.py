"""Exercise role connections through the SDK's real projected Secret reader."""

import json

import pytest
import evalhub.adapter.auth as auth
from evalhub.adapter import JobSpec


@pytest.fixture(params=["k8s", "local"])
def model_secret(tmp_path, monkeypatch, request):
    monkeypatch.setenv("EVALHUB_MODE", request.param)
    if request.param == "local":
        payload = config().model_dump(mode="json")
        payload["model"]["auth"] = {"secret_ref": tmp_path.as_uri()}
        spec_path = tmp_path / "job.json"
        spec_path.write_text(json.dumps(payload))
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setattr(auth, "_MODEL_AUTH_DIR", tmp_path)
    (tmp_path / "api-key").write_text("api-key:ref")
    return tmp_path


def config():
    spec = JobSpec.model_validate(
        {
            "id": "role-routing",
            "provider_id": "tooleval",
            "benchmark_id": "single-tool",
            "benchmark_index": 0,
            "model": {"name": "target-model", "url": "http://localhost:8080/v1"},
            "parameters": {},
            "callback_url": "http://localhost:8081",
        }
    )
    spec.model.url = "http://localhost:8080/target/v1"
    return spec


import main


def judge(spec, params=None):
    return main._build_judge_client(
        config=spec,
        params=params or {},
        model_url=spec.model.url,
        model_name=spec.model.name,
        timeout=30,
    )


@pytest.mark.parametrize(
    "role_url",
    [None, "http://localhost:8080/target/v1", "http://localhost:8080/judge/v1"],
)
def test_judge_uses_matching_secret_reference(model_secret, monkeypatch, role_url):
    monkeypatch.setenv("TOOLEVAL_JUDGE_API_KEY", "unrelated-env-key")
    (model_secret / "judge_api-key").write_text("judge_api-key:ref")
    if role_url:
        (model_secret / "judge_url").write_text(role_url)
    spec = config()
    client, _, url = judge(
        spec, {"judge_url": "https://legacy.example/v1", "judge_api_key": "legacy-key"}
    )
    assert url == (role_url or spec.model.url)
    assert client.auth_headers["Authorization"] == "Bearer judge_api-key:ref"
    client.close()


def test_judge_url_without_key_fails(model_secret):
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    with pytest.raises(ValueError, match="requires judge_api-key"):
        judge(config())


@pytest.mark.parametrize("parameter", ["judge_url", "judge_api_key", "judge_base_url"])
def test_legacy_job_connections_are_ignored_without_role_entries(
    model_secret, parameter
):
    spec = config()
    client, _, url = judge(spec, {parameter: "legacy-value"})
    assert url == spec.model.url
    assert client.auth_headers["Authorization"] == "Bearer api-key:ref"
    client.close()
