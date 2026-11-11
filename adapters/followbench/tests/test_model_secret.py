"""Exercise role connections through the SDK's real projected Secret reader."""

import json
from pathlib import Path

import pytest
import evalhub.adapter.auth as auth
from evalhub.adapter import JobSpec


@pytest.fixture(params=["k8s", "local"])
def model_secret(tmp_path, monkeypatch, request):
    monkeypatch.setenv("EVALHUB_MODE", request.param)
    if request.param == "local":
        payload = json.loads(Path("meta/job.json").read_text())
        payload["model"]["auth"] = {"secret_ref": tmp_path.as_uri()}
        spec_path = tmp_path / "job.json"
        spec_path.write_text(json.dumps(payload))
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setattr(auth, "_MODEL_AUTH_DIR", tmp_path)
    (tmp_path / "api-key").write_text("api-key:ref")
    return tmp_path


def config():
    spec = JobSpec.model_validate_json(Path("meta/job.json").read_text())
    spec.model.url = "http://localhost:8080/target/v1"
    return spec


import main


@pytest.mark.parametrize(
    "role_url",
    [None, "http://localhost:8080/target/v1", "http://localhost:8080/judge/v1"],
)
def test_judge_uses_secret_even_without_judge_model(
    model_secret, monkeypatch, role_url
):
    monkeypatch.setenv("FOLLOWBENCH_JUDGE_API_KEY", "unrelated-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    (model_secret / "judge_api-key").write_text("judge_api-key:ref")
    if role_url:
        (model_secret / "judge_url").write_text(role_url)
    spec = config()
    client, name = main._build_judge_client(
        spec,
        {
            "judge_api_key": "legacy-key",
            "judge_api-key": "legacy-key",
            "judge_url": "https://legacy.example/v1",
            "judge_base_url": "https://legacy.example/v1",
        },
        30,
    )
    assert name == spec.model.name
    assert str(client.base_url).rstrip("/") == (role_url or spec.model.url)
    assert client.auth_headers["Authorization"] == "Bearer judge_api-key:ref"
    client.close()


def test_missing_role_reuses_secret_primary_not_env(model_secret, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    client, _ = main._build_judge_client(
        config(),
        {"judge_api_key": "legacy-key", "judge_url": "https://legacy.example/v1"},
        30,
    )
    assert client.api_key == "api-key:ref"
    client.close()
