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
def test_gpt_model_uses_judge_secret(model_secret, monkeypatch, role_url):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    (model_secret / "judge_api-key").write_text("judge_api-key:ref")
    if role_url:
        (model_secret / "judge_url").write_text(role_url)
    judge = main._resolve_judge_model("gpt-4o-mini", config().model.url)
    client = judge.load_model()
    assert str(client.base_url).rstrip("/") == (role_url or config().model.url)
    assert client.auth_headers["Authorization"] == "Bearer judge_api-key:ref"
    client.close()


def test_judge_url_without_key_fails(model_secret):
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    with pytest.raises(ValueError, match="requires judge_api-key"):
        main._resolve_judge_model("gpt-4o-mini", config().model.url)


@pytest.mark.parametrize("parameter", ["eval_model_url", "judge_url", "judge_api_key"])
def test_legacy_connections_are_rejected(model_secret, parameter):
    from unittest.mock import MagicMock

    spec = config()
    spec.parameters[parameter] = "legacy-value"
    adapter = main.DeepEvalAdapter(job_spec_path="meta/job.json")
    with pytest.raises(ValueError, match="model.auth.secret_ref"):
        adapter.run_benchmark_job(spec, MagicMock())
