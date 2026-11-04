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


@pytest.mark.parametrize("backend", ["litellm", "endpoint"])
def test_clear_judge_secret_overrides_runtime_env(
    model_secret, monkeypatch, tmp_path, backend
):
    (model_secret / "judge_api-key").write_text("judge_api-key:ref")
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    spec = config()
    spec.parameters["inference_backend"] = backend
    spec.parameters.update(
        {
            "judge_api_key": "legacy-key",
            "judge_url": "https://legacy.example/v1",
            "endpoint_url": "https://legacy.example/endpoint",
            "inference_url": "https://legacy.example/inference",
        }
    )
    adapter = main.ClearAdapter(job_spec_path="meta/job.json")
    adapter._ensure_openai_api_key_for_litellm()
    result = adapter._build_agentic_config(spec, "/data", tmp_path)
    assert result["endpoint_url"] == "http://localhost:8080/judge/v1"
    assert result["inference_url"] == "http://localhost:8080/judge/v1"
    import os

    assert os.environ["OPENAI_API_KEY"] == "judge_api-key:ref"


def test_clear_url_requires_matching_secret_key(model_secret):
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    with pytest.raises(ValueError, match="requires judge_api-key"):
        main.ClearAdapter._judge_url(config())


@pytest.mark.parametrize(
    "parameter", ["endpoint_url", "inference_url", "judge_url", "judge_api_key"]
)
def test_clear_legacy_model_connections_ignored_without_role_entries(
    model_secret, parameter
):
    spec = config()
    spec.parameters[parameter] = "legacy-value"
    assert main.ClearAdapter._judge_url(spec) == spec.model.url


@pytest.mark.parametrize("provider", ["rits", "watsonx", "azure"])
def test_native_provider_requires_sidecar_support(model_secret, tmp_path, provider):
    spec = config()
    spec.parameters["provider"] = provider
    adapter = main.ClearAdapter(job_spec_path="meta/job.json")
    with pytest.raises(ValueError, match="need sidecar support"):
        adapter._build_agentic_config(spec, "/data", tmp_path)


def test_model_params_cannot_override_secret(model_secret, tmp_path):
    spec = config()
    spec.parameters["eval_model_params"] = {
        "api_key": "legacy-value",
        "api-key": "legacy-value",
        "api_base": "https://legacy.example/v1",
        "base_url": "https://legacy.example/v1",
        "temperature": 0.3,
        "nested": [{"endpoint_url": "https://legacy.example/v1", "timeout": 30}],
    }
    adapter = main.ClearAdapter(job_spec_path="meta/job.json")
    result = adapter._build_agentic_config(spec, "/data", tmp_path)
    assert result["eval_model_params"] == {
        "temperature": 0.3,
        "nested": [{"timeout": 30}],
    }
    assert result["endpoint_url"] == spec.model.url
    assert spec.parameters["eval_model_params"]["api_key"] == "legacy-value"
