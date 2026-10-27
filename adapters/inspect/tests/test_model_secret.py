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


import _model_auth
import _routing


@pytest.mark.parametrize(
    "role", ["judge", "auditor", "realism", "scenarios", "grader", "custom_role"]
)
@pytest.mark.parametrize(
    "role_url",
    [None, "http://localhost:8080/target/v1", "http://localhost:8080/role/v1"],
)
def test_all_roles_read_secret_reference_and_path(
    model_secret, monkeypatch, role, role_url
):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-env-key")
    (model_secret / f"{role}_api-key").write_text(f"{role}_api-key:ref")
    if role_url:
        (model_secret / f"{role}_url").write_text(role_url)
    spec = config()
    value = json.loads(
        _routing.role_model_spec(
            "deployed-model", role, {}, {"EVALHUB_MODEL_URL": spec.model.url}
        )
    )
    assert value["model_args"]["base_url"] == (role_url or spec.model.url)
    assert value["model_args"]["api_key"] == f"{role}_api-key:ref"
    assert value["model"] == "openai/deployed-model"
    assert value["model_args"]["responses_api"] is False


@pytest.mark.parametrize("role", ["judge", "auditor", "scenarios", "realism", "grader"])
def test_role_url_requires_matching_key(model_secret, role):
    (model_secret / f"{role}_url").write_text("http://localhost:8080/role/v1")
    with pytest.raises(ValueError, match=f"requires {role}_api-key"):
        _model_auth.role_connection(role, config().model.url)


@pytest.mark.parametrize(
    "values",
    [
        {"judge_api_key": "legacy-value"},
        {"auditor_base_url": "legacy-value"},
        {
            "model_roles": {
                "grader": {"model": "openai/judge", "api_key": "legacy-value"}
            }
        },
        {"model_args": {"base_url": "legacy-value"}},
    ],
)
def test_connection_parameters_cannot_bypass_secret(values):
    with pytest.raises(ValueError, match="model.auth.secret_ref"):
        _model_auth.validate_connection_parameters(values)


@pytest.mark.parametrize("model", ["anthropic/claude"])
def test_native_clients_require_sidecar_auth_support(model_secret, model):
    with pytest.raises(ValueError, match="needs sidecar support"):
        _routing.role_model_spec(
            model, "judge", {}, {"EVALHUB_MODEL_URL": config().model.url}
        )


def test_ollama_client_carries_secret_reference(model_secret):
    value = json.loads(
        _routing.role_model_spec(
            "ollama/model", "judge", {}, {"EVALHUB_MODEL_URL": config().model.url}
        )
    )
    assert value["model"] == "ollama/model"
    assert value["model_args"]["api_key"] == "api-key:ref"


def test_bloom_scenarios_use_secret_connection(model_secret, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import _bloom

    (model_secret / "scenarios_api-key").write_text("scenarios_api-key:ref")
    (model_secret / "scenarios_url").write_text("http://localhost:8080/scenarios/v1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_bloom.subprocess, "run", run)
    spec = config()
    spec.parameters = {
        "bloom_template": "test",
        "scenarios_model": "ollama/scenario-model",
    }
    _bloom.bloom_prepare(
        spec, tmp_path, {"OPENAI_API_KEY": "unrelated-env-key"}, MagicMock()
    )
    command, env = calls[1]
    assert "scenarios=openai/scenario-model" in command
    assert env["OPENAI_API_KEY"] == "scenarios_api-key:ref"
    assert env["OPENAI_BASE_URL"] == "http://localhost:8080/scenarios/v1"
