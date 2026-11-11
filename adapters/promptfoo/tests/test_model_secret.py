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


@pytest.mark.parametrize("role", ["judge", "generation_provider"])
@pytest.mark.parametrize(
    "role_url",
    [None, "http://localhost:8080/target/v1", "http://localhost:8080/role/v1"],
)
def test_roles_read_distinct_secret_references(
    model_secret, monkeypatch, role, role_url
):
    monkeypatch.setenv("GENERATION_PROVIDER_API_KEY", "unrelated-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    (model_secret / f"{role}_api-key").write_text(f"{role}_api-key:ref")
    if role_url:
        (model_secret / f"{role}_url").write_text(role_url)
    spec = config()
    url, key = main._role_connection(spec, role)
    env = main._promptfoo_env(api_key=key, base_url=url)
    assert env["OPENAI_API_KEY"] == f"{role}_api-key:ref"
    assert env["OPENAI_BASE_URL"] == (role_url or spec.model.url)
    assert "GENERATION_PROVIDER_API_KEY" not in env


@pytest.mark.parametrize("role", ["judge", "generation_provider"])
def test_role_url_requires_matching_secret_key(model_secret, role):
    (model_secret / f"{role}_url").write_text("http://localhost:8080/role/v1")
    with pytest.raises(ValueError, match=f"requires {role}_api-key"):
        main._role_connection(config(), role)


@pytest.mark.parametrize(
    "value",
    [
        {
            "defaultTest": {
                "options": {"provider": {"config": {"apiKey": "legacy-value"}}}
            }
        },
        {"env": {"OPENAI_BASE_URL": "https://legacy.example/v1"}},
    ],
)
def test_passthrough_connections_are_removed(value):
    sanitized = main._without_connection_options(value)
    assert "legacy" not in json.dumps(sanitized)
    assert "legacy" in json.dumps(value)


@pytest.mark.parametrize("use_yaml", [False, True])
def test_eval_config_ignores_nested_connections_and_preserves_test_data(
    model_secret, use_yaml
):
    import yaml

    spec = config()
    test = {
        "vars": {"url": "https://data.example/input", "api_key": "test-input"},
        "assert": [{"type": "llm-rubric", "value": "Be correct"}],
        "options": {
            "provider": {
                "id": "openai:chat:judge-model",
                "config": {
                    "apiKey": "legacy-key",
                    "apiBaseUrl": "https://legacy.example/v1",
                    "temperature": 0.3,
                },
            }
        },
    }
    params = {"prompts": ["{{url}}"], "tests": [test]}
    if use_yaml:
        params = {
            "config_yaml": yaml.safe_dump(
                {
                    **params,
                    "env": {
                        "OPENAI_API_KEY": "legacy-key",
                        "OPENAI_BASE_URL": "https://legacy.example/v1",
                    },
                }
            )
        }
    spec.parameters = params
    provider = main._build_target_provider(spec, "api-key:ref")
    result = main._build_eval_config(spec, provider, {})

    assert result["providers"] == [provider]
    assert result["tests"][0]["vars"] == test["vars"]
    assert result["tests"][0]["options"]["provider"]["config"] == {"temperature": 0.3}
    assert "legacy" not in json.dumps(result)
