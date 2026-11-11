"""Exercise role connections through the SDK's real projected Secret reader."""

import asyncio
from pathlib import Path

import evalhub.adapter.auth as auth
import httpx
import pytest
import respx
from evalhub.adapter import JobSpec

from main import ATIFAdapter


@pytest.fixture
def model_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr(auth, "_MODEL_AUTH_DIR", tmp_path)
    (tmp_path / "api-key").write_text("api-key:ref")
    return tmp_path


def config():
    spec = JobSpec.model_validate_json(Path("meta/job.json").read_text())
    spec.model.url = "http://localhost:8080/target/v1"
    return spec


@pytest.mark.parametrize("role_entries", ["pair", "key_only", "none"])
def test_atif_uses_secret_and_ignores_inline_connections(
    model_secret, monkeypatch, role_entries
):
    url = config().model.url
    key = "api-key:ref"
    if role_entries != "none":
        key = "judge_api-key:ref"
        (model_secret / "judge_api-key").write_text(key)
    if role_entries == "pair":
        url = "http://localhost:8080/judge/v1"
        (model_secret / "judge_url").write_text(url)
    monkeypatch.setenv("EVALHUB_JUDGE_PROXY_URL", "https://unrelated.example")
    adapter = ATIFAdapter(job_spec_path="meta/job.json")
    payload = config().model_dump(mode="json")
    payload["model"]["auth"] = {"secret_ref": "model-auth"}
    adapter._active_job_spec = JobSpec.model_validate(payload)
    adapter._active_job_spec.parameters.update(
        {
            "judge_url": "https://legacy.example/v1",
            "judge_api_key": "legacy-key",
            "judge_api-key": "legacy-key",
        }
    )
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{url}/chat/completions").mock(
            return_value=httpx.Response(200, json={"score": 0.9})
        )
        asyncio.run(adapter._judge_call({"request": "score"}))
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {key}"


def test_atif_url_requires_matching_key(model_secret):
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    adapter = ATIFAdapter(job_spec_path="meta/job.json")
    with pytest.raises(ValueError, match="requires judge_api-key"):
        asyncio.run(adapter._judge_call({"request": "score"}))
