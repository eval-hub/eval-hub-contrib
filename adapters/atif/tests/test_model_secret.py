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


def test_atif_uses_judge_secret_and_ignores_proxy_env(model_secret, monkeypatch):
    (model_secret / "judge_api-key").write_text("judge_api-key:ref")
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    monkeypatch.setenv("EVALHUB_JUDGE_PROXY_URL", "https://unrelated.example")
    adapter = ATIFAdapter(job_spec_path="meta/job.json")
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("http://localhost:8080/judge/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"score": 0.9})
        )
        asyncio.run(adapter._judge_call({"request": "score"}))
    assert route.calls[0].request.headers["Authorization"] == "Bearer judge_api-key:ref"


def test_atif_url_requires_matching_key(model_secret):
    (model_secret / "judge_url").write_text("http://localhost:8080/judge/v1")
    adapter = ATIFAdapter(job_spec_path="meta/job.json")
    with pytest.raises(ValueError, match="requires judge_api-key"):
        asyncio.run(adapter._judge_call({"request": "score"}))
