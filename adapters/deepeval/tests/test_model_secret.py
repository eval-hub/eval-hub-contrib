"""Exercise role connections through the SDK's real projected Secret reader."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.mark.parametrize("role_entries", ["pair", "key_only", "none"])
@pytest.mark.parametrize(
    "parameters",
    [
        {"eval_model_url": "https://legacy.example/eval/v1"},
        {"judge_url": "https://legacy.example/judge/v1"},
        {"judge_base_url": "https://legacy.example/base/v1"},
        {"judge_api_key": "legacy-key"},
        {
            "eval_model_url": "https://legacy.example/eval/v1",
            "judge_url": "https://legacy.example/judge/v1",
            "judge_base_url": "https://legacy.example/base/v1",
            "judge_api_key": "legacy-key",
        },
    ],
)
def test_job_ignores_legacy_connections(
    model_secret, monkeypatch, parameters, role_entries
):
    secret_url = config().model.url
    secret_key = "api-key:ref"
    if role_entries != "none":
        secret_key = "judge_api-key:ref"
        (model_secret / "judge_api-key").write_text(secret_key)
    if role_entries == "pair":
        secret_url = "http://localhost:8080/judge/v1"
        (model_secret / "judge_url").write_text(secret_url)
    spec = config()
    spec.experiment_name = None
    spec.parameters.update(parameters)
    data_dir = model_secret / "test_data"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text(
        "input,actual_output,retrieval_context\nq,a,ctx\n"
    )
    monkeypatch.setattr(main, "_resolve_data_dir", lambda _config: str(data_dir))

    def create_metric(_benchmark_id, judge, _threshold):
        # Use the real GPTModel/OpenAI client to verify neither legacy value
        # reaches the judge, while avoiding inference calls in this unit test.
        client = judge.load_model()
        try:
            assert str(client.base_url).rstrip("/") == secret_url
            assert client.auth_headers["Authorization"] == f"Bearer {secret_key}"
        finally:
            client.close()
        return SimpleNamespace(name="Faithfulness")

    metric_factory = MagicMock(side_effect=create_metric)
    monkeypatch.setattr(main, "_create_metric", metric_factory)
    metric_data = SimpleNamespace(
        name="Faithfulness", score=1.0, success=True, reason="Supported"
    )
    evaluation = MagicMock(
        return_value=SimpleNamespace(
            test_results=[SimpleNamespace(metrics_data=[metric_data])]
        )
    )
    monkeypatch.setattr(main, "evaluate", evaluation)

    adapter = main.DeepEvalAdapter(job_spec_path="meta/job.json")
    result = adapter.run_benchmark_job(spec, MagicMock())

    metric_factory.assert_called_once()
    evaluation.assert_called_once()
    assert result.num_examples_evaluated == 1
    assert result.overall_score == 1.0
