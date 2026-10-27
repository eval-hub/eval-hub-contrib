"""Tests for the FollowBench adapter orchestration."""

from __future__ import annotations

import copy
import re
from unittest.mock import MagicMock, create_autospec

import pytest
from evalhub.adapter import JobCallbacks, JobPhase

import main as main_module
from _evaluation import FollowBenchExample


@pytest.mark.parametrize("job_spec_path", [None, "/custom/job.json"])
def test_main_loads_job_and_reports_results(monkeypatch, job_spec_path):
    if job_spec_path is None:
        monkeypatch.delenv("EVALHUB_JOB_SPEC_PATH", raising=False)
    else:
        monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", job_spec_path)

    monkeypatch.setattr(main_module, "configure_telemetry", lambda: None)
    adapter = MagicMock()
    adapter_factory = MagicMock(return_value=adapter)
    monkeypatch.setattr(main_module, "FollowBenchAdapter", adapter_factory)
    callbacks = MagicMock()
    monkeypatch.setattr(
        main_module.DefaultCallbacks, "from_adapter", MagicMock(return_value=callbacks)
    )

    main_module.main()

    adapter_factory.assert_called_once_with(
        job_spec_path=job_spec_path or "/meta/job.json"
    )
    main_module.DefaultCallbacks.from_adapter.assert_called_once_with(adapter)
    callbacks.tracer.evaluation_run.assert_called_once_with()
    adapter.run_benchmark_job.assert_called_once_with(adapter.job_spec, callbacks)
    callbacks.report_results.assert_called_once_with(
        adapter.run_benchmark_job.return_value
    )


def test_build_judge_prompt_uses_requested_level():
    group = [
        FollowBenchExample(1, "content", "COGNAC", 0, "base", ""),
        FollowBenchExample(1, "content", "COGNAC", 1, "one constraint", ""),
        FollowBenchExample(1, "content", "COGNAC", 2, "two constraints", ""),
    ]

    prompt = main_module._build_judge_prompt(group, "model response", level=1)

    assert "Return only a Python-style list with 1 values" in prompt
    assert "one constraint" in prompt
    assert "two constraints" not in prompt


def test_build_judge_client_reuses_model_credential_for_default_endpoint(monkeypatch):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    monkeypatch.delenv("FOLLOWBENCH_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolve_model_api_key = MagicMock(return_value="model-secret")
    monkeypatch.setattr(main_module, "_resolve_model_api_key", resolve_model_api_key)
    openai_client = MagicMock(name="openai_client")
    openai_constructor = MagicMock(return_value=openai_client)
    monkeypatch.setattr(main_module.openai, "OpenAI", openai_constructor)

    client, judge_model = main_module._build_judge_client(config, {}, 30)

    assert client is openai_client
    assert judge_model == config.model.name
    assert openai_constructor.call_args.kwargs["api_key"] == "model-secret"
    resolve_model_api_key.assert_called_once_with(config)


def test_build_judge_client_prefers_explicit_judge_credential(monkeypatch):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    monkeypatch.delenv("FOLLOWBENCH_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolve_model_api_key = MagicMock(return_value="model-secret")
    monkeypatch.setattr(main_module, "_resolve_model_api_key", resolve_model_api_key)
    openai_constructor = MagicMock()
    monkeypatch.setattr(main_module.openai, "OpenAI", openai_constructor)

    main_module._build_judge_client(
        config,
        {"judge_api_key": "judge-secret"},
        30,
    )

    assert openai_constructor.call_args.kwargs["api_key"] == "judge-secret"
    resolve_model_api_key.assert_not_called()


def test_build_judge_client_does_not_reuse_model_credential_for_other_endpoint(
    monkeypatch,
):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    monkeypatch.delenv("FOLLOWBENCH_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolve_model_api_key = MagicMock(return_value="model-secret")
    monkeypatch.setattr(main_module, "_resolve_model_api_key", resolve_model_api_key)
    openai_constructor = MagicMock()
    monkeypatch.setattr(main_module.openai, "OpenAI", openai_constructor)

    main_module._build_judge_client(
        config,
        {"judge_url": "https://judge.example/v1"},
        30,
    )

    assert openai_constructor.call_args.kwargs["api_key"] == "DUMMY"
    resolve_model_api_key.assert_not_called()


def test_collection_without_judge_secret_fields_reuses_target_connection(monkeypatch):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    monkeypatch.setenv("FOLLOWBENCH_JUDGE_API_KEY", "unrelated-key")
    monkeypatch.setattr(main_module, "read_model_auth_key", lambda key: None)
    monkeypatch.setattr(main_module, "_resolve_model_api_key", lambda _: "api-key:ref")
    constructor = MagicMock()
    monkeypatch.setattr(main_module.openai, "OpenAI", constructor)

    _, name = main_module._build_judge_client(
        config, {"judge_model": "deployed-judge"}, 30
    )

    assert name == "deployed-judge"
    assert constructor.call_args.kwargs["base_url"] == config.model.url
    assert constructor.call_args.kwargs["api_key"] == "api-key:ref"


@pytest.mark.parametrize("projected_url", [None, "http://localhost:8080/judge/v1"])
def test_collection_judge_secret_fields_select_connection(monkeypatch, projected_url):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    projected = {"judge_api-key": "judge_api-key:ref", "judge_url": projected_url}
    monkeypatch.setattr(main_module, "read_model_auth_key", projected.get)
    constructor = MagicMock()
    monkeypatch.setattr(main_module.openai, "OpenAI", constructor)

    _, name = main_module._build_judge_client(
        config, {"judge_model": "deployed-judge"}, 30
    )

    assert name == "deployed-judge"
    assert constructor.call_args.kwargs["base_url"] == (projected_url or config.model.url)
    assert constructor.call_args.kwargs["api_key"] == "judge_api-key:ref"


@pytest.mark.parametrize("invalid_model", [None, "", "  ", 123])
def test_collection_judge_rejects_invalid_model(monkeypatch, invalid_model):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec

    with pytest.raises(ValueError, match="judge_model"):
        main_module._build_judge_client(config, {"judge_model": invalid_model}, 30)


def test_collection_judge_url_requires_projected_key(monkeypatch):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec
    monkeypatch.setattr(main_module, "read_model_auth_key", {"judge_url": "http://localhost:8080/judge/v1"}.get)

    with pytest.raises(ValueError, match="judge_api-key"):
        main_module._build_judge_client(
            config, {"judge_model": "deployed-judge"}, 30
        )


@pytest.mark.parametrize("legacy_parameter", ["judge_url", "judge_api_key"])
def test_collection_judge_rejects_legacy_connection_parameters(legacy_parameter):
    config = main_module.FollowBenchAdapter(job_spec_path="meta/job.json").job_spec

    with pytest.raises(ValueError, match="judge_model cannot be combined with the legacy"):
        main_module._build_judge_client(
            config,
            {"judge_model": "deployed-judge", legacy_parameter: "legacy-value"},
            30,
        )


def test_malformed_judge_response_is_scored_as_no(monkeypatch):
    example = FollowBenchExample(1, "content", "target", 1, "instruction", "source")
    monkeypatch.setattr(
        main_module,
        "evaluate_rule_constraint",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(main_module, "_build_judge_prompt", lambda *args: "prompt")
    monkeypatch.setattr(
        main_module,
        "_call_chat_model",
        lambda *args, **kwargs: "not a valid judge response",
    )

    score = main_module._score_example(
        example,
        "model response",
        [example],
        MagicMock(name="judge_client"),
        "judge",
        max_tokens=32,
        temperature=0.0,
    )

    assert score.hard_satisfied is False
    assert score.soft_satisfied == 0.0


def test_followbench_judge_and_callbacks_integration(monkeypatch):
    adapter = main_module.FollowBenchAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)
    config = copy.deepcopy(adapter.job_spec)
    # EvalHub serializes this standard limit to the JobSpec root field.
    config.parameters.pop("num_examples", None)
    config.num_examples = 1

    model_client = MagicMock(name="model_client")
    judge_client = MagicMock(name="judge_client")
    judge_calls: list[str] = []

    monkeypatch.setattr(
        main_module,
        "_build_model_client",
        lambda config, request_timeout: model_client,
    )
    monkeypatch.setattr(
        main_module,
        "_build_judge_client",
        lambda config, parameters, request_timeout: (judge_client, "judge"),
    )

    def fake_call_chat_model(client, model_name, prompt, *, max_tokens, temperature):
        if client is model_client:
            return "model response"

        judge_calls.append(prompt)
        count = int(re.search(r"with (\d+) values", prompt).group(1))
        return "YES" if count == 1 else str(["YES"] * count)

    monkeypatch.setattr(main_module, "_call_chat_model", fake_call_chat_model)

    results = adapter.run_benchmark_job(config, callbacks)
    callbacks.report_results(results)
    metrics = {result.metric_name: result.metric_value for result in results.results}

    assert len(judge_calls) == 5
    assert metrics["hsr"] == 1.0
    assert metrics["ssr"] == 1.0
    assert metrics["csl"] == 5.0
    assert metrics["n_evaluated"] == 5
    assert results.overall_score == 1.0
    callbacks.report_results.assert_called_once_with(results)

    phases = [call.args[0].phase for call in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    assert JobPhase.PERSISTING_ARTIFACTS in phases
