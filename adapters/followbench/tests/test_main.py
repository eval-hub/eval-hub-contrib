"""Tests for the FollowBench adapter orchestration."""

from __future__ import annotations

import copy
import re
from unittest.mock import MagicMock, create_autospec

from evalhub.adapter import JobCallbacks, JobPhase

import main as main_module
from _evaluation import FollowBenchExample


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


def test_followbench_judge_and_callbacks_integration(monkeypatch):
    adapter = main_module.FollowBenchAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)
    config = copy.deepcopy(adapter.job_spec)
    config.parameters["num_examples"] = 1

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
