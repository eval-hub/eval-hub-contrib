"""Tests for the IFBench instruction-following adapter.

Dataset loading and OpenAI API calls are monkeypatched — no real network calls.
"""

from __future__ import annotations

import copy
import sys
import types
from unittest.mock import MagicMock, create_autospec

import pytest

from evalhub.adapter import EvaluationResult, JobCallbacks, JobPhase, JobStatus
from evalhub.adapter.models.cards import EvalCardMetadata, EnvironmentCardMetadata
from main import (
    IFBenchAdapter,
    _build_env_card,
    _build_eval_card,
    _local_only_run,
    _resolve_api_key,
)
from _evaluation import InputExample, compute_scores


CANNED_EXAMPLES = [
    InputExample(
        key="0",
        instruction_id_list=["keywords:existence"],
        prompt="Say hello.",
        kwargs=[{"N": None, "keywords": ["zebra"], "keyword1": None}],
    ),
    InputExample(
        key="1",
        instruction_id_list=["keywords:existence"],
        prompt="Write a poem.",
        kwargs=[{"N": None, "keywords": ["nebula"], "keyword1": None}],
    ),
]


def test_resolve_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "test-key"


def test_resolve_api_key_sentinel(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "not-required"


def test_compute_scores_empty():
    scores = compute_scores([])
    assert scores["accuracy"] == 0.0
    assert scores["n_evaluated"] == 0.0


def test_build_eval_card_shape():
    card = _build_eval_card(accuracy=0.34, n_evaluated=299, scoring_mode="loose")
    assert isinstance(card, EvalCardMetadata)
    assert card.modalities_input == ["text"]
    assert len(card.capability_evaluations) == 1
    assert card.capability_evaluations[0].zero_shot == pytest.approx(0.34)


def test_build_env_card_shape():
    env = _build_env_card("llama-3")
    assert isinstance(env, EnvironmentCardMetadata)
    assert env.framework_name == "ifbench"
    assert env.model_id == "llama-3"


def test_evalhub_mode_local_still_reports_to_sidecar(monkeypatch):
    monkeypatch.setenv("EVALHUB_MODE", "local")
    monkeypatch.delenv("IFBENCH_LOCAL_ONLY", raising=False)
    assert _local_only_run() is False


def test_ifbench_local_only_skips_sidecar(monkeypatch):
    monkeypatch.setenv("IFBENCH_LOCAL_ONLY", "1")
    assert _local_only_run() is True


def _inject_fake_ifbench_data(monkeypatch, examples):
    fake_ifbench = types.ModuleType("ifbench")
    fake_ifbench.data_path = lambda: "/tmp/fake-ifbench.jsonl"
    monkeypatch.setitem(sys.modules, "ifbench", fake_ifbench)

    import main as main_mod

    monkeypatch.setattr(main_mod, "_load_test_data", lambda num_examples: examples[:num_examples] if num_examples else examples)


def _inject_fake_openai(monkeypatch):
    if "openai" not in sys.modules:
        fake_openai_mod = types.ModuleType("openai")
        fake_openai_mod.OpenAI = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)


def _inject_call_model(monkeypatch, responses: dict[str, str]):
    import main as main_mod

    def fake_call_model(client, model_name, prompt_text, *, max_tokens, temperature):
        return responses.get(prompt_text, "")

    monkeypatch.setattr(main_mod, "_call_model", fake_call_model)


@pytest.mark.integration
def test_ifbench_happy_path(monkeypatch):
    adapter = IFBenchAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.parameters["num_examples"] = 2
    config.parameters["max_concurrent"] = 1
    config.parameters["scoring_mode"] = "loose"

    _inject_fake_ifbench_data(monkeypatch, CANNED_EXAMPLES)
    _inject_fake_openai(monkeypatch)
    _inject_call_model(
        monkeypatch,
        {
            CANNED_EXAMPLES[0].prompt: "hello zebra",
            CANNED_EXAMPLES[1].prompt: "a nebula poem",
        },
    )

    results = adapter.run_benchmark_job(config, callbacks)

    assert results.id == config.id
    assert results.benchmark_id == "ifbench"
    assert results.duration_seconds > 0

    metric = {r.metric_name: r.metric_value for r in results.results}
    assert metric["n_evaluated"] == 2
    assert metric["accuracy"] == pytest.approx(1.0)
    assert results.overall_score == pytest.approx(1.0)
    assert results.eval_card is not None
    assert results.env_card is not None

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    assert JobPhase.PERSISTING_ARTIFACTS in phases


@pytest.mark.integration
def test_ifbench_api_errors_are_nonfatal(monkeypatch):
    adapter = IFBenchAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.parameters["num_examples"] = 1
    config.parameters["max_concurrent"] = 1

    _inject_fake_ifbench_data(monkeypatch, CANNED_EXAMPLES[:1])
    _inject_fake_openai(monkeypatch)

    import main as main_mod

    def always_raise(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(main_mod, "_call_model", always_raise)

    results = adapter.run_benchmark_job(config, callbacks)
    metric = {r.metric_name: r.metric_value for r in results.results}
    assert metric["accuracy"] == pytest.approx(0.0)
    assert metric["n_evaluated"] == 1

    failed_statuses = [
        c for c in callbacks.report_status.call_args_list if c.args[0].status == JobStatus.FAILED
    ]
    assert len(failed_statuses) == 0


def test_generate_additional_info():
    adapter = IFBenchAdapter(job_spec_path="meta/job.json")
    fake_results = MagicMock()
    fake_results.results = [
        EvaluationResult(metric_name="accuracy", metric_value=0.34, metric_type="float"),
    ]
    fake_results.evaluation_metadata = {"scoring_mode": "loose"}
    info = adapter.generate_additional_info(fake_results)
    assert info is not None
    assert info["zero_shot"] == pytest.approx(0.34)
    assert info["scoring_mode"] == "loose"
