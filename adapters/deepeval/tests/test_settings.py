"""Job parameter validation and runtime tuning."""

import os
from types import SimpleNamespace

import pytest

from deepeval_adapter.settings import (
    DEFAULT_PER_ATTEMPT_TIMEOUT_SECONDS,
    JobSettings,
    apply_runtime_tuning,
    validate_job_spec,
)


def _job(benchmark_id="faithfulness", url="https://judge.example/v1/ ", **params):
    return SimpleNamespace(
        benchmark_id=benchmark_id,
        parameters=params,
        model=SimpleNamespace(url=url, name="target-model"),
    )


def test_validate_requires_a_benchmark_id():
    with pytest.raises(ValueError, match="benchmark_id is required"):
        validate_job_spec(_job(benchmark_id=""))


def test_validate_rejects_unknown_benchmark_and_lists_supported():
    with pytest.raises(ValueError, match=r"Unsupported benchmark_id: 'nope'.*bias"):
        validate_job_spec(_job(benchmark_id="nope"))


@pytest.mark.parametrize(
    "benchmark_id,param",
    [
        ("geval", "criteria"),
        ("conversational-geval", "criteria"),
        ("dag", "dag_criteria_json"),
        ("role-violation", "role"),
    ],
)
def test_validate_requires_the_specs_parameters(benchmark_id, param):
    with pytest.raises(ValueError, match=rf"parameters\.{param} is required"):
        validate_job_spec(_job(benchmark_id=benchmark_id))
    spec = validate_job_spec(_job(benchmark_id=benchmark_id, **{param: "x"}))
    assert spec.id == benchmark_id


def test_settings_defaults_and_judge_fallback():
    settings = JobSettings.from_job_spec(_job())
    assert settings.threshold == 0.5
    assert settings.dataset_format == "csv"
    assert settings.judge_name == "target-model"
    assert settings.judge_url == "https://judge.example/v1"  # stripped


def test_settings_prefer_explicit_judge_and_threshold():
    settings = JobSettings.from_job_spec(
        _job(
            eval_model_name="judge-model",
            eval_model_url="https://other.example/v1",
            threshold="0.8",
            dataset_format="jsonl",
        )
    )
    assert settings.judge_name == "judge-model"
    assert settings.judge_url == "https://other.example/v1"
    assert settings.threshold == 0.8
    assert settings.dataset_format == "jsonl"


@pytest.fixture
def clean_deepeval_env(monkeypatch):
    for name in (
        "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
        "DEEPEVAL_RETRY_MAX_ATTEMPTS",
        "DEEPEVAL_RETRY_CAP_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_runtime_tuning_defaults_only_the_timeout(clean_deepeval_env):
    apply_runtime_tuning({})
    assert os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] == str(
        DEFAULT_PER_ATTEMPT_TIMEOUT_SECONDS
    )
    assert "DEEPEVAL_RETRY_MAX_ATTEMPTS" not in os.environ
    assert "DEEPEVAL_RETRY_CAP_SECONDS" not in os.environ


def test_runtime_tuning_exports_job_parameters(clean_deepeval_env):
    apply_runtime_tuning(
        {
            "per_attempt_timeout_seconds": 60,
            "retry_max_attempts": "4",
            "retry_cap_seconds": 12,
        }
    )
    assert os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] == "60.0"
    assert os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] == "4"
    assert os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] == "12.0"


def test_runtime_tuning_can_unset_the_timeout(clean_deepeval_env):
    apply_runtime_tuning({"per_attempt_timeout_seconds": None})
    assert "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE" not in os.environ
