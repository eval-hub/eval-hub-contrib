"""Integration tests for the DeepEval adapter.

Verifies adapter plumbing by monkeypatching deepeval.evaluate() and
the data-loading layer so no real API calls or test data files are needed.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest

from evalhub.adapter import JobCallbacks, JobPhase
from main import (
    DeepEvalAdapter,
    _build_conversational_test_cases,
    _build_test_cases,
    _load_dataset,
)


def _make_canned_eval_results(score=0.85, name="Faithfulness", reason="All claims supported"):
    """Build a fake object matching deepeval.evaluate()'s return shape."""
    metric_data = SimpleNamespace(score=score, success=score >= 0.5, reason=reason, name=name)
    test_result = SimpleNamespace(metrics_data=[metric_data])
    return SimpleNamespace(test_results=[test_result])


# (benchmark_id, file_content, dataset_format, expected_metrics)
BENCHMARK_CASES = [
    pytest.param(
        "faithfulness",
        "input,actual_output,retrieval_context\nq,a,ctx\n",
        "csv",
        ["faithfulness_score", "claims_count", "supported_claims_count"],
        id="faithfulness",
    ),
    pytest.param(
        "hallucination",
        "input,actual_output,context\nq,a,ctx\n",
        "csv",
        ["hallucination_score", "hallucination_detected"],
        id="hallucination",
    ),
    pytest.param(
        "correctness",
        "input,actual_output,expected_output\nq,a,expected\n",
        "csv",
        ["correctness_score"],
        id="correctness",
    ),
    pytest.param(
        "relevancy",
        "input,actual_output\nq,a\n",
        "csv",
        ["relevancy_score"],
        id="relevancy",
    ),
    pytest.param(
        "summarization",
        "input,actual_output\nq,a\n",
        "csv",
        ["summarization_score"],
        id="summarization",
    ),
    # Multi-turn benchmarks — JSONL format
    pytest.param(
        "conversation-completeness",
        json.dumps({
            "turns": [
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": "Click on 'Forgot password' on the login page."},
            ]
        }),
        "jsonl",
        ["conversation_completeness_score"],
        id="conversation-completeness",
    ),
    pytest.param(
        "role-adherence",
        json.dumps({
            "turns": [
                {"role": "user", "content": "Hello, I need help."},
                {"role": "assistant", "content": "Hi! I'm here to help you."},
            ],
            "chatbot_role": "friendly customer support agent",
        }),
        "jsonl",
        ["role_adherence_score"],
        id="role-adherence",
    ),
    pytest.param(
        "knowledge-retention",
        json.dumps({
            "turns": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What's my name?"},
                {"role": "assistant", "content": "Your name is Alice."},
            ]
        }),
        "jsonl",
        ["knowledge_retention_score"],
        id="knowledge-retention",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id,file_content,dataset_format,expected_metrics", BENCHMARK_CASES)
def test_deepeval_happy_path(tmp_path, monkeypatch, benchmark_id, file_content, dataset_format, expected_metrics):
    """Full run_benchmark_job with mocked evaluate() and canned data."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = benchmark_id
    job["parameters"]["dataset_format"] = dataset_format
    (meta_dir / "job.json").write_text(json.dumps(job))

    adapter = DeepEvalAdapter(job_spec_path=str(meta_dir / "job.json"))
    callbacks = create_autospec(JobCallbacks)

    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    if dataset_format == "csv":
        (data_dir / "data.csv").write_text(file_content)
    else:
        (data_dir / "data.jsonl").write_text(file_content)

    monkeypatch.setattr("main._resolve_data_dir", lambda _config: str(data_dir))
    monkeypatch.setattr("main._resolve_judge_model", lambda _name, _url: SimpleNamespace(name="MockModel"))
    monkeypatch.setattr("main._create_metric", lambda _bid, _model, _threshold: SimpleNamespace(name="MockMetric"))

    canned = _make_canned_eval_results()
    monkeypatch.setattr("main.evaluate", lambda **kwargs: canned)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == benchmark_id
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0
    assert results.num_examples_evaluated == 1

    # Expected aggregate metrics present
    metric_names = [r.metric_name for r in results.results]
    for expected in expected_metrics:
        assert expected in metric_names, f"Missing metric {expected} for {benchmark_id}"

    # Overall score
    assert results.overall_score is not None
    assert results.overall_score == pytest.approx(0.85, abs=0.01)

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    # PERSISTING_ARTIFACTS only emitted when OCI exports are configured


@pytest.mark.integration
def test_validate_config_rejects_unknown_benchmark(tmp_path):
    """_validate_config raises ValueError for an unknown benchmark_id."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")

    adapter = DeepEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    config = MagicMock()
    config.benchmark_id = "deepeval-nonexistent"
    config.parameters = {"eval_model_name": "gpt-4o"}

    with pytest.raises(ValueError, match="Unsupported benchmark_id"):
        adapter._validate_config(config)


@pytest.mark.integration
def test_build_test_cases_skips_incomplete_records():
    """Records missing required columns are skipped, not errored."""
    records = [
        {"input": "q1", "actual_output": "a1", "retrieval_context": "ctx1"},
        {"input": "q2"},  # missing actual_output and retrieval_context
        {"input": "q3", "actual_output": "a3", "retrieval_context": "ctx3"},
    ]
    cases = _build_test_cases(records, "faithfulness")
    assert len(cases) == 2


@pytest.mark.integration
def test_build_test_cases_raises_on_all_invalid():
    """If every record is incomplete, raise ValueError."""
    records = [
        {"input": "q1"},  # missing actual_output and retrieval_context
    ]
    with pytest.raises(ValueError, match="No valid test cases"):
        _build_test_cases(records, "faithfulness")


@pytest.mark.integration
def test_load_dataset_csv(tmp_path):
    """_load_dataset reads CSV files correctly."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("input,actual_output\nq1,a1\nq2,a2\n")
    records = _load_dataset(str(tmp_path), "csv")
    assert len(records) == 2
    assert records[0]["input"] == "q1"


@pytest.mark.integration
def test_load_dataset_jsonl(tmp_path):
    """_load_dataset reads JSONL files correctly."""
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(
        json.dumps({"input": "q1", "actual_output": "a1"}) + "\n"
        + json.dumps({"input": "q2", "actual_output": "a2"}) + "\n"
    )
    records = _load_dataset(str(tmp_path), "jsonl")
    assert len(records) == 2


@pytest.mark.integration
def test_load_dataset_unsupported_format(tmp_path):
    """_load_dataset raises ValueError for unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported dataset_format"):
        _load_dataset(str(tmp_path), "parquet")


# --- Multi-turn unit tests ---

_SAMPLE_TURNS = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


@pytest.mark.integration
def test_build_conversational_test_cases_basic():
    """Valid JSONL records produce ConversationalTestCase objects."""
    records = [{"turns": _SAMPLE_TURNS}]
    cases = _build_conversational_test_cases(records, "conversation-completeness")
    assert len(cases) == 1
    assert len(cases[0].turns) == 2


@pytest.mark.integration
def test_build_conversational_test_cases_json_string_turns():
    """CSV-style JSON-encoded turns string is parsed correctly."""
    records = [{"turns": json.dumps(_SAMPLE_TURNS)}]
    cases = _build_conversational_test_cases(records, "conversation-completeness")
    assert len(cases) == 1
    assert len(cases[0].turns) == 2


@pytest.mark.integration
def test_build_conversational_test_cases_optional_fields():
    """chatbot_role and scenario are forwarded to the test case."""
    records = [{
        "turns": _SAMPLE_TURNS,
        "chatbot_role": "support agent",
        "scenario": "password reset",
    }]
    cases = _build_conversational_test_cases(records, "conversation-completeness")
    assert cases[0].chatbot_role == "support agent"
    assert cases[0].scenario == "password reset"


@pytest.mark.integration
def test_build_conversational_test_cases_skips_missing_turns():
    """Records without the turns column are skipped with a warning."""
    records = [
        {"turns": _SAMPLE_TURNS},
        {"chatbot_role": "agent"},  # missing turns
        {"turns": _SAMPLE_TURNS},
    ]
    cases = _build_conversational_test_cases(records, "conversation-completeness")
    assert len(cases) == 2


@pytest.mark.integration
def test_build_conversational_test_cases_role_adherence_requires_chatbot_role():
    """role-adherence benchmark skips records missing chatbot_role."""
    records = [
        {"turns": _SAMPLE_TURNS},  # missing chatbot_role
    ]
    with pytest.raises(ValueError, match="No valid conversational test cases"):
        _build_conversational_test_cases(records, "role-adherence")


@pytest.mark.integration
def test_build_conversational_test_cases_role_adherence_valid():
    """role-adherence succeeds when chatbot_role is present."""
    records = [{"turns": _SAMPLE_TURNS, "chatbot_role": "assistant"}]
    cases = _build_conversational_test_cases(records, "role-adherence")
    assert len(cases) == 1
    assert cases[0].chatbot_role == "assistant"


@pytest.mark.integration
def test_build_conversational_test_cases_raises_on_all_invalid():
    """If every record is invalid, raise ValueError."""
    records = [{"chatbot_role": "agent"}]  # no turns
    with pytest.raises(ValueError, match="No valid conversational test cases"):
        _build_conversational_test_cases(records, "knowledge-retention")
