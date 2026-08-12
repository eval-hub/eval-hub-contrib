"""Integration tests for the lm-eval adapter.

Monkeypatches _run_lmeval and _parse_results so no subprocess or filesystem
access is required. Never mocks the evalhub-sdk layer — tests adapter plumbing.
"""

import json
from pathlib import Path

import pytest
from evalhub.adapter import JobPhase, JobStatus

from main import LMEvalAdapter

# Shared fixtures (adapter, mock_callbacks) and markers are defined in conftest.py.

# Canned results matching lm-eval's results JSON structure.
CANNED_RESULTS = {
    "results": {
        "gsm8k": {
            "exact_match,none": 0.72,
            "exact_match,none,stderr": 0.014,
        }
    },
    "n-samples": {"gsm8k": {"original": 5, "effective": 5}},
    "config": {
        "model": "local-completions",
        "model_args": "base_url=http://localhost:8000/v1/completions,model=test-model",
        "batch_size": 1,
        "limit": 5,
    },
    "versions": {"gsm8k": 3.0},
}

CANNED_MMLU_RESULTS = {
    "results": {
        "mmlu": {
            "acc,none": 0.651,
            "acc,none,stderr": 0.009,
            "acc_norm,none": 0.648,
            "acc_norm,none,stderr": 0.009,
        }
    },
    "n-samples": {"mmlu": {"original": 5, "effective": 5}},
    "versions": {"mmlu": 2.0},
}


# ── Happy path ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_gsm8k_happy_path(adapter, mock_callbacks, monkeypatch):
    """Full run_benchmark_job with mocked subprocess and result parsing."""
    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda output_dir: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0

    # Metrics
    assert len(results.results) > 0
    metric = next((r for r in results.results if "exact_match" in r.metric_name), None)
    assert metric is not None
    assert metric.metric_value == pytest.approx(0.72)
    assert metric.confidence_interval is not None
    assert metric.num_samples == 5

    # Overall score normalised to [0, 1]
    assert results.overall_score == pytest.approx(0.72)
    assert results.num_examples_evaluated == 5

    # EvalCard and EnvironmentCard populated
    assert results.eval_card is not None
    assert results.env_card is not None
    assert results.eval_card.capability_evaluations
    cap = results.eval_card.capability_evaluations[0]
    assert cap.ability == "math"
    assert cap.benchmark == "lm-eval/gsm8k"

    # Lifecycle phases — all 4 required phases emitted
    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    # PERSISTING_ARTIFACTS only when OCI exports are configured
    assert JobPhase.PERSISTING_ARTIFACTS not in phases


@pytest.mark.integration
def test_oci_export_persists_artifacts(tmp_path, mock_callbacks, monkeypatch):
    """When exports.oci is configured, PERSISTING_ARTIFACTS is emitted."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["exports"] = {
        "oci": {
            "coordinates": {
                "oci_host": "quay.io",
                "oci_repository": "test-org/test-repo",
                "oci_tag": "test-tag",
                "annotations": {},
            }
        }
    }
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda output_dir: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS in phases

    # files_path is the temp output_dir, cleaned up in finally after the job
    # completes. Verify it was a Path object with the correct OCI coordinates.
    spec = mock_callbacks.create_oci_artifact.call_args.args[0]
    assert isinstance(spec.files_path, Path)
    assert spec.coordinates.oci_repository == "test-org/test-repo"

    assert results.oci_artifact is not None
    assert results.oci_artifact.digest == "sha256:fake"


# ── Loglikelihood + chat endpoint pre-flight check ────────────────────────────


@pytest.mark.integration
def test_mmlu_with_chat_endpoint_raises(tmp_path, mock_callbacks, monkeypatch):
    """MMLU (loglikelihood) + chat endpoint raises ValueError at startup."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/mmlu"
    job["model"]["url"] = "https://openrouter.ai/api/v1"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match="loglikelihood scoring"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    # FAILED status reported to callbacks
    statuses = [c.args[0].status for c in mock_callbacks.report_status.call_args_list]
    assert JobStatus.FAILED in statuses


@pytest.mark.integration
def test_mmlu_with_explicit_chat_style_raises(tmp_path, mock_callbacks):
    """api_style=chat + loglikelihood benchmark → ValueError."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/hellaswag"
    job["parameters"]["api_style"] = "chat"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match=r"[Cc]hat-only endpoints"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


@pytest.mark.integration
def test_generation_suite_with_chat_endpoint_succeeds(
    tmp_path, mock_callbacks, monkeypatch
):
    """generation-suite (generate_until only) works fine with chat endpoint."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/generation-suite"
    job["model"]["url"] = "https://openrouter.ai/api/v1"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda output_dir: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert results.overall_score is not None


# ── Excluded tasks ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_humaneval_raises_with_helpful_message(tmp_path, mock_callbacks):
    """Requesting HumanEval raises ValueError with the exclusion reason."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/humaneval"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match="sandboxed Python code execution"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


@pytest.mark.integration
def test_mbpp_raises_with_helpful_message(tmp_path, mock_callbacks):
    """Requesting MBPP raises ValueError with the exclusion reason."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/mbpp"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match="code_eval"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


# ── Custom task ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_custom_task_without_task_param_raises(tmp_path, mock_callbacks):
    """lm-eval/custom without a 'task' parameter raises ValueError."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/custom"
    job["parameters"].pop("task", None)
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match="requires a 'task' parameter"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


@pytest.mark.integration
def test_custom_task_with_task_param_runs(tmp_path, mock_callbacks, monkeypatch):
    """lm-eval/custom with 'task=wikitext' executes successfully."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "lm-eval/custom"
    job["parameters"]["task"] = "wikitext"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    captured_cmd = {}

    def fake_run(cmd, env, timeout):
        captured_cmd["cmd"] = cmd

    monkeypatch.setattr(adapter, "_run_lmeval", fake_run)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    # --tasks should contain the user-specified task name
    assert "wikitext" in " ".join(captured_cmd["cmd"])


# ── Missing model URL ─────────────────────────────────────────────────────────


@pytest.mark.integration
def test_missing_model_url_raises(tmp_path, mock_callbacks):
    """SDK 1.0.0 validates model.url at construction time; empty string fails Pydantic."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["model"]["url"] = ""
    (meta_dir / "job.json").write_text(json.dumps(job))

    # SDK raises ValidationError (not ValueError) on load when url is empty.
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cannot be empty"):
        LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))


# ── Tokenizer warning ─────────────────────────────────────────────────────────


@pytest.mark.integration
def test_tokenizer_warning_logged_when_absent(
    adapter, mock_callbacks, monkeypatch, caplog
):
    """A warning is emitted when no tokenizer is configured."""
    import logging

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    with caplog.at_level(logging.WARNING, logger="main"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert any("tokenizer" in r.message.lower() for r in caplog.records)


# ── EvalCard ability routing ───────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.parametrize(
    "benchmark_id,expected_ability",
    [
        ("lm-eval/gsm8k", "math"),
        ("lm-eval/mmlu", "knowledge"),
        ("lm-eval/hellaswag", "reasoning"),
        ("lm-eval/ifeval", "instruction_following"),
        ("lm-eval/open-llm-leaderboard-v2", "composite"),
        ("lm-eval/custom", "custom"),
    ],
)
def test_evalcard_ability_routing(
    tmp_path, mock_callbacks, monkeypatch, benchmark_id, expected_ability
):
    """EvalCard ability matches the benchmark category for every benchmark."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = benchmark_id
    if benchmark_id == "lm-eval/custom":
        job["parameters"]["task"] = "wikitext"
    if benchmark_id in ("lm-eval/open-llm-leaderboard-v2",):
        # composite suite has loglikelihood tasks — use completions style
        job["parameters"]["api_style"] = "completions"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert results.eval_card.capability_evaluations[0].ability == expected_ability


# ── Few-shot EvalCard prompting field ─────────────────────────────────────────


@pytest.mark.integration
def test_fewshot_populates_alt_prompting(adapter, mock_callbacks, monkeypatch):
    """When num_fewshot > 0, EvalCard uses alt_prompting instead of zero_shot."""
    adapter.job_spec.parameters["num_fewshot"] = 5
    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    cap = results.eval_card.capability_evaluations[0]
    assert cap.alt_prompting is not None
    assert cap.alt_prompting_description == "5-Shot"
    assert cap.zero_shot is None


@pytest.mark.integration
def test_zero_shot_populates_zero_shot_field(tmp_path, mock_callbacks, monkeypatch):
    """When num_fewshot is 0, EvalCard uses zero_shot field."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["parameters"]["num_fewshot"] = 0
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    cap = results.eval_card.capability_evaluations[0]
    # num_fewshot=0 → falsy → zero_shot field used
    assert cap.zero_shot is not None
    assert cap.alt_prompting is None


# ── Unit: _tasks.py ────────────────────────────────────────────────────────────

from _tasks import (
    build_endpoint_url,
    detect_api_style,
    preflight_check,
    resolve_tasks,
)


def test_resolve_known_generation_task():
    assert resolve_tasks("lm-eval/gsm8k", None) == "gsm8k"


def test_resolve_known_loglikelihood_task():
    assert resolve_tasks("lm-eval/mmlu", None) == "mmlu"


def test_resolve_composite_suite():
    tasks = resolve_tasks("lm-eval/generation-suite", None)
    assert "gsm8k" in tasks
    assert "ifeval" in tasks


def test_resolve_custom_with_override():
    assert resolve_tasks("lm-eval/custom", "wikitext") == "wikitext"


def test_resolve_custom_without_override_raises():
    with pytest.raises(ValueError, match="requires a 'task' parameter"):
        resolve_tasks("lm-eval/custom", None)


def test_resolve_excluded_task_raises():
    with pytest.raises(ValueError, match="sandboxed Python code execution"):
        resolve_tasks("lm-eval/humaneval", None)


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        resolve_tasks("lm-eval/nonexistent", None)


def test_detect_api_style_auto_defaults_completions():
    assert detect_api_style({}, "http://vllm:8000") == "completions"


def test_detect_api_style_openrouter_auto():
    assert detect_api_style({}, "https://openrouter.ai/api/v1") == "chat"


def test_detect_api_style_explicit_override():
    assert detect_api_style({"api_style": "chat"}, "http://vllm:8000") == "chat"
    assert (
        detect_api_style({"api_style": "completions"}, "https://openrouter.ai/api/v1")
        == "completions"
    )


def test_preflight_check_passes_generation_with_chat():
    # Should not raise
    preflight_check("lm-eval/gsm8k", "chat")
    preflight_check("lm-eval/generation-suite", "chat")


def test_preflight_check_raises_loglikelihood_with_chat():
    with pytest.raises(ValueError, match="loglikelihood"):
        preflight_check("lm-eval/mmlu", "chat")


def test_preflight_check_passes_loglikelihood_with_completions():
    preflight_check("lm-eval/mmlu", "completions")


def test_build_endpoint_url_appends_completions():
    assert (
        build_endpoint_url("http://vllm:8000", "completions")
        == "http://vllm:8000/v1/completions"
    )


def test_build_endpoint_url_appends_chat():
    assert (
        build_endpoint_url("http://vllm:8000", "chat")
        == "http://vllm:8000/v1/chat/completions"
    )


def test_build_endpoint_url_with_v1_already():
    assert (
        build_endpoint_url("http://vllm:8000/v1", "completions")
        == "http://vllm:8000/v1/completions"
    )


# ── Unit: _results.py ─────────────────────────────────────────────────────────

from _results import compute_overall_score, extract_evaluation_results


def test_extract_results_basic():
    results = extract_evaluation_results(CANNED_RESULTS)
    assert len(results) == 1
    r = results[0]
    assert r.metric_name == "gsm8k.exact_match"
    assert r.metric_value == pytest.approx(0.72)
    assert r.confidence_interval is not None
    assert r.num_samples == 5


def test_extract_results_multiple_metrics():
    results = extract_evaluation_results(CANNED_MMLU_RESULTS)
    metric_names = {r.metric_name for r in results}
    assert "mmlu.acc" in metric_names
    assert "mmlu.acc_norm" in metric_names
    # stderr entries should NOT appear as standalone results
    assert not any("stderr" in n for n in metric_names)


def test_extract_results_strips_fewshot_suffix():
    raw = {
        "results": {
            "gsm8k|5": {
                "exact_match,none": 0.80,
            }
        },
        "n-samples": {"gsm8k|5": {"original": 10, "effective": 10}},
    }
    results = extract_evaluation_results(raw)
    assert results[0].metric_name == "gsm8k.exact_match"


def test_compute_overall_score_normalises():
    from evalhub.adapter import EvaluationResult

    results = [
        EvaluationResult(
            metric_name="gsm8k.exact_match",
            metric_value=0.72,
            metric_type="float",
            num_samples=5,
        )
    ]
    score = compute_overall_score(results)
    assert score == pytest.approx(0.72)


def test_compute_overall_score_percent_normalised():
    from evalhub.adapter import EvaluationResult

    results = [
        EvaluationResult(
            metric_name="task.acc",
            metric_value=72.5,
            metric_type="float",
            num_samples=100,
        )
    ]
    score = compute_overall_score(results)
    assert score == pytest.approx(0.725)


def test_compute_overall_score_empty_returns_none():
    assert compute_overall_score([]) is None
