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

    with pytest.raises(ValueError, match="allow_code_execution"):
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
    """An empty model.url (valid in SDK >= 1.0.5) fails at job start with a clear error."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["model"]["url"] = ""
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    with pytest.raises(ValueError, match="'model.url' is required"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


@pytest.mark.integration
def test_base_url_param_used_when_model_url_empty(tmp_path, mock_callbacks, monkeypatch):
    """params['base_url'] is accepted as a fallback when model.url is empty."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["model"]["url"] = ""
    job["parameters"]["base_url"] = "http://localhost:8000"
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    seen = {}

    def fake_run(cmd, env, timeout):
        seen["cmd"] = cmd
        raise RuntimeError("stop after command build")

    monkeypatch.setattr(adapter, "_run_lmeval", fake_run)
    with pytest.raises(RuntimeError, match="stop after command build"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert any("http://localhost:8000/v1/completions" in str(a) for a in seen["cmd"])


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
    # loglikelihood benchmarks need a tokenizer (lm-eval asserts without one)
    job["parameters"]["tokenizer"] = "org/model"
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
    """GSM8K maps to the 'gsm8k' lm-eval task name."""
    assert resolve_tasks("lm-eval/gsm8k", None) == "gsm8k"


def test_resolve_known_loglikelihood_task():
    """MMLU maps to the 'mmlu' lm-eval task name."""
    assert resolve_tasks("lm-eval/mmlu", None) == "mmlu"


def test_resolve_composite_suite():
    """generation-suite expands to a comma-separated list that includes gsm8k and ifeval."""
    tasks = resolve_tasks("lm-eval/generation-suite", None)
    assert "gsm8k" in tasks
    assert "ifeval" in tasks


def test_resolve_custom_with_override():
    """lm-eval/custom with a task_override passes through the override unchanged."""
    assert resolve_tasks("lm-eval/custom", "wikitext") == "wikitext"


def test_resolve_custom_without_override_raises():
    """lm-eval/custom without a task_override raises ValueError."""
    with pytest.raises(ValueError, match="requires a 'task' parameter"):
        resolve_tasks("lm-eval/custom", None)


def test_resolve_excluded_task_raises():
    """HumanEval is refused unless allow_code_execution is set."""
    with pytest.raises(ValueError, match="sandboxed Python code execution"):
        resolve_tasks("lm-eval/humaneval", None)
    with pytest.raises(ValueError, match="allow_code_execution"):
        resolve_tasks("humaneval", None)


def test_resolve_unknown_raises():
    """An unregistered benchmark ID raises ValueError."""
    with pytest.raises(ValueError, match="Unknown benchmark"):
        resolve_tasks("lm-eval/nonexistent", None)


def test_detect_api_style_auto_defaults_completions():
    """Auto-detection defaults to 'completions' for a plain vLLM host URL."""
    assert detect_api_style({}, "http://vllm:8000") == "completions"


def test_detect_api_style_openrouter_auto():
    """Auto-detection returns 'chat' when the URL contains 'openrouter.ai'."""
    assert detect_api_style({}, "https://openrouter.ai/api/v1") == "chat"


def test_detect_api_style_explicit_override():
    """An explicit api_style parameter overrides URL-based auto-detection."""
    assert detect_api_style({"api_style": "chat"}, "http://vllm:8000") == "chat"
    assert (
        detect_api_style({"api_style": "completions"}, "https://openrouter.ai/api/v1")
        == "completions"
    )


def test_preflight_check_passes_generation_with_chat():
    """Generation-safe benchmarks pass the preflight check with a chat endpoint."""
    preflight_check("lm-eval/gsm8k", "chat")
    preflight_check("lm-eval/generation-suite", "chat")


def test_preflight_check_raises_loglikelihood_with_chat():
    """Loglikelihood benchmarks paired with a chat endpoint raise ValueError."""
    with pytest.raises(ValueError, match="loglikelihood"):
        preflight_check("lm-eval/mmlu", "chat")


def test_preflight_check_passes_loglikelihood_with_completions():
    """Loglikelihood benchmarks paired with a completions endpoint pass the preflight check."""
    preflight_check("lm-eval/mmlu", "completions")


def test_build_endpoint_url_appends_completions():
    """A bare host URL gets /v1/completions appended for the completions API style."""
    assert (
        build_endpoint_url("http://vllm:8000", "completions")
        == "http://vllm:8000/v1/completions"
    )


def test_build_endpoint_url_appends_chat():
    """A bare host URL gets /v1/chat/completions appended for the chat API style."""
    assert (
        build_endpoint_url("http://vllm:8000", "chat")
        == "http://vllm:8000/v1/chat/completions"
    )


def test_build_endpoint_url_with_v1_already():
    """A URL that already ends with /v1 does not get a duplicate /v1 segment."""
    assert (
        build_endpoint_url("http://vllm:8000/v1", "completions")
        == "http://vllm:8000/v1/completions"
    )


# ── Unit: _results.py ─────────────────────────────────────────────────────────

from _results import compute_overall_score, extract_evaluation_results


def test_extract_results_basic():
    """Single-metric result is correctly parsed, including confidence interval and sample count."""
    results = extract_evaluation_results(CANNED_RESULTS)
    assert len(results) == 1
    r = results[0]
    assert r.metric_name == "gsm8k.exact_match"
    assert r.metric_value == pytest.approx(0.72)
    assert r.confidence_interval is not None
    assert r.num_samples == 5


def test_extract_results_multiple_metrics():
    """Multiple metrics per task are all extracted; stderr entries are excluded."""
    results = extract_evaluation_results(CANNED_MMLU_RESULTS)
    metric_names = {r.metric_name for r in results}
    assert "mmlu.acc" in metric_names
    assert "mmlu.acc_norm" in metric_names
    # stderr entries should NOT appear as standalone results
    assert not any("stderr" in n for n in metric_names)


def test_extract_results_strips_fewshot_suffix():
    """Task names with a '|N' few-shot suffix are normalised by stripping the suffix."""
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
    """A score already in [0, 1] is returned unchanged."""
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
    """A score expressed as a percentage (> 1) is divided by 100 before returning."""
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
    """An empty result list returns None rather than raising."""
    assert compute_overall_score([]) is None


# ── LMEvalJob parity ──────────────────────────────────────────────────────────

import yaml  # noqa: E402

from _execution import build_cmd, build_env  # noqa: E402
from _tasks import is_code_execution, requires_completions_endpoint  # noqa: E402

_TESTS_DIR = Path(__file__).resolve().parent
_ADAPTER_DIR = _TESTS_DIR.parent
# Snapshot of benchmark IDs in the lm_evaluation_harness (LMEvalJob) provider: the
# RHOAI 3.5 live list (188) unioned with eval-hub server main (167).
_LMEVALJOB_IDS = (_TESTS_DIR / "lmevaljob_benchmark_ids.txt").read_text().split()


def test_lmevaljob_snapshot_has_expected_size():
    assert len(_LMEVALJOB_IDS) == 193
    assert len(set(_LMEVALJOB_IDS)) == 193


def test_provider_yaml_exposes_every_lmevaljob_benchmark():
    """provider.yaml is a superset of the LMEvalJob provider's benchmark IDs."""
    provider = yaml.safe_load((_ADAPTER_DIR / "provider.yaml").read_text())
    exposed = [b["id"] for b in provider["benchmarks"]]
    assert len(exposed) == len(set(exposed)), "duplicate benchmark IDs"
    assert set(_LMEVALJOB_IDS) <= set(exposed)


@pytest.mark.parametrize("benchmark_id", _LMEVALJOB_IDS)
def test_every_lmevaljob_benchmark_resolves(benchmark_id):
    """Each LMEvalJob ID resolves to its lm-eval task (code-exec ones need opt-in)."""
    task = resolve_tasks(benchmark_id, None, allow_code_execution=True)
    assert task == benchmark_id
    assert resolve_tasks(f"lm-eval/{benchmark_id}", None, allow_code_execution=True)


def test_every_provider_benchmark_resolves():
    """No benchmark listed in provider.yaml is rejected as unknown."""
    provider = yaml.safe_load((_ADAPTER_DIR / "provider.yaml").read_text())
    for b in provider["benchmarks"]:
        if b["id"] == "lm-eval/custom":
            continue
        assert resolve_tasks(b["id"], None, allow_code_execution=True)


def test_api_style_classification():
    assert requires_completions_endpoint("arc_easy")
    assert requires_completions_endpoint("lm-eval/arc_easy")
    assert requires_completions_endpoint("wikitext")  # loglikelihood_rolling
    assert requires_completions_endpoint("lambada_openai")
    assert not requires_completions_endpoint("gsm8k")
    # bbh / mmlu_pro are generate_until in lm-eval 0.4.x
    assert not requires_completions_endpoint("lm-eval/bbh")
    assert not requires_completions_endpoint("bbh")
    assert not requires_completions_endpoint("lm-eval/mmlu_pro")


def test_loglikelihood_bare_id_rejected_on_chat_endpoint():
    with pytest.raises(ValueError, match="loglikelihood"):
        preflight_check("hellaswag_ar", "chat")
    preflight_check("bbh", "chat")  # generation: fine


def test_code_execution_ids():
    for b in ("humaneval", "humaneval_instruct", "mbpp", "lm-eval/mbpp"):
        assert is_code_execution(b)
    assert not is_code_execution("gsm8k")
    assert resolve_tasks("lm-eval/humaneval", None, allow_code_execution=True) == "humaneval"


def _cmd(**kw):
    base = dict(
        model_type="local-completions", model_args="m", tasks="t", output_dir=Path("/o"),
        batch_size=1, num_fewshot=None, limit=2, gen_kwargs=None, include_path=None,
        apply_chat_template=False, system_instruction=None,
    )
    return build_cmd(**{**base, **kw})


def test_build_cmd_safety_flags_off_by_default():
    cmd = _cmd()
    assert "--trust_remote_code" not in cmd
    assert "--confirm_run_unsafe_code" not in cmd


def test_build_cmd_safety_flags_opt_in():
    cmd = _cmd(trust_remote_code=True, confirm_run_unsafe_code=True)
    assert "--trust_remote_code" in cmd
    assert "--confirm_run_unsafe_code" in cmd


def test_build_env_code_eval_opt_in():
    kw = dict(api_key="k", hf_datasets_cache=None, dataset_source="hub",
              s3_endpoint=None, aws_access_key_id=None, aws_secret_access_key=None)
    assert build_env(**kw, allow_code_execution=True)["HF_ALLOW_CODE_EVAL"] == "1"


@pytest.mark.integration
def test_code_execution_job_passes_flags(tmp_path, mock_callbacks, monkeypatch):
    """allow_code_execution=true runs humaneval with the unsafe-code flag and env."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["benchmark_id"] = "humaneval"
    job["parameters"]["allow_code_execution"] = True
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    seen = {}

    def fake_run(cmd, env, timeout):
        seen["cmd"], seen["env"] = cmd, env
        raise RuntimeError("stop")

    monkeypatch.setattr(adapter, "_run_lmeval", fake_run)
    with pytest.raises(RuntimeError, match="stop"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert "--confirm_run_unsafe_code" in seen["cmd"]
    assert seen["env"]["HF_ALLOW_CODE_EVAL"] == "1"


# ── Result naming: filters, bare primary names, effective few-shot ────────────

from _results import effective_num_fewshot  # noqa: E402

CANNED_GSM8K_FILTERS = {
    "results": {
        "gsm8k": {
            "exact_match,strict-match": 0.72,
            "exact_match_stderr,strict-match": 0.014,
            "exact_match,flexible-extract": 0.75,
            "exact_match_stderr,flexible-extract": 0.013,
        }
    },
    "n-samples": {"gsm8k": {"original": 5, "effective": 5}},
    "n-shot": {"gsm8k": 5},
}


def test_filters_keep_metric_names_distinct():
    """strict-match and flexible-extract no longer collapse into gsm8k.exact_match."""
    results = extract_evaluation_results(CANNED_GSM8K_FILTERS)
    by_name = {r.metric_name: r for r in results}

    assert set(by_name) == {
        "gsm8k.exact_match,strict-match",
        "gsm8k.exact_match,flexible-extract",
    }
    assert by_name["gsm8k.exact_match,strict-match"].metric_value == pytest.approx(0.72)
    assert by_name["gsm8k.exact_match,flexible-extract"].metric_value == pytest.approx(0.75)
    assert by_name["gsm8k.exact_match,strict-match"].confidence_interval is not None


def test_unfiltered_names_are_unchanged():
    """The 'none' filter keeps the <task>.<metric> form existing consumers use."""
    names = {r.metric_name for r in extract_evaluation_results(CANNED_MMLU_RESULTS)}
    assert names == {"mmlu.acc", "mmlu.acc_norm"}


def test_primary_task_is_also_reported_under_bare_metric_names():
    """primary_score.metric declares bare names (acc_norm); they must be emitted."""
    raw = {
        "results": {"arc_easy": {"acc,none": 0.8, "acc_norm,none": 0.77}},
        "n-samples": {"arc_easy": {"original": 10, "effective": 10}},
    }
    results = extract_evaluation_results(raw, primary_task="arc_easy")
    by_name = {r.metric_name: r for r in results}

    assert by_name["acc_norm"].metric_value == pytest.approx(0.77)
    assert by_name["acc"].metric_value == pytest.approx(0.8)
    assert by_name["arc_easy.acc_norm"].metric_value == pytest.approx(0.77)
    assert by_name["acc_norm"].metadata["alias_of"] == "arc_easy.acc_norm"


def test_bare_alias_prefers_unfiltered_then_first_filter():
    primary = extract_evaluation_results(CANNED_GSM8K_FILTERS, primary_task="gsm8k")
    bare = {r.metric_name: r for r in primary if "." not in r.metric_name}
    # Only filtered values exist: the first one lm-eval reported (strict-match).
    assert bare["exact_match"].metric_value == pytest.approx(0.72)
    assert bare["exact_match"].metadata["filter"] == "strict-match"

    mixed = {
        "results": {
            "t": {
                "exact_match,flexible-extract": 0.5,
                "exact_match,none": 0.9,
            }
        }
    }
    bare = {
        r.metric_name: r
        for r in extract_evaluation_results(mixed, primary_task="t")
        if "." not in r.metric_name
    }
    assert bare["exact_match"].metric_value == pytest.approx(0.9)


def test_bare_alias_only_for_the_primary_task():
    raw = {
        "results": {
            "mmlu": {"acc,none": 0.65},
            "mmlu_anatomy": {"acc,none": 0.5},
        }
    }
    results = extract_evaluation_results(raw, primary_task="mmlu")
    bare = [r for r in results if "." not in r.metric_name]
    assert [(r.metric_name, r.metric_value) for r in bare] == [("acc", 0.65)]
    assert {r.metric_name for r in results} >= {"mmlu.acc", "mmlu_anatomy.acc"}


def test_no_bare_alias_without_a_primary_task():
    assert all("." in r.metric_name for r in extract_evaluation_results(CANNED_RESULTS))


def test_bare_alias_does_not_inflate_sample_totals():
    """Aliases carry no num_samples, so the per-task total stays 5."""
    results = extract_evaluation_results(CANNED_RESULTS, primary_task="gsm8k")
    assert len(results) == 2
    assert LMEvalAdapter._total_samples(results) == 5


@pytest.mark.parametrize(
    ("raw", "requested", "expected"),
    [
        ({"n-shot": {"gsm8k": 5}}, None, 5),
        ({"n-shot": {"gsm8k": 5}}, 0, 0),  # an explicit value overrides the tasks
        ({"n-shot": {"arc_easy": 0}}, None, 0),
        ({"n-shot": {"a": 0, "b": 3}}, None, None),  # tasks disagree
        ({}, None, None),  # not reported
        ({"n-shot": {"a": 5, "b": 5, "c": 5}}, None, 5),
    ],
)
def test_effective_num_fewshot(raw, requested, expected):
    assert effective_num_fewshot(raw, requested) == expected


@pytest.mark.integration
def test_omitted_num_fewshot_is_not_labelled_zero_shot(tmp_path, mock_callbacks, monkeypatch):
    """GSM8K defaults to 5-shot in the harness; the card must not claim zero-shot."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["parameters"].pop("num_fewshot")
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_GSM8K_FILTERS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    cap = results.eval_card.capability_evaluations[0]
    assert cap.zero_shot is None
    assert cap.alt_prompting is not None
    assert cap.alt_prompting_description == "5-Shot"
    assert results.evaluation_metadata["num_fewshot"] is None
    assert results.evaluation_metadata["num_fewshot_effective"] == 5


@pytest.mark.integration
def test_run_reports_bare_primary_metric(adapter, mock_callbacks, monkeypatch):
    """A single-task benchmark reports the bare name its primary_score declares."""
    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    names = {r.metric_name for r in results.results}
    assert {"exact_match", "gsm8k.exact_match"} <= names
    assert results.num_examples_evaluated == 5


# ── Unsafe-code confirmation ──────────────────────────────────────────────────


def _unsafe_code_run(tmp_path, mock_callbacks, monkeypatch, **job_overrides):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    params = job_overrides.pop("parameters", {})
    job.update(job_overrides)
    job["parameters"].update(params)
    (meta_dir / "job.json").write_text(json.dumps(job))
    adapter = LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    seen = {}

    def fake_run(cmd, env, timeout):
        seen["cmd"], seen["env"] = cmd, env
        raise RuntimeError("stop")

    monkeypatch.setattr(adapter, "_run_lmeval", fake_run)
    with pytest.raises(RuntimeError, match="stop"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    return seen


@pytest.mark.integration
def test_cruxeval_output_forwards_unsafe_code_confirmation(tmp_path, mock_callbacks, monkeypatch):
    """lm-eval marks cruxeval_output unsafe; without the flag the harness refuses it."""
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        benchmark_id="cruxeval_output",
        parameters={"allow_code_execution": True},
    )
    assert "--confirm_run_unsafe_code" in seen["cmd"]
    assert seen["env"]["HF_ALLOW_CODE_EVAL"] == "1"


@pytest.mark.integration
def test_cruxeval_output_is_refused_without_opt_in(tmp_path, mock_callbacks, monkeypatch):
    with pytest.raises(ValueError, match="allow_code_execution"):
        _unsafe_code_run(
            tmp_path, mock_callbacks, monkeypatch, benchmark_id="cruxeval_output"
        )


@pytest.mark.integration
def test_custom_task_list_with_unsafe_task_forwards_confirmation(
    tmp_path, mock_callbacks, monkeypatch
):
    """lm-eval/custom carries its tasks in `task`; the confirmation must follow them."""
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        benchmark_id="lm-eval/custom",
        parameters={"task": "gsm8k,humaneval", "allow_code_execution": True},
    )
    assert "--confirm_run_unsafe_code" in seen["cmd"]


@pytest.mark.integration
def test_safe_tasks_do_not_get_the_confirmation_flag(tmp_path, mock_callbacks, monkeypatch):
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        parameters={"allow_code_execution": True},
    )
    assert "--confirm_run_unsafe_code" not in seen["cmd"]


# ── Generated benchmark table vs the pinned lm_eval ───────────────────────────
#
# These tests need the real lm_eval (requirements.txt); the CI "registry" job
# installs it and sets LMEVAL_REQUIRED=1 so they fail instead of skipping.

import subprocess  # noqa: E402
import sys  # noqa: E402

from _benchmarks import BENCHMARKS, TASKS  # noqa: E402
from _tasks import COMPOSITE_SUITES  # noqa: E402

_PROVIDER = yaml.safe_load((_ADAPTER_DIR / "provider.yaml").read_text())
_MANIFEST = {b["id"]: b for b in _PROVIDER["benchmarks"]}
_NON_TASK_IDS = {"lm-eval/custom", *COMPOSITE_SUITES}


@pytest.mark.lmeval
def test_benchmarks_table_is_current(lmeval):
    """_benchmarks.py is exactly what scripts/generate_benchmarks.py produces."""
    out = subprocess.run(
        [sys.executable, "scripts/generate_benchmarks.py", "--check"],
        cwd=_ADAPTER_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr


@pytest.mark.lmeval
def test_table_matches_pinned_lmeval_version(lmeval):
    import _benchmarks
    from importlib.metadata import version

    assert _benchmarks.LMEVAL_VERSION == version("lm_eval")


def test_provider_and_table_list_the_same_benchmarks():
    """Every manifest ID is classified, and the table adds nothing the manifest hides."""
    assert set(_MANIFEST) - _NON_TASK_IDS == set(BENCHMARKS)


def test_composite_members_are_classified():
    for suite, members in COMPOSITE_SUITES.items():
        assert members and all(m in TASKS for m in members), suite
    assert "hendrycks_math500" in COMPOSITE_SUITES["lm-eval/generation-suite"]
    assert "math_500" not in COMPOSITE_SUITES["lm-eval/generation-suite"]


def test_suite_needs_completions_iff_a_member_does():
    assert requires_completions_endpoint("lm-eval/open-llm-leaderboard-v2")
    assert not requires_completions_endpoint("lm-eval/generation-suite")


def test_unsafe_code_tasks_are_flagged():
    """The tasks lm-eval refuses without --confirm_run_unsafe_code."""
    unsafe = {t for t, s in TASKS.items() if s.unsafe_code}
    assert {"humaneval", "humaneval_instruct", "mbpp", "cruxeval_output"} <= unsafe
    assert not any(TASKS[t].unsafe_code for t in ("gsm8k", "arc_easy", "ifeval"))


@pytest.mark.lmeval
def test_registered_flag_matches_lmeval_index(lmeval):
    from lm_eval.tasks import TaskManager

    index = TaskManager().task_index
    for task, spec in TASKS.items():
        assert spec.registered == (task in index), task


# ── Declared primary scores vs the metrics lm-eval emits ─────────────────────

_SCORED = [i for i in _MANIFEST if i not in _NON_TASK_IDS]


@pytest.mark.parametrize("benchmark_id", _SCORED)
def test_every_benchmark_declares_a_primary_score(benchmark_id):
    """Without primary_score and pass_criteria the server returns no verdict."""
    entry = _MANIFEST[benchmark_id]
    assert entry.get("primary_score", {}).get("metric"), benchmark_id
    assert "threshold" in entry.get("pass_criteria", {}), benchmark_id


@pytest.mark.parametrize("benchmark_id", _SCORED)
def test_declared_primary_metric_is_emitted(benchmark_id):
    """primary_score.metric must be a metric lm-eval writes for the benchmark's task.

    The server matches it exactly and drops the whole test section otherwise.
    Tasks whose metric names come from Python code (humaneval, code2text, ...)
    cannot be read statically and are skipped here.
    """
    spec = TASKS[BENCHMARKS[benchmark_id]]
    if spec.opaque or not spec.registered:
        pytest.skip("metric names are not statically known")
    metric = _MANIFEST[benchmark_id]["primary_score"]["metric"]
    assert metric in spec.emits, f"{benchmark_id} declares {metric!r}, emits {spec.emits}"


@pytest.mark.parametrize("benchmark_id", _SCORED)
def test_declared_direction_matches_lmeval(benchmark_id):
    """lower_is_better must agree with the metric's own higher_is_better in lm-eval."""
    spec = TASKS[BENCHMARKS[benchmark_id]]
    primary = _MANIFEST[benchmark_id]["primary_score"]
    if primary["metric"] not in spec.emits:
        pytest.skip("metric not statically known")
    declared = bool(primary.get("lower_is_better", False))
    assert declared == (primary["metric"] in spec.lower), benchmark_id


@pytest.mark.parametrize("benchmark_id", _SCORED)
def test_listed_metrics_are_emitted_results(benchmark_id):
    """provider.yaml metrics lists no stderr key: the adapter folds it into the CI."""
    metrics = _MANIFEST[benchmark_id].get("metrics") or []
    assert not [m for m in metrics if m.endswith("_stderr")], benchmark_id


def test_tag_benchmark_scores_on_its_all_leaf():
    """winogender is a tag with no row of its own; its overall row is winogender_all."""
    from _tasks import primary_row

    assert primary_row("winogender", "winogender") == "winogender_all"
    assert primary_row("lm-eval/custom", "gsm8k,ifeval") is None
    assert primary_row("lm-eval/custom", "gsm8k") == "gsm8k"
    assert primary_row("arc_easy", "arc_easy") == "arc_easy"


# ── Loglikelihood benchmarks need a tokenizer ────────────────────────────────

from _execution import failure_hint  # noqa: E402
from _tasks import resolve_tokenizer_backend  # noqa: E402


def test_loglikelihood_benchmark_without_tokenizer_is_rejected_at_startup():
    """lm-eval asserts 'Tokenizer is required for loglikelihood tasks' mid-run."""
    for benchmark_id in ("lm-eval/mmlu", "arc_easy", "toxigen", "wikitext"):
        with pytest.raises(ValueError, match="needs a tokenizer"):
            resolve_tokenizer_backend(benchmark_id, None, "None")
    with pytest.raises(ValueError, match="tokenizer_backend=remote"):
        resolve_tokenizer_backend("lm-eval/open-llm-leaderboard-v2", None, "none")


def test_a_tokenizer_without_a_backend_selects_huggingface():
    """lm-eval ignores a tokenizer when the backend is None."""
    assert resolve_tokenizer_backend("arc_easy", "Qwen/Qwen3-8B", "None") == "huggingface"
    assert resolve_tokenizer_backend("lm-eval/gsm8k", "Qwen/Qwen3-8B", "None") == "huggingface"


def test_explicit_backends_and_generation_benchmarks_are_left_alone():
    assert resolve_tokenizer_backend("arc_easy", None, "remote") == "remote"
    assert resolve_tokenizer_backend("arc_easy", "gpt2", "tiktoken") == "tiktoken"
    assert resolve_tokenizer_backend("lm-eval/gsm8k", None, "None") == "None"
    assert resolve_tokenizer_backend("lm-eval/custom", None, "None") == "None"


@pytest.mark.integration
def test_tokenizer_param_reaches_the_command_with_the_huggingface_backend(
    tmp_path, mock_callbacks, monkeypatch
):
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        benchmark_id="arc_easy",
        parameters={"tokenizer": "Qwen/Qwen3-8B", "api_style": "completions"},
    )
    cmd = seen["cmd"]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "tokenizer_backend=huggingface" in model_args
    assert "tokenizer=Qwen/Qwen3-8B" in model_args


@pytest.mark.integration
def test_job_without_a_tokenizer_fails_before_lmeval_starts(tmp_path, mock_callbacks, monkeypatch):
    with pytest.raises(ValueError, match="needs a tokenizer"):
        _unsafe_code_run(
            tmp_path,
            mock_callbacks,
            monkeypatch,
            benchmark_id="arc_easy",
            parameters={"api_style": "completions"},
        )


def test_failure_hints_name_the_adapter_parameter():
    assert "trust_remote_code=true" in failure_hint(
        "ValueError: The repository for X contains custom code which must be executed"
    )
    assert "'tokenizer' parameter" in failure_hint(
        "AssertionError: Tokenizer is required for loglikelihood tasks"
    )
    assert failure_hint("some other failure") == ""


# ── Credentials never reach evaluation_metadata ──────────────────────────────

from _execution import redact_params  # noqa: E402


def test_redact_params_masks_credentials_by_name():
    params = {
        "api_key": "sk-live",
        "aws_secret_access_key": "s3cret",
        "aws_access_key_id": "AKIA123",
        "hf_token": "hf_x",
        "auth_token": "t",
        "nested": {"password": "p", "keep": 1},
        "tokenizer": "org/model",
        "max_gen_tokens": 512,
        "num_concurrent": 4,
        "empty_key": "",
    }
    out = redact_params(params)
    for key in ("api_key", "aws_secret_access_key", "aws_access_key_id", "hf_token", "auth_token"):
        assert out[key] == "***", key
    assert out["nested"] == {"password": "***", "keep": 1}
    assert out["tokenizer"] == "org/model"
    assert out["max_gen_tokens"] == 512
    assert out["num_concurrent"] == 4
    assert params["api_key"] == "sk-live", "input must not be mutated"


def _stub_lmeval(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_run_lmeval", lambda cmd, env, timeout: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda output_dir: CANNED_RESULTS)


@pytest.mark.integration
def test_evaluation_metadata_does_not_carry_credentials(adapter, mock_callbacks, monkeypatch, tmp_path):
    """benchmark_config in the results is the redacted parameter map."""
    adapter.job_spec.parameters.update(
        {"api_key": "sk-live", "aws_secret_access_key": "s3cret", "dataset_source": "hub"}
    )
    _stub_lmeval(adapter, monkeypatch)
    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    cfg = results.evaluation_metadata["benchmark_config"]
    assert cfg["api_key"] == "***"
    assert cfg["aws_secret_access_key"] == "***"
    assert "sk-live" not in json.dumps(results.evaluation_metadata)
    assert "s3cret" not in json.dumps(results.evaluation_metadata)


# ── Chat endpoints build structured messages ─────────────────────────────────


@pytest.mark.integration
def test_chat_endpoint_defaults_to_apply_chat_template(tmp_path, mock_callbacks, monkeypatch):
    """local-chat-completions rejects plain strings; the harness builds messages
    only under --apply_chat_template, so a chat endpoint gets it by default."""
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        benchmark_id="lm-eval/gsm8k",
        parameters={"api_style": "chat"},
    )
    cmd = seen["cmd"]
    assert "--apply_chat_template" in cmd
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "tokenized_requests=False" in model_args


@pytest.mark.integration
def test_auto_detected_chat_endpoint_defaults_to_apply_chat_template(
    tmp_path, mock_callbacks, monkeypatch
):
    """openrouter.ai, groq.com and together.ai resolve to chat without api_style."""
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        benchmark_id="lm-eval/gsm8k",
        model={"url": "https://openrouter.ai/api", "name": "m"},
    )
    cmd = seen["cmd"]
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd


@pytest.mark.integration
def test_completions_endpoint_keeps_apply_chat_template_off(tmp_path, mock_callbacks, monkeypatch):
    seen = _unsafe_code_run(
        tmp_path, mock_callbacks, monkeypatch, parameters={"api_style": "completions"}
    )
    assert "--apply_chat_template" not in seen["cmd"]


@pytest.mark.integration
def test_chat_endpoint_rejects_explicit_apply_chat_template_false(
    tmp_path, mock_callbacks, monkeypatch
):
    with pytest.raises(ValueError, match="apply_chat_template=false is not supported"):
        _unsafe_code_run(
            tmp_path,
            mock_callbacks,
            monkeypatch,
            benchmark_id="lm-eval/gsm8k",
            parameters={"api_style": "chat", "apply_chat_template": False},
        )


@pytest.mark.lmeval
def test_pinned_lmeval_chat_model_needs_structured_messages(lmeval):
    """The pinned local-chat-completions accepts messages and asserts on strings,
    which is why the adapter turns --apply_chat_template on for chat endpoints."""
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.utils import simple_parse_args_string

    args = build_model_args(
        base_url="http://127.0.0.1:1/v1/chat/completions",
        model_name="m",
        num_concurrent=1,
        max_retries=1,
        timeout_http=5,
        max_length=128,
        max_gen_toks=8,
        tokenizer=None,
        tokenizer_backend="None",
        tokenized_requests=False,
    )
    model = LocalChatCompletion(**simple_parse_args_string(args))
    payload = model._create_payload([{"role": "user", "content": "hi"}], generate=True)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    with pytest.raises(AssertionError, match="expects messages as list"):
        model._create_payload("plain string", generate=True)


# ── Chat templating on a completions endpoint ─────────────────────────────────

from _execution import build_model_args  # noqa: E402
from _tasks import resolve_chat_template_mode  # noqa: E402


def test_chat_template_is_a_noop_without_the_flag_or_on_chat_endpoints():
    assert resolve_chat_template_mode("completions", False, None, "None") == ("None", False)
    assert resolve_chat_template_mode("chat", True, None, "None") == ("None", False)
    assert resolve_chat_template_mode("chat", True, "org/m", "None") == ("None", False)


def test_chat_template_on_completions_needs_a_huggingface_tokenizer():
    assert resolve_chat_template_mode("completions", True, "org/m", "None") == (
        "huggingface",
        True,
    )
    assert resolve_chat_template_mode("completions", True, "org/m", "huggingface") == (
        "huggingface",
        True,
    )
    with pytest.raises(ValueError, match="'tokenizer' parameter"):
        resolve_chat_template_mode("completions", True, None, "None")
    with pytest.raises(ValueError, match="tokenizer_backend=huggingface"):
        resolve_chat_template_mode("completions", True, "org/m", "tiktoken")


def test_build_model_args_tokenized_requests_flag():
    kw = dict(
        base_url="http://h/v1/completions", model_name="m", num_concurrent=1,
        max_retries=1, timeout_http=1, max_length=1, max_gen_toks=1,
        tokenizer=None, tokenizer_backend="None",
    )
    assert "tokenized_requests=False" in build_model_args(**kw)
    assert "tokenized_requests=True" in build_model_args(**kw, tokenized_requests=True)


@pytest.mark.integration
def test_chat_template_on_completions_without_tokenizer_fails_fast(
    tmp_path, mock_callbacks, monkeypatch
):
    with pytest.raises(ValueError, match="'tokenizer' parameter"):
        _unsafe_code_run(
            tmp_path,
            mock_callbacks,
            monkeypatch,
            parameters={"api_style": "completions", "apply_chat_template": True},
        )


@pytest.mark.integration
def test_chat_template_on_completions_sends_tokenized_requests(
    tmp_path, mock_callbacks, monkeypatch
):
    seen = _unsafe_code_run(
        tmp_path,
        mock_callbacks,
        monkeypatch,
        parameters={
            "api_style": "completions",
            "apply_chat_template": True,
            "tokenizer": "org/model",
        },
    )
    cmd = seen["cmd"]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "tokenized_requests=True" in model_args
    assert "tokenizer_backend=huggingface" in model_args
    assert "--apply_chat_template" in cmd


def _offline_chat_tokenizer(path: Path) -> str:
    """A tiny local HF tokenizer with a chat template, so no Hub access is needed."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hi": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    fast.save_pretrained(path)
    return str(path)


@pytest.mark.lmeval
def test_pinned_lmeval_renders_the_template_only_with_tokenized_requests(tmp_path, lmeval):
    """lm_eval 0.4.13 on local-completions: JSON string vs a rendered template.

    Documents why the adapter must send tokenized_requests=True here, and checks
    the model args it builds select the rendering path.
    """
    pytest.importorskip("transformers")
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    from lm_eval.utils import simple_parse_args_string

    tokenizer_path = _offline_chat_tokenizer(tmp_path / "tok")
    history = [{"role": "user", "content": "hi"}]

    def _model(tokenized: bool):
        backend, flag = resolve_chat_template_mode(
            "completions", True, tokenizer_path, "None"
        )
        args = build_model_args(
            base_url="http://127.0.0.1:1/v1/completions",
            model_name="m",
            num_concurrent=1,
            max_retries=1,
            timeout_http=5,
            max_length=128,
            max_gen_toks=8,
            tokenizer=tokenizer_path,
            tokenizer_backend=backend,
            tokenized_requests=tokenized and flag,
        )
        return LocalCompletionsAPI(**simple_parse_args_string(args))

    # What the adapter used to send: the chat history as a JSON string.
    legacy = _model(tokenized=False).apply_chat_template(history)
    assert type(legacy).__name__ == "JsonChatStr"
    assert '"role": "user"' in legacy.prompt

    # What it sends now: the tokenizer's rendered template.
    rendered = _model(tokenized=True).apply_chat_template(history)
    assert isinstance(rendered, str)
    assert rendered == "<|user|>hi\n<|assistant|>"
