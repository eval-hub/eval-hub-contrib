"""Integration and unit tests for the Inspect AI adapter.

All tests use monkeypatching or canned log fixtures — no real Inspect CLI
or LLM calls are made. Tests cover all three execution modes:
  - petri mode  (inspect/petri-* benchmarks)
  - bloom mode  (inspect/bloom-* benchmarks)
  - standard mode  (inspect/gsm8k, inspect/mmlu, inspect/custom, …)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from evalhub.adapter import JobPhase, OCIArtifactResult
from main import InspectAdapter
from _benchmarks import (
    PETRI_SEED_MAP,
    BLOOM_TEMPLATE_MAP,
    STANDARD_TASK_MAP,
    PETRI_PRIMARY_METRIC,
)


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("benchmark_id,expected_mode", [
    ("inspect/petri-sycophancy",       "petri"),
    ("inspect/petri-deception",        "petri"),
    ("inspect/petri-full",             "petri"),
    ("inspect/bloom-sycophancy",       "bloom"),
    ("inspect/bloom-custom",           "bloom"),
    ("inspect/gsm8k",                  "standard"),
    ("inspect/mmlu",                   "standard"),
    ("inspect/custom",                 "standard"),
])
def test_detect_mode(job_spec_path, benchmark_id, expected_mode):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    assert adapter._detect_mode(benchmark_id) == expected_mode


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------

def test_resolve_task_petri(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    assert adapter._resolve_task(adapter.job_spec, "petri", None) == "inspect_petri/audit"


def test_resolve_task_bloom(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/bloom-sycophancy"
    assert adapter._resolve_task(adapter.job_spec, "bloom", None) == "petri_bloom/bloom_audit"


def test_resolve_task_standard_gsm8k(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/gsm8k"
    assert adapter._resolve_task(adapter.job_spec, "standard", None) == "inspect_evals/gsm8k"


def test_resolve_task_explicit_override(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.parameters["task"] = "inspect_evals/mmlu"
    assert adapter._resolve_task(adapter.job_spec, "petri", None) == "inspect_evals/mmlu"


def test_resolve_task_custom_requires_parameter(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.parameters.pop("task", None)
    adapter.job_spec.benchmark_id = "inspect/custom"
    with pytest.raises(ValueError, match="task"):
        adapter._resolve_task(adapter.job_spec, "standard", None)


# ---------------------------------------------------------------------------
# Command construction — Petri mode
# ---------------------------------------------------------------------------

def _parse_model_roles(cmd: list[str]) -> dict:
    """Extract --model-role values from the command list, parsing JSON dicts."""
    import json as _json
    roles: dict = {}
    it = iter(cmd)
    for token in it:
        if token == "--model-role":
            val = next(it)
            name, _, spec = val.partition("=")
            try:
                roles[name] = _json.loads(spec)
            except ValueError:
                roles[name] = spec
    return roles


def test_petri_model_roles_via_cli_flags_mixed_apis(job_spec_path, tmp_path, monkeypatch):
    """Petri routing: all roles passed as --model-role CLI flags; model names preserved."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    adapter.job_spec.parameters["auditor_model"] = "claude-sonnet-4-6"
    adapter.job_spec.parameters["judge_model"] = "claude-opus-4-7"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)

    # Roles must be in --model-role CLI flags (Click multiple=True takes CLI OR env, never both)
    assert "--model-role" in cmd
    assert "--model" not in cmd
    roles = _parse_model_roles(cmd)
    assert "auditor" in roles
    assert "target" in roles
    assert "judge" in roles
    # User-supplied model names are preserved in the routing spec
    assert "claude-sonnet-4-6" in str(roles["auditor"])
    assert "claude-opus-4-7" in str(roles["judge"])
    assert "ibm-granite/granite-3.3-8b-instruct" in str(roles["target"])


def test_petri_model_roles_via_cli_flags_all_openai_compat(job_spec_path, tmp_path, monkeypatch):
    """All roles on same endpoint: --model-role flags in command, model names preserved."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    adapter.job_spec.parameters["auditor_model"] = "meta-llama/Llama-3.3-70B-Instruct"
    adapter.job_spec.parameters["judge_model"] = "meta-llama/Llama-3.3-70B-Instruct"
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)

    assert "--model-role" in cmd
    assert "--model" not in cmd
    roles = _parse_model_roles(cmd)
    assert "meta-llama/Llama-3.3-70B-Instruct" in str(roles["auditor"])
    assert "meta-llama/Llama-3.3-70B-Instruct" in str(roles["judge"])
    assert "ibm-granite/granite-3.3-8b-instruct" in str(roles["target"])


def test_petri_command_injects_seed_tag(job_spec_path, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-deception"
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)
    assert "-T" in cmd
    assert "seed_instructions=tags:deception" in cmd


def test_petri_full_no_seed_flag(job_spec_path, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-full"
    adapter.job_spec.parameters.pop("seed_instructions", None)
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)
    assert "seed_instructions" not in " ".join(cmd)


def test_petri_task_args_dish_passthrough(job_spec_path, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    adapter.job_spec.parameters["task_args"] = {"dish_scaffold": "claude-code"}
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)
    assert "dish_scaffold=claude-code" in cmd


# ---------------------------------------------------------------------------
# Command construction — Standard mode
# ---------------------------------------------------------------------------

def test_standard_command_no_model_flag(job_spec_path, tmp_path, monkeypatch):
    """Standard mode: model routed via INSPECT_EVAL_MODEL env var, not CLI flags."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8080/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/gsm8k"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    env = adapter._build_env(adapter.job_spec, "standard")
    cmd = adapter._build_command(adapter.job_spec, "standard", "inspect_evals/gsm8k", tmp_path, None, env)
    # No model flags in command — all model routing through env vars
    assert "--model" not in cmd
    assert "--model-role" not in cmd
    # Env var is set and preserves user-supplied model name
    assert "INSPECT_EVAL_MODEL" in env
    assert "ibm-granite/granite-3.3-8b-instruct" in env["INSPECT_EVAL_MODEL"]


def test_client_selection_url_beats_anthropic_key(job_spec_path):
    """endpoint_url present → OpenAI-compatible API selected even if Anthropic key is set."""
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_BASE_URL": "http://vllm:8080/v1"}
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    env2 = adapter._build_env(adapter.job_spec, "standard")
    # OPENAI_BASE_URL set and INSPECT_EVAL_MODEL contains the bare model name
    assert env2.get("OPENAI_BASE_URL") == "http://vllm:8080/v1"
    assert "ibm-granite/granite-3.3-8b-instruct" in env2.get("INSPECT_EVAL_MODEL", "")


def test_client_selection_anthropic_key(job_spec_path, monkeypatch):
    """ANTHROPIC_API_KEY set, no URL → Anthropic API selected for model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.url = None
    adapter.job_spec.model.name = "claude-opus-4-7"
    env = adapter._build_env(adapter.job_spec, "standard")
    # ANTHROPIC_API_KEY propagated; model name preserved in INSPECT_EVAL_MODEL
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant"
    assert "claude-opus-4-7" in env.get("INSPECT_EVAL_MODEL", "")


def test_client_selection_no_credentials_raises(job_spec_path, monkeypatch):
    """No credentials → ValueError when client cannot be determined."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.url = None
    with pytest.raises(ValueError, match="No API credentials"):
        adapter._select_client({})


def test_target_model_name_preserved_with_url(job_spec_path):
    """model.url present: user-supplied model name preserved in routing spec.

    base_url is NOT included in the spec — build_env sets OPENAI_BASE_URL from
    config.model.url so the provider picks it up via the env var. The spec only
    carries model_args to disable the Responses API.
    """
    import json as _json
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    env = {"OPENAI_BASE_URL": "http://vllm:8080/v1", "OPENAI_API_KEY": "local"}
    spec = adapter._target_model_string(adapter.job_spec, env)
    parsed = _json.loads(spec)
    # User's model name is preserved unchanged in the routing spec
    assert "ibm-granite/granite-3.3-8b-instruct" in parsed["model"]
    # base_url is set via OPENAI_BASE_URL env var, not inlined in the spec
    assert "base_url" not in parsed
    # Responses API is disabled to avoid 405 on /responses/input_tokens
    assert parsed.get("model_args", {}).get("responses_api") is False


def test_target_model_name_preserved_anthropic(job_spec_path):
    """No URL + ANTHROPIC_API_KEY: user-supplied model name preserved in routing spec."""
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.name = "claude-opus-4-7"
    adapter.job_spec.model.url = None
    env = {"ANTHROPIC_API_KEY": "sk-ant"}
    spec = adapter._target_model_string(adapter.job_spec, env)
    # User's model name is preserved unchanged
    assert "claude-opus-4-7" in spec


def test_per_role_anthropic_base_url(job_spec_path, tmp_path, monkeypatch):
    """judge_anthropic_base_url → per-role inline dict; user model names preserved."""
    import json as _json
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    adapter.job_spec.parameters["auditor_model"] = "claude-sonnet-4-6"
    adapter.job_spec.parameters["judge_model"] = "claude-opus-4-7"
    adapter.job_spec.parameters["judge_anthropic_base_url"] = "https://my-anthropic-proxy/v1"
    adapter.job_spec.parameters["judge_anthropic_api_key"] = "sk-proxy-key"
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)
    roles = _parse_model_roles(cmd)

    # auditor uses global ANTHROPIC_API_KEY — bare model name preserved
    assert "claude-sonnet-4-6" in roles["auditor"]
    # judge: per-role proxy → inline dict with user's model name preserved
    assert isinstance(roles["judge"], dict)
    assert "claude-opus-4-7" in roles["judge"]["model"]
    assert roles["judge"]["base_url"] == "https://my-anthropic-proxy/v1"
    assert roles["judge"]["api_key"] == "sk-proxy-key"


def test_per_role_base_url_creates_inline_dict(job_spec_path, tmp_path, monkeypatch):
    """Per-role base_url → inline dict preserving user model name; others use global env."""
    import json as _json
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/petri-sycophancy"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    adapter.job_spec.model.url = "http://vllm-target:8080/v1"
    adapter.job_spec.parameters["auditor_model"] = "claude-sonnet-4-6"
    adapter.job_spec.parameters["judge_model"] = "meta-llama/Llama-3.3-70B-Instruct"
    adapter.job_spec.parameters["judge_base_url"] = "http://vllm-judge:8080/v1"
    adapter.job_spec.parameters["judge_api_key"] = "EMPTY"
    env = adapter._build_env(adapter.job_spec, "petri")
    cmd = adapter._build_command(adapter.job_spec, "petri", "inspect_petri/audit", tmp_path, None, env)
    roles = _parse_model_roles(cmd)

    # auditor: global Anthropic key — user's model name preserved
    assert "claude-sonnet-4-6" in roles["auditor"]
    # judge: per-role endpoint — user's model name preserved in inline dict
    assert isinstance(roles["judge"], dict)
    assert "meta-llama/Llama-3.3-70B-Instruct" in roles["judge"]["model"]
    assert roles["judge"]["base_url"] == "http://vllm-judge:8080/v1"
    assert roles["judge"]["api_key"] == "EMPTY"
    # target: uses OPENAI_BASE_URL (set from model.url in build_env); base_url is not
    # inlined in the spec — it's carried via the env var, not the role spec.
    assert isinstance(roles["target"], dict)
    assert "ibm-granite/granite-3.3-8b-instruct" in roles["target"]["model"]
    assert "base_url" not in roles["target"]


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def test_petri_env_injects_anthropic_key(job_spec_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = adapter._build_env(adapter.job_spec, "petri")
    assert env.get("ANTHROPIC_API_KEY") == "test-anthropic-key"


def test_standard_env_no_anthropic_key(job_spec_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = adapter._build_env(adapter.job_spec, "standard")
    assert "ANTHROPIC_API_KEY" not in env


def test_env_sets_openai_base_url(job_spec_path):
    """OPENAI_BASE_URL is set from config.model.url."""
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.url = "http://vllm:8080/v1"
    adapter.job_spec.model.name = "ibm-granite/granite-3.3-8b-instruct"
    env = adapter._build_env(adapter.job_spec, "standard")
    assert env.get("OPENAI_BASE_URL") == "http://vllm:8080/v1"


def test_env_api_key_param_sets_openai_key(job_spec_path, monkeypatch):
    """api_key parameter sets OPENAI_API_KEY for OpenAI-compatible endpoints."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.url = None
    adapter.job_spec.parameters["api_key"] = "sk-vllm-token"
    env = adapter._build_env(adapter.job_spec, "petri")
    assert env.get("OPENAI_API_KEY") == "sk-vllm-token"


def test_env_anthropic_key_set(job_spec_path, monkeypatch):
    """ANTHROPIC_API_KEY is propagated from env to subprocess."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = adapter._build_env(adapter.job_spec, "petri")
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-test"


def test_env_anthropic_key_param_overrides_env(job_spec_path, monkeypatch):
    """anthropic_api_key parameter overrides ANTHROPIC_API_KEY env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.parameters["anthropic_api_key"] = "param-key"
    env = adapter._build_env(adapter.job_spec, "petri")
    assert env.get("ANTHROPIC_API_KEY") == "param-key"


def test_env_no_url_when_model_url_absent(job_spec_path, monkeypatch):
    """No OPENAI_BASE_URL is set when config.model.url is absent."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.model.url = None
    adapter.job_spec.model.name = "claude-haiku-4-5"
    adapter.job_spec.parameters.pop("api_key", None)
    env = adapter._build_env(adapter.job_spec, "standard")
    assert "OPENAI_BASE_URL" not in env
    assert "claude-haiku-4-5" in env.get("INSPECT_EVAL_MODEL", "")


def test_env_telemetry_disabled(job_spec_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = adapter._build_env(adapter.job_spec, "petri")
    assert env.get("INSPECT_NO_TELEMETRY") == "1"


# ---------------------------------------------------------------------------
# Log parsing and result extraction
# ---------------------------------------------------------------------------

def test_parse_log_raises_on_error_status(job_spec_path, tmp_path):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    log = tmp_path / "err.json"
    log.write_text(json.dumps({"status": "error", "error": {"message": "timeout"}}))
    with pytest.raises(RuntimeError, match="status 'error'"):
        adapter._parse_log(log)


def test_petri_results_extracted_correctly(job_spec_path, petri_eval_log):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    results, _, num = adapter._extract_results(
        petri_eval_log, "inspect/petri-sycophancy", "petri"
    )
    metric_names = {r.metric_name for r in results}
    # 5 dimensions × 2 metrics (mean, stderr) = 10 results
    assert len(results) == 10
    assert "concerning/mean" in metric_names
    assert "eval_awareness/mean" in metric_names
    assert num == 5


def test_petri_concerning_is_overall_score(job_spec_path, petri_eval_log):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    results, _, _ = adapter._extract_results(petri_eval_log, "inspect/petri-sycophancy", "petri")
    score = adapter._compute_overall_score(results, "petri")
    # concerning/mean = 3.2 in fixture
    assert score == pytest.approx(3.2)


def test_standard_results_extracted_correctly(job_spec_path, standard_eval_log):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    results, _, num = adapter._extract_results(standard_eval_log, "inspect/gsm8k", "standard")
    assert len(results) == 2  # accuracy/accuracy + accuracy/stderr
    assert any(r.metric_name == "accuracy/accuracy" for r in results)
    assert num == 10


def test_eval_awareness_mapped_to_alignment_meta(job_spec_path, petri_eval_log):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    _, caps, _ = adapter._extract_results(petri_eval_log, "inspect/petri-sycophancy", "petri")
    eval_awareness_caps = [c for c in caps if "eval_awareness" in c.metric]
    assert all(c.ability == "alignment_meta" for c in eval_awareness_caps)


def test_admirable_mapped_to_alignment_positive(job_spec_path, petri_eval_log):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    _, caps, _ = adapter._extract_results(petri_eval_log, "inspect/petri-sycophancy", "petri")
    admirable_caps = [c for c in caps if "admirable" in c.metric]
    assert all(c.ability == "alignment_positive" for c in admirable_caps)


# ---------------------------------------------------------------------------
# Full run_benchmark_job integration (Petri mode)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_petri_happy_path(monkeypatch, job_spec_path, petri_log_file):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    callbacks = MagicMock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )
    callbacks.mlflow.save.return_value = None

    import main as _main
    monkeypatch.setattr(_main, "run_inspect", lambda *_: petri_log_file)
    monkeypatch.setattr(adapter, "_get_inspect_version", lambda: "0.3.40")

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0
    assert results.num_examples_evaluated == 5
    assert results.overall_score == pytest.approx(3.2)

    metric_names = {r.metric_name for r in results.results}
    assert "concerning/mean" in metric_names
    assert "eval_awareness/mean" in metric_names

    assert results.eval_card is not None
    assert results.env_card is not None
    assert results.evaluation_metadata["mode"] == "petri"
    assert results.evaluation_metadata["framework"] == "inspect-ai"

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    # PERSISTING_ARTIFACTS only emitted when OCI exports are configured


@pytest.mark.integration
def test_oci_export_persists_artifacts(monkeypatch, job_spec_path, petri_eval_log):
    """When exports.oci is configured, PERSISTING_ARTIFACTS is emitted and create_oci_artifact is called."""
    # Patch job spec to include OCI exports
    job_spec = Path(job_spec_path)
    job = json.loads(job_spec.read_text())
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
    patched_spec = job_spec.parent / "job_oci.json"
    patched_spec.write_text(json.dumps(job))

    adapter = InspectAdapter(job_spec_path=str(patched_spec))
    callbacks = MagicMock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )
    callbacks.mlflow.save.return_value = None

    # run_inspect must place the log file inside the adapter's log_dir
    def fake_run_inspect(cmd, env, log_dir):
        log_file = log_dir / "petri_sycophancy_001.json"
        log_file.write_text(json.dumps(petri_eval_log))
        return log_file

    import main as _main
    monkeypatch.setattr(_main, "run_inspect", fake_run_inspect)
    monkeypatch.setattr(adapter, "_get_inspect_version", lambda: "0.3.40")

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    # PERSISTING_ARTIFACTS phase was reported
    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS in phases

    # create_oci_artifact was called with log_dir as files_path
    call_args = callbacks.create_oci_artifact.call_args
    assert call_args is not None
    spec = call_args.args[0]
    # work_dir is cleaned up in a finally block, so we verify the path shape
    assert spec.files_path.name == "logs"

    # OCI artifact is attached to results
    assert results.oci_artifact is not None
    assert results.oci_artifact.digest == "sha256:fake"


# ---------------------------------------------------------------------------
# Full run_benchmark_job integration (Standard mode)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_standard_happy_path(monkeypatch, job_spec_path, standard_log_file):
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/gsm8k"

    callbacks = MagicMock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )
    callbacks.mlflow.save.return_value = None

    import main as _main
    monkeypatch.setattr(_main, "run_inspect", lambda *_: standard_log_file)
    monkeypatch.setattr(adapter, "_get_inspect_version", lambda: "0.3.40")

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.evaluation_metadata["mode"] == "standard"
    assert results.num_examples_evaluated == 10
    assert any(r.metric_name == "accuracy/accuracy" for r in results.results)


# ---------------------------------------------------------------------------
# Catalog integrity checks
# ---------------------------------------------------------------------------

def test_all_petri_benchmarks_have_seed_or_none():
    for bid, seed in PETRI_SEED_MAP.items():
        if bid == "inspect/petri-full":
            assert seed is None, f"{bid} should have None seed (all seeds)"
        else:
            assert isinstance(seed, str) and seed.startswith("tags:"), (
                f"{bid} seed should be a 'tags:...' string, got {seed!r}"
            )


def test_all_bloom_benchmarks_have_template_or_none():
    for bid, template in BLOOM_TEMPLATE_MAP.items():
        if bid == "inspect/bloom-custom":
            assert template is None
        else:
            assert isinstance(template, str) and len(template) > 0


def test_all_standard_benchmarks_have_task_or_none():
    for bid, task in STANDARD_TASK_MAP.items():
        if bid == "inspect/custom":
            assert task is None
        else:
            assert isinstance(task, str) and len(task) > 0


def test_petri_primary_metric_is_defined():
    assert PETRI_PRIMARY_METRIC == "concerning"
