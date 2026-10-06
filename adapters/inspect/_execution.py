"""Command building, environment setup, and subprocess execution."""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from evalhub.adapter import JobSpec
from evalhub.adapter.auth import resolve_model_credentials

from _benchmarks import PETRI_SEED_MAP
from _hf_auth import apply_hf_hub_auth, refresh_hf_hub_auth
from _hf_offline import (
    TEST_DATA_DIR,
    configure_hf_offline_environment,
    ensure_test_data_ready_for_offline,
    should_use_hf_offline,
)
from _routing import (
    _is_ollama_endpoint,
    grader_model_spec,
    role_model_spec,
    route_model,
    select_client,
    target_model_spec,
)

logger = logging.getLogger(__name__)

_API_KEY_RE = re.compile(r'"api_key"\s*:\s*"[^"]*"')


def redact_cmd(cmd: list[str]) -> str:
    """Return a loggable representation of cmd with api_key values redacted."""
    return " ".join(_API_KEY_RE.sub('"api_key": "[REDACTED]"', arg) for arg in cmd)


def build_env(config: JobSpec, mode: str) -> dict[str, str]:
    """Build the subprocess environment from job credentials and model routing.

    For standard mode, INSPECT_EVAL_MODEL is injected via env var to keep the
    model name out of the CLI command log. For petri/bloom mode, all model roles
    are passed as --model-role CLI flags (see build_command) because Click's
    multiple=True option takes CLI args OR the env var, never both — so mixing
    them silently drops whichever source loses.

      INSPECT_EVAL_MODEL  — main model for standard mode
      OPENAI_BASE_URL     — endpoint for OpenAI-compatible APIs
      OPENAI_API_KEY      — key for OpenAI-compatible APIs
      ANTHROPIC_API_KEY   — key for Anthropic Messages API
    """
    env = os.environ.copy()
    p = config.parameters

    if config.model.url:
        if _is_ollama_endpoint(config.model.url):
            env["OLLAMA_BASE_URL"] = config.model.url
        else:
            env["OPENAI_BASE_URL"] = config.model.url

    # Resolve API key: sidecar-mounted secret takes precedence, then job spec
    # parameter, then existing env var, then a dummy fallback for endpoints that
    # don't require auth (e.g. bare vLLM without a proxy).
    creds = resolve_model_credentials()
    openai_key = creds.api_key or p.get("api_key") or env.get("OPENAI_API_KEY", "")
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    elif env.get("OPENAI_BASE_URL") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "local"

    anthropic_key = p.get("anthropic_api_key") or env.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key

    # StrongREJECT can use an independent grader endpoint and credential. The
    # key is supplied by the runtime (e.g. a Secret-backed env var), never by a
    # job parameter, so it is not persisted in the job spec or CLI arguments.
    if p.get("grader_model") and config.benchmark_id == "inspect/strong-reject":
        grader_key = env.get("OPENAI_JUDGE_API_KEY", "").strip()
        if not grader_key:
            raise ValueError(
                "parameters.grader_model requires OPENAI_JUDGE_API_KEY in the adapter environment. "
                "Provide it from a Kubernetes Secret; do not put API keys in the evaluation YAML."
            )
        env["OPENAI_JUDGE_API_KEY"] = grader_key
        env["OPENAI_JUDGE_BASE_URL"] = (
            p.get("grader_base_url")
            or env.get("OPENAI_JUDGE_BASE_URL")
            or "https://api.openai.com/v1"
        )

    # ANTHROPIC_BASE_URL passes through from host env.

    if mode not in ("petri", "bloom"):
        client = select_client(env, endpoint_url=config.model.url)
        env["INSPECT_EVAL_MODEL"] = route_model(config.model.name, client)

    # Staged S3/PVC/git test data (test_data_ref) or tokenizer+/test_data layout → offline Hub.
    if should_use_hf_offline(p):
        configure_hf_offline_environment(TEST_DATA_DIR, env)
        logger.info(
            "HF offline mode: HF_HOME=%s (staged test data), Hub downloads disabled",
            TEST_DATA_DIR,
        )

    # Gated datasets (Open-Telco, humaneval, mmlu, …) need Hub auth when not offline.
    if env.get("HF_HUB_OFFLINE") != "1":
        apply_hf_hub_auth(env)

    env["INSPECT_NO_TELEMETRY"] = "1"
    return env


def build_command(
    config: JobSpec,
    mode: str,
    task_spec: str,
    log_dir: Path,
    behavior_dir: Path | None,
    env: dict[str, str],
) -> list[str]:
    """Build the inspect eval CLI command."""
    cmd = [
        "inspect", "eval", task_spec,
        "--log-dir", str(log_dir),
        "--log-format", "json",
        "--no-ansi",
    ]

    if mode in ("petri", "bloom"):
        # All model roles via --model-role CLI flags. Click's multiple=True
        # option takes CLI args OR env var (INSPECT_EVAL_MODEL_ROLE), never
        # both — so we use only CLI flags to avoid silently dropping roles.
        cmd += _petri_model_role_flags(config, env)
        cmd += _petri_task_flags(config, mode, behavior_dir)
    else:
        # Main model is set via INSPECT_EVAL_MODEL env var — no --model flag needed.
        # Default to "local" sandbox — Docker is not available in Kubernetes pods.
        # Override by setting parameters.sandbox to another provider (e.g. "docker", "k8s").
        sandbox = config.parameters.get("sandbox", "local")
        if sandbox not in ("none", None):
            cmd += ["--sandbox", sandbox]

        # Optional model-role overrides for benchmarks that use judge/grader
        # models (e.g. HLE defaults to an OpenRouter judge). Provider YAML can
        # specify parameters.model_roles: {grader: "openai/gpt-4o-mini"} to
        # override at runtime without changing inspect-evals upstream defaults.
        model_roles = config.parameters.get("model_roles") or {}
        if not isinstance(model_roles, dict):
            raise ValueError(
                f"parameters.model_roles must be a dict (got {type(model_roles).__name__}). "
                "Example: model_roles: {grader: openai/gpt-4o-mini}"
            )
        for role, spec in model_roles.items():
            cmd += ["--model-role", f"{role}={spec}"]

        # StrongREJECT's task-specific judge_llm argument otherwise bypasses
        # Inspect's named grader role. Setting it to None activates the task's
        # grader-role fallback; the model itself is routed through the isolated
        # openai_judge provider namespace above.
        grader_model = config.parameters.get("grader_model")
        if grader_model and config.benchmark_id == "inspect/strong-reject":
            if not isinstance(grader_model, str) or not grader_model.strip():
                raise ValueError("parameters.grader_model must be a non-empty model name")
            if "grader" in model_roles:
                raise ValueError(
                    "Configure the StrongREJECT grader with parameters.grader_model, "
                    "not both grader_model and model_roles.grader."
                )
            # Insert the role before task args are appended below.
            cmd += ["--model-role", f"grader={grader_model_spec(grader_model.strip())}"]

    max_tasks = config.parameters.get("max_tasks")
    if max_tasks:
        cmd += ["--max-tasks", str(max_tasks)]

    limit = _sample_limit(config, mode)
    if limit is not None:
        cmd += ["--limit", str(limit)]

    epochs = config.parameters.get("epochs")
    if epochs and epochs > 1:
        cmd += ["--epochs", str(epochs)]

    cmd += ["--log-level", config.parameters.get("log_level", "info")]

    task_args = _task_args(config)
    if config.benchmark_id == "inspect/strong-reject" and config.parameters.get("grader_model"):
        # Ignore a legacy explicit judge_llm (often set to the target model) so
        # StrongREJECT uses the separately configured grader role.
        task_args["judge_llm"] = None

    for key, value in task_args.items():
        if isinstance(value, bool):
            value = str(value).lower()
        cmd += ["-T", f"{key}={value}"]

    for key, value in config.parameters.get("model_args", {}).items():
        cmd += ["-M", f"{key}={value}"]

    return cmd


# Petri/Bloom run every matching seed (170+ for the full audit), so they keep a small
# default cap. Standard inspect-evals benchmarks run the full dataset unless capped.
_PETRI_BLOOM_DEFAULT_LIMIT = 5


def _positive_int(value: Any, name: str) -> int:
    """Coerce a sample-limit value to an int >= 1, naming the parameter on failure."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer (got {value!r})") from None
    if n < 1:
        raise ValueError(f"{name} must be a positive integer (got {value!r})")
    return n


def _sample_limit(config: JobSpec, mode: str) -> int | None:
    """Resolve the Inspect ``--limit`` (number of samples), or None for no cap.

    Precedence:
      1. JobSpec.num_examples (lifted from benchmarks[].parameters.num_examples by
         eval-hub; the convention shared by all contrib adapters).
      2. parameters.max_samples — deprecated alias. It was the sample cap before
         num_examples was introduced and eval-hub does not strip it, so existing
         collections still carry it. Note Inspect's own ``--max-samples`` means
         *parallel* samples, not a sample cap, which is why it is being retired.
      3. Petri/Bloom default; otherwise unbounded.
    """
    if config.num_examples is not None:
        return _positive_int(config.num_examples, "num_examples")

    legacy = config.parameters.get("max_samples")
    if legacy is not None:
        logger.warning(
            "parameters.max_samples is deprecated; use num_examples "
            "(treating max_samples=%s as num_examples)",
            legacy,
        )
        return _positive_int(legacy, "parameters.max_samples")

    return _PETRI_BLOOM_DEFAULT_LIMIT if mode in ("petri", "bloom") else None


# Open-Telco (and similar) first-class -T parameters — keep flat under parameters, not task_args.
_FIRST_CLASS_TASK_PARAMS = ("full", "subject", "eval_type")


def _task_args(config: JobSpec) -> dict:
    """Build Inspect -T args from first-class parameters and optional task_args escape hatch.

    First-class keys are the supported API surface for Open-Telco and similar tasks.
    parameters.task_args remains for Dish / ad-hoc Inspect flags only; it must not
    redefine first-class keys.
    """
    args: dict = {}
    for key in _FIRST_CLASS_TASK_PARAMS:
        if key in config.parameters and config.parameters[key] is not None:
            args[key] = config.parameters[key]
    # Escape hatch last so unknown -T flags still work; first-class keys win on conflict.
    for key, value in (config.parameters.get("task_args") or {}).items():
        if key in args:
            continue
        args[key] = value
    return args


def _petri_model_role_flags(config: JobSpec, env: dict[str, str]) -> list[str]:
    """Build --model-role CLI flags for all petri/bloom roles.

    Target gets a JSON dict with model_args to explicitly disable the Responses
    API — vLLM doesn't implement POST /responses/input_tokens which inspect_ai
    calls during compaction when responses_api is True or inferred True.

    auditor_model and judge_model can be overridden via job parameters;
    defaults to claude-sonnet-4-6 / claude-opus-4-7 when not specified.
    """
    p = config.parameters
    auditor = role_model_spec(p.get("auditor_model", "claude-sonnet-4-6"), "auditor", p, env)
    judge   = role_model_spec(p.get("judge_model",   "claude-opus-4-7"),   "judge",   p, env)
    target  = target_model_spec(config.model.name, config.model.url, p, env)

    flags = [
        "--model-role", f"auditor={auditor}",
        "--model-role", f"target={target}",
        "--model-role", f"judge={judge}",
    ]

    realism_name = p.get("realism_model")
    if realism_name:
        flags += ["--model-role", f"realism={role_model_spec(realism_name, 'realism', p, env)}"]

    return flags


def _petri_task_flags(
    config: JobSpec, mode: str, behavior_dir: Path | None
) -> list[str]:
    flags: list[str] = []

    if mode == "petri":
        seed = config.parameters.get("seed_instructions") or PETRI_SEED_MAP.get(config.benchmark_id)
        if seed:
            flags += ["-T", f"seed_instructions={seed}"]

        flags += ["-T", f"max_turns={config.parameters.get('max_turns', 30)}"]
        flags += ["-T", f"enable_rollback={str(config.parameters.get('enable_rollback', True)).lower()}"]

        realism_filter = config.parameters.get("realism_filter", False)
        if realism_filter:
            flags += ["-T", f"realism_filter={realism_filter}"]

        target_tools = config.parameters.get("target_tools")
        if target_tools:
            flags += ["-T", f"target_tools={target_tools}"]

        judge_dims = config.parameters.get("judge_dimensions")
        if judge_dims:
            flags += ["-T", f"judge_dimensions={judge_dims}"]

    elif mode == "bloom" and behavior_dir is not None:
        flags += ["-T", f"behavior={behavior_dir}"]
        flags += ["-T", f"max_turns={config.parameters.get('max_turns', 30)}"]
        flags += ["-T", f"enable_rollback={str(config.parameters.get('enable_rollback', True)).lower()}"]

    return flags


def run_inspect(cmd: list[str], env: dict[str, str], log_dir: Path) -> Path:
    refresh_hf_hub_auth(env)
    try:
        if env.get("EVALHUB_MODE", "") == "k8s":
            # allows long running benchmarks in k8s to run indefinitely
            timeout = None
        else:
            timeout = 7200
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"inspect eval timed out after {timeout}.") from e
    except (subprocess.SubprocessError, OSError):
        logger.exception("Subprocess failed")
        raise

    if result.returncode != 0:
        logger.error(f"inspect eval stdout:\n{result.stdout[-3000:]}")
        logger.error(f"inspect eval stderr:\n{result.stderr[-3000:]}")
        raise RuntimeError(
            f"inspect eval failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    if result.stdout:
        logger.debug(f"stdout (tail): {result.stdout[-2000:]}")
    if result.stderr:
        logger.debug(f"stderr (tail): {result.stderr[-2000:]}")

    log_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        raise RuntimeError(
            f"inspect eval produced no JSON log in {log_dir}. stdout: {result.stdout[-500:]}"
        )

    log_file = log_files[-1]
    logger.info(f"Inspect log: {log_file}")
    return log_file


def get_inspect_version() -> str:
    try:
        result = subprocess.run(["inspect", "--version"], capture_output=True, text=True, timeout=10)
        raw = result.stdout.strip() or result.stderr.strip()
        return raw.split()[-1] if raw else "unknown"
    except Exception:
        return "unknown"
