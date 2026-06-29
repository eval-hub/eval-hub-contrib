"""Command building, environment setup, and subprocess execution."""

import logging
import os
import re
import subprocess
from pathlib import Path

from evalhub.adapter import JobSpec
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials

from _benchmarks import PETRI_SEED_MAP
from _routing import _is_ollama_endpoint, role_model_spec, route_model, select_client, target_model_spec

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

    # ANTHROPIC_BASE_URL passes through from host env.

    if mode not in ("petri", "bloom"):
        client = select_client(env, endpoint_url=config.model.url)
        env["INSPECT_EVAL_MODEL"] = route_model(config.model.name, client)

    # Inject HF_TOKEN from sidecar-mounted secret if not already in env.
    # inspect-evals benchmarks (e.g. humaneval) download datasets from HF Hub.
    if not env.get("HF_TOKEN"):
        hf_token = read_model_auth_key("hf-token")
        if hf_token:
            env["HF_TOKEN"] = hf_token
            logger.info("Injected HF_TOKEN from mounted secret")

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

    max_tasks = config.parameters.get("max_tasks")
    if max_tasks:
        cmd += ["--max-tasks", str(max_tasks)]

    max_samples = config.parameters.get("max_samples")
    if max_samples is not None:
        cmd += ["--limit", str(max_samples)]

    epochs = config.parameters.get("epochs")
    if epochs and epochs > 1:
        cmd += ["--epochs", str(epochs)]

    cmd += ["--log-level", config.parameters.get("log_level", "info")]

    for key, value in config.parameters.get("task_args", {}).items():
        cmd += ["-T", f"{key}={value}"]

    return cmd


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
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("inspect eval timed out after 7200s.") from e

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
