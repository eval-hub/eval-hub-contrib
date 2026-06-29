"""Command building, environment setup, and subprocess execution."""

import logging
import os
import subprocess
from pathlib import Path

from evalhub.adapter import JobSpec

from _benchmarks import PETRI_SEED_MAP
from _routing import _is_ollama_endpoint, role_model_spec, route_model, select_client, target_model_spec

logger = logging.getLogger(__name__)


def build_env(config: JobSpec, mode: str) -> dict[str, str]:
    """Build the subprocess environment from job credentials and model routing.

    Model routing is injected via Inspect AI env vars so that model names
    never appear in the CLI command string (and therefore not in logs).
    The user-supplied model names are read as-is from the job spec; the
    internal routing details required by the Inspect AI framework are an
    implementation detail of this adapter.

      INSPECT_EVAL_MODEL       — main model for standard mode
      INSPECT_EVAL_MODEL_ROLE  — per-role models for Petri/Bloom mode
      OPENAI_BASE_URL          — endpoint for OpenAI-compatible APIs
      OPENAI_API_KEY           — key for OpenAI-compatible APIs
      ANTHROPIC_API_KEY        — key for Anthropic Messages API
    """
    env = os.environ.copy()
    p = config.parameters

    if config.model.url:
        if _is_ollama_endpoint(config.model.url):
            env["OLLAMA_BASE_URL"] = config.model.url
        else:
            env["OPENAI_BASE_URL"] = config.model.url

    # Inspect AI requires OPENAI_API_KEY for OpenAI-compat endpoints that don't
    # validate authentication. Not needed for Ollama (it uses its own auth).
    openai_key = p.get("api_key") or env.get("OPENAI_API_KEY", "")
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    elif env.get("OPENAI_BASE_URL") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "local"

    anthropic_key = p.get("anthropic_api_key") or env.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key

    # ANTHROPIC_BASE_URL passes through from host env.

    # Inject model routing via Inspect AI env vars — hidden from CLI command logs.
    if mode in ("petri", "bloom"):
        env["INSPECT_EVAL_MODEL_ROLE"] = _build_model_role_env(config, env)
    else:
        client = select_client(env, endpoint_url=config.model.url)
        env["INSPECT_EVAL_MODEL"] = route_model(config.model.name, client)

    env["INSPECT_NO_TELEMETRY"] = "1"
    return env


def _build_model_role_env(config: JobSpec, env: dict[str, str]) -> str:
    """Build the INSPECT_EVAL_MODEL_ROLE env var value (comma-separated roles).

    Uses simple provider/model strings for all roles. OPENAI_BASE_URL and
    ANTHROPIC_API_KEY env vars handle endpoint routing — no inline base_url
    needed. This keeps the inspect eval CLI command free of model names.

    Per-role URL overrides (via {role}_base_url params) use --model-role CLI
    flags as fallback since JSON dicts with commas break Click's env var parsing.
    """
    p = config.parameters
    # Detect if any role needs a per-role URL override (JSON dict spec)
    has_per_role_url = any(
        p.get(f"{role}_base_url") or p.get(f"{role}_anthropic_base_url")
        for role in ("auditor", "judge", "target", "realism")
    )
    if has_per_role_url:
        return ""  # sentinel: use --model-role CLI flags instead

    # Build simple provider/model strings — URL routing via env vars
    client = select_client(env, endpoint_url=config.model.url)
    target_str  = route_model(config.model.name, client)
    auditor_str = route_model(p.get("auditor_model", "claude-sonnet-4-6"), select_client(env))
    judge_str   = route_model(p.get("judge_model",   "claude-opus-4-7"),   select_client(env))

    parts = [f"auditor={auditor_str}", f"target={target_str}", f"judge={judge_str}"]

    realism_name = p.get("realism_model")
    if realism_name:
        parts.append(f"realism={route_model(realism_name, select_client(env))}")

    # Click splits INSPECT_EVAL_MODEL_ROLE on newlines for multiple=True options.
    # Commas cannot be used: Inspect AI treats them as multi-model separators
    # within a single role spec, which causes a resolution error.
    return "\n".join(parts)


def build_command(
    config: JobSpec,
    mode: str,
    task_spec: str,
    log_dir: Path,
    behavior_dir: Path | None,
    env: dict[str, str],
) -> list[str]:
    """Build the inspect eval CLI command.

    Model routing is handled entirely via env vars set by build_env().
    Model names never appear in the command string.
    """
    cmd = [
        "inspect", "eval", task_spec,
        "--log-dir", str(log_dir),
        "--log-format", "json",
        "--no-ansi",
    ]

    if mode in ("petri", "bloom"):
        # Model roles are normally set via INSPECT_EVAL_MODEL_ROLE env var.
        # Fall back to --model-role flags only when per-role JSON dict specs
        # are in use (i.e. env var was set to empty sentinel by build_env).
        if not env.get("INSPECT_EVAL_MODEL_ROLE"):
            cmd += _petri_model_role_flags(config, env)
        cmd += _petri_task_flags(config, mode, behavior_dir)
    else:
        # Main model is set via INSPECT_EVAL_MODEL env var — no --model flag needed.
        if config.parameters.get("sandbox", "none") not in ("none", None):
            cmd += ["--sandbox", config.parameters["sandbox"]]

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
    """Fallback: --model-role CLI flags used only when per-role JSON dict specs
    are present (e.g. per-role URL overrides). Normally replaced by env var."""
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

    if result.stdout:
        logger.debug(f"stdout (tail): {result.stdout[-2000:]}")
    if result.stderr:
        logger.debug(f"stderr (tail): {result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"inspect eval failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}"
        )

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
