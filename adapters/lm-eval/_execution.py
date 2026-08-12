"""Command building, environment setup, and subprocess execution for lm-eval."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_REDACT_PATTERN = re.compile(r"((?:api_key|auth_token)=)[^,]*")


def get_lmeval_version() -> str:
    try:
        from importlib.metadata import version

        return version("lm_eval")
    except Exception:  # noqa: BLE001
        return "unknown"


def build_model_args(
    *,
    base_url: str,
    model_name: str,
    num_concurrent: int,
    max_retries: int,
    timeout_http: int,
    max_length: int,
    max_gen_toks: int,
    tokenizer: str | None,
    tokenizer_backend: str,
) -> str:
    """Build the comma-separated model_args string passed to --model_args."""
    parts: list[tuple[str, str]] = [
        ("base_url", base_url),
        ("model", model_name),
        ("num_concurrent", str(num_concurrent)),
        ("max_retries", str(max_retries)),
        ("timeout", str(timeout_http)),
        ("max_length", str(max_length)),
        ("max_gen_toks", str(max_gen_toks)),
        ("tokenizer_backend", tokenizer_backend),
        # Send prompts as strings, not token-ID lists.
        # Without a local tokenizer loaded, tokenized_requests=True would cause
        # the harness to guess token IDs from characters — send strings instead.
        ("tokenized_requests", "False"),
    ]
    if tokenizer:
        parts.append(("tokenizer", tokenizer))
    return ",".join(f"{k}={v}" for k, v in parts)


def build_cmd(
    *,
    model_type: str,
    model_args: str,
    tasks: str,
    output_dir: Path,
    batch_size: int,
    num_fewshot: int | None,
    limit: int | None,
    gen_kwargs: str | None,
    include_path: str | None,
    apply_chat_template: bool,
    system_instruction: str | None,
) -> list[str]:
    cmd = [
        "lm_eval",
        "--model",
        model_type,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--output_path",
        str(output_dir),
        "--log_samples",
        "--batch_size",
        str(batch_size),
    ]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    if include_path:
        cmd += ["--include_path", include_path]
    if apply_chat_template:
        cmd += ["--apply_chat_template"]
    if system_instruction:
        cmd += ["--system_instruction", system_instruction]
    return cmd


def build_env(
    *,
    api_key: str,
    hf_datasets_cache: str | None,
    dataset_source: str,
    s3_endpoint: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
) -> dict[str, str]:
    """Build environment variables for the lm_eval subprocess."""
    env = dict(os.environ)
    # lm-eval's LocalCompletionsAPI reads OPENAI_API_KEY. For open endpoints,
    # any non-empty value works — the server ignores it.
    env["OPENAI_API_KEY"] = api_key if api_key else "dummy"
    # Prevent HuggingFace tokenizer parallelism warnings in the subprocess.
    env["TOKENIZERS_PARALLELISM"] = "false"
    if hf_datasets_cache:
        env["HF_DATASETS_CACHE"] = hf_datasets_cache
    if dataset_source == "s3":
        if s3_endpoint:
            env["AWS_ENDPOINT_URL"] = s3_endpoint
        if aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    return env


def redact_model_args(model_args: str) -> str:
    """Remove api_key values from model_args for safe logging."""
    return _REDACT_PATTERN.sub(r"\1***", model_args)


def redact_cmd(cmd: list[str]) -> list[str]:
    """Return a copy of cmd with credential values redacted."""
    return [_REDACT_PATTERN.sub(r"\1***", arg) for arg in cmd]


def run_lmeval(
    cmd: list[str],
    env: dict[str, str],
    timeout: int = 7200,
) -> tuple[str, str]:
    """Execute lm_eval as a subprocess.

    Returns (stdout, stderr).
    Raises RuntimeError on non-zero exit or timeout.
    """
    logger.info("Command: %s", " ".join(redact_cmd(cmd)))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"lm_eval timed out after {exc.timeout}s. "
            "Increase subprocess_timeout or reduce num_examples (--limit)."
        ) from exc

    if result.stdout:
        logger.info("lm_eval stdout (tail):\n%s", result.stdout[-4000:])
    if result.stderr:
        log_fn = logger.warning if result.returncode != 0 else logger.debug
        log_fn("lm_eval stderr (tail):\n%s", result.stderr[-4000:])

    if result.returncode != 0:
        raise RuntimeError(
            f"lm_eval exited {result.returncode}.\n"
            f"stderr (last 2000 chars):\n{result.stderr[-2000:]}"
        )
    return result.stdout, result.stderr
