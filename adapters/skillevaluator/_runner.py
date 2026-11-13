"""Invoke the pinned ``skillevaluator`` CLI and collect its JSON report.

SkillEvaluator documents a CLI and an on-disk JSON contract, not a Python API.
This module is the only place that spawns a process. The command line is built
from a fixed argument list (no shell), the child environment is an allow-list
(job secrets and the pod's service-account variables are not forwarded), and
neither stdout nor stderr is logged: both can echo content of the skill under
evaluation, which is untrusted.

The CLI exits ``0`` when every gate passed, ``1`` when a check failed or
evidence was incomplete (a report is still written) and ``2`` on a usage error.
Exit codes ``0`` and ``1`` therefore both yield a report; the verdict is read
from the report, not from the exit code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CLI_BIN_ENV = "SKILLEVALUATOR_BIN"
CLI_NAME = "skillevaluator"

#: Environment variables forwarded to the CLI. Everything else is dropped.
_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SKILLEVALUATOR_SKILLSPECTOR_PATH",
    "SKILLEVALUATOR_SEMGREP_PATH",
    "SKILLEVALUATOR_BANDIT_PATH",
    "SKILLEVALUATOR_SUBMITTER",
)

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+[^\s]*)")


class RunnerError(RuntimeError):
    """The CLI could not be run or did not produce a usable report."""


@dataclass(frozen=True)
class CliRun:
    """Outcome of one CLI invocation: its exit code and the JSON report it wrote."""

    returncode: int
    report_path: Path


def find_cli() -> str:
    """Return the path of the ``skillevaluator`` executable."""
    override = os.environ.get(CLI_BIN_ENV, "").strip()
    if override:
        return override
    sibling = Path(sys.executable).parent / CLI_NAME
    if sibling.is_file():
        return str(sibling)
    found = shutil.which(CLI_NAME)
    if found:
        return found
    raise RunnerError(f"{CLI_NAME} executable not found (set {CLI_BIN_ENV})")


def build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Allow-listed child environment, plus validated ``extra`` entries."""
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    if extra:
        env.update({k: v for k, v in extra.items() if v})
    return env


def get_version(executable: str, *, env: dict[str, str] | None = None) -> str | None:
    """Best-effort ``--version`` of a tool; ``None`` when it cannot be read."""
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env if env is not None else build_env(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION_RE.search(proc.stdout or proc.stderr or "")
    return match.group(1) if match else None


def run_cli(
    args: list[str],
    *,
    output_dir: Path,
    timeout: int,
    env: dict[str, str],
    executable: str | None = None,
) -> CliRun:
    """Run ``skillevaluator <args> -r json -o <output_dir>`` and locate the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [executable or find_cli(), *args, "-r", "json", "-o", str(output_dir)]
    logger.info("Running %s %s", CLI_NAME, args[0] if args else "")
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=output_dir,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(f"{CLI_NAME} {args[0]} timed out after {timeout}s") from exc
    except OSError as exc:
        raise RunnerError(
            f"cannot execute {CLI_NAME}: {exc.__class__.__name__}"
        ) from exc

    if proc.returncode not in (0, 1):
        raise RunnerError(
            f"{CLI_NAME} {args[0]} exited with code {proc.returncode} (usage or internal error)"
        )

    reports = sorted(output_dir.glob("*.json"))
    if len(reports) != 1:
        raise RunnerError(
            f"expected exactly one JSON report in the output directory, found {len(reports)}"
        )
    return CliRun(returncode=proc.returncode, report_path=reports[0])


def load_report(path: Path) -> dict[str, Any]:
    """Parse a report file; the top level must be a JSON object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"report is not valid JSON: {exc.__class__.__name__}"
        ) from exc
    if not isinstance(data, dict):
        raise RunnerError("report top level is not a JSON object")
    return data
