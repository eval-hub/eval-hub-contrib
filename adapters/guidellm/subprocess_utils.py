"""Utilities for streaming subprocess output and reporting bounded diagnostics."""

from __future__ import annotations

import logging
import re
import subprocess
from collections import deque
from collections.abc import Iterable, Mapping, Sequence

_MAX_OUTPUT_LINES = 40
_MAX_ERROR_OUTPUT_CHARS = 4_000

_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"authorization|password|secret)[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|(?:(?:bearer|basic|token)\s+)?[^\s,;{}]+)"
)
_URL_CREDENTIAL_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password)=)"
    r"[^&\s]+"
)


def _sanitize_output_line(line: str, sensitive_values: Iterable[str]) -> str:
    """Redact known secrets and common credential assignments from one output line."""
    for value in sorted((value for value in sensitive_values if value), key=len, reverse=True):
        line = line.replace(value, "[REDACTED]")

    line = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1[REDACTED]", line)
    return _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]", line)


def _bounded_output_tail(lines: Iterable[str]) -> str:
    """Join recent output lines and cap the diagnostic text to the configured limit."""
    output = "\n".join(lines)
    if len(output) <= _MAX_ERROR_OUTPUT_CHARS:
        return output

    marker = "[earlier subprocess output truncated]\n"
    return marker + output[-(_MAX_ERROR_OUTPUT_CHARS - len(marker)) :]


def run_streaming_subprocess(
    command: Sequence[str],
    *,
    name: str,
    logger: logging.Logger,
    env: Mapping[str, str] | None = None,
    sensitive_values: Iterable[str] = (),
) -> None:
    """Run a command, stream sanitized output to logs, and include a bounded tail on failure.

    Standard error is merged into standard output so both streams are logged and
    retained together. The tail is deliberately bounded because it is included in
    the adapter's structured job error message.
    """
    sensitive_values = tuple(value for value in sensitive_values if value)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_tail: deque[str] = deque(maxlen=_MAX_OUTPUT_LINES)
    if process.stdout:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                sanitized_line = _sanitize_output_line(line, sensitive_values)
                output_tail.append(sanitized_line)
                logger.info("%s: %s", name, sanitized_line)

    return_code = process.wait()
    if return_code != 0:
        error = f"{name} failed with exit code {return_code}"
        output = _bounded_output_tail(output_tail)
        if output:
            error += f"\nLast subprocess output (stdout/stderr):\n{output}"
        raise RuntimeError(error)
