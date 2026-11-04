"""Helpers for detecting staged RULER inputs and resolving local tokenizers."""

import json
import os
from pathlib import Path
from typing import Any

TEST_DATA_DIR_ENV = "RULER_TEST_DATA_DIR"
DEFAULT_TEST_DATA_DIR = Path("/test_data")
JOB_SPEC_PATH_ENV = "EVALHUB_JOB_SPEC_PATH"
DEFAULT_JOB_SPEC_PATH = Path("/meta/job.json")


def get_test_data_root() -> Path:
    """Return the configured staged-data root (``/test_data`` in Kubernetes)."""
    return Path(os.environ.get(TEST_DATA_DIR_ENV, str(DEFAULT_TEST_DATA_DIR)))


def job_spec_requests_test_data(job_spec_path: str | Path | None = None) -> bool:
    """Check the raw mounted job spec for a staged-data reference when present."""
    path = Path(
        job_spec_path
        or os.environ.get(JOB_SPEC_PATH_ENV, str(DEFAULT_JOB_SPEC_PATH))
    )
    try:
        with path.open(encoding="utf-8") as stream:
            spec: Any = json.load(stream)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    reference = spec.get("test_data_ref") if isinstance(spec, dict) else None
    return isinstance(reference, dict) and any(
        reference.get(key) for key in ("s3", "pvc", "git", "hf")
    )


def is_staged_data_available(root: str | Path) -> bool:
    """Recognize a mounted /test_data volume, including an empty failed stage."""
    path = Path(root)
    try:
        return path.is_mount() or (
            path.is_dir() and any(not entry.name.startswith(".") for entry in path.iterdir())
        )
    except OSError:
        return False


def staged_data_requested(
    job_spec_path: str | Path | None = None,
    root: str | Path | None = None,
) -> bool:
    """Return true when the job spec or mounted staged-data volume requests staging."""
    staged_root = Path(root) if root is not None else get_test_data_root()
    return job_spec_requests_test_data(job_spec_path) or is_staged_data_available(staged_root)


def resolve_staged_tokenizer_path(
    tokenizer_path: str | None,
    model_name: str,
    tokenizer_type: str,
    root: str | Path,
) -> Path:
    """Resolve the local HF tokenizer directory for a disconnected RULER job.

    If ``tokenizer_path`` is omitted or is the evaluated model ID (the online
    default), the staged tokenizer is expected in ``<root>/tokenizer``. Other
    relative paths are resolved beneath the staged root; absolute paths are
    accepted for callers that mount a tokenizer elsewhere in the pod.
    """
    if tokenizer_type != "hf":
        raise ValueError(
            "Disconnected RULER jobs currently require tokenizer_type='hf' and a "
            "staged Hugging Face tokenizer. Stage it under /test_data/tokenizer "
            "or set parameters.tokenizer_path to a local tokenizer directory."
        )

    configured = (tokenizer_path or "").strip()
    if not configured or configured == model_name:
        path = Path(root) / "tokenizer"
    else:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            path = configured_path
        else:
            staged_root = Path(root).resolve()
            path = (staged_root / configured_path).resolve()
            try:
                path.relative_to(staged_root)
            except ValueError as exc:
                raise ValueError(
                    "Relative parameters.tokenizer_path must resolve beneath /test_data."
                ) from exc

    if not path.is_dir():
        raise ValueError(
            f"Staged RULER tokenizer directory not found: {path}. Stage the complete "
            "tokenizer files under /test_data/tokenizer, or set parameters.tokenizer_path "
            "to a local directory."
        )
    return path
