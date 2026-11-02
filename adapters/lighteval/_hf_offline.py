"""Configure Hugging Face caches for LightEval jobs with staged test data.

EvalHub mounts assets referenced by a benchmark's ``test_data_ref`` at
``/test_data``. LightEval reads Hugging Face datasets through ``datasets`` and
the Hub, so these cache paths must point at the staged tree and network access
to the Hub must be disabled for the duration of the job.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

TEST_DATA_DIR = Path("/test_data")
_TEST_DATA_SOURCES = ("s3", "pvc", "git", "hf")


def job_spec_requests_test_data(job_spec_path: str | Path) -> bool:
    """Return whether the raw job spec contains a configured test-data source.

    The adapter SDK intentionally ignores the service-only ``test_data_ref``
    field when parsing ``JobSpec``, so inspect the mounted JSON directly.
    """
    try:
        with Path(job_spec_path).open(encoding="utf-8") as job_file:
            spec = json.load(job_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    if not isinstance(spec, dict):
        return False
    test_data_ref = spec.get("test_data_ref")
    if not isinstance(test_data_ref, dict):
        return False
    return any(test_data_ref.get(source) for source in _TEST_DATA_SOURCES)


def is_test_data_mount_usable(test_data_root: str | Path = TEST_DATA_DIR) -> bool:
    """Return whether the staged test-data directory contains visible entries."""
    root = Path(test_data_root)
    try:
        return root.is_dir() and any(
            not entry.name.startswith(".") for entry in root.iterdir()
        )
    except OSError:
        return False


def configure_hf_offline_environment(
    test_data_root: str | Path,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Use the staged HF cache tree and disable Hub downloads.

    When ``env`` is omitted, configure the current process so framework imports
    and the LightEval subprocess use the same cache and offline settings.
    """
    root = Path(test_data_root)
    target_env = os.environ if env is None else env
    target_env.update(
        {
            "HF_HOME": str(root),
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_DATASETS_CACHE": str(root / "datasets"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_EVALUATE_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )


def build_hf_offline_environment(
    job_spec_path: str | Path,
    test_data_root: str | Path = TEST_DATA_DIR,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Build an offline environment for an explicit staged-data job.

    Returns ``None`` for ordinary jobs, preserving LightEval's existing online
    behavior. A requested but unavailable mount fails early with a clear error.
    """
    if not job_spec_requests_test_data(job_spec_path):
        return None

    if not is_test_data_mount_usable(test_data_root):
        raise RuntimeError(
            f"Job spec includes test_data_ref but {test_data_root} is missing or empty. "
            "Ensure the test-data init container populated the mounted HF cache "
            "before the LightEval adapter starts."
        )

    env = dict(os.environ if base_env is None else base_env)
    configure_hf_offline_environment(test_data_root, env)
    return env
