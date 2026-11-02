"""Configure Hugging Face caches for LightEval jobs with staged test data.

EvalHub mounts assets referenced by a benchmark's ``test_data_ref`` at
``/test_data``. LightEval reads Hugging Face datasets through ``datasets`` and
the Hub, so these cache paths must point at the staged tree and network access
to the Hub must be disabled for the duration of the job.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path

TEST_DATA_DIR = Path("/test_data")
_TEST_DATA_SOURCES = ("s3", "pvc", "git", "hf")
LIGHTEVAL_TASK_REGISTRY_DIR: Path | None = None


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
    writable_datasets_cache_dir: str | Path | None = None,
) -> Path:
    """Use the staged HF cache tree and disable Hub downloads.

    When ``env`` is omitted, configure the current process so framework imports
    and the LightEval subprocess use the same cache and offline settings. The
    datasets cache is mirrored into a writable directory with symlinked data
    files so Hugging Face can create lock files without copying staged assets.
    """
    root = Path(test_data_root)
    target_env = os.environ if env is None else env
    datasets_cache = _prepare_writable_datasets_cache(
        root,
        writable_datasets_cache_dir,
    )
    target_env.update(
        {
            "HF_HOME": str(root),
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_DATASETS_CACHE": str(datasets_cache),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_EVALUATE_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return datasets_cache


def _prepare_writable_datasets_cache(
    test_data_root: Path,
    writable_datasets_cache_dir: str | Path | None,
) -> Path:
    """Mirror dataset-cache directories while symlinking staged files in place."""
    staged_root = test_data_root.resolve()
    staged_cache = test_data_root / "datasets"
    cache_dir = (
        Path(writable_datasets_cache_dir)
        if writable_datasets_cache_dir is not None
        else Path(tempfile.mkdtemp(prefix="lighteval-datasets-cache-"))
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_dir.resolve().is_relative_to(staged_root):
            raise RuntimeError(
                f"Writable datasets cache {cache_dir} must be outside staged "
                f"test-data directory {test_data_root}."
            )
        if any(cache_dir.iterdir()):
            raise RuntimeError(
                f"Writable datasets cache directory {cache_dir} must be empty."
            )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create writable Hugging Face datasets cache at {cache_dir}: {exc}"
        ) from exc

    def mirror(source: Path, destination: Path, active_directories: set[Path]) -> None:
        if source.name.endswith(".lock"):
            return
        try:
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(staged_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Staged datasets cache entry {source} is missing or points outside "
                f"{test_data_root}."
            ) from exc

        if resolved_source.is_dir():
            if resolved_source in active_directories:
                raise RuntimeError(
                    f"Staged datasets cache contains a directory cycle at {source}."
                )
            destination.mkdir(parents=True, exist_ok=True)
            next_active = {*active_directories, resolved_source}
            for child in resolved_source.iterdir():
                mirror(child, destination / child.name, next_active)
            return

        if not resolved_source.is_file():
            raise RuntimeError(f"Unsupported entry in staged datasets cache: {source}.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(resolved_source)

    if staged_cache.is_dir():
        try:
            for entry in staged_cache.iterdir():
                mirror(entry, cache_dir / entry.name, set())
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Cannot prepare writable Hugging Face datasets cache "
                f"at {cache_dir}: {exc}"
            ) from exc

    return cache_dir


def ensure_lighteval_registry_assets(
    test_data_root: str | Path,
    task_registry_dir: str | Path | None = None,
) -> None:
    """Seed LightEval's eagerly imported TinyBenchmarks registry asset offline.

    LightEval 0.13.0 downloads this auxiliary file with ``requests`` while
    importing every task config. That request bypasses the HF offline flags,
    even when the selected task does not use TinyBenchmarks.
    """
    root = Path(test_data_root)
    destination_dir = (
        Path(task_registry_dir)
        if task_registry_dir is not None
        else LIGHTEVAL_TASK_REGISTRY_DIR or _default_lighteval_task_registry_dir()
    )
    destination = destination_dir / "tinyBenchmarks.pkl"
    staged_asset = root / "lighteval" / "tinyBenchmarks.pkl"

    if staged_asset.is_file() and staged_asset.stat().st_size > 0:
        if not destination_dir.is_dir():
            raise RuntimeError(
                f"LightEval task registry directory {destination_dir} does not exist."
            )
        shutil.copyfile(staged_asset, destination)
        return

    if destination.is_file() and destination.stat().st_size > 0:
        return

    raise RuntimeError(
        "LightEval's task registry requires tinyBenchmarks.pkl even when the selected "
        "benchmark does not use TinyBenchmarks. Stage this file at "
        f"{root / 'lighteval' / 'tinyBenchmarks.pkl'} to prevent a runtime GitHub download."
    )


def _default_lighteval_task_registry_dir() -> Path:
    """Return the installed LightEval task-module directory without importing its registry."""
    spec = importlib.util.find_spec("lighteval")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "LightEval is not installed; cannot locate its task registry assets."
        )
    return Path(spec.origin).parent / "tasks" / "tasks"


def build_hf_offline_environment(
    job_spec_path: str | Path,
    test_data_root: str | Path = TEST_DATA_DIR,
    base_env: Mapping[str, str] | None = None,
    writable_datasets_cache_dir: str | Path | None = None,
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
    configure_hf_offline_environment(test_data_root, env, writable_datasets_cache_dir)
    return env
