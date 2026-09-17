"""Pin Hugging Face caches to Eval Hub staged test data at ``/test_data``.

When the init container (or PVC) populates ``/test_data``, the inspect subprocess
uses that tree as ``HF_HOME`` so ``datasets`` / the Hub resolve locally first.
On cache miss, libraries may still attempt a Hub download (or fail on air-gapped
clusters). No ``HF_*_OFFLINE`` flags are set by this module.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TEST_DATA_DIR = "/test_data"

# Ensure subprocess env does not inherit forced-offline mode when using staged caches.
_OFFLINE_ENV_KEYS = (
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_EVALUATE_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


def _test_data_mount_usable(test_data_root: str | Path | None = None) -> bool:
    root = Path(test_data_root if test_data_root is not None else TEST_DATA_DIR)
    try:
        if not root.is_dir():
            return False
        return any(root.iterdir())
    except OSError:
        return False


def should_pin_hf_cache_to_test_data(
    test_data_root: str | Path | None = None,
) -> bool:
    """True when ``/test_data`` exists and is non-empty (staged benchmark data present)."""
    return _test_data_mount_usable(test_data_root)


def configure_hf_staged_cache_environment(
    hf_home: str,
    env: dict[str, str] | None = None,
) -> None:
    """Point Hugging Face cache dirs at staged data; allow Hub fallback on cache miss."""
    root = Path(hf_home)
    values = {
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(root / "hub"),
        "HF_DATASETS_CACHE": str(root / "datasets"),
    }
    for key, value in values.items():
        os.environ[key] = value
        if env is not None:
            env[key] = value
    for key in _OFFLINE_ENV_KEYS:
        os.environ.pop(key, None)
        if env is not None:
            env.pop(key, None)


def seed_hf_staged_cache_from_mount() -> None:
    """Best-effort cache pin at import time when ``/test_data`` is already populated."""
    if not should_pin_hf_cache_to_test_data():
        return
    configure_hf_staged_cache_environment(TEST_DATA_DIR)
    logger.info(
        "HF staged cache: HF_HOME=%s (local /test_data first, Hub fallback on miss)",
        TEST_DATA_DIR,
    )


def _seed_hf_staged_cache_before_adapter_import() -> None:
    try:
        seed_hf_staged_cache_from_mount()
    except Exception as exc:  # noqa: BLE001 — import-time seed must never block startup
        print(f"WARNING: HF staged cache seed skipped: {exc}", file=sys.stderr)


_seed_hf_staged_cache_before_adapter_import()
