"""Dataset discovery and reading (CSV, JSONL, JSON)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from evalhub.adapter import JobSpec

logger = logging.getLogger(__name__)

Records = list[dict[str, Any]]


def _read_csv(path: Path) -> Records:
    return pd.read_csv(path).to_dict("records")


def _read_jsonl(path: Path) -> Records:
    with open(path) as fh:
        return [json.loads(line) for line in (raw.strip() for raw in fh) if line]


def _read_json(path: Path) -> Records:
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


# format -> (file glob, reader)
_READERS: dict[str, tuple[str, Callable[[Path], Records]]] = {
    "csv": ("*.csv", _read_csv),
    "jsonl": ("*.jsonl", _read_jsonl),
    "json": ("*.json", _read_json),
}


def load_dataset(data_dir: str, fmt: str) -> Records:
    """Load every file of the given format from ``data_dir``, in name order."""
    if fmt not in _READERS:
        raise ValueError(
            f"Unsupported dataset_format: {fmt!r}. Use csv, jsonl, or json."
        )
    pattern, read = _READERS[fmt]
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise ValueError(f"No {fmt.upper()} files found in {data_dir}")

    records: Records = []
    for path in files:
        records.extend(read(path))

    if not records:
        raise ValueError(f"No records loaded from {data_dir} (format={fmt})")

    logger.info("Loaded %d records from %s (format=%s)", len(records), data_dir, fmt)
    return records


def resolve_data_dir(config: JobSpec) -> str:
    """Find the directory containing test data, checking standard mount paths first."""
    for candidate in ("/test_data", "/data"):
        p = Path(candidate)
        if p.is_dir() and any(p.iterdir()):
            logger.info("Using data from %s", candidate)
            return candidate

    data_dir = config.parameters.get("data_dir")
    if data_dir and Path(data_dir).is_dir():
        logger.info("Using data_dir from parameters: %s", data_dir)
        return data_dir

    raise ValueError(
        "No input data found: mount data under /test_data or /data, "
        "or set parameters.data_dir"
    )
