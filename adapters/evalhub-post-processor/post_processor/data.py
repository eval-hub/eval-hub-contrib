"""Read per-example numeric scores and form the three arrays required by PPI.

Aggregate benchmark metrics are deliberately never treated as observations.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMATS = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".parquet": "parquet",
}
ROLES = {
    "sample_id",
    "metric",
    "prediction",
    "label",
    "benchmark_id",
    "provider_id",
}


def within(root: Path, name: str) -> Path:
    """Resolve an untrusted relative artifact path under its download directory."""
    if not isinstance(name, str) or "\\" in name or Path(name).is_absolute():
        raise ValueError("Artifact paths must be relative POSIX paths")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes the download directory")
    return result


def field_value(row: dict, key: str) -> Any:
    # Prefer literal keys: benchmark metric names frequently contain dots.
    if key in row:
        return row[key]
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def numeric(value: Any, name: str) -> float:
    if isinstance(value, dict) and set(value) == {"value"}:
        value = value["value"]
    if value is None or isinstance(value, (bool, list, dict)):
        raise ValueError(f"{name} must be a finite numeric score")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric score") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric score")
    return number


@dataclass
class Table:
    rows: list[dict]
    config: dict = field(default_factory=dict)
    primary_score: str | None = None
    estimand: str = "mean"

    def get(self, row: dict, role: str) -> Any:
        columns = self.config.get("columns", {})
        if role in columns:
            value = field_value(row, columns[role])
        else:
            value = row.get(role)
            if value is None and role == "sample_id":
                value = row.get("doc_id")
        return self.map_value(value, role)

    def map_value(self, value: Any, role: str) -> Any:
        mapping = self.config.get("value_mappings", {}).get(role)
        if mapping is None or value is None:
            return value
        if not isinstance(value, str) or value not in mapping:
            raise ValueError(f"Unmapped {role} value; every category needs a value_mappings entry")
        return mapping[value]


def read_table(root: Path, config: dict | None = None) -> Table:
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("data_config must be an object")
    fmt = config.get("format", "auto")
    if fmt not in {"auto", *FORMATS.values()}:
        raise ValueError(f"Unsupported data format {fmt!r}")
    columns = config.get("columns", {})
    if not isinstance(columns, dict) or set(columns) - ROLES:
        raise ValueError(f"columns must map these roles to field names: {sorted(ROLES)}")
    if any(not isinstance(v, str) or not v for v in columns.values()):
        raise ValueError("Column mappings must be nonempty field names")
    mappings = config.get("value_mappings", {})
    if not isinstance(mappings, dict) or set(mappings) - {"prediction", "label"}:
        raise ValueError("value_mappings supports prediction and label roles")
    for role, mapping in mappings.items():
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(k, str) for k in mapping)
        ):
            raise ValueError(f"value_mappings.{role} must be a nonempty string-to-number map")
        for value in mapping.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"value_mappings.{role} outputs must be finite numeric scores")
            numeric(value, f"value_mappings.{role} output")
    selection = config.get("selection", {})
    if not isinstance(selection, dict) or set(selection) - {
        "benchmark_id",
        "provider_id",
        "metric",
    }:
        raise ValueError("selection supports benchmark_id, provider_id, and metric")
    if "path" in config:
        root = within(root if root.is_dir() else root.parent, config["path"])
    if not root.exists():
        raise ValueError("Selected data path does not exist")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    table = Table([], config=config)
    primary_scores: set[str] = set()
    estimands: set[str] = set()
    for path in files:
        if path.is_symlink():
            raise ValueError("Data artifacts must not contain symbolic links")
        kind = FORMATS.get(path.suffix.lower())
        if kind is None or (fmt != "auto" and fmt != kind and path.name != "manifest.json"):
            continue
        records: Any
        if kind == "json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                primary = obj.get("primary_score")
                if isinstance(primary, dict) and isinstance(primary.get("metric"), str):
                    primary_scores.add(primary["metric"])
                if "estimand" in obj:
                    estimands.add(obj["estimand"])
                records = obj.get("samples", obj.get("records", [obj]))
            else:
                records = obj
        elif kind == "jsonl":
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif kind == "csv":
            with path.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
        else:
            import pyarrow.parquet as pq

            records = pq.read_table(path).to_pylist()
        if not isinstance(records, list):
            raise ValueError(f"Expected a record array in {path.name}")
        # Artifacts often contain aggregate summaries alongside sample files.
        if not any(
            isinstance(row, dict) and table.get(row, "sample_id") is not None for row in records
        ):
            continue
        if any(not isinstance(row, dict) or table.get(row, "sample_id") is None for row in records):
            raise ValueError(f"Missing sample_id in a per-example data file: {path.name}")
        table.rows.extend(records)
    if len(primary_scores) > 1:
        raise ValueError("External result data has conflicting primary_score declarations")
    table.primary_score = next(iter(primary_scores), None)
    if estimands - {"mean"}:
        raise ValueError(
            "The initial PPI operation supports mean per-example scores; this artifact declares another estimand"
        )
    if not table.rows:
        raise ValueError(
            "No per-example scores found; aggregate metrics cannot be used for PPI. Supply sample_id and score columns."
        )
    return table


def sample_id(table: Table, row: dict) -> str:
    value = table.get(row, "sample_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
        raise ValueError("sample_id must be a nonempty string or integer")
    return str(value)


def matching(table: Table, row: dict, target: dict, *, shared: bool) -> bool:
    selection = table.config.get("selection", {})
    selectors = {
        role: selection.get(role, table.get(row, role))
        for role in ("benchmark_id", "provider_id", "metric")
    }
    if shared and any(selectors[role] in (None, "") for role in ("benchmark_id", "provider_id")):
        raise ValueError(
            "Calibration shared by multiple benchmarks needs both benchmark_id and provider_id "
            "in each row or selection"
        )
    for role, value in selectors.items():
        if value is None or value == "":
            continue
        if str(value) != str(target.get(role, "")):
            return False
    return True


def prediction(table: Table, row: dict, metric: str) -> Any:
    value = table.get(row, "prediction")
    if value is not None or "prediction" in table.config.get("columns", {}):
        return value
    # Existing frameworks often store per-example scores by metric name.
    for scores in (row, row.get("metrics", {}), row.get("scores", {})):
        if isinstance(scores, dict) and metric in scores:
            return table.map_value(scores[metric], "prediction")
    return None


def ppi_arrays(
    results: Table, calibration: list[Table], target: dict, *, shared: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = target["metric"]
    scores: dict[str, float] = {}
    for row in results.rows:
        if not matching(results, row, target, shared=False):
            continue
        key = sample_id(results, row)
        if key in scores:
            raise ValueError(f"Duplicate evaluation sample_id {key!r}")
        scores[key] = numeric(prediction(results, row, metric), f"{metric} prediction")
    labels: list[float] = []
    predicted: list[float] = []
    seen: set[str] = set()
    for table in calibration:
        for row in table.rows:
            if not matching(table, row, target, shared=shared):
                continue
            key = sample_id(table, row)
            if key in seen:
                raise ValueError(f"Duplicate calibration sample_id {key!r}")
            seen.add(key)
            label = numeric(table.get(row, "label"), f"{metric} calibration label")
            estimate = prediction(table, row, metric)
            if estimate is None:
                if key not in scores:
                    raise ValueError(
                        "Calibration needs a prediction column or a matching evaluation sample_id"
                    )
                estimate = scores[key]
            estimate = numeric(estimate, f"{metric} calibration prediction")
            if key in scores and not math.isclose(
                estimate, scores[key], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(
                    "Calibration and evaluation predictions disagree for the same sample_id"
                )
            labels.append(label)
            predicted.append(estimate)
    # Calibration rows are labeled observations, never also unlabeled observations.
    unlabeled = [score for key, score in scores.items() if key not in seen]
    if len(labels) < 2 or len(unlabeled) < 2:
        raise ValueError(
            "PPI needs at least two paired calibration examples and two remaining unlabeled examples"
        )
    return np.asarray(labels), np.asarray(predicted), np.asarray(unlabeled)
