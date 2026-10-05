"""Validation of the operation protocol and the API's data reference shapes."""

from __future__ import annotations

import math
from typing import Any

SOURCES = frozenset({"eval_job", "mlflow", "oci", "s3", "pvc", "git", "hf"})
CALIBRATION_SOURCES = frozenset({"s3", "pvc", "git", "hf"})


def object_value(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def text_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def source_type(ref: Any, *, calibration: bool = False) -> str:
    ref = object_value(ref, "data reference")
    keys = SOURCES.intersection(ref)
    if len(keys) != 1:
        raise ValueError("A data reference must contain exactly one source")
    key = next(iter(keys))
    if calibration and key not in CALIBRATION_SOURCES:
        raise ValueError("calibration_data_ref supports s3, pvc, git, or hf")
    object_value(ref[key], key)
    metadata = {"type", "resolved_sha"}
    if calibration:
        metadata.add("data_config")
        if "data_config" in ref:
            object_value(ref["data_config"], "calibration_data_ref[].data_config")
    unknown = set(ref) - SOURCES - metadata
    if unknown:
        raise ValueError(f"Unknown data reference fields: {sorted(unknown)}")
    return key


def ci_config(raw: Any) -> dict:
    config = dict(object_value(raw, "confidence_interval"))
    source = object_value(config.get("results_data_ref"), "results_data_ref")
    kind = source_type(source)
    if kind == "eval_job":
        text_value(source[kind].get("id"), "results_data_ref.eval_job.id")
    calibration = config.get("calibration_data_ref")
    # The API specifies a list; also accept the single reference in the runtime proposal.
    if isinstance(calibration, dict):
        calibration = [calibration]
    if not isinstance(calibration, list) or not calibration:
        raise ValueError("calibration_data_ref must contain at least one reference")
    for ref in calibration:
        source_type(ref, calibration=True)
    config["calibration_data_ref"] = calibration
    alpha = config.get("significance_level")
    if isinstance(alpha, bool):
        raise ValueError("significance_level must be a number between 0 and 1")
    try:
        alpha = float(alpha)
    except (ValueError, TypeError) as exc:
        raise ValueError("significance_level must be a number between 0 and 1") from exc
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("significance_level must be strictly between 0 and 1")
    config["significance_level"] = alpha
    nested_threads = source.get("eval_job", {}).get("num_parallel_threads")
    threads = config.get(
        "num_parallel_threads", nested_threads if nested_threads is not None else 1
    )
    if (
        nested_threads is not None
        and "num_parallel_threads" in config
        and threads != nested_threads
    ):
        raise ValueError("Conflicting num_parallel_threads settings")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("num_parallel_threads must be an integer between 1 and 64")
    config["num_parallel_threads"] = threads
    return config


def operations(parameters: dict) -> list[tuple[str, dict]]:
    raw = parameters.get("operations")
    if raw is None and "results_data_ref" in parameters:
        # Compatibility with eval-hub/postprocessing.ToEvaluationJob's current mapping.
        config = dict(parameters)
        config.pop("operation_order", None)
        raw = {"confidence_interval": config}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("parameters.operations must be a nonempty object")

    parsed = {}
    for name, raw_config in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Operation names must be nonempty strings")
        parsed[name] = dict(object_value(raw_config, name))

    if "operation_order" not in parameters:
        return list(parsed.items())

    order = parameters["operation_order"]
    if not isinstance(order, list) or any(not isinstance(name, str) for name in order):
        raise ValueError("parameters.operation_order must be an array of operation names")
    names = set(parsed)
    if (
        len(order) != len(names)
        or len(set(order)) != len(order)
        or set(order) != names
    ):
        raise ValueError("parameters.operation_order must list each operation name exactly once")
    return [(name, parsed[name]) for name in order]
