"""Bounded ATIF collection loading and source identities (no storage transport)."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

from evalhub.models.atif import Trajectory
from pydantic import ValidationError

MAX_COLLECTION_BYTES = 64 * 1024 * 1024


class ATIFLoadError(ValueError):
    """Invalid or unsafe ATIF input, with a safe diagnostic code."""

    def __init__(self, message: str, code: str = "invalid_trajectory") -> None:
        super().__init__(message)
        self.code = code


class CollectionBudgetError(ATIFLoadError):
    """Fatal collection-wide resource limit, independent of invalid-file policy."""


def read_bounded(path: Path, limit: int) -> bytes:
    """Do not follow symlinks, including in ancestor directories, or read devices."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ATIFLoadError("Symlinks are not allowed in input paths", "unsafe_path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ATIFLoadError("Input must be a regular file", "unsafe_path")
        if info.st_size > limit:
            raise ATIFLoadError(f"Input exceeds maximum size of {limit} bytes")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ATIFLoadError(f"Input exceeds maximum size of {limit} bytes")
    return raw


def read_bounded_config(path: Path, limit: int) -> bytes:
    """Bounded read for operator-mounted configuration files.

    Kubernetes projected volumes (ConfigMap, Secret) expose mounted entries
    as symlinks, so symlinks are followed here. The opened target must still
    be a regular file within ``limit`` bytes. Use :func:`read_bounded` for
    untrusted trajectory input, where symlinks stay forbidden.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ATIFLoadError("Configuration must be a regular file", "unsafe_path")
    if info.st_size > limit:
        os.close(fd)
        raise ATIFLoadError(f"Configuration exceeds maximum size of {limit} bytes")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ATIFLoadError(f"Configuration exceeds maximum size of {limit} bytes")
    return raw


def load_collection(
    files: list[Path],
    *,
    max_file_bytes: int,
    max_files: int,
    max_collection_bytes: int,
    max_total_steps: int,
    invalid_input_policy: str,
    supported_versions: frozenset[str],
    validate_limits: Callable[[Trajectory, Path], None],
    source_path: Callable[[Path], str],
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for name, value in (
        ("max_file_bytes", max_file_bytes),
        ("max_trajectory_files", max_files),
        ("max_collection_bytes", max_collection_bytes),
        ("max_total_steps", max_total_steps),
    ):
        if value < 1:
            raise ATIFLoadError(f"{name} must be positive")
    if invalid_input_policy not in {"fail_fast", "skip_invalid"}:
        raise ATIFLoadError("invalid_input_policy must be fail_fast or skip_invalid")
    if not files:
        raise ATIFLoadError("No ATIF JSON trajectory files were found")
    if len(files) > max_files:
        raise ATIFLoadError(
            f"ATIF input contains {len(files)} files; maximum is {max_files}"
        )
    trajectories: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_bytes = total_steps = 0
    for path in files:
        outcome = {"path": str(path)}
        outcomes.append(outcome)
        try:
            # The budget includes ignored and invalid files, not just accepted input.
            size = path.lstat().st_size
            if total_bytes + size > max_collection_bytes:
                raise CollectionBudgetError("Collection exceeds max_collection_bytes")
            remaining = max_collection_bytes - total_bytes
            total_bytes += size
            raw = read_bounded(path, min(max_file_bytes, remaining))
            total_bytes += max(0, len(raw) - size)
            data = json.loads(raw)
            candidate = path.name == "trajectory.json" or path.name.endswith(
                ".atif.json"
            )
            candidate = (
                candidate
                or isinstance(data, dict)
                and any(
                    key in data
                    for key in (
                        "schema_version",
                        "atif_schema_version",
                        "steps",
                        "agent",
                    )
                )
            )
            if not candidate and len(files) > 1:
                outcome.update(status="ignored", reason="not_atif")
                continue
            if not isinstance(data, dict):
                raise ATIFLoadError("top-level JSON value must be an object")
            alias = data.pop("atif_schema_version", None)
            if alias is not None and data.get("schema_version", alias) != alias:
                raise ATIFLoadError(
                    "conflicting schema_version and atif_schema_version values",
                    "conflicting_schema",
                )
            data.setdefault("schema_version", alias)
            pending = [data]
            while pending:
                node = pending.pop()
                version = node.get("schema_version")
                if not isinstance(version, str) or version not in supported_versions:
                    raise ATIFLoadError(
                        "Missing or unsupported schema_version", "unsupported_schema"
                    )
                pending.extend(node.get("subagent_trajectories") or [])
            parsed = Trajectory.model_validate(data)
            validate_limits(parsed, path)
            trajectory = parsed.model_dump(mode="json")
            digest = hashlib.sha256(raw).hexdigest()
            uri = source_path(path)
            source_pending: list[tuple[dict[str, Any], str, str | None]] = [
                (trajectory, "", None)
            ]
            node_ids: set[str] = set()
            step_count = 0
            while source_pending:
                node, selector, parent_id = source_pending.pop()
                producer_id = node.get("trajectory_id")
                # A source-qualified ID also prevents collision for identical ID-less files.
                node_id = (
                    producer_id
                    if producer_id is not None
                    else "sha256:"
                    + hashlib.sha256(f"{uri}:{digest}:{selector}".encode()).hexdigest()
                )
                if node_id in seen_ids or node_id in node_ids:
                    raise ATIFLoadError(
                        "duplicate trajectory_id in accepted collection",
                        "duplicate_identity",
                    )
                node_ids.add(node_id)
                node["trajectory_id"] = node_id
                node["_source_path"] = uri
                node["_provenance"] = {
                    "source_uri": uri,
                    "local_path": str(path),
                    "sha256": digest,
                    "selector": selector,
                    "producer_trajectory_id": producer_id,
                }
                node["_parent_id"] = parent_id
                step_count += len(node["steps"])
                source_pending.extend(
                    (child, f"{selector}/subagent_trajectories/{i}", node_id)
                    for i, child in enumerate(node.get("subagent_trajectories") or [])
                )
        except CollectionBudgetError:
            raise
        except (
            ATIFLoadError,
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
            AttributeError,
            RecursionError,
            ValidationError,
        ) as exc:
            # Do not retain validation errors containing raw producer data.
            reason = getattr(
                exc,
                "code",
                {
                    json.JSONDecodeError: "invalid_json",
                    ValidationError: "invalid_schema",
                }.get(type(exc), "unreadable_or_invalid_input"),
            )
            outcome.update(status="invalid", reason=reason)
            if invalid_input_policy == "fail_fast":
                message = str(exc) if isinstance(exc, ATIFLoadError) else reason
                raise ATIFLoadError(
                    f"invalid ATIF trajectory {path}: {message}", reason
                ) from None
            continue
        # Resource limits are fatal, never silently converted into partial acceptance.
        total_steps += step_count
        if total_steps > max_total_steps:
            raise ATIFLoadError(f"Collection exceeds {max_total_steps} total steps")
        seen_ids.update(node_ids)
        outcome.update(status="accepted", trajectory_id=trajectory["trajectory_id"])
        trajectories.append(trajectory)
    if not trajectories:
        raise ATIFLoadError("No usable ATIF trajectories remain after input validation")
    return trajectories
