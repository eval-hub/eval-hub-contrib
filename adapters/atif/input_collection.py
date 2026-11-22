"""Bounded ATIF collection loading and source identities (no storage transport)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from evalhub.models.atif import Trajectory
from pydantic import ValidationError

MAX_COLLECTION_BYTES = 64 * 1024 * 1024
# Producer provenance the ATIF model does not declare is preserved from the raw payload
# before validation, so a rejected file still reports what its producer wrote. The model
# stays the contract: the fields are not moved into the trajectory.
MAX_UNDECLARED_ROOT_BYTES = 4 * 1024
DECLARED_ROOT_FIELDS = frozenset(Trajectory.model_fields)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ATIFManifestEntry:
    path: Path
    task_index: int
    rollout_index: int
    expected_sha256: str | None


class ATIFLoadError(ValueError):
    """Invalid or unsafe ATIF input, with a safe diagnostic code."""

    def __init__(self, message: str, code: str = "invalid_trajectory") -> None:
        super().__init__(message)
        self.code = code


class CollectionBudgetError(ATIFLoadError):
    """Fatal collection-wide resource limit, independent of invalid-file policy."""


class UnsafeManifestRowError(ValueError):
    """A manifest row that breaks a boundary rather than carrying bad content.

    An unsafe path or a repeated identity makes the manifest's own account of the
    export untrustworthy, so no ``invalid_input_policy`` may skip it: skipping would
    score the rest of an export whose file list cannot be believed.
    """


class ATIFDigestMismatchError(ATIFLoadError):
    """Provenance failure governed by ``partial_result_policy``, not by
    ``invalid_input_policy``: the bytes were readable, they just did not match
    the digest the producer declared for them."""


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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _bounded_provenance(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep producer fields small enough to belong in a report.

    A file is bounded by ``max_file_bytes``, but the report copies what is preserved, so
    an oversized blob is replaced by its key names rather than duplicated into MLflow.
    """
    encoded = json.dumps(fields, default=str)
    if len(encoded) <= MAX_UNDECLARED_ROOT_BYTES:
        return fields
    return {"_keys": sorted(fields), "_preserved": False, "_bytes": len(encoded)}


def load_manifest(
    path: Path,
    root: Path,
    *,
    max_entries: int,
    invalid_input_policy: str = "fail_fast",
    outcomes: list[dict[str, Any]] | None = None,
) -> list[ATIFManifestEntry]:
    """Read Gym's JSONL manifest and safely resolve its ordered file entries.

    Rows are validated one at a time. ``fail_fast`` names the offending row and
    aborts; ``skip_invalid`` records that row through ``outcomes`` as an invalid
    input and keeps the remaining entries, so a partially corrupt export still
    accounts for what it can. An empty manifest is treated the same way. The
    manifest object itself is never charged to trajectory or collection-byte
    accounting.
    """
    if invalid_input_policy not in {"fail_fast", "skip_invalid"}:
        raise ATIFLoadError("invalid_input_policy must be fail_fast or skip_invalid")
    try:
        raw = read_bounded(path, MAX_MANIFEST_BYTES)
    except (OSError, ATIFLoadError) as exc:
        raise ATIFLoadError(
            f"cannot read ATIF manifest: {exc}", "invalid_manifest"
        ) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ATIFLoadError("ATIF manifest is not UTF-8", "invalid_manifest") from exc
    root = root.resolve(strict=True)
    entries: list[ATIFManifestEntry] = []
    seen_paths: set[Path] = set()
    seen_rollouts: set[tuple[int, int]] = set()
    required = {"trajectory_path", "_ng_task_index", "_ng_rollout_index"}
    allowed = required | {"expected_sha256"}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if (
                not isinstance(row, dict)
                or not required <= row.keys()
                or row.keys() - allowed
            ):
                raise ValueError("manifest row has an invalid field set")
            relative = row["trajectory_path"]
            if (
                not isinstance(relative, str)
                or not relative.strip()
                or "\\" in relative
                or ":" in relative
            ):
                raise ValueError("unsafe manifest path")
            posix_path = PurePosixPath(relative)
            if (
                posix_path.is_absolute()
                or ".." in posix_path.parts
                or "." in posix_path.parts
            ):
                raise UnsafeManifestRowError("unsafe manifest path")
            task_index, rollout_index = row["_ng_task_index"], row["_ng_rollout_index"]
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or task_index < 0
                or isinstance(rollout_index, bool)
                or not isinstance(rollout_index, int)
                or rollout_index < 0
            ):
                raise ValueError("manifest indices must be non-negative integers")
            digest = row.get("expected_sha256")
            if digest is not None and (
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError(
                    "expected_sha256 must be a lowercase SHA-256 hex digest"
                )
            candidate = root.joinpath(*posix_path.parts)
            if any(part.is_symlink() for part in (candidate, *candidate.parents)):
                raise UnsafeManifestRowError(
                    "unsafe manifest path: symlinks are not allowed"
                )
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError:
                raise UnsafeManifestRowError(
                    "unsafe manifest path: outside the input root"
                ) from None
            if not resolved.is_file():
                raise UnsafeManifestRowError(
                    "manifest trajectory is not a regular file"
                )
            if resolved in seen_paths or (task_index, rollout_index) in seen_rollouts:
                # A duplicate is one row the producer wrote twice: `skip_invalid` drops
                # the second and scores the first, `fail_fast` names it. Only a path that
                # cannot be trusted is fatal regardless of policy.
                raise ValueError("manifest contains duplicate path or rollout identity")
        except UnsafeManifestRowError as exc:
            detail = f"invalid ATIF manifest row {line_number}: {exc}"
            raise ATIFLoadError(detail, "invalid_manifest") from None
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
            detail = f"invalid ATIF manifest row {line_number}: {exc}"
            if invalid_input_policy == "fail_fast":
                raise ATIFLoadError(detail, "invalid_manifest") from None
            if outcomes is not None:
                outcomes.append(
                    {
                        "path": str(path),
                        "status": "invalid",
                        "reason": "invalid_manifest",
                        "detail": detail,
                    }
                )
            continue
        seen_paths.add(resolved)
        seen_rollouts.add((task_index, rollout_index))
        entries.append(ATIFManifestEntry(resolved, task_index, rollout_index, digest))
        if len(entries) > max_entries:
            raise ATIFLoadError(
                f"ATIF manifest exceeds maximum of {max_entries} entries", "file_limit"
            )
    if not entries:
        detail = "ATIF manifest contains no entries"
        if invalid_input_policy == "fail_fast":
            raise ATIFLoadError(detail, "invalid_manifest")
        if outcomes is not None:
            outcomes.append(
                {
                    "path": str(path),
                    "status": "invalid",
                    "reason": "invalid_manifest",
                    "detail": detail,
                }
            )
    return entries


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
    manifest_entries: dict[Path, ATIFManifestEntry] | None = None,
    partial_result_policy: str = "fail_fast",
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """Load and validate every collection entry.

    ``allow_empty`` lets a manifest-driven collection complete with no accepted
    trajectories instead of failing, so an export whose entries are all invalid
    reports unscored with its invalid counts rather than looking like a crash.
    """
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
    if partial_result_policy not in {"fail_fast", "skip_failed_trajectory"}:
        raise ATIFLoadError(
            "partial_result_policy must be fail_fast or skip_failed_trajectory"
        )
    if not files:
        if allow_empty:
            return []
        raise ATIFLoadError("No ATIF JSON trajectory files were found")
    if len(files) > max_files:
        raise ATIFLoadError(
            f"ATIF input contains {len(files)} files; maximum is {max_files}"
        )
    trajectories: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_bytes = total_steps = 0
    digest_withheld = 0
    for path in files:
        manifest_entry = (manifest_entries or {}).get(path.resolve())
        manifest_provenance = (
            {
                "task_index": manifest_entry.task_index,
                "rollout_index": manifest_entry.rollout_index,
                "expected_sha256": manifest_entry.expected_sha256,
                # Set only once the comparison below succeeded: a declared digest is
                # a claim, not a verdict.
                "digest_verified": False,
            }
            if manifest_entry
            else None
        )
        outcome = {"path": str(path)}
        if manifest_provenance:
            outcome["nemo_gym_manifest"] = manifest_provenance
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
            digest = hashlib.sha256(raw).hexdigest()
            if (
                manifest_entry
                and manifest_entry.expected_sha256
                and digest != manifest_entry.expected_sha256
            ):
                if partial_result_policy == "fail_fast":
                    raise ATIFDigestMismatchError(
                        f"manifest SHA-256 mismatch for {path}", "digest_mismatch"
                    )
                # skip_failed_trajectory: the entry contributes no score and is
                # counted as a failed trajectory, not as an invalid input.
                outcome.update(status="failed", reason="digest_mismatch")
                digest_withheld += 1
                continue
            if manifest_provenance is not None and manifest_entry.expected_sha256:
                manifest_provenance["digest_verified"] = True
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
            # Recursive discovery meets files that are not trajectories at all, so a
            # collection of several files ignores one. A manifest-listed file is not
            # that case: the producer named it, so it has to reach schema validation and
            # be reported as invalid input rather than silently dropped.
            if not candidate and len(files) > 1 and manifest_entry is None:
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
            undeclared_root = {
                key: value
                for key, value in data.items()
                if key not in DECLARED_ROOT_FIELDS
            }
            if undeclared_root:
                outcome["producer_undeclared_root"] = _bounded_provenance(
                    undeclared_root
                )
            pending = [data]
            while pending:
                node = pending.pop()
                version = node.get("schema_version")
                supported = sorted(
                    supported_versions,
                    key=lambda item: tuple(
                        int(part) for part in re.findall(r"\d+", item)
                    ),
                )
                version_range = (
                    f"{supported[0]} through {supported[-1]}" if supported else "none"
                )
                if not isinstance(version, str) or version not in supported_versions:
                    detail = (
                        "Missing schema_version"
                        if not isinstance(version, str) or not version
                        else f"Unsupported schema_version {version!r}"
                    )
                    raise ATIFLoadError(
                        f"{detail}; supported range is {version_range}",
                        "unsupported_schema",
                    )
                pending.extend(node.get("subagent_trajectories") or [])
            parsed = Trajectory.model_validate(data)
            validate_limits(parsed, path)
            trajectory = parsed.model_dump(mode="json")
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
                    # Producer provenance belongs in ATIF's declared extra object;
                    # the SDK rejects undeclared root fields, so this is the only
                    # producer metadata the adapter can preserve verbatim.
                    "producer_extra": node.get("extra"),
                    **(
                        {"nemo_gym_manifest": manifest_provenance}
                        if manifest_provenance
                        else {}
                    ),
                }
                node["_parent_id"] = parent_id
                step_count += len(node["steps"])
                source_pending.extend(
                    (child, f"{selector}/subagent_trajectories/{i}", node_id)
                    for i, child in enumerate(node.get("subagent_trajectories") or [])
                )
        except ATIFDigestMismatchError:
            raise
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
    # An export whose every entry was withheld for provenance is a complete, correctly
    # scored result with nothing in it: the adapter reports the counts and withholds the
    # score. `invalid_input_policy` governs malformed input, not a producer's wrong
    # digest, so that case may not be turned into a failure by `fail_fast`.
    if not trajectories and not allow_empty and not digest_withheld:
        raise ATIFLoadError("No usable ATIF trajectories remain after input validation")
    return trajectories
