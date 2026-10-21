"""Resolve the agent skill under evaluation from material staged at /test_data.

An agent skill is a directory that contains a ``SKILL.md`` file (Agent Skills
specification, https://agentskills.io/). A job submission carries
``test_data_ref`` (git, hf, s3 or pvc on RHOAI 3.6+); the eval-hub init container
stages it read-only under ``/test_data`` before the adapter starts. Skills are
untrusted input, so this module only locates and validates a directory: it
never reads, imports or executes skill content.

Resolution order:

1. Explicit ``parameters.skill_path`` (absolute, or relative to ``/test_data``).
   It is authoritative: a path that is missing, has no ``SKILL.md`` or escapes
   ``/test_data`` fails the job instead of falling through to discovery.
2. ``/test_data/SKILL.md`` itself (the staged repository *is* the skill).
3. Exactly one directory below ``/test_data`` that contains ``SKILL.md``.
   Zero or several matches raise; the adapter never picks one silently.

Hidden directories (``.git`` ...) and symlinked directories are never
descended into, and every resolved path must stay inside the ``/test_data``
root, so a symlink inside a staged repository cannot point the scanner at
files outside the staged material.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEST_DATA_DIR = "/test_data"
SKILL_MANIFEST = "SKILL.md"

#: Discovery gives up below this depth; staged repositories keep skills shallow.
MAX_DISCOVERY_DEPTH = 4

#: Cap on candidate names echoed in an ambiguity error.
_MAX_LISTED_CANDIDATES = 10


class SkillSourceError(RuntimeError):
    """Raised when no usable skill directory can be resolved."""


class AmbiguousSkillError(SkillSourceError):
    """Raised when discovery finds more than one skill directory."""


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` is ``root`` or lies underneath it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_skill_dir(candidate: Path, root: Path) -> Path:
    """Return the resolved skill directory or raise ``SkillSourceError``."""
    resolved = candidate.resolve()
    if not _is_within(resolved, root):
        raise SkillSourceError(f"skill path escapes {root}: {candidate}")
    if not resolved.is_dir():
        raise SkillSourceError(f"skill path is not a directory: {candidate}")
    manifest = (resolved / SKILL_MANIFEST).resolve()
    if not manifest.is_file():
        raise SkillSourceError(f"no {SKILL_MANIFEST} in skill directory: {candidate}")
    if not _is_within(manifest, root):
        raise SkillSourceError(f"{SKILL_MANIFEST} resolves outside {root}: {candidate}")
    return resolved


def _discover(root: Path) -> list[Path]:
    """Find skill directories below ``root`` without following symlinked dirs."""
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = (
            sorted(d for d in dirs if not d.startswith("."))
            if depth < MAX_DISCOVERY_DEPTH
            else []
        )
        if SKILL_MANIFEST in files:
            try:
                found.append(_validate_skill_dir(Path(current), root))
            except SkillSourceError:
                logger.debug("Ignoring discovered skill dir that failed validation")
    return found


def resolve_skill_path(
    parameters: dict[str, Any], *, test_data_dir: str | None = None
) -> Path:
    """Locate the single skill directory staged for this job."""
    test_data_dir = test_data_dir or TEST_DATA_DIR
    root = Path(test_data_dir).resolve()
    if not root.is_dir():
        raise SkillSourceError(
            f"{test_data_dir} is not mounted: submit the job with test_data_ref pointing at the skill (git, hf, s3 or pvc)"
        )

    explicit = parameters.get("skill_path")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise SkillSourceError("parameters.skill_path must be a non-empty string")
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        return _validate_skill_dir(candidate, root)

    if (root / SKILL_MANIFEST).is_file():
        return _validate_skill_dir(root, root)

    found = _discover(root)
    if not found:
        raise SkillSourceError(
            f"no skill ({SKILL_MANIFEST}) found under {test_data_dir}"
        )
    if len(found) > 1:
        names = ", ".join(
            str(p.relative_to(root)) for p in found[:_MAX_LISTED_CANDIDATES]
        )
        suffix = ", ..." if len(found) > _MAX_LISTED_CANDIDATES else ""
        raise AmbiguousSkillError(
            f"{len(found)} skills found under {test_data_dir} ({names}{suffix}); set parameters.skill_path to choose one"
        )
    return found[0]
