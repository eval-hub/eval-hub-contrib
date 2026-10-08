"""Airgap dataset-source resolution for the WildGuard adapter.

On disconnected clusters a job submission carries ``test_data_ref`` (exactly one
of s3/pvc/git/hf, verified in eval-hub-sdk ``evalhub.models.api.TestDataRef``).
The eval-hub init container stages the referenced material into ``/test_data``
(read-only) before the adapter starts:

* ``pvc``  — the PVC is mounted directly at ``/test_data`` (no init container).
* ``s3``   — objects are downloaded from the bucket into ``/test_data``.
* ``git``  — the repository is cloned and checked out into ``/test_data``.
* ``hf``   — raw repository files are staged preserving repo structure (for
  ``allenai/wildguard`` these are parquet files).

Without this module the adapter calls ``load_dataset("allenai/wildguard")``
unconditionally, which contacts huggingface.co and fails on air-gapped clusters
even when the dataset is already staged under ``/test_data``.

Resolution order (mirrors the inspect adapter's offline module and the
``swebench``/``ragas``/``deepeval`` ``/test_data``-first patterns):

1. Explicit ``parameters.dataset_path`` (absolute, or relative to ``/test_data``).
2. A staged WildGuard dataset under ``/test_data`` — files whose name or path
   contains the requested split are preferred; hidden dirs (``.git``) are
   skipped; ambiguous matches raise instead of silently picking the first file.
3. Fail with a clear error — staged-data jobs never fall through to the Hub.

Expected staged format: JSONL (one object per line, bare JSON arrays also
accepted), JSON (a list or ``{"data": [...]}``), or Parquet files whose rows
carry the columns this adapter reads: ``prompt``, ``response`` and
``safety_label``. Note the Hub id ``allenai/wildguard`` is the *model* repo
(gated; the HF datasets API returns 401 for it) — the actual dataset is
``allenai/wildguardmix`` with different column names, so the Hub fallback
cannot serve this benchmark and the staged path is the supported route.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEST_DATA_DIR = "/test_data"
JOB_SPEC_PATH_ENV = "EVALHUB_JOB_SPEC_PATH"
DEFAULT_JOB_SPEC_PATH = "/meta/job.json"

#: File extensions recognised as staged dataset material.
DATASET_EXTENSIONS = (".parquet", ".json", ".jsonl")

#: Candidate directory names / layouts for staged WildGuard data.
_STAGED_DIR_CANDIDATES = ("wildguard", "allenai/wildguard", "allenai_wildguard")

#: Directories never descended into during dataset discovery.
_SKIPPED_DIRS = (".git", ".hg", ".svn", "__pycache__", ".ipynb_checkpoints")


class DatasetSourceError(RuntimeError):
    """Raised when no usable dataset source can be resolved."""


class AmbiguousDatasetError(DatasetSourceError):
    """Raised when staged discovery matches more than one dataset file."""


def _read_job_spec_parameters(job_spec_path: str | None = None) -> dict[str, Any]:
    """Best-effort read of ``parameters`` from the mounted job spec JSON."""
    path = job_spec_path or os.environ.get(JOB_SPEC_PATH_ENV, DEFAULT_JOB_SPEC_PATH)
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return {}
    try:
        with open(resolved, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(spec, dict):
        return {}
    params = spec.get("parameters")
    return params if isinstance(params, dict) else {}


def job_spec_requests_test_data(job_spec_path: str | None = None) -> bool:
    """True when the mounted job spec JSON includes a ``test_data_ref``."""
    path = job_spec_path or os.environ.get(JOB_SPEC_PATH_ENV, DEFAULT_JOB_SPEC_PATH)
    try:
        resolved = Path(path).resolve()
        with open(resolved, encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    ref = spec.get("test_data_ref") if isinstance(spec, dict) else None
    if not isinstance(ref, dict):
        return False
    return any(ref.get(key) for key in ("s3", "pvc", "git", "hf"))


def is_test_data_mount_usable(test_data_root: str | Path = TEST_DATA_DIR) -> bool:
    """True when ``/test_data`` exists and holds at least one non-hidden entry.

    (Named ``is_test_data_mount_usable`` so pytest does not collect it as a test.)
    """
    root = Path(test_data_root)
    try:
        if not root.is_dir():
            return False
        return any(not entry.name.startswith(".") for entry in root.iterdir())
    except OSError:
        return False


def configure_hf_offline_environment(
    hf_home: str = TEST_DATA_DIR, env: dict[str, str] | None = None
) -> None:
    """Pin HF caches to staged data and disable Hub network calls."""
    root = Path(hf_home)
    values = {
        "HF_HOME": str(root),
        "HF_HUB_CACHE": str(root / "hub"),
        "HF_DATASETS_CACHE": str(root / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_EVALUATE_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for key, value in values.items():
        os.environ[key] = value
        if env is not None:
            env[key] = value


def _find_dataset_files(root: Path) -> list[Path]:
    """Return staged dataset files under ``root`` (hidden-safe, shallow-first).

    Skips ``_SKIPPED_DIRS`` and dot-directories entirely (never descends into
    them). Split preference and ambiguity handling live in ``_pick_best_file``.
    """
    candidates: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in DATASET_EXTENSIONS:
                continue
            if any(
                part.startswith(".") or part in _SKIPPED_DIRS
                for part in path.relative_to(root).parts[:-1]
            ):
                continue
            candidates.append(path)
    except OSError:
        return []
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates


def _pick_best_file(files: list[Path], split: str | None, context: str) -> Path | None:
    """Pick the single best dataset file from ``files``, raising on ambiguity.

    Ranking: files whose stem or parent path mentions the requested split first,
    then shallowest. Only files tied at the best rank are ambiguous — silently
    picking the first would make results depend on filesystem order, so raise a
    clear error naming every tied candidate instead.
    """
    if not files:
        return None

    if split:

        def _rank(p: Path) -> tuple[int, int]:
            # Match the split against the file STEM, or the parent DIRECTORY
            # NAME exactly. A substring match on the full parent path is wrong:
            # a "test_data" mount (or a pytest tmp dir named "test_*") contains
            # "test" and would rank every file 0, defeating disambiguation.
            stem = p.stem.lower()
            parent = p.parent.name.lower()
            return (
                0 if (split.lower() in stem or parent == split.lower()) else 1,
                len(p.parts),
            )

        def _order(p: Path) -> tuple[int, int, str]:
            return (*_rank(p), str(p))

    else:

        def _rank(p: Path) -> tuple[int, int]:
            return (len(p.parts), 0)

        def _order(p: Path) -> tuple[int, int, str]:
            return (*_rank(p), str(p))

    ordered = sorted(files, key=_order)
    best = _rank(ordered[0])
    tied = [p for p in ordered if _rank(p) == best]
    if len(tied) > 1:
        names = ", ".join(str(p.relative_to(context)) for p in tied[:8])
        raise AmbiguousDatasetError(
            f"Ambiguous staged dataset under {context} (split={split!r}): {names}. "
            "Set parameters.dataset_path to the exact file to evaluate."
        )
    return tied[0]


def _find_staged_wildguard_dir(root: Path) -> Path | None:
    """Return a staged WildGuard directory under ``root`` if one exists."""
    for candidate in _STAGED_DIR_CANDIDATES:
        staged = root / candidate
        try:
            if staged.is_dir() and any(staged.iterdir()):
                return staged
        except OSError:
            continue
    return None


def _resolve_staged_dataset(
    parameters: dict[str, Any],
    test_data_root: str | Path = TEST_DATA_DIR,
    split: str | None = None,
) -> Path | None:
    """Resolve a staged dataset path under ``/test_data``.

    Order: explicit ``parameters.dataset_path`` (absolute, or relative to the
    root), a named WildGuard directory, then split-aware discovery over the
    whole root. Raises ``AmbiguousDatasetError`` when discovery matches more
    than one dataset file and no explicit path disambiguates.
    """
    root = Path(test_data_root)

    explicit = parameters.get("dataset_path")
    if isinstance(explicit, str) and explicit.strip():
        p = Path(explicit.strip())
        if not p.is_absolute():
            p = root / p
        try:
            if p.is_file():
                return p
            if p.is_dir() and any(p.iterdir()):
                found = _pick_best_file(_find_dataset_files(p), split, str(p))
                if found is not None:
                    return found
        except OSError:
            pass

    staged_dir = _find_staged_wildguard_dir(root)
    if staged_dir is not None:
        found = _pick_best_file(_find_dataset_files(staged_dir), split, str(staged_dir))
        if found is not None:
            return found

    return _pick_best_file(_find_dataset_files(root), split, str(root))


def should_use_staged_data(
    parameters: dict[str, Any],
    *,
    split: str | None = None,
    job_spec_path: str | None = None,
    test_data_root: str | Path = TEST_DATA_DIR,
) -> bool:
    """Whether staged ``/test_data`` data should take priority over the HF Hub.

    True when the ``/test_data`` mount is usable, when the job spec carries
    ``test_data_ref``, or when staged dataset material is actually present.

    The mount is the reliable signal on a cluster: eval-hub mounts ``/test_data``
    only for jobs that set ``test_data_ref`` and does NOT copy that key into the
    ``/meta/job.json`` the adapter reads (verified on-cluster), so the spec check
    alone never fires there.
    """
    if is_test_data_mount_usable(test_data_root):
        return True
    if job_spec_requests_test_data(job_spec_path):
        return True
    return _resolve_staged_dataset(parameters, test_data_root, split=split) is not None


def ensure_staged_data_ready(
    *,
    job_spec_path: str | None = None,
    test_data_root: str | Path = TEST_DATA_DIR,
) -> None:
    """Fail fast when the job expects staged data but ``/test_data`` is empty.

    A job submitted with ``test_data_ref`` whose init container failed to
    populate ``/test_data`` should produce a clear error, not a confusing HF
    download failure or an empty-dataset metric.
    """
    if not job_spec_requests_test_data(job_spec_path):
        return
    if not is_test_data_mount_usable(test_data_root):
        raise DatasetSourceError(
            f"Job spec includes test_data_ref but {test_data_root} is missing or "
            "empty. Ensure the test-data init container populated /test_data "
            "before the adapter starts."
        )


def load_rows_from_file(
    path: Path, num_examples: int | None = None
) -> list[dict[str, Any]]:
    """Load dataset rows from a staged parquet/json/jsonl file.

    Returns a list of row dicts. ``num_examples`` caps the number of rows
    returned (matching the Hub path's ``dataset.select`` behaviour). Unparseable
    JSONL lines are skipped with a warning; the total skipped count is logged.
    """
    if not path.is_file():
        raise DatasetSourceError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        # Deferred import — pyarrow/pandas may be absent in slim deployments.
        try:
            import pyarrow.parquet

            table = pyarrow.parquet.read_table(str(path))
            rows = table.to_pylist()
        except ImportError as exc:
            raise DatasetSourceError(
                "pyarrow is required to read staged parquet datasets"
            ) from exc
    elif suffix == ".jsonl":
        rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    logger.warning("Skipping unparseable JSONL line in %s", path)
                    continue
                # JSONL is one object per line, but tolerate a single JSON
                # array dumped onto one line (a common dataset export shape).
                if isinstance(obj, dict):
                    rows.append(obj)
                elif isinstance(obj, list):
                    rows.extend(r for r in obj if isinstance(r, dict))
        if skipped:
            logger.warning("Skipped %d unparseable JSONL lines in %s", skipped, path)
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            rows = [r for r in data["data"] if isinstance(r, dict)]
        else:
            raise DatasetSourceError(
                f"JSON dataset must be a list or {{'data': list}}: {path}"
            )
    else:
        raise DatasetSourceError(f"Unsupported dataset format: {path.suffix}")

    if num_examples is not None:
        rows = rows[:num_examples]
    return rows


def validate_staged_rows(rows: list[dict[str, Any]]) -> None:
    """Validate staged rows carry the columns the adapter reads, once, up front.

    A parquet/jsonl with different columns must fail with a clear schema error
    before evaluation starts — not per row, deep inside the worker threads.
    """
    required = ("prompt", "response", "safety_label")
    missing = [col for col in required if rows and col not in rows[0]]
    if missing:
        found = sorted(rows[0].keys()) if rows else []
        raise DatasetSourceError(
            f"Staged dataset is missing required column(s) {missing}. "
            f"Columns found: {found}. Expected columns: {list(required)} "
            "(WildGuard layout: prompt, response, safety_label)."
        )


def load_dataset_rows(
    parameters: dict[str, Any],
    *,
    hf_dataset_id: str,
    split: str = "test",
    num_examples: int | None = None,
    job_spec_path: str | None = None,
    test_data_root: str | Path = TEST_DATA_DIR,
) -> tuple[list[dict[str, Any]], str]:
    """Load WildGuard rows from staged ``/test_data`` data, or fail clearly.

    Returns ``(rows, source)`` where ``source`` is ``"staged"`` when rows came
    from ``/test_data`` (offline/airgap path). When a job requests staged data
    (``test_data_ref``) but no usable dataset material resolves, this raises a
    clear error instead of silently falling through to the Hub — a silent
    fall-through masks a broken staging step and, on an air-gapped cluster,
    turns a diagnosable config error into a confusing network failure. Explicit
    ``parameters.dataset_path`` is honoured; an ambiguous staged layout raises
    ``AmbiguousDatasetError`` naming every candidate.
    """
    if should_use_staged_data(
        parameters,
        split=split,
        job_spec_path=job_spec_path,
        test_data_root=test_data_root,
    ):
        ensure_staged_data_ready(
            job_spec_path=job_spec_path, test_data_root=test_data_root
        )
        staged = _resolve_staged_dataset(parameters, test_data_root, split=split)
        if staged is None:
            raise DatasetSourceError(
                f"Job requests staged data (test_data_ref) but no usable dataset "
                f"material was found under {test_data_root} (split={split!r}). "
                "Set parameters.dataset_path or fix the staged layout."
            )
        logger.info("Using staged dataset from %s (offline/airgap path)", staged)
        configure_hf_offline_environment(str(test_data_root))
        rows = load_rows_from_file(staged, num_examples=None)
        if num_examples is not None:
            rows = rows[:num_examples]
        if not rows:
            raise DatasetSourceError(f"Staged dataset at {staged} contains no rows")
        validate_staged_rows(rows)
        return rows, "staged"

    raise DatasetSourceError(
        "No staged dataset available and the Hub fallback is disabled for "
        "staged-data jobs. Provide parameters.dataset_path or stage the "
        f"dataset under {test_data_root}."
    )
