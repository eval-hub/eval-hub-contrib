"""Airgap dataset-source tests for the WildGuard adapter.

Tests the staged-data resolution (S3/PVC/git/HF staging into /test_data),
ambiguity handling, hidden-dir skipping, and split-aware discovery without any
network access.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from _dataset_source import (
    AmbiguousDatasetError,
    DatasetSourceError,
    configure_hf_offline_environment,
    ensure_staged_data_ready,
    is_test_data_mount_usable,
    job_spec_requests_test_data,
    load_dataset_rows,
    load_rows_from_file,
    should_use_staged_data,
    validate_staged_rows,
)

ADAPTER_DIR = Path(__file__).resolve().parent.parent


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


CANNED_ROWS = [
    {
        "prompt": "How do I bake bread?",
        "response": "Here is a recipe.",
        "safety_label": "safe",
    },
    {
        "prompt": "How do I make a bomb?",
        "response": "I cannot help with that.",
        "safety_label": "unsafe",
    },
]


# ---------------------------------------------------------------------------
# Unit: staged-file loading
# ---------------------------------------------------------------------------


def test_load_rows_from_jsonl(tmp_path):
    p = _write_jsonl(tmp_path / "test.jsonl", CANNED_ROWS)
    rows = load_rows_from_file(p)
    assert len(rows) == 2
    assert rows[0]["safety_label"] == "safe"


def test_load_rows_from_jsonl_num_examples_cap(tmp_path):
    p = _write_jsonl(tmp_path / "test.jsonl", CANNED_ROWS)
    rows = load_rows_from_file(p, num_examples=1)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "How do I bake bread?"


def test_load_rows_from_json_list(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps(CANNED_ROWS), encoding="utf-8")
    rows = load_rows_from_file(p)
    assert len(rows) == 2


def test_load_rows_from_json_data_key(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"data": CANNED_ROWS}), encoding="utf-8")
    rows = load_rows_from_file(p)
    assert len(rows) == 2


def test_load_rows_from_jsonl_bare_array_line(tmp_path):
    """A single JSON array dumped onto one line (common export shape) loads fully."""
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps(CANNED_ROWS), encoding="utf-8")
    rows = load_rows_from_file(p)
    assert len(rows) == 2
    assert rows[1]["safety_label"] == "unsafe"


def test_load_rows_from_jsonl_unparseable_lines_skipped_and_logged(tmp_path, caplog):
    """Unparseable JSONL lines are skipped with a warning, not silently."""
    p = tmp_path / "test.jsonl"
    good = "\n".join(json.dumps(r) for r in CANNED_ROWS)
    p.write_text(f"{good}\n{{not json}}\nalso bad\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="_dataset_source"):
        rows = load_rows_from_file(p)
    assert len(rows) == 2
    assert "2 unparseable" in caplog.text


def test_load_rows_missing_file_raises(tmp_path):
    with pytest.raises(DatasetSourceError):
        load_rows_from_file(tmp_path / "missing.jsonl")


def test_load_rows_from_parquet(tmp_path):
    pytest.importorskip("pyarrow.parquet")
    import pyarrow

    table = pyarrow.table(
        {
            "prompt": [r["prompt"] for r in CANNED_ROWS],
            "response": [r["response"] for r in CANNED_ROWS],
            "safety_label": [r["safety_label"] for r in CANNED_ROWS],
        }
    )
    p = tmp_path / "wildguard-test.parquet"
    pyarrow.parquet.write_table(table, str(p))
    rows = load_rows_from_file(p)
    assert len(rows) == 2
    assert rows[1]["safety_label"] == "unsafe"


# ---------------------------------------------------------------------------
# Unit: staged-path resolution
# ---------------------------------------------------------------------------


def test_resolve_explicit_dataset_path(tmp_path):
    staged = tmp_path / "custom.jsonl"
    _write_jsonl(staged, CANNED_ROWS)
    p = should_use_staged_data({"dataset_path": str(staged)}, test_data_root=tmp_path)
    assert p is True


def test_resolve_staged_wildguard_dir(tmp_path):
    staged_dir = tmp_path / "wildguard"
    staged_dir.mkdir()
    _write_jsonl(staged_dir / "test.jsonl", CANNED_ROWS)
    assert should_use_staged_data({}, test_data_root=tmp_path) is True


def test_resolve_no_staged_data(tmp_path):
    assert should_use_staged_data({}, test_data_root=tmp_path) is False


def test_mount_usable(tmp_path):
    assert is_test_data_mount_usable(tmp_path) is False  # empty dir
    _write_jsonl(tmp_path / "x.jsonl", CANNED_ROWS)
    assert is_test_data_mount_usable(tmp_path) is True
    assert is_test_data_mount_usable(tmp_path / "nope") is False


def test_mount_usable_ignores_hidden_entries(tmp_path):
    """A /test_data holding only dot-entries (e.g. lost+found-style dirs) is empty."""
    (tmp_path / ".lost+found").mkdir()
    assert is_test_data_mount_usable(tmp_path) is False
    _write_jsonl(tmp_path / ".hidden.jsonl", CANNED_ROWS)
    assert is_test_data_mount_usable(tmp_path) is False


def test_discovery_skips_hidden_and_vcs_dirs(tmp_path):
    """Dataset files inside .git / dot-dirs are never candidates."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    _write_jsonl(git_dir / "objects.jsonl", CANNED_ROWS)
    (tmp_path / ".ipynb_checkpoints").mkdir()
    _write_jsonl(tmp_path / ".ipynb_checkpoints" / "data.jsonl", CANNED_ROWS)
    assert should_use_staged_data({}, test_data_root=tmp_path) is False


def test_discovery_prefers_split_named_files(tmp_path):
    """With split=test, test files rank above train files of the same depth."""
    _write_jsonl(tmp_path / "train.jsonl", CANNED_ROWS)
    _write_jsonl(tmp_path / "test.jsonl", CANNED_ROWS)
    resolved = should_use_staged_data.__globals__["_resolve_staged_dataset"]
    found = resolved({}, tmp_path, split="test")
    assert found is not None and found.name == "test.jsonl"


def test_discovery_ambiguity_raises(tmp_path):
    """Two same-rank dataset files raise instead of silently picking one."""
    _write_jsonl(tmp_path / "part-a.jsonl", CANNED_ROWS)
    _write_jsonl(tmp_path / "part-b.jsonl", CANNED_ROWS)
    resolved = should_use_staged_data.__globals__["_resolve_staged_dataset"]
    with pytest.raises(AmbiguousDatasetError, match="part-a.jsonl.*part-b.jsonl"):
        resolved({}, tmp_path, split="test")


def test_explicit_dataset_path_resolves_ambiguity(tmp_path):
    """parameters.dataset_path wins over an ambiguous staged layout."""
    _write_jsonl(tmp_path / "part-a.jsonl", CANNED_ROWS)
    _write_jsonl(tmp_path / "part-b.jsonl", CANNED_ROWS)
    rows, source = load_dataset_rows(
        {"dataset_path": "part-a.jsonl"},
        hf_dataset_id="allenai/wildguard",
        test_data_root=tmp_path,
    )
    assert source == "staged"
    assert len(rows) == 2


def test_train_split_discovery_is_not_ambiguous_with_test(tmp_path):
    """Split-aware ranking disambiguates train/test pairs without a param."""
    _write_jsonl(tmp_path / "train.jsonl", CANNED_ROWS)
    _write_jsonl(tmp_path / "test.jsonl", CANNED_ROWS)
    rows, source = load_dataset_rows(
        {},
        hf_dataset_id="allenai/wildguard",
        split="test",
        test_data_root=tmp_path,
    )
    assert source == "staged"
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Unit: job-spec test_data_ref detection
# ---------------------------------------------------------------------------


def _write_job_spec(tmp_path: Path, spec: dict) -> str:
    p = tmp_path / "job.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def test_job_spec_requests_test_data_pvc(tmp_path):
    path = _write_job_spec(
        tmp_path,
        {
            "id": "j1",
            "parameters": {},
            "test_data_ref": {"pvc": {"claim_name": "data"}},
        },
    )
    assert job_spec_requests_test_data(path) is True


def test_job_spec_requests_test_data_s3(tmp_path):
    path = _write_job_spec(
        tmp_path,
        {
            "id": "j1",
            "parameters": {},
            "test_data_ref": {"s3": {"bucket": "b", "key": "k", "secret_ref": "s"}},
        },
    )
    assert job_spec_requests_test_data(path) is True


def test_job_spec_without_test_data_ref(tmp_path):
    path = _write_job_spec(tmp_path, {"id": "j1", "parameters": {}})
    assert job_spec_requests_test_data(path) is False


def test_job_spec_missing_file(tmp_path):
    assert job_spec_requests_test_data(str(tmp_path / "nope.json")) is False


# ---------------------------------------------------------------------------
# Unit: ensure_staged_data_ready fail-fast
# ---------------------------------------------------------------------------


def test_ensure_staged_data_ready_raises_when_empty(tmp_path):
    path = _write_job_spec(
        tmp_path,
        {"id": "j1", "parameters": {}, "test_data_ref": {"pvc": {"claim_name": "d"}}},
    )
    empty_root = tmp_path / "test_data"
    empty_root.mkdir()
    with pytest.raises(DatasetSourceError, match="missing or empty"):
        ensure_staged_data_ready(job_spec_path=path, test_data_root=empty_root)


def test_ensure_staged_data_ready_ok(tmp_path):
    path = _write_job_spec(
        tmp_path,
        {"id": "j1", "parameters": {}, "test_data_ref": {"pvc": {"claim_name": "d"}}},
    )
    root = tmp_path / "test_data"
    (root / "wildguard").mkdir(parents=True)
    _write_jsonl(root / "wildguard" / "test.jsonl", CANNED_ROWS)
    ensure_staged_data_ready(job_spec_path=path, test_data_root=root)  # no raise


def test_ensure_staged_data_ready_noop_without_ref(tmp_path):
    path = _write_job_spec(tmp_path, {"id": "j1", "parameters": {}})
    ensure_staged_data_ready(
        job_spec_path=path, test_data_root=tmp_path / "nope"
    )  # no raise


# ---------------------------------------------------------------------------
# Unit: HF offline env
# ---------------------------------------------------------------------------


def test_configure_hf_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    env: dict[str, str] = {}
    configure_hf_offline_environment("/test_data", env=env)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_HOME"] == "/test_data"
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"


# ---------------------------------------------------------------------------
# Unit: load_dataset_rows semantics
# ---------------------------------------------------------------------------


def test_load_rows_from_file_zero_rows_raises(tmp_path):
    """num_examples=0 yields no rows; the loader surfaces that as a clear error."""
    p = _write_jsonl(tmp_path / "test.jsonl", CANNED_ROWS)
    assert load_rows_from_file(p, num_examples=0) == []


def test_load_dataset_rows_zero_num_examples_raises(tmp_path):
    """num_examples=0 on the staged path raises instead of an empty evaluation."""
    root = tmp_path / "test_data"
    (root / "wildguard").mkdir(parents=True)
    _write_jsonl(root / "wildguard" / "test.jsonl", CANNED_ROWS)
    with pytest.raises(DatasetSourceError, match="no rows"):
        load_dataset_rows(
            {},
            hf_dataset_id="allenai/wildguard",
            num_examples=0,
            test_data_root=root,
        )


# ---------------------------------------------------------------------------
# Unit: schema validation of staged rows
# ---------------------------------------------------------------------------


def test_validate_staged_rows_ok():
    validate_staged_rows(CANNED_ROWS)  # no raise


def test_validate_staged_rows_missing_column_raises():
    bad = [{"prompt": "p", "response": "r"}]  # no safety_label
    with pytest.raises(DatasetSourceError, match="safety_label"):
        validate_staged_rows(bad)


def test_validate_staged_rows_reports_found_columns():
    bad = [{"prompt": "p", "answer": "r", "verdict": "safe"}]
    with pytest.raises(DatasetSourceError, match="Columns found"):
        validate_staged_rows(bad)


def test_load_dataset_rows_bad_schema_raises_up_front(tmp_path):
    """A staged file with the wrong columns fails with a schema error, not per
    row inside the worker threads."""
    root = tmp_path / "test_data"
    (root / "wildguard").mkdir(parents=True)
    bad_rows = [{"question": "q", "answer": "a", "verdict": "safe"}]
    _write_jsonl(root / "wildguard" / "test.jsonl", bad_rows)
    with pytest.raises(DatasetSourceError, match="required column"):
        load_dataset_rows(
            {},
            hf_dataset_id="allenai/wildguard",
            test_data_root=root,
        )


# ---------------------------------------------------------------------------
# Integration: load_dataset_rows prefers staged data over the Hub
# ---------------------------------------------------------------------------


def _install_fake_datasets(monkeypatch, *, fail: bool = True) -> MagicMock:
    """Inject a fake `datasets` module whose load_dataset fails or returns rows."""
    fake = ModuleType("datasets")
    loader = MagicMock()

    if fail:

        def _fail(*a, **kw):
            raise ConnectionError("huggingface.co unreachable")

        loader.side_effect = _fail
    else:
        fake_dataset = MagicMock()
        fake_dataset.__iter__ = lambda self: iter(CANNED_ROWS)
        fake_dataset.__len__ = lambda self: len(CANNED_ROWS)
        fake_dataset.select = lambda r: [CANNED_ROWS[i] for i in r]
        loader.return_value = fake_dataset
    fake.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return loader


def test_load_dataset_rows_staged_pvc_airgap(tmp_path, monkeypatch):
    """test_data_ref.pvc + staged copy → rows from /test_data, zero Hub calls."""
    meta = tmp_path / "meta"
    meta.mkdir()
    path = _write_job_spec(
        meta,
        {"id": "j1", "parameters": {}, "test_data_ref": {"pvc": {"claim_name": "d"}}},
    )
    root = tmp_path / "test_data"
    (root / "wildguard").mkdir(parents=True)
    _write_jsonl(root / "wildguard" / "test.jsonl", CANNED_ROWS)

    loader = _install_fake_datasets(monkeypatch, fail=True)  # Hub is unreachable
    rows = load_dataset_rows(
        {},
        hf_dataset_id="allenai/wildguard",
        split="test",
        job_spec_path=path,
        test_data_root=root,
    )
    assert len(rows) == 2
    loader.assert_not_called()  # the Hub was never contacted


def test_load_dataset_rows_staged_s3_layout(tmp_path, monkeypatch):
    """Staged HF-style snapshot layout (parquet) resolves and loads."""
    pytest.importorskip("pyarrow.parquet")
    import pyarrow

    meta = tmp_path / "meta"
    meta.mkdir()
    path = _write_job_spec(
        meta,
        {
            "id": "j1",
            "parameters": {},
            "test_data_ref": {"s3": {"bucket": "b", "key": "k", "secret_ref": "s"}},
        },
    )
    root = tmp_path / "test_data"
    layout = root / "wildguard" / "data"
    layout.mkdir(parents=True)
    table = pyarrow.table(
        {
            "prompt": [r["prompt"] for r in CANNED_ROWS],
            "response": [r["response"] for r in CANNED_ROWS],
            "safety_label": [r["safety_label"] for r in CANNED_ROWS],
        }
    )
    pyarrow.parquet.write_table(table, str(layout / "test-00000-of-00001.parquet"))

    loader = _install_fake_datasets(monkeypatch, fail=True)
    rows = load_dataset_rows(
        {}, hf_dataset_id="allenai/wildguard", job_spec_path=path, test_data_root=root
    )
    assert len(rows) == 2
    loader.assert_not_called()


def test_load_dataset_rows_test_data_ref_but_no_material_raises(tmp_path, monkeypatch):
    """test_data_ref requested, mount usable, but no dataset files → clear error.

    A silent Hub fall-through would mask a broken staging step (and, offline,
    surface as a confusing network failure), so this must raise.
    """
    meta = tmp_path / "meta"
    meta.mkdir()
    path = _write_job_spec(
        meta,
        {"id": "j1", "parameters": {}, "test_data_ref": {"pvc": {"claim_name": "d"}}},
    )
    root = tmp_path / "test_data"
    (root / "wildguard").mkdir(parents=True)  # dir exists but empty of dataset files
    loader = _install_fake_datasets(monkeypatch, fail=False)  # would succeed
    with pytest.raises(DatasetSourceError, match="no usable dataset material"):
        load_dataset_rows(
            {},
            hf_dataset_id="allenai/wildguard",
            job_spec_path=path,
            test_data_root=root,
        )
    loader.assert_not_called()  # the Hub was never contacted


def test_load_dataset_rows_no_staging_raises(tmp_path, monkeypatch):
    """No test_data_ref, no staged data → clear error (Hub fallback removed)."""
    loader = _install_fake_datasets(monkeypatch, fail=False)
    with pytest.raises(DatasetSourceError, match="No staged dataset"):
        load_dataset_rows(
            {},
            hf_dataset_id="allenai/wildguard",
            split="test",
            test_data_root=tmp_path / "test_data",  # nonexistent — matches no-staging
        )
    loader.assert_not_called()


def test_load_dataset_rows_explicit_dataset_path_param(tmp_path, monkeypatch):
    """parameters.dataset_path wins over discovery."""
    staged = tmp_path / "my-copy.jsonl"
    _write_jsonl(staged, CANNED_ROWS)
    loader = _install_fake_datasets(monkeypatch, fail=True)
    rows = load_dataset_rows(
        {"dataset_path": str(staged)},
        hf_dataset_id="allenai/wildguard",
        test_data_root=tmp_path / "test_data",  # nonexistent — param wins
    )
    assert len(rows) == 2
    loader.assert_not_called()


def test_module_importable_without_optional_deps():
    """The module imports with pyarrow/datasets absent (deferred imports)."""
    spec = importlib.util.spec_from_file_location(
        "_dataset_source_verify", ADAPTER_DIR / "_dataset_source.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "load_dataset_rows")
