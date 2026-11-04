"""Disconnected RULER jobs consume only validated staged data and local tokenizers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from offline import ensure_nltk_resources
from prepare_assets import ESSAY_FILE, prepare_assets
from staged_assets import (
    is_staged_data_available,
    job_spec_requests_test_data,
    resolve_staged_tokenizer_path,
    staged_data_requested,
)
import main


def test_prepare_assets_copies_only_task_required_staged_inputs(monkeypatch, tmp_path):
    import prepare_assets as assets_module

    staged = tmp_path / "test_data"
    cache = tmp_path / "writable" / "assets"
    staged.mkdir()
    payloads = {
        "english_words.json": b'{"1":"word"}',
        "squad.json": b'[{"question":"Q","answers":{"text":["A"]}}]',
        ESSAY_FILE: b'{"text":"A staged essay."}',
    }
    for name, payload in payloads.items():
        if name != ESSAY_FILE:
            digest = hashlib.sha256(payload).hexdigest()
            monkeypatch.setitem(assets_module.ASSETS, name, ("unused", digest))
        (staged / name).write_bytes(payload)

    prepare_assets(
        [
            {"task": "common_words_extraction"},
            {"task": "qa", "args": {"dataset": "squad"}},
            {"task": "niah", "args": {"type_haystack": "essay"}},
        ],
        cache,
        tmp_path / "scripts",
        staged_data_dir=staged,
    )

    assert {path.name for path in cache.iterdir()} == set(payloads)
    for name, payload in payloads.items():
        assert (cache / name).read_bytes() == payload


def test_staged_asset_validation_reports_all_missing_files_without_downloads(tmp_path):
    staged = tmp_path / "test_data"
    staged.mkdir()
    with (
        patch("prepare_assets.urlopen") as download,
        patch("prepare_assets.subprocess.run") as subprocess_run,
        pytest.raises(ValueError) as exc_info,
    ):
        prepare_assets(
            [
                {"task": "common_words_extraction"},
                {"task": "qa", "args": {"dataset": "hotpotqa"}},
            ],
            tmp_path / "cache",
            tmp_path / "scripts",
            staged_data_dir=staged,
        )

    message = str(exc_info.value)
    assert "english_words.json" in message
    assert "hotpotqa.json" in message
    download.assert_not_called()
    subprocess_run.assert_not_called()


def test_staged_asset_checksum_failure_does_not_copy_any_assets(monkeypatch, tmp_path):
    import prepare_assets as assets_module

    staged = tmp_path / "test_data"
    staged.mkdir()
    (staged / "english_words.json").write_text('{"1":"word"}')
    monkeypatch.setitem(assets_module.ASSETS, "english_words.json", ("unused", "0" * 64))
    cache = tmp_path / "cache"

    with patch("prepare_assets.urlopen") as download:
        with pytest.raises(ValueError, match="checksum mismatch"):
            prepare_assets(
                [{"task": "common_words_extraction"}],
                cache,
                tmp_path / "scripts",
                staged_data_dir=staged,
            )

    download.assert_not_called()
    assert not cache.exists()


@pytest.mark.parametrize("payload", [b"not-json", b'{"text":" "}', b"[]"])
def test_invalid_staged_essay_is_rejected_without_fallback(tmp_path, payload):
    staged = tmp_path / "test_data"
    staged.mkdir()
    (staged / ESSAY_FILE).write_bytes(payload)
    cache = tmp_path / "cache"

    with patch("prepare_assets.subprocess.run") as downloader:
        with pytest.raises(ValueError, match=ESSAY_FILE):
            prepare_assets(
                [{"task": "niah", "args": {"type_haystack": "essay"}}],
                cache,
                tmp_path / "scripts",
                staged_data_dir=staged,
            )

    downloader.assert_not_called()
    assert not cache.exists()


def test_job_spec_and_mount_detect_staged_mode(tmp_path):
    spec_path = tmp_path / "job.json"
    spec_path.write_text(json.dumps({"test_data_ref": {"pvc": {"claim_name": "assets"}}}))
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    assert job_spec_requests_test_data(spec_path)
    assert staged_data_requested(spec_path, empty_root)

    staged_root = tmp_path / "mounted-data"
    staged_root.mkdir()
    (staged_root / "tokenizer").mkdir()
    assert is_staged_data_available(staged_root)
    assert staged_data_requested(tmp_path / "missing-job.json", staged_root)


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, "tokenizer"),
        ("model/name", "tokenizer"),
        ("custom-tokenizer", "custom-tokenizer"),
    ],
)
def test_resolve_staged_tokenizer_path(configured, expected, tmp_path):
    root = tmp_path / "test_data"
    tokenizer = root / expected
    tokenizer.mkdir(parents=True)

    result = resolve_staged_tokenizer_path(configured, "model/name", "hf", root)

    assert result == tokenizer


def test_resolve_staged_tokenizer_requires_local_hf_directory(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()

    with pytest.raises(ValueError, match="not found"):
        resolve_staged_tokenizer_path(None, "model/name", "hf", root)
    with pytest.raises(ValueError, match="tokenizer_type='hf'"):
        resolve_staged_tokenizer_path(None, "model/name", "openai", root)


def test_relative_tokenizer_path_cannot_escape_staged_root(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()

    with pytest.raises(ValueError, match="beneath /test_data"):
        resolve_staged_tokenizer_path("../outside", "model/name", "hf", root)


def test_disconnected_job_routes_staged_bundle_and_local_tokenizer(monkeypatch, tmp_path):
    staged = tmp_path / "test_data"
    (staged / "tokenizer").mkdir(parents=True)
    (staged / "english_words.json").write_text('{"1":"word"}')
    job_spec = tmp_path / "job.json"
    job_spec.write_text(json.dumps({"test_data_ref": {"s3": {"bucket": "assets", "key": "ruler"}}}))
    monkeypatch.setenv("RULER_TEST_DATA_DIR", str(staged))
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_spec))

    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    config = SimpleNamespace(
        id="offline-ruler-test",
        benchmark_id="common-words-extraction",
        benchmark_index=0,
        num_examples=1,
        model=SimpleNamespace(name="model/name", url="http://model/v1"),
        parameters={"context_lengths": [4096]},
    )
    callbacks = SimpleNamespace(report_status=lambda _update: None)
    work_dir = tmp_path / "work"
    monkeypatch.setattr(main.tempfile, "mkdtemp", lambda **_kwargs: str(work_dir))

    verified = []

    def verify(path, tokenizer_type, offline):
        verified.append((path, tokenizer_type, offline))
        return True

    monkeypatch.setattr(adapter, "_verify_tokenizer", verify)
    prepared = []

    def prepare(tasks, data_dir, timeout, staged_data_dir=None):
        prepared.append((tasks, data_dir, timeout, staged_data_dir))

    monkeypatch.setattr(adapter, "_prepare_task_assets", prepare)

    def generate(**kwargs):
        assert kwargs["task_id"] == "cwe"
        assert kwargs["offline"] is True
        raise RuntimeError("stop after staged data setup")

    monkeypatch.setattr(adapter, "_generate_task_data", generate)
    with pytest.raises(RuntimeError, match="stop after staged data setup"):
        adapter.run_benchmark_job(config, callbacks)

    assert verified == [(str(staged / "tokenizer"), "hf", True)]
    assert prepared == [(["cwe"], work_dir / "data", 600, staged)]


def test_container_image_copies_staged_asset_resolver():
    containerfile = Path(__file__).resolve().parents[1] / "Containerfile"
    assert "COPY prepare_assets.py staged_assets.py DATA_LICENSES.md ./" in containerfile.read_text()


def test_missing_nltk_resource_fails_offline_without_downloading():
    class FakeNLTK:
        class data:
            @staticmethod
            def find(_resource):
                raise LookupError("not installed")

        @staticmethod
        def download(*_args, **_kwargs):
            pytest.fail("offline RULER must not download NLTK data")

    with pytest.raises(RuntimeError, match="running offline.*punkt.*missing"):
        ensure_nltk_resources(FakeNLTK, offline=True)


def test_connected_nltk_setup_downloads_only_missing_resources():
    found = {"tokenizers/punkt"}
    downloaded = []

    class FakeNLTK:
        class data:
            @staticmethod
            def find(resource):
                if resource not in found:
                    raise LookupError("not installed")

        @staticmethod
        def download(package, **kwargs):
            downloaded.append((package, kwargs))
            found.add(f"tokenizers/{package}")
            return True

    ensure_nltk_resources(FakeNLTK, offline=False)

    assert downloaded == [("punkt_tab", {"raise_on_error": True})]
