"""Check build-time asset integrity and failure handling without runtime packages."""

import hashlib
import importlib.util
import io
from pathlib import Path
from urllib.error import URLError

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_task_data", Path(__file__).parents[1] / "scripts" / "prepare_task_data.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_download_publishes_verified_archive(tmp_path, monkeypatch):
    payload = b"verified task archive"
    monkeypatch.setattr(prepare, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    target = tmp_path / "data" / "archive.zip"
    prepare.download_archive(
        "https://example.test/archive", target, hashlib.sha256(payload).hexdigest()
    )
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]


def test_checksum_failure_leaves_no_archive_or_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt")
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare.download_archive(
            "https://example.test/archive", tmp_path / "archive.zip", "0" * 64
        )
    assert not list(tmp_path.iterdir())


def test_failed_download_cleans_partial_file(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("download unavailable")

    monkeypatch.setattr(prepare, "urlopen", unavailable)
    with pytest.raises(URLError):
        prepare.download_archive(
            "https://example.test/archive", tmp_path / "archive.zip", "0" * 64
        )
    assert not list(tmp_path.iterdir())


def test_verified_existing_archive_needs_no_network(tmp_path, monkeypatch):
    payload = b"already bundled"
    target = tmp_path / "archive.zip"
    target.write_bytes(payload)

    def forbidden(*args, **kwargs):
        pytest.fail("An existing verified archive must not be downloaded again")

    monkeypatch.setattr(prepare, "urlopen", forbidden)
    prepare.download_archive(
        "https://example.test/archive", target, hashlib.sha256(payload).hexdigest()
    )
    assert target.read_bytes() == payload


def test_pre_downloaded_archive_uses_same_integrity_check(tmp_path):
    payload = b"offline task archive"
    source = tmp_path / "staged.zip"
    source.write_bytes(payload)
    target = tmp_path / "installed.zip"
    prepare.download_archive(
        source.as_uri(), target, hashlib.sha256(payload).hexdigest()
    )
    assert target.read_bytes() == payload


def test_dependency_upgrade_requires_matching_asset_pin(monkeypatch):
    monkeypatch.setattr(prepare, "version", lambda name: "different-release")
    with pytest.raises(
        RuntimeError, match="update the source commit and archive checksum"
    ):
        prepare.prepare_self_proliferation()
