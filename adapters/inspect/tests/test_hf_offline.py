"""Tests for Hugging Face offline detection and env configuration."""

import json
import stat
from pathlib import Path

import pytest

from _hf_offline import (
    _link_staged_datasets_cache,
    configure_hf_offline_environment,
    job_spec_requests_test_data,
    should_use_hf_offline,
)
from _execution import build_env


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")


@pytest.fixture
def fake_test_data(tmp_path: Path) -> Path:
    root = tmp_path / "test_data"
    tok = root / "tokenizer"
    tok.mkdir(parents=True)
    _touch(tok / "config.json")
    bundle = root / "GSMA--ot-full--telemath"
    bundle.mkdir(parents=True)
    _touch(bundle / "dataset_dict.json")
    return root


def test_infer_offline_from_tokenizer_and_bundle(fake_test_data: Path) -> None:
    tok = fake_test_data / "tokenizer"
    params = {"tokenizer": str(tok.resolve())}
    assert should_use_hf_offline(params, test_data_root=fake_test_data)


def test_infer_offline_from_test_data_ref_in_job_spec(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "hub" / "datasets--GSMA--ot-full" / "refs" / "main")

    spec = {
        "test_data_ref": {
            "s3": {"bucket": "mlpipeline", "key": "offline", "secret_ref": "minio-test"},
        },
    }
    monkeypatch.setattr(
        "_hf_offline._read_job_spec_dict_from_path",
        lambda _path: spec,
    )

    assert job_spec_requests_test_data("/meta/job.json")
    assert should_use_hf_offline({}, job_spec_path="/meta/job.json", test_data_root=root)


def test_no_offline_without_test_data_ref_or_tokenizer(tmp_path: Path) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "placeholder.txt")
    assert not should_use_hf_offline({}, test_data_root=root)


def test_build_env_sets_hf_offline(monkeypatch, fake_test_data: Path, job_spec_path) -> None:
    from main import InspectAdapter

    import _execution as execution_mod
    import _hf_offline as hf_offline_mod

    monkeypatch.setattr(hf_offline_mod, "TEST_DATA_DIR", str(fake_test_data))
    monkeypatch.setattr(execution_mod, "TEST_DATA_DIR", str(fake_test_data))

    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.parameters["tokenizer"] = str((fake_test_data / "tokenizer").resolve())
    env = build_env(adapter.job_spec, "standard")
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("HF_HOME") == "/data/huggingface"
    assert env.get("HF_HUB_CACHE") == str(fake_test_data / "hub")
    assert env.get("HF_DATASETS_CACHE") == "/data/huggingface/datasets"


def test_configure_hf_offline_environment_updates_os_environ(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    env: dict[str, str] = {}
    writable = tmp_path / "writable-cache"
    configure_hf_offline_environment(str(root), env, writable_cache_home=str(writable))
    assert env["HF_HOME"] == str(writable)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(writable / "datasets")
    assert env["HF_XET_CACHE"] == str(writable / "xet")
    assert env["HF_ASSETS_CACHE"] == str(writable / "assets")
    assert env["HF_HUB_OFFLINE"] == "1"


def test_read_only_staged_dataset_directories_mirror_to_writable_cache(tmp_path: Path) -> None:
    staged_root = tmp_path / "test_data"
    source = staged_root / "datasets" / "datasets--example" / "snapshots" / "revision"
    source.mkdir(parents=True)
    data_file = source / "dataset.arrow"
    data_file.write_bytes(b"staged data")
    (source / "builder.lock").write_text("lock")
    (source / "partial.incomplete").write_text("partial")

    source_dirs = [source, *source.parents]
    for directory in source_dirs:
        if directory == tmp_path:
            break
        directory.chmod(0o555)

    writable_root = tmp_path / "writable"
    destination_root = writable_root / "datasets"
    existing_destination = (
        destination_root / "datasets--example" / "snapshots" / "revision"
    )
    existing_destination.mkdir(parents=True)
    destination_dirs = [existing_destination, *existing_destination.parents]
    for directory in destination_dirs:
        if directory == writable_root:
            break
        directory.chmod(0o555)

    try:
        linked = _link_staged_datasets_cache(staged_root, writable_root)
        destination = existing_destination

        assert linked == 1
        assert (destination / "dataset.arrow").is_symlink()
        assert (destination / "dataset.arrow").resolve() == data_file.resolve()
        assert not (destination / "builder.lock").exists()
        assert not (destination / "partial.incomplete").exists()
        for directory in [destination, *destination.parents]:
            if directory == writable_root:
                break
            assert stat.S_IMODE(directory.stat().st_mode) & stat.S_IWUSR
            assert stat.S_IMODE(directory.stat().st_mode) & stat.S_IXUSR
    finally:
        for directory in source_dirs:
            if directory == tmp_path:
                break
            directory.chmod(0o755)
        for directory in destination_dirs:
            if directory == writable_root:
                break
            directory.chmod(0o755)
