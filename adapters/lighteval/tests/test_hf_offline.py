"""Unit tests for LightEval's staged Hugging Face cache support."""

import pytest

from _hf_offline import (
    build_hf_offline_environment,
    configure_hf_offline_environment,
    ensure_lighteval_registry_assets,
    is_test_data_mount_usable,
)


def test_is_test_data_mount_usable_requires_visible_content(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    assert not is_test_data_mount_usable(root)

    (root / ".metadata").mkdir()
    assert not is_test_data_mount_usable(root)

    (root / "hub").mkdir()
    assert is_test_data_mount_usable(root)


def test_is_test_data_mount_usable_recognizes_empty_mount(tmp_path, monkeypatch):
    root = tmp_path / "test_data"
    root.mkdir()
    monkeypatch.setattr(type(root), "is_mount", lambda path: path == root)

    assert is_test_data_mount_usable(root)


def test_configure_hf_offline_environment_uses_writable_cache_overlay(tmp_path):
    root = tmp_path / "test_data"
    dataset_dir = root / "datasets" / "fixture" / "default" / "1.0.0" / "abc123"
    dataset_dir.mkdir(parents=True)
    staged_file = dataset_dir / "train.arrow"
    staged_file.write_bytes(b"prepared dataset")
    staged_lock = root / "datasets" / "fixture.lock"
    staged_lock.write_text("staged lock")
    writable_cache = tmp_path / "writable-datasets"
    env = {"HF_TOKEN": "retained"}

    configured_cache = configure_hf_offline_environment(root, env, writable_cache)

    cached_file = (
        writable_cache
        / "fixture"
        / "default"
        / "1.0.0"
        / "abc123"
        / "train.arrow"
    )
    assert configured_cache == writable_cache
    assert env["HF_TOKEN"] == "retained"
    assert env["HF_HOME"] == str(root)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(writable_cache)
    assert cached_file.is_symlink()
    assert cached_file.read_bytes() == b"prepared dataset"
    assert not (writable_cache / "fixture.lock").exists()
    (writable_cache / "fixture.lock").write_text("writable lock")
    assert (writable_cache / "fixture.lock").read_text() == "writable lock"
    assert staged_lock.read_text() == "staged lock"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_EVALUATE_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_build_hf_offline_environment_is_noop_without_staged_mount(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    env = {"HF_HUB_OFFLINE": "existing"}

    assert build_hf_offline_environment(root, env) is None
    assert env == {"HF_HUB_OFFLINE": "existing"}


def test_build_hf_offline_environment_configures_staged_mount(tmp_path):
    root = tmp_path / "test_data"
    (root / "hub").mkdir(parents=True)
    dataset_cache = root / "datasets" / "fixture" / "default" / "1.0.0"
    dataset_cache.mkdir(parents=True)
    (dataset_cache / "dataset_info.json").write_text("{}")
    env = {"HF_TOKEN": "retained"}

    writable_cache = tmp_path / "writable-datasets"
    configured_env = build_hf_offline_environment(
        root,
        env,
        writable_cache,
    )
    assert configured_env is not None
    assert env == {"HF_TOKEN": "retained"}
    env = configured_env
    assert env["HF_TOKEN"] == "retained"
    assert env["HF_HOME"] == str(root)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(writable_cache)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"


def test_build_hf_offline_environment_rejects_raw_hf_repository_layout(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    (root / "README.md").write_text("raw repository file")

    with pytest.raises(RuntimeError, match="test_data_ref.hf source stages raw repository"):
        build_hf_offline_environment(root)


def test_build_hf_offline_environment_rejects_empty_staged_mount(tmp_path, monkeypatch):
    root = tmp_path / "test_data"
    root.mkdir()
    monkeypatch.setattr(type(root), "is_mount", lambda path: path == root)

    with pytest.raises(RuntimeError, match="does not contain a populated Hugging Face cache"):
        build_hf_offline_environment(root)


def test_configure_hf_offline_environment_allows_no_prepared_dataset_cache(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    writable_cache = tmp_path / "writable-datasets"
    env = {}

    configure_hf_offline_environment(root, env, writable_cache)

    assert env["HF_DATASETS_CACHE"] == str(writable_cache)
    assert writable_cache.is_dir()
    assert list(writable_cache.iterdir()) == []


def test_ensure_lighteval_registry_assets_copies_staged_tinybenchmarks(tmp_path):
    root = tmp_path / "test_data"
    staged_asset = root / "lighteval" / "tinyBenchmarks.pkl"
    staged_asset.parent.mkdir(parents=True)
    staged_asset.write_bytes(b"staged registry data")
    registry_dir = tmp_path / "site-packages" / "lighteval" / "tasks" / "tasks"
    registry_dir.mkdir(parents=True)

    ensure_lighteval_registry_assets(root, registry_dir)

    assert (registry_dir / "tinyBenchmarks.pkl").read_bytes() == b"staged registry data"


def test_ensure_lighteval_registry_assets_requires_staged_or_cached_file(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    registry_dir = tmp_path / "site-packages" / "lighteval" / "tasks" / "tasks"
    registry_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="tinyBenchmarks.pkl.*Stage this file"):
        ensure_lighteval_registry_assets(root, registry_dir)


def test_ensure_lighteval_registry_assets_keeps_existing_cached_file(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    registry_dir = tmp_path / "site-packages" / "lighteval" / "tasks" / "tasks"
    registry_dir.mkdir(parents=True)
    cached_asset = registry_dir / "tinyBenchmarks.pkl"
    cached_asset.write_bytes(b"already bundled")

    ensure_lighteval_registry_assets(root, registry_dir)

    assert cached_asset.read_bytes() == b"already bundled"
