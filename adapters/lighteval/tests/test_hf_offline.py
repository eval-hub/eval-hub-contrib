"""Unit tests for LightEval's staged Hugging Face cache support."""

import json

import pytest

from _hf_offline import (
    build_hf_offline_environment,
    configure_hf_offline_environment,
    ensure_lighteval_registry_assets,
    is_test_data_mount_usable,
    job_spec_requests_test_data,
)


@pytest.mark.parametrize("source", ["s3", "pvc", "git", "hf"])
def test_job_spec_requests_test_data_for_supported_sources(tmp_path, source):
    job_spec = tmp_path / "job.json"
    job_spec.write_text(json.dumps({"test_data_ref": {source: {"location": "assets"}}}))

    assert job_spec_requests_test_data(job_spec)


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"test_data_ref": None},
        {"test_data_ref": {}},
        {"test_data_ref": {"pvc": {}}},
    ],
)
def test_job_spec_without_configured_source_does_not_request_staged_data(
    tmp_path, spec
):
    job_spec = tmp_path / "job.json"
    job_spec.write_text(json.dumps(spec))

    assert not job_spec_requests_test_data(job_spec)


def test_invalid_or_missing_job_spec_does_not_request_staged_data(tmp_path):
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")

    assert not job_spec_requests_test_data(missing)
    assert not job_spec_requests_test_data(invalid)


def test_is_test_data_mount_usable_requires_visible_content(tmp_path):
    root = tmp_path / "test_data"
    root.mkdir()
    assert not is_test_data_mount_usable(root)

    (root / ".metadata").mkdir()
    assert not is_test_data_mount_usable(root)

    (root / "hub").mkdir()
    assert is_test_data_mount_usable(root)


def test_configure_hf_offline_environment_points_to_staged_cache(tmp_path):
    env = {"HF_TOKEN": "retained"}

    configure_hf_offline_environment(tmp_path, env)

    assert env["HF_TOKEN"] == "retained"
    assert env["HF_HOME"] == str(tmp_path)
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hub")
    assert env["HF_DATASETS_CACHE"] == str(tmp_path / "datasets")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_EVALUATE_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_build_hf_offline_environment_is_noop_without_test_data_ref(tmp_path):
    job_spec = tmp_path / "job.json"
    job_spec.write_text(json.dumps({"parameters": {}}))
    env = {"HF_HUB_OFFLINE": "existing"}

    assert build_hf_offline_environment(job_spec, tmp_path / "test_data", env) is None
    assert env == {"HF_HUB_OFFLINE": "existing"}


def test_build_hf_offline_environment_configures_staged_mount(tmp_path):
    job_spec = tmp_path / "job.json"
    job_spec.write_text(
        json.dumps({"test_data_ref": {"pvc": {"claim_name": "hf-cache"}}})
    )
    root = tmp_path / "test_data"
    (root / "hub").mkdir(parents=True)
    (root / "datasets").mkdir()
    env = {"HF_TOKEN": "retained"}

    configured_env = build_hf_offline_environment(job_spec, root, env)
    assert configured_env is not None
    assert env == {"HF_TOKEN": "retained"}
    env = configured_env
    assert env["HF_TOKEN"] == "retained"
    assert env["HF_HOME"] == str(root)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(root / "datasets")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"


def test_build_hf_offline_environment_fails_for_missing_staged_mount(tmp_path):
    job_spec = tmp_path / "job.json"
    job_spec.write_text(json.dumps({"test_data_ref": {"s3": {"bucket": "bucket"}}}))

    with pytest.raises(RuntimeError, match="test_data_ref.*missing or empty"):
        build_hf_offline_environment(job_spec, tmp_path / "missing")


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
