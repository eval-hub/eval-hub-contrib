"""Tests for Hugging Face staged-cache pinning to /test_data."""

from pathlib import Path

from _execution import build_env
from _hf_offline import (
    configure_hf_staged_cache_environment,
    should_pin_hf_cache_to_test_data,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")


def test_pin_when_test_data_mount_non_empty(tmp_path: Path) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "placeholder.txt")
    assert should_pin_hf_cache_to_test_data(root)


def test_no_pin_when_test_data_missing_or_empty(tmp_path: Path) -> None:
    assert not should_pin_hf_cache_to_test_data(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not should_pin_hf_cache_to_test_data(empty)


def test_build_env_pins_hf_home_without_offline_flags(
    monkeypatch, tmp_path: Path, job_spec_path
) -> None:
    from main import InspectAdapter

    import _execution as execution_mod
    import _hf_offline as hf_offline_mod

    root = tmp_path / "test_data"
    root.mkdir()
    _touch(root / "hub" / "datasets--GSMA--ot-full" / "refs" / "main")

    monkeypatch.setattr(hf_offline_mod, "TEST_DATA_DIR", str(root))
    monkeypatch.setattr(execution_mod, "TEST_DATA_DIR", str(root))

    adapter = InspectAdapter(job_spec_path=job_spec_path)
    env = build_env(adapter.job_spec, "standard")
    assert env.get("HF_HOME") == str(root)
    assert env.get("HF_HUB_CACHE") == str(root / "hub")
    assert env.get("HF_DATASETS_CACHE") == str(root / "datasets")
    assert env.get("HF_HUB_OFFLINE") is None


def test_configure_hf_staged_cache_environment_sets_paths_only(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    env: dict[str, str] = {"HF_HUB_OFFLINE": "1"}
    configure_hf_staged_cache_environment(str(root), env)
    assert env["HF_HOME"] == str(root)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(root / "datasets")
    assert "HF_HUB_OFFLINE" not in env
