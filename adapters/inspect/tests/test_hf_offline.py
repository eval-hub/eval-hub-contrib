"""Tests for Hugging Face offline detection and env configuration."""

import json
from pathlib import Path

import pytest

from _hf_offline import (
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
    assert env.get("HF_HOME") == str(fake_test_data)


def test_configure_hf_offline_environment_updates_os_environ(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    env: dict[str, str] = {}
    configure_hf_offline_environment(str(root), env)
    assert env["HF_HOME"] == str(root)
    assert env["HF_HUB_CACHE"] == str(root / "hub")
    assert env["HF_DATASETS_CACHE"] == str(root / "datasets")
    assert env["HF_HUB_OFFLINE"] == "1"
