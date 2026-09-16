"""Tests for HuggingFace token resolution in the Inspect adapter."""

from unittest.mock import patch

import pytest

from _hf_auth import apply_hf_hub_auth, resolve_hf_token


@pytest.fixture
def env():
    return {"PATH": "/usr/bin"}


def test_resolve_prefers_mount_over_ref_in_env(env):
    env["HF_TOKEN"] = "hf-token:ref"
    with patch("_hf_auth.read_model_auth_key", return_value="hf_real_secret"):
        token, source = resolve_hf_token(env, wait_timeout_s=0)
    assert token == "hf_real_secret"
    assert source == "mount"


def test_resolve_uses_environment_when_mount_missing(env):
    env["HF_TOKEN"] = "hf_from_env"
    with patch("_hf_auth.read_model_auth_key", return_value=None):
        token, source = resolve_hf_token(env, wait_timeout_s=0)
    assert token == "hf_from_env"
    assert source == "environment"


def test_resolve_retries_until_mount_appears(env):
    reads = iter([None, None, "hf_delayed"])

    def fake_read(key):
        assert key == "hf-token"
        return next(reads, "hf_delayed")

    with patch("_hf_auth.read_model_auth_key", side_effect=fake_read):
        with patch("_hf_auth.time.sleep") as sleep:
            token, source = resolve_hf_token(
                env, wait_timeout_s=5, poll_interval_s=0.1
            )
    assert token == "hf_delayed"
    assert source == "mount"
    assert sleep.call_count == 2


def test_resolve_rejects_ref_placeholder_from_mount(env):
    with patch("_hf_auth.read_model_auth_key", return_value="hf-token:ref"):
        token, source = resolve_hf_token(env, wait_timeout_s=0)
    assert token is None
    assert source == ""


def test_apply_sets_both_hub_env_vars(env):
    with patch("_hf_auth.read_model_auth_key", return_value="hf_abc"):
        apply_hf_hub_auth(env, wait_timeout_s=0)
    assert env["HF_TOKEN"] == "hf_abc"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_abc"


def test_apply_strips_invalid_env_tokens(env):
    env["HF_TOKEN"] = "hf-token:ref"
    env["HUGGING_FACE_HUB_TOKEN"] = "also:ref"
    with patch("_hf_auth.read_model_auth_key", return_value=None):
        apply_hf_hub_auth(env, wait_timeout_s=0)
    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env


def test_default_wait_zero_outside_k8s(monkeypatch):
    monkeypatch.delenv("EVALHUB_MODE", raising=False)
    from _hf_auth import _default_wait_timeout_s

    assert _default_wait_timeout_s() == 0.0


def test_default_wait_sixty_in_k8s(monkeypatch):
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    from _hf_auth import _K8S_WAIT_TIMEOUT_S, _default_wait_timeout_s

    assert _default_wait_timeout_s() == _K8S_WAIT_TIMEOUT_S


def test_build_env_integrates_hf_auth(job_spec_path):
    from main import InspectAdapter

    adapter = InspectAdapter(job_spec_path=job_spec_path)
    with patch("_hf_auth.read_model_auth_key", return_value="hf_mount"):
        env = adapter._build_env(adapter.job_spec, "standard")
    assert env["HF_TOKEN"] == "hf_mount"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_mount"
