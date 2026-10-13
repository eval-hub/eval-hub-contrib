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


@pytest.fixture
def k8s_mount(monkeypatch, tmp_path):
    """Pretend to be an EvalHub k8s job pod whose model-auth mount is ``tmp_path/model``."""
    mount = tmp_path / "model"
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.delenv("INSPECT_HF_TOKEN_WAIT_S", raising=False)
    monkeypatch.setattr("_hf_auth._MODEL_MOUNT_DIR", mount)
    return mount


def test_default_wait_zero_outside_k8s(monkeypatch):
    monkeypatch.delenv("EVALHUB_MODE", raising=False)
    monkeypatch.delenv("INSPECT_HF_TOKEN_WAIT_S", raising=False)
    from _hf_auth import _default_wait_timeout_s

    assert _default_wait_timeout_s() == 0.0


def test_default_wait_zero_when_no_model_auth_mount(k8s_mount):
    """No model.auth.secret_ref means no mount, so an hf-token can never appear."""
    from _hf_auth import _default_wait_timeout_s

    assert not k8s_mount.exists()
    assert _default_wait_timeout_s() == 0.0


def test_default_wait_zero_when_mount_is_populated(k8s_mount):
    """A populated projected volume is complete: a missing hf-token is final."""
    k8s_mount.mkdir()
    (k8s_mount / "api-key").write_text("ref")
    from _hf_auth import _default_wait_timeout_s

    assert _default_wait_timeout_s() == 0.0


def test_default_wait_grace_when_mount_is_empty(k8s_mount):
    """An empty mount (only kubelet's hidden entries) is ambiguous: allow a short grace."""
    k8s_mount.mkdir()
    (k8s_mount / "..2026_10_06_00_00_00.1").mkdir()
    (k8s_mount / "..data").symlink_to("..2026_10_06_00_00_00.1")
    from _hf_auth import _K8S_EMPTY_MOUNT_GRACE_S, _default_wait_timeout_s

    assert _default_wait_timeout_s() == _K8S_EMPTY_MOUNT_GRACE_S
    assert _K8S_EMPTY_MOUNT_GRACE_S <= 10


@pytest.mark.parametrize("value, expected", [("0", 0.0), ("30", 30.0), ("-5", 0.0)])
def test_wait_override_env(k8s_mount, monkeypatch, value, expected):
    """INSPECT_HF_TOKEN_WAIT_S overrides the computed wait (never negative)."""
    monkeypatch.setenv("INSPECT_HF_TOKEN_WAIT_S", value)
    from _hf_auth import _default_wait_timeout_s

    assert _default_wait_timeout_s() == expected


def test_wait_override_invalid_falls_back_to_default(k8s_mount, monkeypatch):
    """A non-numeric override is ignored, not fatal."""
    monkeypatch.setenv("INSPECT_HF_TOKEN_WAIT_S", "soon")
    from _hf_auth import _default_wait_timeout_s

    assert _default_wait_timeout_s() == 0.0  # no mount in this fixture


def test_apply_does_not_sleep_when_no_mount_in_k8s(env, k8s_mount):
    """Regression: a job without model auth used to poll for 60s before starting."""
    with patch("_hf_auth.read_model_auth_key", return_value=None):
        with patch("_hf_auth.time.sleep") as sleep:
            apply_hf_hub_auth(env)
    sleep.assert_not_called()
    assert "HF_TOKEN" not in env


def test_build_env_integrates_hf_auth(job_spec_path):
    from main import InspectAdapter

    adapter = InspectAdapter(job_spec_path=job_spec_path)
    with patch("_hf_auth.read_model_auth_key", return_value="hf_mount"):
        env = adapter._build_env(adapter.job_spec, "standard")
    assert env["HF_TOKEN"] == "hf_mount"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_mount"
