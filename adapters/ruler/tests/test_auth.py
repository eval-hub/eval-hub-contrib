"""Credential propagation for proxy inference and gated tokenizer generation."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import main


def test_proxy_credential_precedes_legacy_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "legacy")
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: SimpleNamespace(api_key="api-key:ref"))
    client = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    main.RulerAdapter.__new__(main.RulerAdapter)._make_api_client("http://localhost:8080/v1")
    client.assert_called_once_with(base_url="http://localhost:8080/v1", api_key="api-key:ref")


def test_missing_credentials_fail(monkeypatch):
    for key in ("MODEL_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None))
    with pytest.raises(ValueError, match="model.auth.secret_ref"):
        main.RulerAdapter.__new__(main.RulerAdapter)._make_api_client("https://model/v1")


def test_hf_token_reaches_precheck_and_child(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: "mounted-hf-token")
    tokenizer = Mock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    assert adapter._verify_tokenizer("gated/model", "hf")
    assert tokenizer.from_pretrained.call_args.kwargs["token"] == "mounted-hf-token"
    monkeypatch.setattr(adapter, "_load_task_config", lambda task: {})
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        path = tmp_path / "4096" / "niah_single_1" / "validation.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(main.subprocess, "run", run)
    adapter._generate_task_data("niah_single_1", 4096, tmp_path, "gated/model", "hf", "base", 10, 42, timeout=600)
    assert calls[0][1]["env"]["HF_TOKEN"] == "mounted-hf-token"
    assert "mounted-hf-token" not in str(calls[0][0])


def test_legacy_credentials_and_hf_env_remain_supported(monkeypatch):
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None))
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    monkeypatch.setenv("MODEL_API_KEY", "direct-key")
    monkeypatch.setenv("HF_TOKEN", "env-hf-token")
    client = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    adapter._make_api_client("https://model/v1")
    assert client.call_args.kwargs["api_key"] == "direct-key"
    assert adapter._hf_token() == "env-hf-token"
