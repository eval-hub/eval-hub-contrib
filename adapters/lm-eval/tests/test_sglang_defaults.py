"""Keep the installed SGLang CLI backend opt-in for remote model code."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.harden_sglang import harden_sglang_default


def test_harden_sglang_changes_only_the_default(tmp_path):
    backend = tmp_path / "sglang_causallms.py"
    before = (
        "class SGLangLM:\n"
        "    def __init__(self,\n"
        "        trust_remote_code: bool = True,\n"
        "    ):\n"
        '        self.model_args = {"trust_remote_code": trust_remote_code}\n'
    )
    backend.write_text(before)

    harden_sglang_default(backend)

    expected = before.replace(
        "trust_remote_code: bool = True", "trust_remote_code: bool = False"
    )
    assert backend.read_text() == expected
    harden_sglang_default(backend)
    assert backend.read_text() == expected


@pytest.mark.parametrize(
    "source",
    [
        "        trust_remote_code: bool = None,\n",
        "        trust_remote_code: bool = True,\n" * 2,
        "        trust_remote_code: bool = True,\n        trust_remote_code: bool = False,\n",
    ],
    ids=["unknown-default", "ambiguous-default", "mixed-defaults"],
)
def test_harden_sglang_rejects_unexpected_source(tmp_path, source):
    backend = tmp_path / "sglang_causallms.py"
    backend.write_text(source)

    with pytest.raises(RuntimeError, match="Unexpected SGLang"):
        harden_sglang_default(backend)

    assert backend.read_text() == source


@pytest.mark.lmeval
@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ("", False),
        (",trust_remote_code=false", False),
        (",trust_remote_code=true", True),
    ],
    ids=["default-disabled", "explicit-disabled", "explicit-opt-in"],
)
def test_installed_sglang_forwards_remote_code_opt_in(
    lmeval, monkeypatch, extra_args, expected
):
    backend_module = importlib.import_module("lm_eval.models.sglang_causallms")
    engine = Mock(
        return_value=SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=object())
        )
    )
    monkeypatch.setattr(
        backend_module, "sgl", SimpleNamespace(Engine=engine), raising=False
    )
    monkeypatch.setattr(backend_module, "find_spec", lambda _name: object())

    backend = backend_module.SGLangLM.create_from_arg_string(
        "pretrained=test/model" + extra_args
    )

    engine.assert_called_once()
    assert engine.call_args.kwargs["trust_remote_code"] is expected
    assert backend.model_args["trust_remote_code"] is expected
