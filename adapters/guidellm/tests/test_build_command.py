"""Unit tests for GuideLLM CLI command construction from job_spec.parameters."""

from pathlib import Path
from types import SimpleNamespace
import os

import pytest
import main

from main import GuideLLMAdapter


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


@pytest.fixture
def adapter() -> GuideLLMAdapter:
    adapter = GuideLLMAdapter(job_spec_path="meta/job.json")
    adapter.results_dir = Path("/tmp/guidellm-test")
    return adapter


@pytest.fixture(autouse=True)
def clear_offline_environment(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "case,expected_offline",
    [("tokenizer", True), ("processor", True), ("override", False),
     ("missing", False), ("outside", False), ("symlink", False),
     ("root", False), ("file", False), ("relative", False)],
)
def test_subprocess_offline_detection(adapter, monkeypatch, tmp_path, case, expected_offline):
    root = tmp_path / "test_data"
    tokenizer = root / "tokenizer"
    tokenizer.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(adapter, "TEST_DATA_DIR", root)
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["tokenizer"] = str(tokenizer)
    if case == "processor":
        job_spec.parameters = {"processor": str(tokenizer)}
    elif case == "override":
        job_spec.parameters["processor"] = "gpt2"
    elif case in {"missing", "outside", "symlink", "root", "file", "relative"}:
        job_spec.parameters["tokenizer"] = {
            "missing": str(root / "missing"), "outside": str(outside),
            "symlink": str(root / "linked"), "root": str(root),
            "file": str(root / "tokenizer.json"), "relative": "tokenizer",
        }[case]

    captured = {}
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None))
    monkeypatch.setattr(main, "run_streaming_subprocess", lambda cmd, **kwargs: captured.update(kwargs))

    adapter._run_guidellm(adapter._build_guidellm_command(job_spec))

    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert captured["env"].get(name) == ("1" if expected_offline else None)
        assert name not in os.environ  # Offline settings belong only to the subprocess.


@pytest.mark.parametrize("parameter", ["tokenizer", "processor"])
@pytest.mark.parametrize("offline", [False, True])
def test_local_tokenizer_used_as_processor(adapter, monkeypatch, parameter, offline):
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters[parameter] = "/test_data/tokenizer"

    cmd = adapter._build_guidellm_command(job_spec)

    assert _flag_value(cmd, "--processor") == "/test_data/tokenizer"
    assert cmd.count("--processor") == 1
    assert "gpt2" not in cmd


def test_explicit_processor_takes_precedence(adapter):
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters.update(processor="/test_data/custom", tokenizer="/test_data/tokenizer")

    cmd = adapter._build_guidellm_command(job_spec)

    assert _flag_value(cmd, "--processor") == "/test_data/custom"


def test_connected_synthetic_data_defaults_to_gpt2(adapter):
    cmd = adapter._build_guidellm_command(adapter.job_spec)

    assert _flag_value(cmd, "--processor") == "gpt2"


@pytest.mark.parametrize("name", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"])
@pytest.mark.parametrize("value", ["1", "true", "YES", "ON"])
def test_offline_synthetic_data_requires_explicit_tokenizer(adapter, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="processor or tokenizer in offline mode"):
        adapter._build_guidellm_command(adapter.job_spec)


def test_offline_file_data_does_not_require_processor(adapter, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["data"] = "/test_data/prompts.jsonl"

    cmd = adapter._build_guidellm_command(job_spec)

    assert "--processor" not in cmd


def test_outputs_defaults_when_not_in_parameters(adapter: GuideLLMAdapter):
    cmd = adapter._build_guidellm_command(adapter.job_spec)

    assert _flag_value(cmd, "--outputs") == "json,csv,html,yaml"


@pytest.mark.parametrize("source", ["staged", *GuideLLMAdapter.OFFLINE_ENV_VARS])
@pytest.mark.parametrize("outputs", [None, "json", "json,csv,yaml", "HTML", "json, HTML"])
def test_offline_output_selection(adapter, monkeypatch, tmp_path, source, outputs):
    job_spec = adapter.job_spec.model_copy(deep=True)
    if source == "staged":
        root = tmp_path / "test_data"
        tokenizer = root / "tokenizer"
        tokenizer.mkdir(parents=True)
        monkeypatch.setattr(adapter, "TEST_DATA_DIR", root)
        job_spec.parameters["tokenizer"] = str(tokenizer)
    else:
        monkeypatch.setenv(source, "1")
        # File input isolates output behavior from synthetic-data validation.
        job_spec.parameters["data"] = "/test_data/prompts.jsonl"
    if outputs is not None:
        job_spec.parameters["outputs"] = outputs

    if outputs is not None and "html" in outputs.lower():
        with pytest.raises(ValueError, match="HTML output is unavailable in offline mode"):
            adapter._build_guidellm_command(job_spec)
    else:
        cmd = adapter._build_guidellm_command(job_spec)
        assert _flag_value(cmd, "--outputs") == (outputs or "json,csv,yaml")


@pytest.mark.parametrize(
    "outputs",
    ["json", "json,html", "json,csv,html,yaml"],
)
def test_outputs_from_job_spec_parameters(adapter: GuideLLMAdapter, outputs: str):
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["outputs"] = outputs

    cmd = adapter._build_guidellm_command(job_spec)

    assert _flag_value(cmd, "--outputs") == outputs


def test_outputs_normalizes_case(adapter: GuideLLMAdapter):
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["outputs"] = "JSON,HTML"

    cmd = adapter._build_guidellm_command(job_spec)

    assert _flag_value(cmd, "--outputs") == "json,html"


@pytest.mark.parametrize("outputs", ["josn", "json,foo", ""])
def test_outputs_rejects_invalid_formats(adapter: GuideLLMAdapter, outputs: str):
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["outputs"] = outputs

    with pytest.raises(ValueError, match="Invalid output format|at least one format"):
        adapter._build_guidellm_command(job_spec)
