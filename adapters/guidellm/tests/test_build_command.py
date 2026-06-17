"""Unit tests for GuideLLM CLI command construction from job_spec.parameters."""

from pathlib import Path

import pytest

from main import GuideLLMAdapter


def _flag_value(cmd: list[str], flag: str) -> str:
    idx = cmd.index(flag)
    return cmd[idx + 1]


@pytest.fixture
def adapter() -> GuideLLMAdapter:
    adapter = GuideLLMAdapter(job_spec_path="meta/job.json")
    adapter.results_dir = Path("/tmp/guidellm-test")
    return adapter


def test_outputs_defaults_when_not_in_parameters(adapter: GuideLLMAdapter):
    cmd = adapter._build_guidellm_command(adapter.job_spec)

    assert _flag_value(cmd, "--outputs") == "json,csv,html,yaml"


@pytest.mark.parametrize(
    "outputs",
    ["json", "json,html", "json,csv,html,yaml"],
)
def test_outputs_from_job_spec_parameters(adapter: GuideLLMAdapter, outputs: str):
    job_spec = adapter.job_spec.model_copy(deep=True)
    job_spec.parameters["outputs"] = outputs

    cmd = adapter._build_guidellm_command(job_spec)

    assert _flag_value(cmd, "--outputs") == outputs
