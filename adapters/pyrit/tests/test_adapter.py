"""Tests for the PyRIT adapter.

Scaffold for RHAIRFE-3602 — cases below mark the planned coverage mirrored
from adapters/promptfoo/tests/test_adapter.py. Implement alongside the
CLI bridge and metrics mapping.
"""

from __future__ import annotations

import pytest
from main import SECRET_PARAM_NAMES, PyritAdapter, _strip_secret_params


def test_adapter_loads_job_fixture(pyrit_adapter: PyritAdapter):
    """Scaffold smoke: PyritAdapter loads meta/job.json."""
    assert pyrit_adapter.job_spec.provider_id == "pyrit"
    assert pyrit_adapter.job_spec.benchmark_id == "pyrit-scan"


def test_secret_param_names():
    """Attacker/scorer API key params must stay on the secret denylist."""
    assert "attacker_provider_api_key" in SECRET_PARAM_NAMES
    assert "scorer_provider_api_key" in SECRET_PARAM_NAMES


def test_strip_secret_params_removes_keys():
    """_strip_secret_params drops credential parameters and keeps the rest."""
    cleaned = _strip_secret_params(
        {
            "scenario": "prompt_injection",
            "attacker_provider_api_key": "sk-attacker",
            "scorer_provider_api_key": "sk-scorer",
            "max_objectives": 5,
        }
    )
    assert cleaned == {"scenario": "prompt_injection", "max_objectives": 5}


@pytest.mark.skip(reason="scaffold: implement env/conf builders from JobSpec")
def test_build_pyrit_env_and_conf():
    pass


@pytest.mark.skip(reason="scaffold: implement CLI argv / scenario validation")
def test_build_scan_argv_and_scenario_validation():
    pass


@pytest.mark.skip(reason="scaffold: implement metrics mapping from fixture JSON")
def test_compute_metrics_from_fixture():
    pass


@pytest.mark.skip(reason="scaffold: happy path with mocked subprocess")
def test_run_benchmark_job_happy_path(pyrit_adapter, mock_callbacks):
    pass


@pytest.mark.skip(reason="scaffold: fatal vs non-fatal exit codes and FAILED status")
def test_run_benchmark_job_exit_codes(pyrit_adapter, mock_callbacks):
    pass


@pytest.mark.skip(reason="scaffold: OCI excludes secrets")
def test_oci_excludes_secrets(pyrit_adapter, mock_callbacks):
    pass


@pytest.mark.skip(reason="scaffold: PERSISTING_ARTIFACTS reported without OCI")
def test_persisting_artifacts_without_oci(pyrit_adapter, mock_callbacks):
    pass
