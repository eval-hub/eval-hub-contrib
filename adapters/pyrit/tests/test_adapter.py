"""Tests for the PyRIT adapter.

The pyrit_scan subprocess boundary (_run_pyrit_scan / _fetch_scan_results /
_stop_pyrit_server) is monkeypatched — no real pyrit_scan binary or network
calls are needed to run these tests.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from evalhub.adapter import JobPhase, JobStatus
from evalhub.adapter.models.job import (
    JobSpecExports,
    JobSpecExportsOCI,
    OCIArtifactResult,
)
from evalhub.models.api import OCICoordinates
from main import (
    DEFAULT_SCAN_TIMEOUT_SECONDS,
    SECRET_PARAM_NAMES,
    PyritAdapter,
    _build_pyrit_conf,
    _build_pyrit_env,
    _build_scan_argv,
    _compute_metrics,
    _dotenv_payload,
    _format_pyrit_scan_failure,
    _normalize_openai_base_url,
    _parse_scenario_result_id,
    _redact_secrets,
    _resolve_api_key,
    _resolve_scan_timeout,
    _strip_secret_params,
)


class _FakeCompletedProcess:
    """Minimal subprocess.CompletedProcess stand-in."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _scan_export(
    *,
    successes: int,
    failures: int,
    errors: int = 0,
    scenario: str = "foundry.red_team_agent",
) -> dict:
    """Return a minimal ScenarioResult-shaped export for metrics tests."""
    rows = []
    for _ in range(successes):
        rows.append({"outcome": "success", "objective": "do bad thing"})
    for _ in range(failures):
        rows.append({"outcome": "failure", "objective": "do bad thing"})
    for _ in range(errors):
        rows.append({"outcome": "error", "objective": "do bad thing"})
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "scenario_result_id": "11111111-1111-1111-1111-111111111111",
        "scenario_name": scenario,
        "attack_results": {"prompt_sending__harmbench": rows},
        "objective_achieved_rate": int(
            round(100 * successes / max(successes + failures + errors, 1))
        ),
        "total_attacks": successes + failures + errors,
    }


# ---------------------------------------------------------------------------
# Unit tests: secrets / api key / URL
# ---------------------------------------------------------------------------


def test_adapter_loads_job_fixture(pyrit_adapter: PyritAdapter):
    """PyritAdapter loads meta/job.json."""
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
            "scenario": "foundry.red_team_agent",
            "attacker_provider_api_key": "sk-attacker",
            "scorer_provider_api_key": "sk-scorer",
            "max_objectives": 5,
        }
    )
    assert cleaned == {
        "scenario": "foundry.red_team_agent",
        "max_objectives": 5,
    }


def test_resolve_api_key_env(monkeypatch):
    """OPENAI_API_KEY env var is returned when set and no secret_ref is configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "my-key-from-env")
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "my-key-from-env"


def test_resolve_api_key_sentinel(monkeypatch):
    """'not-required' sentinel is returned when no env key and no secret_ref."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "not-required"


def test_normalize_openai_base_url_appends_v1():
    """Bare base URLs get /v1 appended; chat/completions suffix is stripped."""
    assert (
        _normalize_openai_base_url("http://localhost:8080")
        == "http://localhost:8080/v1"
    )
    assert (
        _normalize_openai_base_url("http://localhost:8080/v1")
        == "http://localhost:8080/v1"
    )
    assert (
        _normalize_openai_base_url("http://localhost:8080/v1/chat/completions")
        == "http://localhost:8080/v1"
    )


def test_normalize_openai_base_url_missing_raises():
    """Empty model URL raises ValueError."""
    with pytest.raises(ValueError, match="model.url"):
        _normalize_openai_base_url("")


# ---------------------------------------------------------------------------
# Unit tests: env / conf / argv
# ---------------------------------------------------------------------------


def test_build_pyrit_env_and_conf(tmp_path, monkeypatch):
    """Env uses secret_ref role keys for attacker/scorer, not job params."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = MagicMock()
    config.model.url = "http://localhost:8080"
    config.model.name = "test-model"
    config.parameters = {
        "attacker_provider": "attacker-model",
        "scorer_provider": "scorer-model",
    }

    role_values = {
        "attacker_url": "http://attacker:8000/v1",
        "attacker_api-key": "sk-attacker",
        "scorer_url": "http://scorer:8000/v1",
        "scorer_api-key": "sk-scorer",
    }
    monkeypatch.setattr(
        "main.read_model_auth_key",
        lambda name: role_values.get(name),
    )

    env = _build_pyrit_env(config, tmp_path, "sk-target")
    assert env["HOME"] == str(tmp_path)
    assert env["OPENAI_CHAT_ENDPOINT"] == "http://localhost:8080/v1"
    assert env["OPENAI_CHAT_KEY"] == "sk-target"
    assert env["OPENAI_CHAT_MODEL"] == "test-model"
    assert env["ADVERSARIAL_CHAT_MODEL"] == "attacker-model"
    assert env["ADVERSARIAL_CHAT_ENDPOINT"] == "http://attacker:8000/v1"
    assert env["ADVERSARIAL_CHAT_KEY"] == "sk-attacker"
    assert env["OBJECTIVE_SCORER_CHAT_MODEL"] == "scorer-model"
    assert env["OBJECTIVE_SCORER_CHAT_ENDPOINT"] == "http://scorer:8000/v1"
    assert env["OBJECTIVE_SCORER_CHAT_KEY"] == "sk-scorer"

    dotenv = _dotenv_payload(env)
    assert "OPENAI_CHAT_KEY=sk-target" in dotenv
    assert "attacker_provider_api_key" not in dotenv

    env_file = tmp_path / ".env"
    conf = _build_pyrit_conf(env_file=env_file)
    assert conf["memory_db_type"] == "in_memory"
    assert conf["env_files"] == [str(env_file)]
    assert conf["server"]["auth_mode"] == "none"
    assert conf["initializers"][0] == "technique"
    assert any(
        (isinstance(i, dict) and i.get("name") == "target") or i == "target"
        for i in conf["initializers"]
    )


def test_build_pyrit_env_rejects_plaintext_api_key_params(tmp_path):
    """Plaintext attacker/scorer API key job params are rejected."""
    config = MagicMock()
    config.model.url = "http://localhost:8080"
    config.model.name = "test-model"
    config.parameters = {"attacker_provider_api_key": "sk-attacker"}
    with pytest.raises(ValueError, match="plaintext credential parameters"):
        _build_pyrit_env(config, tmp_path, "sk-target")


def test_build_pyrit_env_falls_back_to_target_without_roles(tmp_path, monkeypatch):
    """Missing role keys fall back to the target model connection."""
    monkeypatch.setattr("main.read_model_auth_key", lambda _name: None)
    config = MagicMock()
    config.model.url = "http://localhost:8080"
    config.model.name = "test-model"
    config.parameters = {}
    env = _build_pyrit_env(config, tmp_path, "sk-target")
    assert env["ADVERSARIAL_CHAT_ENDPOINT"] == "http://localhost:8080/v1"
    assert env["ADVERSARIAL_CHAT_KEY"] == "sk-target"
    assert env["ADVERSARIAL_CHAT_MODEL"] == "test-model"
    assert env["OBJECTIVE_SCORER_CHAT_KEY"] == "sk-target"


def test_resolve_scan_timeout_null_uses_default():
    """Explicit null scan_timeout must not raise TypeError."""
    assert _resolve_scan_timeout(None) == DEFAULT_SCAN_TIMEOUT_SECONDS
    assert _resolve_scan_timeout({}) == DEFAULT_SCAN_TIMEOUT_SECONDS
    assert _resolve_scan_timeout({"scan_timeout": None}) == DEFAULT_SCAN_TIMEOUT_SECONDS
    assert _resolve_scan_timeout({"scan_timeout": 30}) == 30


def test_build_scan_argv_and_scenario_validation(tmp_path):
    """CLI argv includes scenario, target, --start-server, and max-dataset-size."""
    config = MagicMock()
    config.benchmark_id = "pyrit-scan"
    config.parameters = {
        "scenario": "foundry.red_team_agent",
        "max_objectives": 3,
        "techniques": ["base64", "rot13"],
    }
    conf_path = tmp_path / ".pyrit_conf"
    argv = _build_scan_argv(config, conf_path)
    assert argv[:4] == [
        "pyrit_scan",
        "run",
        "foundry.red_team_agent",
        "--target",
    ]
    assert "openai_chat" in argv
    assert "--start-server" in argv
    assert "--auth-mode" in argv
    assert argv[argv.index("--max-dataset-size") + 1] == "3"
    assert "base64" in argv
    assert "rot13" in argv

    config.parameters = {}
    with pytest.raises(ValueError, match="parameters.scenario"):
        _build_scan_argv(config, conf_path)

    config.benchmark_id = "not-a-pyrit-benchmark"
    config.parameters = {"scenario": "x"}
    with pytest.raises(ValueError, match="Unsupported benchmark_id"):
        _build_scan_argv(config, conf_path)


def test_parse_scenario_result_id():
    """Result ID is extracted from pyrit_scan summary stdout."""
    stdout = (
        "\nScenario: foundry.red_team_agent\n"
        "  Result ID:      605d715b-7c07-4bde-a8f9-22fea0b50c4f\n"
        "  Status:         COMPLETED\n"
        "  Attack Results: 4\n"
        "  Success Rate:   25%\n"
    )
    assert (
        _parse_scenario_result_id(stdout)
        == "605d715b-7c07-4bde-a8f9-22fea0b50c4f"
    )


# ---------------------------------------------------------------------------
# Unit tests: metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_from_fixture():
    """pass_rate == failures / scored outcomes (success|failure only)."""
    export = _scan_export(successes=1, failures=3)
    results, pass_rate, n_evaluated = _compute_metrics(export)
    assert n_evaluated == 4
    assert pass_rate == pytest.approx(0.75)
    by_name = {r.metric_name: r.metric_value for r in results}
    assert by_name["pass_rate"] == pytest.approx(0.75)
    assert by_name["attack_success_rate"] == pytest.approx(0.25)
    assert by_name["n_evaluated"] == 4


def test_compute_metrics_excludes_error_and_undetermined():
    """ERROR/UNDETERMINED must not inflate pass_rate (Julian high finding)."""
    export = {
        "attack_results": {
            "baseline": [
                {"outcome": "error", "objective": "a"},
                {"outcome": "undetermined", "objective": "b"},
                {"outcome": "error", "objective": "c"},
            ]
        }
    }
    results, pass_rate, n_evaluated = _compute_metrics(export)
    assert n_evaluated == 0
    assert pass_rate is None
    by_name = {r.metric_name: r.metric_value for r in results}
    assert "pass_rate" not in by_name
    assert "attack_success_rate" not in by_name


def test_compute_metrics_mixed_scored_and_errors():
    """Only success/failure count toward n_evaluated when errors are present."""
    export = {
        "attack_results": {
            "base64": [
                {"outcome": "success", "objective": "a"},
                {"outcome": "failure", "objective": "b"},
                {"outcome": "error", "objective": "c"},
                {"outcome": "undetermined", "objective": "d"},
            ]
        }
    }
    _results, pass_rate, n_evaluated = _compute_metrics(export)
    assert n_evaluated == 2
    assert pass_rate == pytest.approx(0.5)


def test_compute_metrics_empty_attack_results_is_unknown():
    """Empty attack_results never invent a score from CLI summary rates."""
    export = {
        "attack_results": {},
        "objective_achieved_rate": 40,
        "total_attacks": 10,
    }
    results, pass_rate, n_evaluated = _compute_metrics(export)
    assert n_evaluated == 0
    assert pass_rate is None
    by_name = {r.metric_name: r.metric_value for r in results}
    assert "pass_rate" not in by_name
    assert "attack_success_rate" not in by_name


def test_compute_metrics_zero_attacks_is_unknown():
    """n_evaluated=0 must not report pass_rate=1.0 (Julian medium finding)."""
    export = {
        "attack_results": {},
        "objective_achieved_rate": 0,
        "total_attacks": 0,
    }
    results, pass_rate, n_evaluated = _compute_metrics(export)
    assert n_evaluated == 0
    assert pass_rate is None
    by_name = {r.metric_name: r.metric_value for r in results}
    assert "pass_rate" not in by_name


# ---------------------------------------------------------------------------
# Integration-style: run_benchmark_job with mocked subprocess
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_benchmark_job_happy_path(pyrit_adapter, mock_callbacks, monkeypatch):
    """Happy path: mocked pyrit_scan + results fetch → JobResults."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {
        "scenario": "foundry.red_team_agent",
        "max_objectives": 2,
    }
    export = _scan_export(successes=1, failures=1)
    result_id = "605d715b-7c07-4bde-a8f9-22fea0b50c4f"

    def fake_run(argv, env, work_dir, timeout=7200):
        assert argv[0] == "pyrit_scan"
        assert "foundry.red_team_agent" in argv
        assert env["OPENAI_CHAT_MODEL"] == config.model.name
        # Conf + dotenv must exist before the CLI would run.
        assert (Path(work_dir) / ".pyrit_conf").exists()
        assert (Path(work_dir) / ".env").exists()
        conf = yaml.safe_load((Path(work_dir) / ".pyrit_conf").read_text())
        assert conf["server"]["auth_mode"] == "none"
        return _FakeCompletedProcess(
            0,
            stdout=(
                f"Result ID:      {result_id}\n"
                "Attack Results: 2\n"
                "Success Rate:   50%\n"
            ),
        )

    monkeypatch.setattr(main_mod, "_run_pyrit_scan", fake_run)
    monkeypatch.setattr(
        main_mod,
        "_fetch_scan_results",
        lambda **kwargs: export,
    )
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    results = pyrit_adapter.run_benchmark_job(config, mock_callbacks)

    assert results.benchmark_id == "pyrit-scan"
    assert results.overall_score == pytest.approx(0.5)
    assert results.num_examples_evaluated == 2
    assert results.additional_info["scenario"] == "foundry.red_team_agent"
    assert results.additional_info["scenario_result_id"] == result_id
    assert results.additional_info["metrics_source"] == "results_api"
    assert results.additional_info["attacker_model"] == config.model.name
    assert results.additional_info["scorer_model"] == config.model.name
    assert results.additional_info["attacker_uses_target"] is True
    assert results.additional_info["scorer_uses_target"] is True
    assert "attacker_provider_api_key" not in results.additional_info
    assert "pyrit_scan_json_b64" in (results.evaluation_metadata or {})
    assert "attacker=" in (results.eval_card.safety_evaluations[0].benchmark or "")
    assert "same as target" in (results.eval_card.developer_footnotes or "")

    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    # No OCI export configured on this fixture → no PERSISTING_ARTIFACTS.
    assert JobPhase.PERSISTING_ARTIFACTS not in phases


@pytest.mark.integration
def test_run_benchmark_job_fails_when_nothing_scored(
    pyrit_adapter, mock_callbacks, monkeypatch
):
    """All error/undetermined outcomes must fail the job (William #1)."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {"scenario": "foundry.red_team_agent"}
    export = {
        "attack_results": {
            "baseline": [
                {"outcome": "error", "objective": "a"},
                {"outcome": "undetermined", "objective": "b"},
            ]
        }
    }

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(
            0, stdout="Result ID: 605d715b-7c07-4bde-a8f9-22fea0b50c4f\n"
        ),
    )
    monkeypatch.setattr(main_mod, "_fetch_scan_results", lambda **k: export)
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    with pytest.raises(RuntimeError, match="no probes were scored"):
        pyrit_adapter.run_benchmark_job(config, mock_callbacks)

    failed = [
        c
        for c in mock_callbacks.report_status.call_args_list
        if c.args[0].status == JobStatus.FAILED
    ]
    assert len(failed) == 1


@pytest.mark.integration
def test_run_benchmark_job_fails_without_result_id(
    pyrit_adapter, mock_callbacks, monkeypatch
):
    """Missing Result ID must not score from CLI summary (William #2)."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {"scenario": "foundry.red_team_agent"}

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(
            0, stdout="Attack Results: 4\nSuccess Rate:   0%\n"
        ),
    )
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    with pytest.raises(RuntimeError, match="refusing to score from CLI summary"):
        pyrit_adapter.run_benchmark_job(config, mock_callbacks)


@pytest.mark.integration
def test_run_benchmark_job_fails_when_results_fetch_fails(
    pyrit_adapter, mock_callbacks, monkeypatch
):
    """Results API failure must not fall back to summary scoring (William #2)."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {"scenario": "foundry.red_team_agent"}

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(
            0, stdout="Result ID: 605d715b-7c07-4bde-a8f9-22fea0b50c4f\n"
        ),
    )

    def boom(**_kwargs):
        raise TimeoutError("backend down")

    monkeypatch.setattr(main_mod, "_fetch_scan_results", boom)
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    with pytest.raises(RuntimeError, match="Failed to fetch pyrit scenario results"):
        pyrit_adapter.run_benchmark_job(config, mock_callbacks)


def test_redact_secrets_masks_keys_and_patterns():
    """Redaction covers explicit secrets and common token shapes."""
    text = (
        "Bearer sk-live-ABCDEFG123456 "
        "api_key=sk-secret-value "
        "OPENAI_CHAT_KEY=sk-env-secret"
    )
    out = _redact_secrets(text, ["sk-env-secret"])
    assert "sk-env-secret" not in out
    assert "sk-live-ABCDEFG123456" not in out
    assert "sk-secret-value" not in out
    assert "***REDACTED***" in out


def test_format_pyrit_scan_failure_includes_backend_log(tmp_path, monkeypatch):
    """Failure message appends redacted backend log tail (William live #1)."""
    log_path = tmp_path / "pyrit_backend.log"
    log_path.write_text(
        "ok\n"
        "openai.AuthenticationError: Error code: 401 - invalid api key "
        "sk-should-not-leak\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "main.BACKEND_LOG_CANDIDATES",
        (log_path,),
    )
    completed = _FakeCompletedProcess(
        1,
        stdout="Error sending prompt with conversation ID xxx\n" * 50,
        stderr="progress bar noise",
    )
    msg = _format_pyrit_scan_failure(
        completed,
        env={"OPENAI_CHAT_KEY": "sk-should-not-leak"},
        work_dir=tmp_path,
    )
    assert "pyrit_scan failed (exit 1)" in msg
    assert "pyrit_backend.log" in msg
    assert "AuthenticationError" in msg
    assert "sk-should-not-leak" not in msg
    assert "***REDACTED***" in msg
    # CLI stdout is truncated — not the full repeated blob.
    assert msg.count("Error sending prompt") < 50


@pytest.mark.integration
def test_run_benchmark_job_exit_codes(pyrit_adapter, mock_callbacks, monkeypatch):
    """Non-zero pyrit_scan exit raises RuntimeError and reports FAILED."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {"scenario": "foundry.red_team_agent"}

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(1, "", "boom"),
    )
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    with pytest.raises(RuntimeError, match="pyrit_scan failed"):
        pyrit_adapter.run_benchmark_job(config, mock_callbacks)

    failed = [
        c
        for c in mock_callbacks.report_status.call_args_list
        if c.args[0].status == JobStatus.FAILED
    ]
    assert len(failed) == 1
    assert "boom" in failed[0].args[0].error_message.message


@pytest.mark.integration
def test_oci_excludes_secrets(pyrit_adapter, mock_callbacks, monkeypatch):
    """OCI export copies only scan_results.json, never the job-local .env."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {
        "scenario": "foundry.red_team_agent",
        "attacker_provider": "attacker-model",
        "scorer_provider": "scorer-model",
    }
    monkeypatch.setattr(
        "main.read_model_auth_key",
        lambda name: {
            "attacker_api-key": "sk-attacker-secret",
            "scorer_api-key": "sk-scorer-secret",
        }.get(name),
    )
    config.exports = JobSpecExports(
        oci=JobSpecExportsOCI(
            coordinates=OCICoordinates(
                oci_host="quay.io",
                oci_repository="evalhub/pyrit-results",
                oci_tag="test",
            )
        )
    )

    export = _scan_export(successes=0, failures=2)
    seen_exported_files: set[str] = set()
    seen_exported_raw: list[str] = []

    def fake_create_oci(spec):
        # Snapshot before run_benchmark_job's finally deletes work_dir.
        for path in Path(spec.files_path).iterdir():
            seen_exported_files.add(path.name)
            if path.name == "scan_results.json":
                seen_exported_raw.append(path.read_text())
        return OCIArtifactResult(
            digest="sha256:" + "0" * 64,
            reference="quay.io/evalhub/pyrit-results:test",
        )

    mock_callbacks.create_oci_artifact.side_effect = fake_create_oci

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(
            0, stdout="Result ID: 605d715b-7c07-4bde-a8f9-22fea0b50c4f\n"
        ),
    )
    monkeypatch.setattr(main_mod, "_fetch_scan_results", lambda **k: export)
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-target")

    results = pyrit_adapter.run_benchmark_job(config, mock_callbacks)

    assert results.oci_artifact is not None
    assert seen_exported_files == {"scan_results.json"}
    raw = seen_exported_raw[0]
    assert "sk-attacker-secret" not in raw
    assert "sk-scorer-secret" not in raw
    assert "sk-target" not in raw
    assert "attacker_provider_api_key" not in (results.additional_info or {})


@pytest.mark.integration
def test_persisting_artifacts_without_oci(pyrit_adapter, mock_callbacks, monkeypatch):
    """PERSISTING_ARTIFACTS is skipped when OCI export is not configured."""
    import main as main_mod

    config = copy.deepcopy(pyrit_adapter.job_spec)
    config.parameters = {"scenario": "garak.encoding"}
    config.exports = None
    export = _scan_export(successes=0, failures=1, scenario="garak.encoding")

    monkeypatch.setattr(
        main_mod,
        "_run_pyrit_scan",
        lambda *a, **k: _FakeCompletedProcess(
            0, stdout="Result ID: 605d715b-7c07-4bde-a8f9-22fea0b50c4f\n"
        ),
    )
    monkeypatch.setattr(main_mod, "_fetch_scan_results", lambda **k: export)
    monkeypatch.setattr(main_mod, "_stop_pyrit_server", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_resolve_api_key", lambda _c: "sk-test")

    results = pyrit_adapter.run_benchmark_job(config, mock_callbacks)
    assert results.oci_artifact is None
    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS not in phases
    mock_callbacks.create_oci_artifact.assert_not_called()

    # MLflow artifact payload is base64 eval JSON, independent of OCI.
    b64 = (results.evaluation_metadata or {}).get("pyrit_scan_json_b64")
    assert b64
    assert json.loads(base64.b64decode(b64))["total_attacks"] == 1
