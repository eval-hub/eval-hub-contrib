"""Tests for the promptfoo adapter.

The promptfoo CLI subprocess boundary (_run_promptfoo_cli) is monkeypatched —
no real `promptfoo` binary or network calls are needed to run these tests.
Canned eval.json fixtures below match the REAL shape produced by promptfoo
0.123.1 (verified 2026-09-21 against `promptfoo eval` / `promptfoo redteam
run` + `promptfoo export eval`), not a guessed schema.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
from evalhub.adapter import JobCallbacks, JobPhase, JobStatus
from evalhub.adapter.models.cards import EnvironmentCardMetadata, EvalCardMetadata
from main import (
    PromptfooAdapter,
    _build_eval_config,
    _build_redteam_config,
    _build_target_provider,
    _compute_metrics,
    _compute_plugin_breakdown,
    _extract_eval_id,
    _resolve_api_key,
)

# ---------------------------------------------------------------------------
# Canned promptfoo eval.json fixtures (real shape, promptfoo 0.123.1)
# ---------------------------------------------------------------------------


def _eval_json(
    successes: int, failures: int, errors: int, plugin_rows: list[dict] | None = None
) -> dict:
    results = []
    for _ in range(successes):
        results.append({"success": True, "score": 1, "metadata": {}})
    for _ in range(failures):
        results.append({"success": False, "score": 0, "metadata": {}})
    for _ in range(errors):
        results.append(
            {"success": False, "score": 0, "metadata": {}, "failureReason": "error"}
        )
    if plugin_rows is not None:
        results = plugin_rows
    return {
        "evalId": "eval-Test-2026-09-21T00:00:00",
        "results": {
            "version": 3,
            "stats": {"successes": successes, "failures": failures, "errors": errors},
            "results": results,
        },
        "config": {},
    }


# ---------------------------------------------------------------------------
# Unit tests: _resolve_api_key
# ---------------------------------------------------------------------------


def test_resolve_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-key-from-env")
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "my-key-from-env"


def test_resolve_api_key_sentinel(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = MagicMock()
    config.model.auth = None
    assert _resolve_api_key(config) == "not-required"


# ---------------------------------------------------------------------------
# Unit tests: config generation
# ---------------------------------------------------------------------------


def test_build_target_provider_appends_v1():
    config = MagicMock()
    config.model.url = "http://localhost:8080"
    config.model.name = "my-model"
    provider = _build_target_provider(config, "sk-test", request_timeout=60)
    assert provider["id"] == "openai:chat:my-model"
    assert provider["config"]["apiBaseUrl"] == "http://localhost:8080/v1"
    assert provider["config"]["apiKey"] == "sk-test"
    assert provider["config"]["timeoutMs"] == 60_000


def test_build_target_provider_missing_url_raises():
    config = MagicMock()
    config.model.url = ""
    with pytest.raises(ValueError, match="model.url"):
        _build_target_provider(config, "key", 60)


def test_build_eval_config_from_prompts_and_tests():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "prompts": ["hi {{name}}"],
        "tests": [{"vars": {"name": "World"}}],
    }
    provider = {"id": "openai:chat:m"}
    pf_config = _build_eval_config(config, provider)
    assert pf_config["prompts"] == ["hi {{name}}"]
    assert pf_config["providers"] == [provider]
    assert pf_config["tests"] == [{"vars": {"name": "World"}}]


def test_build_eval_config_missing_params_raises():
    config = MagicMock()
    config.parameters = {}
    with pytest.raises(ValueError, match="config_yaml"):
        _build_eval_config(config, {"id": "p"})


def test_build_eval_config_passthrough_overwrites_providers():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "config_yaml": "description: mine\nproviders:\n  - id: should-be-replaced\ntests:\n  - vars: {}\n"
    }
    provider = {"id": "openai:chat:m"}
    pf_config = _build_eval_config(config, provider)
    assert pf_config["providers"] == [provider]
    assert pf_config["description"] == "mine"


def test_build_redteam_config_defaults():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {}
    provider = {"id": "openai:chat:m"}
    pf_config = _build_redteam_config(config, provider)
    assert pf_config["targets"] == [provider]
    assert pf_config["redteam"]["purpose"] == "An AI assistant"
    assert pf_config["redteam"]["numTests"] == 5
    assert len(pf_config["redteam"]["plugins"]) > 0
    assert all(p["numTests"] == 5 for p in pf_config["redteam"]["plugins"])


def test_build_redteam_config_custom_plugins():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "plugins": ["sql-injection"],
        "num_tests_per_plugin": 2,
        "purpose": "a bank assistant",
    }
    pf_config = _build_redteam_config(config, {"id": "p"})
    assert pf_config["redteam"]["plugins"] == [{"id": "sql-injection", "numTests": 2}]
    assert pf_config["redteam"]["purpose"] == "a bank assistant"


# ---------------------------------------------------------------------------
# Unit tests: eval ID extraction
# ---------------------------------------------------------------------------


def test_extract_eval_id_eval_complete():
    stdout = "some output\n✓ Eval complete (ID: eval-Piu-2026-09-21T15:41:24)\nmore"
    assert _extract_eval_id(stdout) == "eval-Piu-2026-09-21T15:41:24"


def test_extract_eval_id_redteam_complete():
    stdout = "✓ Red team complete (ID: eval-RBy-2026-09-21T15:46:08)"
    assert _extract_eval_id(stdout) == "eval-RBy-2026-09-21T15:46:08"


def test_extract_eval_id_missing_raises():
    with pytest.raises(RuntimeError, match="eval ID"):
        _extract_eval_id("nothing useful here")


# ---------------------------------------------------------------------------
# Unit tests: metrics extraction (real promptfoo 0.123.1 eval.json shape)
# ---------------------------------------------------------------------------


def test_compute_metrics_all_pass():
    ej = _eval_json(successes=2, failures=0, errors=0)
    results, pass_rate, n = _compute_metrics(ej)
    metric = {r.metric_name: r.metric_value for r in results}
    assert pass_rate == pytest.approx(1.0)
    assert n == 2
    assert metric["n_passed"] == 2
    assert metric["n_failed"] == 0
    assert metric["n_errors"] == 0


def test_compute_metrics_mixed():
    ej = _eval_json(successes=3, failures=1, errors=1)
    results, pass_rate, n = _compute_metrics(ej)
    metric = {r.metric_name: r.metric_value for r in results}
    assert n == 5
    assert pass_rate == pytest.approx(3 / 5)
    assert metric["n_errors"] == 1


def test_compute_metrics_empty_no_pass_rate():
    ej = _eval_json(successes=0, failures=0, errors=0)
    results, pass_rate, n = _compute_metrics(ej)
    assert pass_rate is None
    assert n == 0
    metric_names = {r.metric_name for r in results}
    assert "pass_rate" not in metric_names


def test_compute_plugin_breakdown_real_shape():
    """Matches the real metadata shape captured from a live `redteam run`."""
    rows = [
        {
            "success": True,
            "metadata": {"pluginId": "sql-injection", "severity": "high"},
        },
        {
            "success": False,
            "metadata": {"pluginId": "sql-injection", "severity": "high"},
        },
        {
            "success": True,
            "metadata": {"pluginId": "ssrf", "severity": "medium"},
        },
    ]
    ej = _eval_json(0, 0, 0, plugin_rows=rows)
    breakdown = _compute_plugin_breakdown(ej)
    assert breakdown["pass_rate_by_plugin"]["sql-injection"] == pytest.approx(0.5)
    assert breakdown["pass_rate_by_plugin"]["ssrf"] == pytest.approx(1.0)
    assert breakdown["severity_by_plugin"]["sql-injection"] == "high"


def test_compute_plugin_breakdown_no_metadata_returns_empty():
    ej = _eval_json(2, 0, 0)
    assert _compute_plugin_breakdown(ej) == {}


# ---------------------------------------------------------------------------
# Integration: happy path (promptfoo-eval), CLI monkeypatched
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str, stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.integration
def test_promptfoo_eval_happy_path(monkeypatch, tmp_path):
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"

    eval_json = _eval_json(successes=2, failures=0, errors=0)
    eval_id = "eval-Fake-2026-09-21T00:00:00"

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        if args[0] == "eval":
            return _FakeCompletedProcess(0, f"✓ Eval complete (ID: {eval_id})")
        if args[0] == "export":
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_text(json.dumps(eval_json))
            return _FakeCompletedProcess(0, "exported")
        raise AssertionError(f"unexpected promptfoo invocation: {args}")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    results = adapter.run_benchmark_job(config, callbacks)

    assert results.id == config.id
    assert results.benchmark_id == "promptfoo-eval"
    assert results.overall_score == pytest.approx(1.0)
    assert results.num_examples_evaluated == 2

    metric = {r.metric_name: r.metric_value for r in results.results}
    assert metric["n_passed"] == 2

    assert results.eval_card is not None
    assert isinstance(results.eval_card, EvalCardMetadata)
    assert len(results.eval_card.capability_evaluations) == 1
    assert len(results.eval_card.safety_evaluations) == 0

    assert results.env_card is not None
    assert isinstance(results.env_card, EnvironmentCardMetadata)
    assert results.env_card.framework_name == "promptfoo"

    # eval.json triple-path: always-on additional_info embed (path 1 of 3)
    assert results.additional_info is not None
    assert results.additional_info["promptfoo_eval_json"] == eval_json
    assert results.additional_info["promptfoo_eval_id"] == eval_id

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert phases[0] == JobPhase.INITIALIZING
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases


@pytest.mark.integration
def test_promptfoo_redteam_happy_path(monkeypatch):
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-redteam"
    config.parameters = {"plugins": ["sql-injection"], "num_tests_per_plugin": 1}

    rows = [
        {
            "success": True,
            "metadata": {"pluginId": "sql-injection", "severity": "high"},
        },
        {
            "success": False,
            "metadata": {"pluginId": "sql-injection", "severity": "high"},
        },
    ]
    eval_json = _eval_json(1, 1, 0, plugin_rows=rows)
    eval_id = "eval-FakeRT-2026-09-21T00:00:00"

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        if args[0] == "redteam" and args[1] == "run":
            return _FakeCompletedProcess(0, f"✓ Red team complete (ID: {eval_id})")
        if args[0] == "export":
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_text(json.dumps(eval_json))
            return _FakeCompletedProcess(0, "exported")
        raise AssertionError(f"unexpected promptfoo invocation: {args}")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    results = adapter.run_benchmark_job(config, callbacks)

    assert results.benchmark_id == "promptfoo-redteam"
    assert results.overall_score == pytest.approx(0.5)
    assert results.eval_card.safety_evaluations[0].zero_shot == pytest.approx(0.5)
    assert len(results.eval_card.capability_evaluations) == 0

    assert results.additional_info["pass_rate_by_plugin"][
        "sql-injection"
    ] == pytest.approx(0.5)
    assert results.additional_info["severity_by_plugin"]["sql-injection"] == "high"


@pytest.mark.integration
def test_promptfoo_cli_failure_reports_failed_status(monkeypatch):
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        return _FakeCompletedProcess(1, "", "config error: bad yaml")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    with pytest.raises(RuntimeError, match="promptfoo CLI failed"):
        adapter.run_benchmark_job(config, callbacks)

    failed_statuses = [
        c
        for c in callbacks.report_status.call_args_list
        if c.args[0].status == JobStatus.FAILED
    ]
    assert len(failed_statuses) == 1
