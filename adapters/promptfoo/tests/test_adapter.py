"""Tests for the promptfoo adapter.

The promptfoo CLI subprocess boundary (_run_promptfoo_cli) is monkeypatched —
no real `promptfoo` binary or network calls are needed to run these tests.
Canned eval.json fixtures below match the REAL shape produced by promptfoo
0.123.1 (verified 2026-09-21 against `promptfoo eval` / `promptfoo redteam
run` + `promptfoo export eval`), not a guessed schema.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
from evalhub.adapter import JobCallbacks, JobPhase, JobStatus
from evalhub.adapter.models.cards import EnvironmentCardMetadata, EvalCardMetadata
from evalhub.adapter.models.job import (
    JobSpecExports,
    JobSpecExportsOCI,
    OCIArtifactResult,
)
from evalhub.models.api import OCICoordinates
from main import (
    PromptfooAdapter,
    _build_eval_config,
    _build_evaluate_options,
    _build_redteam_config,
    _build_target_provider,
    _compute_metrics,
    _compute_plugin_breakdown,
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
    provider = _build_target_provider(config, "sk-test")
    assert provider["id"] == "openai:chat:my-model"
    assert provider["config"]["apiBaseUrl"] == "http://localhost:8080/v1"
    assert provider["config"]["apiKey"] == "sk-test"


def test_build_target_provider_missing_url_raises():
    config = MagicMock()
    config.model.url = ""
    with pytest.raises(ValueError, match="model.url"):
        _build_target_provider(config, "key")


def test_build_evaluate_options():
    """Regression test: promptfoo 0.123.1's OpenAI provider family does not
    consume a per-provider config.timeoutMs — the per-test timeout that
    actually applies is the top-level evaluateOptions.timeoutMs, read at
    the evaluator level (verified against promptfoo's own source)."""
    opts = _build_evaluate_options(request_timeout=60, max_concurrency=8)
    assert opts == {"maxConcurrency": 8, "timeoutMs": 60_000}


def test_build_eval_config_from_prompts_and_tests():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "prompts": ["hi {{name}}"],
        "tests": [{"vars": {"name": "World"}}],
    }
    provider = {"id": "openai:chat:m"}
    evaluate_options = {"maxConcurrency": 4, "timeoutMs": 120_000}
    pf_config = _build_eval_config(config, provider, evaluate_options)
    assert pf_config["prompts"] == ["hi {{name}}"]
    assert pf_config["providers"] == [provider]
    assert pf_config["tests"] == [{"vars": {"name": "World"}}]
    assert pf_config["evaluateOptions"] == evaluate_options


def test_build_eval_config_missing_params_raises():
    config = MagicMock()
    config.parameters = {}
    with pytest.raises(ValueError, match="config_yaml"):
        _build_eval_config(config, {"id": "p"}, {})


def test_build_eval_config_passthrough_overwrites_providers():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "config_yaml": "description: mine\nproviders:\n  - id: should-be-replaced\ntests:\n  - vars: {}\n"
    }
    provider = {"id": "openai:chat:m"}
    pf_config = _build_eval_config(
        config, provider, {"maxConcurrency": 4, "timeoutMs": 120_000}
    )
    assert pf_config["providers"] == [provider]
    assert pf_config["description"] == "mine"
    assert pf_config["evaluateOptions"] == {"maxConcurrency": 4, "timeoutMs": 120_000}


def test_build_eval_config_passthrough_drops_stale_targets_key():
    """Regression test: promptfoo accepts `targets:` as an alias for
    `providers:` (rewriting one into the other internally). A pass-through
    config_yaml with a `targets:` key must not survive alongside our
    injected `providers:` — that could point part of the run at the user's
    original, non-EvalHub-controlled endpoint/credentials."""
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "config_yaml": "description: mine\ntargets:\n  - id: user-supplied-endpoint\ntests:\n  - vars: {}\n"
    }
    provider = {"id": "openai:chat:m"}
    pf_config = _build_eval_config(config, provider, {})
    assert pf_config["providers"] == [provider]
    assert "targets" not in pf_config


def test_build_eval_config_passthrough_preserves_other_evaluate_options():
    """CodeRabbit-requested behavior: EvalHub's timeout/concurrency win, but
    other evaluateOptions keys a passed-through config_yaml set (e.g.
    `repeat`) must survive."""
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "config_yaml": (
            "description: mine\ntests:\n  - vars: {}\n"
            "evaluateOptions:\n  repeat: 3\n  timeoutMs: 999\n"
        )
    }
    provider = {"id": "openai:chat:m"}
    pf_config = _build_eval_config(
        config, provider, {"maxConcurrency": 4, "timeoutMs": 120_000}
    )
    assert pf_config["evaluateOptions"]["repeat"] == 3
    assert pf_config["evaluateOptions"]["timeoutMs"] == 120_000
    assert pf_config["evaluateOptions"]["maxConcurrency"] == 4


def test_build_redteam_config_defaults():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {}
    provider = {"id": "openai:chat:m"}
    pf_config = _build_redteam_config(
        config, provider, {"maxConcurrency": 4, "timeoutMs": 120_000}
    )
    assert pf_config["targets"] == [provider]
    assert pf_config["redteam"]["purpose"] == "An AI assistant"
    assert pf_config["redteam"]["numTests"] == 5
    assert len(pf_config["redteam"]["plugins"]) > 0
    assert all(p["numTests"] == 5 for p in pf_config["redteam"]["plugins"])
    assert pf_config["evaluateOptions"] == {"maxConcurrency": 4, "timeoutMs": 120_000}


def test_build_redteam_config_custom_plugins():
    config = MagicMock()
    config.id = "job-1"
    config.parameters = {
        "plugins": ["sql-injection"],
        "num_tests_per_plugin": 2,
        "purpose": "a bank assistant",
    }
    pf_config = _build_redteam_config(config, {"id": "p"}, {})
    assert pf_config["redteam"]["plugins"] == [{"id": "sql-injection", "numTests": 2}]
    assert pf_config["redteam"]["purpose"] == "a bank assistant"


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
    eval_id = eval_json["evalId"]

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        if args[0] == "eval":
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_text(json.dumps(eval_json))
            return _FakeCompletedProcess(0, "")
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

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        if args[0] == "redteam" and args[1] == "generate":
            return _FakeCompletedProcess(0, "")
        if args[0] == "eval":
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_text(json.dumps(eval_json))
            return _FakeCompletedProcess(0, "")
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
def test_promptfoo_redteam_passes_grader_when_generation_provider_set(monkeypatch):
    """Regression test for a live-cluster finding (2026-09-21): promptfoo's
    redteam GRADING step uses its own hardcoded default model, entirely
    independent of --provider (which only controls attack generation). Left
    unset, grading 404s against a nonexistent hosted model and every test
    silently reports as failed. generation_provider must also be passed as
    --grader on the final eval step."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-redteam"
    config.parameters = {
        "plugins": ["sql-injection"],
        "num_tests_per_plugin": 1,
        "generation_provider": "openai:chat:internal-model",
    }

    eval_json = _eval_json(1, 0, 0)
    seen_eval_args = []

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        if args[0] == "redteam" and args[1] == "generate":
            assert "--provider" in args
            assert args[args.index("--provider") + 1] == "openai:chat:internal-model"
            return _FakeCompletedProcess(0, "")
        if args[0] == "eval":
            seen_eval_args.extend(args)
            out_path = Path(args[args.index("-o") + 1])
            out_path.write_text(json.dumps(eval_json))
            return _FakeCompletedProcess(0, "")
        raise AssertionError(f"unexpected promptfoo invocation: {args}")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    adapter.run_benchmark_job(config, callbacks)

    assert "--grader" in seen_eval_args
    assert (
        seen_eval_args[seen_eval_args.index("--grader") + 1]
        == "openai:chat:internal-model"
    )


def test_promptfoo_eval_does_not_pass_grader_flag(monkeypatch):
    """--grader is a redteam-specific concern; promptfoo-eval must never pass it."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"

    eval_json = _eval_json(1, 0, 0)
    seen_eval_args = []

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        seen_eval_args.extend(args)
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_text(json.dumps(eval_json))
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    adapter.run_benchmark_job(config, callbacks)

    assert "--grader" not in seen_eval_args


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


@pytest.mark.integration
def test_promptfoo_exit_zero_no_output_file_raises(monkeypatch):
    """Regression test: on a real OpenShift cluster (2026-09-21), `promptfoo eval`
    exited 0 with completely empty stdout — it suppresses its decorated results
    table when stdout is not a TTY (as in a container), which broke the original
    stdout-ID-parsing approach. The fix writes eval.json directly via `-o` and
    must fail loudly, not silently, if that file is somehow still missing."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        # Exit 0, empty stdout, but never writes the -o path — exactly what
        # was observed against the real cluster before this fix.
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    with pytest.raises(RuntimeError, match="did not write"):
        adapter.run_benchmark_job(config, callbacks)


@pytest.mark.integration
def test_promptfoo_max_concurrency_passed_to_cli(monkeypatch):
    """max_concurrency is a documented parameter — it must actually reach the
    promptfoo CLI (`-j <value>`), not just be accepted and ignored."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"
    config.parameters = {**config.parameters, "max_concurrency": 8}

    eval_json = _eval_json(1, 0, 0)
    seen_eval_args: list[str] = []

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        seen_eval_args.extend(args)
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_text(json.dumps(eval_json))
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    adapter.run_benchmark_job(config, callbacks)

    assert "-j" in seen_eval_args
    assert seen_eval_args[seen_eval_args.index("-j") + 1] == "8"


@pytest.mark.integration
def test_promptfoo_persisting_artifacts_phase_reported_without_oci(monkeypatch):
    """PERSISTING_ARTIFACTS must be reported for every job, not only when
    config.exports.oci happens to be set — OCI export is one thing that can
    happen during that phase, not a precondition for reporting it."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"
    assert config.exports is None

    eval_json = _eval_json(1, 0, 0)

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_text(json.dumps(eval_json))
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    adapter.run_benchmark_job(config, callbacks)

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS in phases
    callbacks.create_oci_artifact.assert_not_called()


@pytest.mark.integration
def test_promptfoo_oci_export_excludes_config_with_credentials(monkeypatch, tmp_path):
    """Regression test: the OCI artifact must contain ONLY eval.json. The
    work directory also holds promptfooconfig.yaml, which embeds the target
    model's plaintext apiKey (see _build_target_provider) — persisting the
    whole directory would leak that credential into the OCI artifact."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"
    config.exports = JobSpecExports(
        oci=JobSpecExportsOCI(
            coordinates=OCICoordinates(
                oci_host="registry.example.com", oci_repository="evals/job-1"
            )
        )
    )

    eval_json = _eval_json(1, 0, 0)

    import main as main_mod

    def fake_run_cli(args, cwd, timeout=3600):
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_text(json.dumps(eval_json))
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    seen_exported_files: set[str] = set()

    def fake_create_oci_artifact(spec):
        # Snapshot files_path here — run_benchmark_job's finally block
        # deletes the whole work_dir (including this artifact subdir)
        # before returning to the caller.
        seen_exported_files.update(p.name for p in spec.files_path.iterdir())
        return OCIArtifactResult(
            digest="sha256:" + "0" * 64,
            reference="registry.example.com/evals/job-1:latest",
        )

    callbacks.create_oci_artifact.side_effect = fake_create_oci_artifact

    adapter.run_benchmark_job(config, callbacks)

    assert seen_exported_files == {"eval.json"}
    assert "promptfooconfig.yaml" not in seen_exported_files


@pytest.mark.integration
def test_promptfoo_eval_json_preserved_in_metadata_beyond_size_gate(monkeypatch):
    """Regression test: additional_info is size-gated for the /events
    payload, but the MLflow artifact path (built in main() from
    evaluation_metadata) must not depend on that gate — a large eval.json
    must not silently lose its MLflow-artifact availability."""
    adapter = PromptfooAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)

    config = copy.deepcopy(adapter.job_spec)
    config.benchmark_id = "promptfoo-eval"

    eval_json = _eval_json(1, 0, 0)

    import main as main_mod

    monkeypatch.setattr(main_mod, "PROMPTFOO_EVAL_JSON_MAX_BYTES", 1)

    def fake_run_cli(args, cwd, timeout=3600):
        out_path = Path(args[args.index("-o") + 1])
        out_path.write_text(json.dumps(eval_json))
        return _FakeCompletedProcess(0, "")

    monkeypatch.setattr(main_mod, "_run_promptfoo_cli", fake_run_cli)

    results = adapter.run_benchmark_job(config, callbacks)

    # additional_info omits it (over the artificially tiny limit)...
    assert results.additional_info.get("promptfoo_eval_json_omitted") is True
    assert "promptfoo_eval_json" not in results.additional_info
    # ...but evaluation_metadata still carries the exact original bytes.
    b64 = results.evaluation_metadata["promptfoo_eval_json_b64"]
    assert json.loads(base64.b64decode(b64)) == eval_json
