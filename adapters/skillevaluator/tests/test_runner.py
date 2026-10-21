"""Tests for the CLI runner: lookup, environment isolation, exit codes and report loading."""

import json

import pytest

import _runner as runner


def test_env_is_allowlisted(monkeypatch):
    """The child environment keeps only allow-listed names."""
    monkeypatch.setenv("EVALHUB_CALLBACK_TOKEN", "secret")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv(
        "SKILLEVALUATOR_SKILLSPECTOR_PATH", "/opt/skillspector/bin/skillspector"
    )
    env = runner.build_env({"SKILLEVALUATOR_SUBMITTER": "alice", "EMPTY": ""})
    assert "SKILLEVALUATOR_SKILLSPECTOR_PATH" in env
    assert env["SKILLEVALUATOR_SUBMITTER"] == "alice"
    assert "EMPTY" not in env
    for leaked in (
        "EVALHUB_CALLBACK_TOKEN",
        "KUBERNETES_SERVICE_HOST",
        "OPENAI_API_KEY",
    ):
        assert leaked not in env


def test_find_cli_prefers_override(monkeypatch):
    """``SKILLEVALUATOR_BIN`` takes precedence when locating the CLI."""
    monkeypatch.setenv(runner.CLI_BIN_ENV, "/custom/skillevaluator")
    assert runner.find_cli() == "/custom/skillevaluator"


def test_find_cli_missing(monkeypatch, tmp_path):
    """A missing CLI raises ``RunnerError``."""
    monkeypatch.delenv(runner.CLI_BIN_ENV, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(runner.sys, "executable", str(tmp_path / "python"))
    with pytest.raises(runner.RunnerError, match="not found"):
        runner.find_cli()


def test_get_version_parses_click_output(stub_cli):
    """The version is parsed from the CLI's ``--version`` output."""
    stub = stub_cli("quality-pass")
    assert runner.get_version(str(stub)) == "9.9.9"


def test_get_version_is_none_for_missing_executable(tmp_path):
    """A missing executable yields no version instead of an error."""
    assert runner.get_version(str(tmp_path / "nope")) is None


@pytest.mark.parametrize("code", [0, 1])
def test_exit_0_and_1_both_yield_a_report(stub_cli, tmp_path, code):
    """Exit codes 0 and 1 both mean a report was produced."""
    stub = stub_cli("security-failed", exit_code=code)
    run = runner.run_cli(
        ["security-scan", "--no-llm", str(tmp_path)],
        output_dir=tmp_path / "out",
        timeout=30,
        env=runner.build_env(),
        executable=str(stub),
    )
    assert run.returncode == code
    assert runner.load_report(run.report_path)["overall_status"] == "failed"


def test_exit_2_is_a_usage_error(stub_cli, tmp_path):
    """Exit code 2 is treated as a usage error."""
    stub = stub_cli("quality-pass", exit_code=2)
    with pytest.raises(runner.RunnerError, match="code 2"):
        runner.run_cli(
            ["quality-check", "x"],
            output_dir=tmp_path / "o",
            timeout=30,
            env={},
            executable=str(stub),
        )


def test_timeout_is_reported(stub_cli, tmp_path):
    """A run that exceeds its timeout raises ``RunnerError``."""
    stub = stub_cli("quality-pass", sleep=5)
    with pytest.raises(runner.RunnerError, match="timed out after 1s"):
        runner.run_cli(
            ["quality-check", "x"],
            output_dir=tmp_path / "o",
            timeout=1,
            env={},
            executable=str(stub),
        )


def test_missing_executable_is_a_runner_error(tmp_path):
    """A nonexistent executable raises ``RunnerError``."""
    with pytest.raises(runner.RunnerError, match="cannot execute"):
        runner.run_cli(
            ["q"],
            output_dir=tmp_path / "o",
            timeout=5,
            env={},
            executable=str(tmp_path / "absent"),
        )


def test_zero_or_many_reports_are_rejected(stub_cli, tmp_path):
    """Anything other than exactly one report file is rejected."""
    stub = stub_cli("quality-pass")
    out = tmp_path / "o"
    out.mkdir()
    (out / "stale.json").write_text("{}")
    with pytest.raises(runner.RunnerError, match="exactly one"):
        runner.run_cli(
            ["quality-check", "x"],
            output_dir=out,
            timeout=30,
            env={},
            executable=str(stub),
        )


def test_argv_has_no_shell_and_forces_json_output(stub_cli, tmp_path):
    """The command is an argument list with no shell and JSON output forced."""
    stub = stub_cli("quality-pass")
    skill = tmp_path / "a skill; rm -rf x"
    skill.mkdir()
    run = runner.run_cli(
        ["quality-check", str(skill)],
        output_dir=tmp_path / "o",
        timeout=30,
        env={},
        executable=str(stub),
    )
    argv = runner.load_report(run.report_path)["_argv"]
    assert argv[:2] == ["quality-check", str(skill)]
    assert argv[2:4] == ["-r", "json"]


def test_child_env_contains_only_allowlisted_names(stub_cli, tmp_path, monkeypatch):
    """The child process sees only allow-listed environment names."""
    monkeypatch.setenv("EVALHUB_CALLBACK_TOKEN", "secret")
    stub = stub_cli("quality-pass")
    run = runner.run_cli(
        ["quality-check", "x"],
        output_dir=tmp_path / "o",
        timeout=30,
        env=runner.build_env(),
        executable=str(stub),
    )
    seen = set(runner.load_report(run.report_path)["_env_keys"])
    assert "EVALHUB_CALLBACK_TOKEN" not in seen


@pytest.mark.parametrize("payload", ["not json", "[1, 2]", "\xff"])
def test_load_report_rejects_non_object_json(tmp_path, payload):
    """A report that is not a JSON object is rejected."""
    path = tmp_path / "r.json"
    path.write_bytes(payload.encode("latin-1"))
    with pytest.raises(runner.RunnerError):
        runner.load_report(path)


def test_load_report_accepts_object(tmp_path):
    """A JSON object report loads unchanged."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"overall_status": "passed"}))
    assert runner.load_report(path) == {"overall_status": "passed"}
