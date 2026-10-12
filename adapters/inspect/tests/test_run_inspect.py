"""Tests for the streaming, bounded inspect runner and its job-parameter overrides."""

import sys
import time

import pytest

import _execution as ex
from main import InspectAdapter

_OFF = {"INSPECT_PREFLIGHT": "off"}


def _cmd(script: str) -> list[str]:
    """Return a command list that runs script via sys.executable -c; cmd[2] acts as the task spec."""
    return [sys.executable, "-c", script]


def _write_log_script(log_dir) -> str:
    """Return a Python one-liner that writes an empty JSON file to log_dir for run_inspect to find."""
    return f"import json,pathlib; pathlib.Path({str(log_dir)!r}, 'run.json').write_text('{{}}')"


# -- streaming ----------------------------------------------------------------------------------

def test_output_is_streamed_to_the_adapter_log(tmp_path, caplog):
    """Each line of inspect's stdout is logged with the 'inspect |' prefix as it arrives."""
    script = "print('loading dataset', flush=True); print('2 of 4 samples', flush=True)\n" + _write_log_script(tmp_path)
    with caplog.at_level("INFO"):
        ex.run_inspect(_cmd(script), dict(_OFF), tmp_path)
    assert "inspect | loading dataset" in caplog.text
    assert "inspect | 2 of 4 samples" in caplog.text


def test_stderr_is_merged_into_the_stream(tmp_path, caplog):
    """stderr is merged with stdout so diagnostic messages appear in the adapter log."""
    script = "import sys; print('warn on stderr', file=sys.stderr, flush=True)\n" + _write_log_script(tmp_path)
    with caplog.at_level("INFO"):
        ex.run_inspect(_cmd(script), dict(_OFF), tmp_path)
    assert "inspect | warn on stderr" in caplog.text


def test_stream_logs_can_be_disabled(tmp_path, caplog):
    """INSPECT_STREAM_LOGS=0 suppresses per-line logging; only the tail is kept for failures."""
    script = "print('quiet please', flush=True)\n" + _write_log_script(tmp_path)
    with caplog.at_level("INFO"):
        ex.run_inspect(_cmd(script), {**_OFF, "INSPECT_STREAM_LOGS": "0"}, tmp_path)
    assert "quiet please" not in caplog.text


def test_lines_arrive_before_the_process_exits(tmp_path):
    """The point of streaming: a line is logged while inspect is still running."""
    marker = tmp_path / "seen"
    script = (
        "import time,pathlib; print('first', flush=True); time.sleep(1.5);"
        f"pathlib.Path({str(marker)!r}).write_text('done');"
        + _write_log_script(tmp_path)
    )
    seen_while_running: list[bool] = []

    import logging

    class _Probe(logging.Handler):
        def emit(self, record):
            """Record whether the marker file exists when the first streamed line is logged."""
            if "inspect | first" in record.getMessage():
                seen_while_running.append(not marker.exists())

    handler = _Probe()
    ex.logger.addHandler(handler)
    ex.logger.setLevel(logging.INFO)
    try:
        ex.run_inspect(_cmd(script), dict(_OFF), tmp_path)
    finally:
        ex.logger.removeHandler(handler)
    assert seen_while_running == [True]


# -- results and failures ---------------------------------------------------------------------

def test_returns_newest_json_log(tmp_path):
    """run_inspect returns the path to the most recently modified JSON log in log_dir."""
    ex.run_inspect(_cmd(_write_log_script(tmp_path)), dict(_OFF), tmp_path)
    assert ex.run_inspect(_cmd(_write_log_script(tmp_path)), dict(_OFF), tmp_path).name == "run.json"


def test_failure_reports_exit_code_and_output_tail(tmp_path):
    """A non-zero exit raises RuntimeError with the exit code and the last lines of output."""
    script = "print('boom detail', flush=True); raise SystemExit(3)"
    with pytest.raises(RuntimeError, match=r"exit 3") as exc:
        ex.run_inspect(_cmd(script), dict(_OFF), tmp_path)
    assert "boom detail" in str(exc.value)


def test_missing_json_log_is_an_error(tmp_path):
    """run_inspect raises RuntimeError when inspect exits 0 but writes no JSON log file."""
    with pytest.raises(RuntimeError, match="no JSON log"):
        ex.run_inspect(_cmd("print('ok')"), dict(_OFF), tmp_path)


def test_unlaunchable_command_raises_oserror(tmp_path):
    """OSError is raised directly when the subprocess binary does not exist."""
    with pytest.raises(OSError):
        ex.run_inspect(["/nonexistent/inspect", "eval", "x"], dict(_OFF), tmp_path)


def test_interrupted_run_reaps_the_child_and_closes_the_pipe(tmp_path, monkeypatch):
    """If logging raises mid-run, the process group is killed and waited on (no zombie)."""
    import os

    pids: list[int] = []
    created: list = []
    real_popen = ex.subprocess.Popen

    def spy_popen(*args, **kwargs):
        """Wrap Popen to record the created process object for later assertions."""
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    def boom(msg, *args):
        """Raise RuntimeError on the first streamed inspect line, simulating a logging failure."""
        if str(msg).startswith("inspect |"):
            pids.append(int(args[0]))
            raise RuntimeError("logging failed")

    monkeypatch.setattr(ex.subprocess, "Popen", spy_popen)
    monkeypatch.setattr(ex.logger, "info", boom)
    script = "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
    with pytest.raises(RuntimeError, match="logging failed"):
        ex.run_inspect(_cmd(script), dict(_OFF), tmp_path)

    proc = created[0]
    assert proc.returncode is not None            # reaped
    assert proc.stdout.closed                      # pipe released
    with pytest.raises(ProcessLookupError):        # and really gone
        os.kill(pids[0], 0)


# -- timeout ----------------------------------------------------------------------------------

def test_timeout_stops_a_hung_run(tmp_path):
    """A run that exceeds timeout_s is killed and raises RuntimeError naming the timeout."""
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        ex.run_inspect(_cmd("import time; time.sleep(60)"), {**_OFF, "INSPECT_TIMEOUT_S": "1"}, tmp_path)
    assert time.monotonic() - started < 15


def test_timeout_also_stops_child_processes(tmp_path):
    """A hung `git clone`-style child must not keep the output pipe (and the job) alive."""
    script = "import subprocess,time; subprocess.Popen(['sleep','60']); time.sleep(60)"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        ex.run_inspect(_cmd(script), {**_OFF, "INSPECT_TIMEOUT_S": "1"}, tmp_path)
    assert time.monotonic() - started < 15


def test_timeout_kills_term_resistant_descendants_keeping_pipe_open(tmp_path):
    """SIGKILL is sent to the process group after the grace period even when the direct child already exited.

    os.fork() preserves the open pipe write-end in the grandchild, so the stdout loop blocks
    until the grandchild is killed. Without an unconditional post-grace SIGKILL to the group
    the loop would hang for the full grandchild sleep duration.
    """
    script = "\n".join([
        "import os, signal, sys, time",
        "pid = os.fork()",
        "if pid == 0:",
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "    time.sleep(60)",
        "    sys.exit(0)",
        "else:",
        "    sys.exit(0)",
    ])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        ex.run_inspect(_cmd(script), {**_OFF, "INSPECT_TIMEOUT_S": "1"}, tmp_path)
    assert time.monotonic() - started < 20


def test_default_timeout_k8s_unbounded_local_two_hours():
    """k8s mode is unbounded by default; non-k8s modes default to 7200s."""
    assert ex._resolve_job_timeout({"EVALHUB_MODE": "k8s"}) is None
    assert ex._resolve_job_timeout({}) == 7200.0


@pytest.mark.parametrize("raw,expected", [("90", 90.0), ("0", None), ("none", None), ("1.5", 1.5)])
def test_timeout_override(raw, expected):
    """INSPECT_TIMEOUT_S overrides the default; 0 and 'none' disable the limit."""
    assert ex._resolve_job_timeout({"INSPECT_TIMEOUT_S": raw, "EVALHUB_MODE": "k8s"}) == expected


@pytest.mark.parametrize("raw", ["soon", "-5"])
def test_invalid_timeout_names_the_parameter(raw):
    """Non-numeric or negative INSPECT_TIMEOUT_S raises ValueError naming the parameter."""
    with pytest.raises(ValueError, match="timeout_s"):
        ex._resolve_job_timeout({"INSPECT_TIMEOUT_S": raw})


# -- pre-flight is wired into the run ---------------------------------------------------------

def test_unreachable_host_fails_before_inspect_starts(tmp_path, monkeypatch):
    """Pre-flight failure raises before Popen is called so inspect never starts."""
    cmd = ["inspect", "eval", "inspect_evals/mmlu_pro"]
    env = {"HF_ENDPOINT": "http://127.0.0.1:1", "INSPECT_PREFLIGHT_TIMEOUT_S": "0.5"}
    monkeypatch.setattr(ex.subprocess, "Popen", lambda *_a, **_k: pytest.fail("inspect must not start"))
    with pytest.raises(RuntimeError, match="Pre-flight check"):
        ex.run_inspect(cmd, env, tmp_path)


# -- job parameters -> env --------------------------------------------------------------------

def _adapter(job_spec_path, **params):
    """Create an InspectAdapter for mmlu-pro with the given job parameters."""
    adapter = InspectAdapter(job_spec_path=job_spec_path)
    adapter.job_spec.benchmark_id = "inspect/mmlu-pro"
    adapter.job_spec.parameters.update(params)
    return adapter


def test_parameters_map_to_env(job_spec_path, monkeypatch):
    """Each diagnosability job parameter is written to its corresponding env var."""
    for v in ("INSPECT_PREFLIGHT", "INSPECT_PREFLIGHT_TIMEOUT_S", "INSPECT_PREFLIGHT_HOSTS", "INSPECT_TIMEOUT_S"):
        monkeypatch.delenv(v, raising=False)
    adapter = _adapter(
        job_spec_path,
        preflight="warn", preflight_timeout_s=3, preflight_hosts=["https://corp.example"],
        timeout_s=600, stream_logs=False,
    )
    env = adapter._build_env(adapter.job_spec, "standard")
    assert env["INSPECT_PREFLIGHT"] == "warn"
    assert env["INSPECT_PREFLIGHT_TIMEOUT_S"] == "3.0"
    assert env["INSPECT_PREFLIGHT_HOSTS"] == "https://corp.example"
    assert env["INSPECT_TIMEOUT_S"] == "600"
    assert env["INSPECT_STREAM_LOGS"] == "0"


def test_parameter_beats_inherited_env(job_spec_path, monkeypatch):
    """A job parameter overrides the same-named env var already in the process environment."""
    monkeypatch.setenv("INSPECT_PREFLIGHT", "off")
    adapter = _adapter(job_spec_path, preflight="fail")
    assert adapter._build_env(adapter.job_spec, "standard")["INSPECT_PREFLIGHT"] == "fail"


def test_env_alone_still_applies_when_no_parameter(job_spec_path, monkeypatch):
    """When no parameter is set, the env var passes through to the subprocess env unchanged."""
    monkeypatch.setenv("INSPECT_PREFLIGHT", "warn")
    adapter = _adapter(job_spec_path)
    assert adapter._build_env(adapter.job_spec, "standard")["INSPECT_PREFLIGHT"] == "warn"


@pytest.mark.parametrize(
    "params,match",
    [
        ({"preflight": "sometimes"}, "preflight must be one of"),
        ({"preflight_timeout_s": 0}, "preflight_timeout_s"),
        ({"preflight_hosts": ["ftp://x"]}, "preflight_hosts"),
        ({"timeout_s": "soon"}, "timeout_s"),
    ],
)
def test_bad_parameters_fail_at_start_naming_the_parameter(job_spec_path, params, match):
    """An invalid parameter value raises ValueError at env-build time, naming the bad parameter."""
    adapter = _adapter(job_spec_path, **params)
    with pytest.raises(ValueError, match=match):
        adapter._build_env(adapter.job_spec, "standard")
