import io
import logging

import pytest

import subprocess_utils


class FakeProcess:
    def __init__(self, output: str, return_code: int):
        self.stdout = io.StringIO(output)
        self.return_code = return_code

    def wait(self) -> int:
        return self.return_code


def test_failure_includes_only_the_bounded_tail_and_logs_it(monkeypatch, caplog):
    output = "".join(f"line-{index}\n" for index in range(25))
    process = FakeProcess(output, return_code=1)
    popen_kwargs = {}

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError) as error:
            subprocess_utils.run_streaming_subprocess(
                ["guidellm", "benchmark"],
                name="GuideLLM",
                logger=logging.getLogger("test"),
            )

    message = str(error.value)
    assert message.startswith("GuideLLM failed with exit code 1")
    assert "Last subprocess output (stdout/stderr):" in message
    assert "\nline-4\n" not in message
    assert "line-5" in message
    assert "line-24" in message
    assert "GuideLLM: line-24" in caplog.text
    assert popen_kwargs["stderr"] is subprocess_utils.subprocess.STDOUT


def test_failure_redacts_known_and_labeled_credentials(monkeypatch, caplog):
    output = (
        "api_key=known-secret\n"
        "Authorization: Bearer bearer-secret\n"
        "request failed at https://example.test/?access_token=url-secret\n"
        '{"api_key": "quoted secret with spaces"}\n'
    )
    process = FakeProcess(output, return_code=2)
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", lambda *args, **kwargs: process)

    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError) as error:
            subprocess_utils.run_streaming_subprocess(
                ["tool"],
                name="Tool",
                logger=logging.getLogger("test"),
                sensitive_values=("known-secret",),
            )

    combined = str(error.value) + caplog.text
    for secret in (
        "known-secret",
        "bearer-secret",
        "url-secret",
        "quoted secret with spaces",
    ):
        assert secret not in combined
    assert combined.count("[REDACTED]") >= 3


def test_failure_output_is_truncated_by_character_limit(monkeypatch):
    output = "".join(f"{index}: {'x' * 300}\n" for index in range(20))
    process = FakeProcess(output, return_code=1)
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError) as error:
        subprocess_utils.run_streaming_subprocess(
            ["tool"], name="Tool", logger=logging.getLogger("test")
        )

    message = str(error.value)
    output_tail = message.split("Last subprocess output (stdout/stderr):\n", 1)[1]
    assert len(output_tail) <= subprocess_utils._MAX_ERROR_OUTPUT_CHARS
    assert output_tail.startswith("[earlier subprocess output truncated]\n")
    assert output_tail.endswith("x" * 300)


def test_success_does_not_raise(monkeypatch):
    process = FakeProcess("finished successfully\n", return_code=0)
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", lambda *args, **kwargs: process)

    assert subprocess_utils.run_streaming_subprocess(
        ["tool"], name="Tool", logger=logging.getLogger("test")
    ) is None


def test_failure_without_output_keeps_concise_error(monkeypatch):
    process = FakeProcess("", return_code=1)
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", lambda *args, **kwargs: process)

    with pytest.raises(RuntimeError, match="^Tool failed with exit code 1$"):
        subprocess_utils.run_streaming_subprocess(
            ["tool"], name="Tool", logger=logging.getLogger("test")
        )
