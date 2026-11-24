"""Shared fixtures: golden reports, a stub skillevaluator CLI, a staged skill and an adapter factory."""

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
ADAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ADAPTER_DIR))

from evalhub.adapter import JobCallbacks

import _runner
import _skill_source
from main import SkillEvaluatorAdapter

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config):
    """Register the custom markers used by this suite."""
    config.addinivalue_line(
        "markers", "integration: integration tests for adapter plumbing"
    )
    config.addinivalue_line(
        "markers",
        "live_cli: runs the real skillevaluator and skillspector CLIs (deselected when they are not installed)",
    )


def load_report(name: str) -> dict:
    """Load a recorded (redacted) CLI report from ``tests/fixtures/reports``."""
    return json.loads((FIXTURES / "reports" / f"{name}.json").read_text())


@pytest.fixture()
def report():
    """Expose ``load_report`` as a fixture."""
    return load_report


@pytest.fixture()
def mock_callbacks():
    """Return an autospecced ``JobCallbacks`` that records status updates and artifact uploads."""
    return create_autospec(JobCallbacks)


@pytest.fixture()
def staged_skill(tmp_path, monkeypatch):
    """Stage a minimal skill under a temp /test_data and point the adapter at it."""
    root = tmp_path / "test_data"
    skill = root / "my-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: my-skill\ndescription: x\n---\n# x\n")
    monkeypatch.setattr(_skill_source, "TEST_DATA_DIR", str(root))
    return skill


@pytest.fixture()
def stub_cli(tmp_path, monkeypatch):
    """Factory for a fake ``skillevaluator`` executable that emits a canned report.

    The stub honours the real CLI contract: ``<command> ... PATH -r json -o DIR``,
    writes one JSON file into DIR, exits with the given code. It also records the
    names of the environment variables it received (``_env_keys``) and its argv
    (``_argv``) inside the report so tests can assert on isolation.
    """

    def make(
        report_name: str,
        *,
        exit_code: int = 0,
        report_override: dict | None = None,
        sleep: float = 0.0,
    ):
        """Install a stub CLI that replays ``report_name``, point ``SKILLEVALUATOR_BIN`` at it and return its path."""
        data = (
            report_override if report_override is not None else load_report(report_name)
        )
        stub = tmp_path / "skillevaluator-stub"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys, time\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('skillevaluator, version 9.9.9'); sys.exit(0)\n"
            f"time.sleep({sleep})\n"
            f"data = json.loads({json.dumps(json.dumps(data))})\n"
            "data['_env_keys'] = sorted(os.environ)\n"
            "data['_argv'] = sys.argv[1:]\n"
            "out = sys.argv[sys.argv.index('-o') + 1]\n"
            "os.makedirs(out, exist_ok=True)\n"
            "open(os.path.join(out, 'report.json'), 'w').write(json.dumps(data))\n"
            f"sys.exit({exit_code})\n"
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("SKILLEVALUATOR_BIN", str(stub))
        return stub

    return make


@pytest.fixture()
def adapter_for(tmp_path):
    """Build an adapter whose JobSpec uses ``benchmark_id`` and ``parameters``."""

    def build(
        benchmark_id: str, parameters: dict | None = None, exports: dict | None = None
    ):
        """Create an adapter for ``benchmark_id`` with the given parameters and exports."""
        spec = json.loads((ADAPTER_DIR / "meta" / "job.json").read_text())
        spec["benchmark_id"] = benchmark_id
        spec["parameters"] = parameters or {}
        if exports:
            spec["exports"] = exports
        path = tmp_path / "job.json"
        path.write_text(json.dumps(spec))
        return SkillEvaluatorAdapter(job_spec_path=str(path))

    return build


def have_live_clis() -> bool:
    """Return True when the real SkillEvaluator and SkillSpector executables are available."""
    try:
        _runner.find_cli()
    except _runner.RunnerError:
        return False
    return bool(os.environ.get("SKILLEVALUATOR_SKILLSPECTOR_PATH"))
