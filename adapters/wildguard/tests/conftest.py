import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalhub.adapter import JobCallbacks  # noqa: E402
from main import WildGuardAdapter  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests for adapter plumbing")


@pytest.fixture()
def wildguard_adapter():
    return WildGuardAdapter(job_spec_path="meta/job.json")


@pytest.fixture()
def mock_callbacks():
    return create_autospec(JobCallbacks)
