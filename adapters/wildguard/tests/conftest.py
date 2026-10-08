import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalhub.adapter import JobCallbacks

from main import WildGuardAdapter


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: integration tests for adapter plumbing"
    )
    config.addinivalue_line(
        "markers",
        "live_hf: needs HuggingFace Hub access (deselected in CI with -m 'not live_hf')",
    )


@pytest.fixture()
def wildguard_adapter():
    return WildGuardAdapter(job_spec_path="meta/job.json")


@pytest.fixture()
def mock_callbacks():
    return create_autospec(JobCallbacks)
