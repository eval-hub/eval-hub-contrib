import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalhub.adapter import JobCallbacks
from main import PromptfooAdapter


def pytest_configure(config):
    """Register the 'integration' marker so pytest does not warn on unknown marks."""
    config.addinivalue_line(
        "markers", "integration: integration tests for adapter plumbing"
    )


@pytest.fixture()
def promptfoo_adapter():
    """Return a PromptfooAdapter loaded from the canonical job fixture."""
    return PromptfooAdapter(job_spec_path="meta/job.json")


@pytest.fixture()
def mock_callbacks():
    """Return a spec-constrained mock for JobCallbacks."""
    return create_autospec(JobCallbacks)
