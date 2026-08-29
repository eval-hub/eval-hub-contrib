import shutil
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalhub.adapter import JobCallbacks, OCIArtifactResult

from main import LMEvalAdapter


def pytest_configure(config):
    """Register custom pytest markers for integration, local, and endpoint tests."""
    config.addinivalue_line(
        "markers", "integration: integration tests for adapter plumbing"
    )
    config.addinivalue_line(
        "markers", "local: tests requiring local infrastructure (Ollama, vLLM)"
    )
    config.addinivalue_line(
        "markers", "openai_endpoint: tests requiring a local OpenAI-compatible endpoint"
    )


@pytest.fixture
def adapter(tmp_path):
    """Return a LMEvalAdapter loaded from a copy of the canonical job.json fixture."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")
    return LMEvalAdapter(job_spec_path=str(meta_dir / "job.json"))


@pytest.fixture
def mock_callbacks():
    """Return a mock JobCallbacks with a pre-configured OCI artifact response."""
    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake",
        reference="fake:latest",
    )
    return callbacks
