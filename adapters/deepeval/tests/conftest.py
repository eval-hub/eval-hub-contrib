import sys
from pathlib import Path

import pytest
from deepeval.models import DeepEvalBaseLLM

# Add the adapter directory to sys.path so `import main` and
# `import deepeval_adapter` work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests for adapter plumbing")
    config.addinivalue_line("markers", "local: tests requiring local infrastructure (API keys, etc.)")


class StubJudge(DeepEvalBaseLLM):
    """Judge model that fails if a test accidentally asks it for a completion."""

    def load_model(self):
        return None

    def generate(self, *args, **kwargs):
        raise AssertionError("no judge call expected")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("no judge call expected")

    def get_model_name(self):
        return "stub"


@pytest.fixture
def stub_judge() -> StubJudge:
    return StubJudge()
