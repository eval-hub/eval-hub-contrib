import sys
from pathlib import Path

import pytest

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests for adapter plumbing")
    config.addinivalue_line("markers", "local: tests requiring local infrastructure (Ollama, HuggingFace, etc.)")
    config.addinivalue_line("markers", "openai_endpoint: tests requiring a real OpenAI-compatible endpoint")
