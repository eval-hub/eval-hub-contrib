import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock CLEAR framework modules BEFORE main.py is imported.
# CLEAR is imported at module level and would
# fail with ImportError if clear-eval is not installed.
# setdefault preserves the real module if clear-eval IS installed.
for mod in [
    "clear_eval",
    "clear_eval.agentic",
    "clear_eval.agentic.pipeline",
    "clear_eval.agentic.pipeline.run_clear_agentic_eval",
    "clear_eval.agentic.pipeline.utils",
]:
    sys.modules.setdefault(mod, MagicMock())

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests for adapter plumbing")
