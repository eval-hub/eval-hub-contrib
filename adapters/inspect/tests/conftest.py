"""Shared fixtures for Inspect AI adapter tests."""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Add the adapter directory to sys.path so `from main import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock inspect-ai and petri packages at collection time — they are not installed
# in the test environment (only eval-hub-sdk and pytest are). The adapter calls
# inspect via subprocess, so no real import of inspect_ai occurs at runtime.
for _mod in [
    "inspect_ai",
    "inspect_petri",
    "petri_bloom",
]:
    sys.modules.setdefault(_mod, MagicMock())


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: full run_benchmark_job with monkeypatched subprocess"
    )


def _make_petri_log(status: str = "success", num_samples: int = 5) -> dict:
    """Minimal well-formed Inspect AI log for a Petri sycophancy audit."""
    scores = []
    for dim in [
        "concerning", "unprompted_sycophancy", "eval_awareness",
        "admirable", "broken_tool_use",
    ]:
        scores.append(
            {
                "name": dim,
                "scorer": "petri_judge",
                "metrics": {
                    "mean": {"name": "mean", "value": 3.2},
                    "stderr": {"name": "stderr", "value": 0.4},
                },
            }
        )
    return {
        "version": 2,
        "status": status,
        "eval": {
            "task": "inspect_petri/audit",
            "model": "granite-3.3-8b-instruct",
        },
        "results": {
            "scores": scores,
            "total_samples": num_samples,
        },
        "samples": [{"id": i} for i in range(num_samples)],
        "error": None,
    }


def _make_standard_log(status: str = "success", num_samples: int = 10) -> dict:
    """Minimal well-formed Inspect AI log for a standard (gsm8k-style) benchmark."""
    return {
        "version": 2,
        "status": status,
        "eval": {
            "task": "inspect_evals/gsm8k",
            "model": "granite-3.3-8b-instruct",
        },
        "results": {
            "scores": [
                {
                    "name": "accuracy",
                    "scorer": "accuracy",
                    "metrics": {
                        "accuracy": {"name": "accuracy", "value": 0.82},
                        "stderr":   {"name": "stderr",   "value": 0.03},
                    },
                }
            ],
            "total_samples": num_samples,
        },
        "samples": [{"id": i} for i in range(num_samples)],
        "error": None,
    }


@pytest.fixture
def petri_eval_log() -> dict:
    return _make_petri_log()


@pytest.fixture
def standard_eval_log() -> dict:
    return _make_standard_log()


@pytest.fixture
def petri_log_file(tmp_path: Path, petri_eval_log: dict) -> Path:
    p = tmp_path / "petri_sycophancy_001.json"
    p.write_text(json.dumps(petri_eval_log))
    return p


@pytest.fixture
def standard_log_file(tmp_path: Path, standard_eval_log: dict) -> Path:
    p = tmp_path / "gsm8k_001.json"
    p.write_text(json.dumps(standard_eval_log))
    return p


@pytest.fixture
def job_spec_path() -> str:
    return str(Path(__file__).parent.parent / "meta" / "job.json")
