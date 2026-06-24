"""Unit tests for the SWE-bench adapter."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main import SWEBenchAdapter

# -- Sample data -------------------------------------------------------------

SAMPLE_PREDICTIONS = {
    "astropy__astropy-7336": {
        "instance_id": "astropy__astropy-7336",
        "model_patch": "diff --git a/astropy/units/decorators.py b/astropy/units/decorators.py\n",
        "model_name_or_path": "test-model",
    },
    "django__django-11099": {
        "instance_id": "django__django-11099",
        "model_patch": "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n",
        "model_name_or_path": "test-model",
    },
}

# -- Fixtures ----------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "meta"


@pytest.fixture
def job_spec_path():
    return str(FIXTURE_DIR / "job.json")


@pytest.fixture
def adapter(job_spec_path):
    return SWEBenchAdapter(job_spec_path=job_spec_path)


@pytest.fixture
def predictions_json(tmp_path):
    """Sample predictions as a JSON file (dict format)."""
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(SAMPLE_PREDICTIONS))
    return str(path)


@pytest.fixture
def predictions_jsonl(tmp_path):
    """Sample predictions as a JSONL file (list format)."""
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(v) for v in SAMPLE_PREDICTIONS.values()))
    return str(path)


# -- Tests -------------------------------------------------------------------

class TestResultsProcessing:

    def test_all_resolved(self, adapter):
        results = {
            "a": {"completed": True, "resolved": True},
            "b": {"completed": True, "resolved": True},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 1.0
        assert summary["resolved"] == 2
        assert summary["errors"] == 0

    def test_none_resolved(self, adapter):
        results = {
            "a": {"completed": True, "resolved": False},
            "b": {"completed": True, "resolved": False},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 0.0

    def test_with_errors(self, adapter):
        results = {
            "a": {"completed": True, "resolved": True},
            "b": {"completed": False, "resolved": False},
        }
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 1.0
        assert summary["errors"] == 1

    def test_with_no_results(self, adapter):
        results = {}
        evals, score, summary = adapter._process_results(results, adapter.job_spec)
        assert score == 0.0
        assert summary["total"] == 0

class TestPredictionsLoading:

    def test_load_json_format(self, adapter, predictions_json):
        """Load predictions from a JSON dict file."""
        preds = adapter._load_predictions(predictions_json, "", "test")
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds
        assert preds["astropy__astropy-7336"]["model_patch"].startswith("diff")

    def test_load_jsonl_format(self, adapter, predictions_jsonl):
        """Load predictions from a JSONL file."""
        preds = adapter._load_predictions(predictions_jsonl, "", "test")
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds

    def test_missing_file_raises(self, adapter):
        with pytest.raises(FileNotFoundError):
            adapter._load_predictions("/nonexistent/path.json", "", "test")

    def test_load_from_test_data_path(self, adapter, tmp_path):
        """Test loading predictions via explicit path (eval-hub init container pattern)."""
        test_data = tmp_path / "test_data"
        test_data.mkdir()
        (test_data / "predictions.jsonl").write_text(
            "\n".join(json.dumps(v) for v in SAMPLE_PREDICTIONS.values())
        )
        preds = adapter._load_predictions(str(test_data / "predictions.jsonl"), "", "test")
        assert "astropy__astropy-7336" in preds
        assert "django__django-11099" in preds

class TestEmptyResults:

    def test_empty_results(self, adapter):
        
        job_result = adapter._empty_results(adapter.job_spec, 1.0)
        
        assert job_result.id == "swebench-eval-001"
        assert job_result.benchmark_id == "swebench_verified"
        assert job_result.benchmark_index == 0
        assert job_result.model_name == "gold-patches"
        assert job_result.results[0].metric_name == "resolve_rate"
        assert job_result.results[0].metric_value == 0.0
        assert job_result.results[0].metric_type == "accuracy"
        assert job_result.overall_score == 0.0
        assert job_result.num_examples_evaluated == 0
    
        



