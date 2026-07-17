"""Integration tests for the LightEval adapter.

Verifies adapter plumbing with a single monkeypatch point -- _run_lighteval
returns parsed results inline, so no filesystem setup is needed.
"""

import copy
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest

from evalhub.adapter import JobCallbacks, JobPhase, JobResults, OCIArtifactResult
from main import LightEvalAdapter

# Canned output matching LightEval's results JSON structure.
# Based losely on https://github.com/huggingface/lighteval/blob/main/docs/source/saving-and-reading-results.mdx#general-configuration
CANNED_RESULTS = {
    "results": {
        "boolq|0": {
            "accuracy": 0.78,
            "accuracy_stderr": 0.02,
        }
    },
    "config_general": {
        "max_samples": 5,
        "model_config": {
            "generation_parameters": {
                "temperature": 0,
                "max_new_tokens": None,
                "top_p": None,
                "top_k": None,
                "seed": None,
                "stop_tokens": None,
                "repetition_penalty": None,
            },
        },
    },
    "config_tasks": {
        "boolq|0": {
            "name": "boolq",
            "hf_repo": "google/boolq",
            "hf_subset": "default",
            "num_fewshots": 0,
        }
    },
}


@pytest.fixture
def adapter(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(Path("meta/job.json"), meta_dir / "job.json")
    return LightEvalAdapter(job_spec_path=str(meta_dir / "job.json"))


@pytest.fixture
def mock_callbacks():
    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )
    return callbacks


@pytest.mark.integration
def test_lighteval_happy_path(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """Full run_benchmark_job with mocked _run_lighteval returning canned results."""

    # Single patch -- _run_lighteval returns parsed results directly
    monkeypatch.setattr(
        adapter, "_run_lighteval",
        lambda **kwargs: CANNED_RESULTS,
    )

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    # FrameworkAdapter contract
    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds > 0

    # Metrics extracted from canned data
    assert len(results.results) > 0
    assert any(r.metric_name == "boolq.accuracy" for r in results.results)
    boolq_acc = next(r for r in results.results if r.metric_name == "boolq.accuracy")
    assert boolq_acc.metric_value == 0.78
    assert boolq_acc.confidence_interval is not None

    # Overall score and example count
    assert results.overall_score == 0.78
    assert results.num_examples_evaluated == 5

    # Callback lifecycle phases
    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases
    # PERSISTING_ARTIFACTS only emitted when OCI exports are configured


@pytest.mark.integration
def test_oci_export_persists_artifacts(tmp_path, mock_callbacks, monkeypatch, mock_hf_api):
    """When exports.oci is configured, PERSISTING_ARTIFACTS is emitted and create_oci_artifact is called."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["exports"] = {
        "oci": {
            "coordinates": {
                "oci_host": "quay.io",
                "oci_repository": "test-org/test-repo",
                "oci_tag": "test-tag",
                "annotations": {},
            }
        }
    }
    (meta_dir / "job.json").write_text(json.dumps(job))

    adapter = LightEvalAdapter(job_spec_path=str(meta_dir / "job.json"))

    monkeypatch.setattr(
        adapter, "_run_lighteval",
        lambda **kwargs: CANNED_RESULTS,
    )

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    # PERSISTING_ARTIFACTS phase was reported
    phases = [c.args[0].phase for c in mock_callbacks.report_status.call_args_list]
    assert JobPhase.PERSISTING_ARTIFACTS in phases

    # create_oci_artifact was called with a directory containing result files
    call_args = mock_callbacks.create_oci_artifact.call_args
    assert call_args is not None
    spec = call_args.args[0]
    assert spec.files_path.exists()
    assert spec.files_path.is_dir()

    # OCI artifact is attached to results
    assert results.oci_artifact is not None
    assert results.oci_artifact.digest == "sha256:fake"


@pytest.mark.integration
def test_results_use_local_jobs_base_path(adapter, tmp_path):
    """Results are saved under local_jobs_base_path/results, not hardcoded /tmp paths."""
    expected_base = adapter.local_jobs_base_path
    assert expected_base is not None, "local mode should provide local_jobs_base_path"
    assert expected_base == tmp_path
    expected_results_dir = expected_base / "results"

    saved_files = adapter._save_detailed_results(
        job_id=adapter.job_spec.id,
        benchmark_id=adapter.job_spec.benchmark_id,
        model_name=adapter.job_spec.model.name,
        lighteval_results=CANNED_RESULTS,
        evaluation_results=[],
    )

    assert len(saved_files) > 0
    for f in saved_files:
        assert str(f).startswith(str(expected_results_dir)), (
            f"Result file {f} should be under {expected_results_dir}"
        )
        assert "/tmp/lighteval_results" not in str(f)


@pytest.mark.integration
def test_additional_info_zero_shot(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """Zero-shot run populates zero_shot with the overall score."""
    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    info = results.additional_info
    assert info["zero_shot"] == results.overall_score
    assert "alt_prompting" not in info
    assert "alt_prompting_description" not in info
    assert len(info["dataset"]) == 1
    assert info["dataset"][0]["hf_repo"] == "google/boolq"
    assert info["dataset"][0]["hf_subset"] == "default"


@pytest.mark.integration
def test_additional_info_few_shot(tmp_path, mock_callbacks, monkeypatch, mock_hf_api):
    """Few-shot run populates alt_prompting with the overall score and a description."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()

    with open(Path("meta/job.json")) as f:
        job = json.load(f)
    job["parameters"]["num_few_shot"] = 3
    (meta_dir / "job.json").write_text(json.dumps(job))

    few_shot_results = copy.deepcopy(CANNED_RESULTS)
    few_shot_results["config_tasks"] = {
        "boolq|3": {
            "name": "boolq",
            "hf_repo": "google/boolq",
            "hf_subset": "default",
            "num_fewshots": 3,
        }
    }
    few_shot_results["results"] = {"boolq|3": {"accuracy": 0.78, "accuracy_stderr": 0.02}}

    adapter = LightEvalAdapter(job_spec_path=str(meta_dir / "job.json"))
    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: few_shot_results)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    info = results.additional_info
    assert "zero_shot" not in info
    assert info["alt_prompting"] == results.overall_score
    assert info["alt_prompting_description"] == "3-Shot"


@pytest.mark.integration
def test_additional_info_in_results_json(adapter):
    """additional_info is written into the structured results.json file."""
    sample_info = {
        "dataset": [{"hf_repo": "google/boolq", "hf_subset": "default"}],
        "zero_shot": 0.78,
    }

    saved_files = adapter._save_detailed_results(
        job_id=adapter.job_spec.id,
        benchmark_id=adapter.job_spec.benchmark_id,
        model_name=adapter.job_spec.model.name,
        lighteval_results=CANNED_RESULTS,
        evaluation_results=[],
        additional_info=sample_info,
    )

    results_file = next(f for f in saved_files if f.name == "results.json")
    with open(results_file) as f:
        data = json.load(f)

    assert "additional_info" in data
    assert data["additional_info"]["zero_shot"] == 0.78
    assert data["additional_info"]["dataset"][0]["hf_repo"] == "google/boolq"


@pytest.mark.integration
def test_generate_additional_info_fallback(adapter):
    """generate_additional_info() works as a fallback using evaluation_metadata."""
    results = JobResults(
        id="test-job",
        benchmark_id="boolq",
        benchmark_index=0,
        model_name="test-model",
        results=[],
        overall_score=0.85,
        num_examples_evaluated=10,
        duration_seconds=1.0,
        completed_at=datetime.now(UTC),
        evaluation_metadata={"num_few_shot": 5},
    )

    info = adapter.generate_additional_info(results)

    assert info is not None
    assert "zero_shot" not in info
    assert info["alt_prompting"] == 0.85
    assert info["alt_prompting_description"] == "5-Shot"
    assert info["dataset"] == []


@pytest.fixture
def mock_hf_api(monkeypatch):
    """Stub huggingface_hub so _resolve_dataset_shas never hits the network.

    Returns the mock HfApi instance; SHA-specific tests can reconfigure
    mock_api.dataset_info to return custom metadata.
    """
    mock_api = MagicMock()
    mock_api.dataset_info.return_value = SimpleNamespace(sha="deterministic-sha")
    stub = ModuleType("huggingface_hub")
    stub.HfApi = MagicMock(return_value=mock_api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
    return mock_api


@pytest.mark.integration
def test_resolve_dataset_shas_success(mock_hf_api):
    """_resolve_dataset_shas resolves SHAs via HfApi."""
    config_tasks = {
        "gsm8k|3": {
            "name": "gsm8k",
            "hf_repo": "openai/gsm8k",
            "hf_subset": "main",
            "hf_revision": None,
            "num_fewshots": 3,
        },
        "math:algebra|3": {
            "name": "math:algebra",
            "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
            "hf_subset": "algebra",
            "hf_revision": None,
            "num_fewshots": 3,
        },
        "math:counting_and_probability|3": {
            "name": "math:counting_and_probability",
            "hf_repo": "DigitalLearningGmbH/MATH-lighteval",
            "hf_subset": "counting_and_probability",
            "hf_revision": None,
            "num_fewshots": 3,
        },
    }

    mock_hf_api.dataset_info.side_effect = lambda repo_id, revision="main": {
        "openai/gsm8k": SimpleNamespace(sha="aaa111"),
        "DigitalLearningGmbH/MATH-lighteval": SimpleNamespace(sha="bbb222"),
    }[repo_id]

    dataset = LightEvalAdapter._resolve_dataset_shas(config_tasks)

    assert len(dataset) == 3
    assert dataset[0] == {"hf_repo": "openai/gsm8k", "hf_subset": "main", "sha": "aaa111"}
    assert dataset[1] == {"hf_repo": "DigitalLearningGmbH/MATH-lighteval", "hf_subset": "algebra", "sha": "bbb222"}
    assert dataset[2] == {"hf_repo": "DigitalLearningGmbH/MATH-lighteval", "hf_subset": "counting_and_probability", "sha": "bbb222"}
    assert mock_hf_api.dataset_info.call_count == 2


@pytest.mark.integration
def test_resolve_dataset_shas_different_revisions(mock_hf_api):
    """Two tasks sharing a repo but with different revisions resolve independently."""
    config_tasks = {
        "task_a|0": {
            "name": "task_a",
            "hf_repo": "shared/repo",
            "hf_subset": "default",
            "hf_revision": "rev-aaa",
            "num_fewshots": 0,
        },
        "task_b|0": {
            "name": "task_b",
            "hf_repo": "shared/repo",
            "hf_subset": "default",
            "hf_revision": "rev-bbb",
            "num_fewshots": 0,
        },
    }

    mock_hf_api.dataset_info.side_effect = lambda repo_id, revision="main": {
        "rev-aaa": SimpleNamespace(sha="sha-aaa"),
        "rev-bbb": SimpleNamespace(sha="sha-bbb"),
    }[revision]

    dataset = LightEvalAdapter._resolve_dataset_shas(config_tasks)

    assert len(dataset) == 2
    assert dataset[0] == {"hf_repo": "shared/repo", "hf_subset": "default", "sha": "sha-aaa"}
    assert dataset[1] == {"hf_repo": "shared/repo", "hf_subset": "default", "sha": "sha-bbb"}
    assert mock_hf_api.dataset_info.call_count == 2


@pytest.mark.integration
def test_resolve_dataset_shas_fault_tolerant(mock_hf_api):
    """_resolve_dataset_shas skips SHA on API failure without crashing."""
    config_tasks = {
        "gsm8k|0": {
            "name": "gsm8k",
            "hf_repo": "openai/gsm8k",
            "hf_subset": "main",
            "num_fewshots": 0,
        }
    }

    mock_hf_api.dataset_info.side_effect = Exception("network error")

    dataset = LightEvalAdapter._resolve_dataset_shas(config_tasks)

    assert len(dataset) == 1
    assert dataset[0] == {"hf_repo": "openai/gsm8k", "hf_subset": "main"}
    assert "sha" not in dataset[0]


@pytest.mark.integration
def test_additional_info_includes_sha(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """Full run_benchmark_job includes dataset SHA when HfApi succeeds."""
    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: CANNED_RESULTS)
    mock_hf_api.dataset_info.return_value = SimpleNamespace(sha="abc123def456")

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    ds = results.additional_info["dataset"]
    assert len(ds) == 1
    assert ds[0]["hf_repo"] == "google/boolq"
    assert ds[0]["sha"] == "abc123def456"


@pytest.mark.integration
def test_additional_info_generation_parameters(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """generation_parameters includes only non-null values from config_general."""
    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: CANNED_RESULTS)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    gen = results.additional_info["generation_parameters"]
    assert gen == {"temperature": 0}
    assert "max_new_tokens" not in gen
    assert "top_p" not in gen


@pytest.mark.integration
def test_generation_parameters_all_null(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """generation_parameters key is omitted when all values are null."""
    all_null_results = copy.deepcopy(CANNED_RESULTS)
    all_null_results["config_general"]["model_config"]["generation_parameters"] = {
        "temperature": None,
        "max_new_tokens": None,
    }

    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: all_null_results)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    assert "generation_parameters" not in results.additional_info


@pytest.mark.integration
def test_generation_parameters_rich(adapter, mock_callbacks, monkeypatch, mock_hf_api):
    """generation_parameters preserves multiple non-null values."""
    rich_results = copy.deepcopy(CANNED_RESULTS)
    rich_results["config_general"]["model_config"]["generation_parameters"] = {
        "temperature": 0.1,
        "max_new_tokens": 512,
        "top_p": None,
        "seed": 42,
        "stop_tokens": None,
    }

    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: rich_results)

    results = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert results.additional_info is not None
    gen = results.additional_info["generation_parameters"]
    assert gen == {"temperature": 0.1, "max_new_tokens": 512, "seed": 42}
