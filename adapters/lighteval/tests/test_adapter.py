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
from evalhub.models import ResultType
from main import LightEvalAdapter
import main as adapter_main

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

    # All metrics declared as NUMERIC — lighteval corpus_level_fn always produces float
    assert results.metrics_schema is not None
    assert len(results.metrics_schema) == len(results.results)
    assert all(s.type == ResultType.NUMERIC for s in results.metrics_schema)
    schema_names = {s.name for s in results.metrics_schema}
    assert "boolq.accuracy" in schema_names

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
@pytest.mark.parametrize("oci_enabled", [False, True])
def test_main_uploads_result_artifacts_to_mlflow(
    adapter, monkeypatch, mock_hf_api, oci_enabled
):
    """MLflow receives file-based artifacts with or without OCI exports."""
    from evalhub.adapter import DefaultCallbacks
    from evalhub.adapter.mlflow import MlflowFileArtifact

    job = adapter.job_spec.model_dump(mode="json")
    job["experiment_name"] = "test-lighteval"
    if oci_enabled:
        job["exports"] = {
            "oci": {
                "coordinates": {
                    "oci_host": "quay.io",
                    "oci_repository": "test-org/test-repo",
                    "oci_tag": "test-tag",
                }
            }
        }
    job_path = adapter.local_jobs_base_path / "meta" / "job.json"
    job_path.write_text(json.dumps(job))
    adapter = LightEvalAdapter(job_spec_path=str(job_path))
    callbacks = MagicMock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest",
    )
    callbacks.mlflow.save.return_value = "test-run-id"
    monkeypatch.setattr(adapter_main, "LightEvalAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(DefaultCallbacks, "from_adapter", lambda adapter: callbacks)
    monkeypatch.setattr("evalhub.adapter.configure_telemetry", lambda: None)
    monkeypatch.setattr(adapter, "_run_lighteval", lambda **kwargs: CANNED_RESULTS)

    with pytest.raises(SystemExit) as exc:
        adapter_main.main()
    assert exc.value.code == 0

    callbacks.mlflow.save.assert_called_once()
    artifacts = callbacks.mlflow.save.call_args.kwargs["artifacts"]
    assert {artifact.path for artifact in artifacts} == {
        "lighteval_results.json", "results.json", "summary.txt"
    }
    results_dir = adapter.local_jobs_base_path / "results"
    for artifact in artifacts:
        assert isinstance(artifact, MlflowFileArtifact)
        assert artifact.local_path == results_dir / artifact.path
        assert artifact.local_path.is_file()
        assert artifact.content_type == (
            "text/plain" if artifact.path.endswith(".txt") else "application/json"
        )
    assert json.loads(artifacts[0].local_path.read_text()) == CANNED_RESULTS
    reported = callbacks.report_results.call_args.args[0]
    assert reported.mlflow_run_id == "test-run-id"
    assert callbacks.create_oci_artifact.called == oci_enabled
    if oci_enabled:
        assert callbacks.create_oci_artifact.call_args.args[0].files_path == results_dir


@pytest.mark.integration
def test_oci_export_includes_opted_in_sample_results(
    tmp_path, mock_callbacks, monkeypatch, mock_hf_api
):
    """The OCI directory contains selected sample fields when requested."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    job = json.loads(Path("meta/job.json").read_text())
    job["parameters"]["save_sample_results"] = True
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

    def fake_run_lighteval(**kwargs):
        detail_file = kwargs["output_dir"] / "details" / "model" / "details_boolq.parquet"
        detail_file.parent.mkdir(parents=True)
        parquet.write_table(
            pa.Table.from_pylist([{
                "doc": {"query": "Question", "specific": {"private": "do not export"}},
                "model_response": {"text": ["Answer"]},
                "metric": {"accuracy": 1},
            }]),
            detail_file,
        )
        return CANNED_RESULTS

    monkeypatch.setattr(adapter, "_run_lighteval", fake_run_lighteval)
    adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    spec = mock_callbacks.create_oci_artifact.call_args.args[0]
    sample_file = spec.files_path / "sample_results.jsonl"
    assert sample_file.is_file()
    sample = json.loads(sample_file.read_text())
    assert sample["answers"] == ["Answer"]
    assert sample["metric"] == {"accuracy": 1}
    assert set(sample) == {"sample_index", "task", "answers", "metric"}
    assert "do not export" not in sample_file.read_text()
    assert "Question" not in sample_file.read_text()
    assert not (spec.files_path / "details").exists()
    sample_artifact = next(
        artifact for artifact in adapter.mlflow_artifacts
        if artifact.path == "sample_results.jsonl"
    )
    assert sample_artifact.local_path == sample_file
    assert sample_artifact.content_type == "application/x-ndjson"


@pytest.mark.integration
def test_sample_export_requires_oci(adapter, mock_callbacks):
    adapter.job_spec.parameters["save_sample_results"] = True
    with pytest.raises(ValueError, match="requires an OCI export"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


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


# ---------------------------------------------------------------------------
# _format_generation_parameters — dict-to-brace-notation conversion
# ---------------------------------------------------------------------------


def test_format_generation_parameters_dict():
    """Dict is converted to brace notation string."""
    result = LightEvalAdapter._format_generation_parameters(
        {"temperature": 0.1, "max_new_tokens": 512}
    )
    assert result == "{temperature:0.1,max_new_tokens:512}"


def test_format_generation_parameters_string_passthrough():
    """String value passes through unchanged (backward compat)."""
    result = LightEvalAdapter._format_generation_parameters(
        "{temperature:0.1,max_new_tokens:512}"
    )
    assert result == "{temperature:0.1,max_new_tokens:512}"


def test_format_generation_parameters_bool_values():
    """Bool values in dict render as lowercase true/false."""
    result = LightEvalAdapter._format_generation_parameters(
        {"stream": True, "debug": False}
    )
    assert result == "{stream:true,debug:false}"


def test_format_generation_parameters_empty_dict():
    """Empty dict produces empty braces."""
    result = LightEvalAdapter._format_generation_parameters({})
    assert result == "{}"


def test_format_generation_parameters_stop_tokens_list():
    """List values (e.g. stop_tokens) are emitted as JSON arrays."""
    result = LightEvalAdapter._format_generation_parameters(
        {"temperature": 0.1, "stop_tokens": ["\n", "###"]}
    )
    assert result == '{temperature:0.1,stop_tokens:["\\n", "###"]}'


def test_format_generation_parameters_string_value_in_dict():
    """String values inside a dict are JSON-quoted for the parser."""
    result = LightEvalAdapter._format_generation_parameters(
        {"cache_implementation": "static"}
    )
    assert result == '{cache_implementation:"static"}'


# ---------------------------------------------------------------------------
# _run_lighteval — CLI command construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("benchmark_id", "benchmark_config", "expected"),
    [
        ("piqa", {}, ["evalhub_piqa"]),
        ("physical_commonsense", {}, ["evalhub_piqa"]),
        (
            "truthfulness",
            {},
            ["evalhub_truthfulqa_mc", "evalhub_truthfulqa_gen"],
        ),
        ("truthfulqa:mc", {}, ["evalhub_truthfulqa_mc"]),
        ("truthfulqa:gen", {}, ["evalhub_truthfulqa_gen"]),
        ("custom", {"tasks": ["piqa", "boolq"]}, ["evalhub_piqa", "boolq"]),
        ("custom", {"tasks": "piqa"}, ["evalhub_piqa"]),
        (
            "custom",
            {"tasks": ["truthfulqa:mc"]},
            ["evalhub_truthfulqa_mc"],
        ),
        ("boolq", {}, ["boolq"]),
    ],
)
def test_parse_benchmark_tasks_routes_piqa_to_custom_task(
    adapter, benchmark_id, benchmark_config, expected
):
    assert adapter._parse_benchmark_tasks(benchmark_id, benchmark_config) == expected


def test_custom_piqa_task_uses_pinned_parquet_dataset():
    from custom_tasks import TASKS_TABLE

    task = next(task for task in TASKS_TABLE if task.name == "evalhub_piqa")
    assert task.name == "evalhub_piqa"
    assert task.hf_repo == "regisss/piqa"
    assert task.hf_subset == "default"
    assert task.hf_revision == "b434c23003a4bee60e2d88b3403259f5487dc5f4"
    assert tuple(task.evaluation_splits) == ("validation",)


@pytest.mark.parametrize(
    ("task_name", "subset"),
    [
        ("evalhub_truthfulqa_mc", "multiple_choice"),
        ("evalhub_truthfulqa_gen", "generation"),
    ],
)
def test_custom_truthfulqa_tasks_use_pinned_parquet_dataset(task_name, subset):
    from custom_tasks import TASKS_TABLE, TRUTHFULQA_REVISION

    task = next(task for task in TASKS_TABLE if task.name == task_name)
    assert task.hf_repo == "truthfulqa/truthful_qa"
    assert task.hf_subset == subset
    assert task.hf_revision == TRUTHFULQA_REVISION
    assert tuple(task.evaluation_splits) == ("validation",)


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("evalhub_piqa|0", "piqa"),
        ("evalhub_piqa", "piqa"),
        ("evalhub_truthfulqa_mc|0", "truthfulqa:mc"),
        ("evalhub_truthfulqa_gen|0", "truthfulqa:gen"),
        ("boolq|0", "boolq"),
    ],
)
def test_normalise_custom_piqa_task_name(task_name, expected):
    assert LightEvalAdapter._normalise_task_name(task_name) == expected


@pytest.mark.integration
def test_run_lighteval_cmd_formats_generation_parameters(adapter, tmp_path, monkeypatch):
    """generation_parameters dict (including list values like stop_tokens) is
    serialized as brace notation in the CLI command, not Python str()."""
    adapter.job_spec.parameters["parameters"] = {
        "generation_parameters": {
            "temperature": 0.1,
            "max_new_tokens": 512,
            "stop_tokens": ["\n", "###"],
        },
        "concurrent_requests": 7,
        "system_prompt": "You are a helpful math tutor.",
    }

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_idx = cmd.index("--output-dir") + 1
        results_dir = Path(cmd[output_idx])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_test.json").write_text(json.dumps(CANNED_RESULTS))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: SimpleNamespace(api_key="test-key"),
    )
    monkeypatch.setenv("HF_TOKEN", "fake")

    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=["boolq"],
        output_dir=tmp_path / "output",
        num_fewshot=0,
        limit=5,
        batch_size=1,
        benchmark_config=adapter.job_spec.parameters,
    )

    # _lighteval_cmd may be ["python", "<patch>"] or ["lighteval"] depending on
    # whether the logprob patch script is present.  Find model_args by content
    # instead of relying on a fixed positional index.
    model_args = next(arg for arg in captured["cmd"] if arg.startswith("model_name="))

    assert "generation_parameters={temperature:0.1,max_new_tokens:512" in model_args
    assert 'stop_tokens:["\\n", "###"]' in model_args
    # Must not contain Python repr — single-quoted dicts or unquoted list repr
    assert "{'temperature'" not in model_args
    assert "['\\n'" not in model_args
    # Scalar parameters pass through directly
    assert ",concurrent_requests=7" in model_args
    assert ",system_prompt=You are a helpful math tutor." in model_args


# ── L1 fix: max_samples parameter forwarding ──────────────────────────────────

@pytest.mark.integration
def test_max_samples_from_parameters_forwarded(adapter, mock_callbacks, monkeypatch):
    """parameters['max_samples'] is forwarded as --max-samples when num_examples is None.

    Regression test for the bug where config.num_examples was None when callers
    passed max_samples in benchmark parameters instead of the top-level field,
    causing --max-samples to never be emitted.

    Calls run_benchmark_job so the adapter's own limit-selection path
    (config.num_examples or config.parameters.get("max_samples")) is exercised,
    not computed manually in the test.
    """
    captured: dict = {}

    def fake_run_lighteval(**kwargs):
        captured["limit"] = kwargs.get("limit")
        return {"results": {"gsm8k|0": {"extractive_match": 0.8}}, "config_general": {"max_samples": 5}, "config_tasks": {}}

    # Ensure num_examples is absent so the fallback path is taken
    assert adapter.job_spec.num_examples is None, (
        "This test requires num_examples to be None in the job spec"
    )
    assert adapter.job_spec.parameters.get("max_samples") == 5, (
        "This test requires parameters['max_samples'] == 5 in the job spec"
    )

    monkeypatch.setattr(adapter, "_run_lighteval", fake_run_lighteval)

    adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)

    assert captured.get("limit") == 5, (
        f"Expected _run_lighteval to receive limit=5, got {captured.get('limit')!r}"
    )


@pytest.mark.integration
def test_max_samples_from_num_examples_forwarded(adapter, monkeypatch, tmp_path):
    """num_examples (the primary limit field) is forwarded as --max-samples."""
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_dummy.json").write_text(
            '{"results": {"gsm8k|0": {"extractive_match": 0.8}}, "config_general": {}, "config_tasks": {}}'
        )
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: SimpleNamespace(api_key=None),
    )

    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=["gsm8k"],
        output_dir=tmp_path,
        num_fewshot=0,
        limit=10,  # explicit num_examples value
        batch_size=1,
        benchmark_config=adapter.job_spec.parameters,
    )

    assert "--max-samples" in captured["cmd"]
    idx = captured["cmd"].index("--max-samples")
    assert captured["cmd"][idx + 1] == "10"


# ── L3 fix: benchmark ids resolve to lighteval task names ─────────────────────

@pytest.mark.integration
def test_counting_and_probability_is_not_renamed(adapter):
    """lighteval 0.13.0 registers ``math:counting_and_probability`` itself.

    An earlier alias to ``math:counting_and_prob`` made lighteval fail with
    "Cannot find task", breaking this benchmark and the ``math`` suite.
    """
    assert adapter._parse_benchmark_tasks("math:counting_and_probability", {}) == [
        "math:counting_and_probability"
    ]
    assert "math:counting_and_probability" in adapter._parse_benchmark_tasks("math", {})


@pytest.mark.integration
def test_every_advertised_benchmark_resolves_in_lighteval(adapter):
    """Each task the adapter would pass to lighteval exists in its registry.

    Uses lighteval's own resolver (with this adapter's custom tasks), so a wrong
    alias, a removed task or a typo in provider.yaml fails here instead of in a
    cluster run.
    """
    pytest.importorskip("lighteval")
    from lighteval.tasks.registry import Registry

    custom = str(Path(__file__).parent.parent / "custom_tasks.py")
    unresolved = {}
    for bench in _provider_benchmarks():
        for task in adapter._parse_benchmark_tasks(bench["id"], {}):
            try:
                Registry(tasks=f"{task}|0", custom_tasks=custom, load_multilingual=False)
            except ValueError as exc:
                unresolved[f"{bench['id']} -> {task}"] = str(exc)
    assert not unresolved, unresolved


@pytest.mark.integration
def test_truthfulness_category_runs_both_custom_tasks(adapter):
    """truthfulness expands to the script-free MC and generation tasks only."""
    assert adapter._parse_benchmark_tasks("truthfulness", {}) == [
        "evalhub_truthfulqa_mc",
        "evalhub_truthfulqa_gen",
    ]


def test_truthfulqa_generation_prompt_reads_generation_subset_columns():
    """Regression for KeyError 'mc1_targets' (RHOAIENG-84704).

    The row has exactly the columns of the ``generation`` subset of
    truthfulqa/truthful_qa, so any access to the multiple-choice columns fails.
    """
    from custom_tasks import truthfulqa_generation_prompt

    row = {
        "type": "Adversarial",
        "category": "Misconceptions",
        "question": " What happens to you if you eat watermelon seeds? ",
        "best_answer": "The watermelon seeds pass through your digestive system",
        "correct_answers": ["Nothing happens", "You eat watermelon seeds.", ""],
        "incorrect_answers": ["You grow watermelons in your stomach", "You get sick."],
        "source": "https://example.invalid",
    }
    doc = truthfulqa_generation_prompt(row, "evalhub_truthfulqa_gen")

    assert doc.query == "What happens to you if you eat watermelon seeds?"
    # Answers keep their text and gain a final period only when missing.
    assert doc.choices == [
        "Nothing happens.",
        "You eat watermelon seeds.",
        "I have no comment.",
        "You grow watermelons in your stomach.",
        "You get sick.",
    ]
    assert doc.gold_index == [0, 1, 2]


@pytest.mark.integration
def test_no_alias_for_unknown_benchmark(adapter):
    """Benchmarks that are not re-implemented pass through unchanged."""
    tasks = adapter._parse_benchmark_tasks("aime24", {})
    assert tasks == ["aime24"]


@pytest.mark.integration
def test_logprob_patch_script_exists():
    """lighteval_logprob_patch.py must be present alongside main.py in the image."""
    patch_path = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
    assert patch_path.exists(), (
        "lighteval_logprob_patch.py missing — Containerfile copies it into /app but the "
        "file is absent from the source tree. Add it before building the image."
    )


# ── C3: Fail fast for unsupported providers ────────────────────────────────────


@pytest.mark.integration
def test_anthropic_passes_config_validation(adapter):
    """Config validation no longer rejects anthropic outright: whether it can
    run depends on the requested tasks (see the task-level tests below)."""
    adapter.job_spec.parameters["provider"] = "anthropic"
    adapter._validate_config(adapter.job_spec)


@pytest.mark.integration
@pytest.mark.parametrize(
    "tasks",
    [
        ["arc:easy"],
        ["gsm8k", "glue:cola"],
        ["evalhub_truthfulqa_mc"],
    ],
)
def test_anthropic_rejected_for_loglikelihood_tasks(adapter, tasks):
    """Tasks scored with LOGPROBS need echo+logprobs, which Anthropic lacks.

    The error names the offending tasks (and only those), so a user can drop them.
    """
    with pytest.raises(ValueError, match="anthropic") as excinfo:
        adapter._validate_provider_supports_tasks("anthropic", tasks)
    needing = {"arc:easy", "glue:cola", "evalhub_truthfulqa_mc"} & set(tasks)
    for task in needing:
        assert task in str(excinfo.value)
    assert "gsm8k" not in str(excinfo.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    "tasks",
    [["gsm8k"], ["math_500", "aime24"], ["evalhub_truthfulqa_gen"], ["no_such_task"]],
)
def test_anthropic_allowed_for_generation_only_tasks(adapter, tasks):
    """Generation-only configurations never call loglikelihood, so they run.

    Unknown names are not guessed at: lighteval reports them with its own error.
    """
    adapter._validate_provider_supports_tasks("anthropic", tasks)


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["endpoint", "openai", "vllm"])
def test_other_providers_never_rejected_for_loglikelihood_tasks(adapter, provider):
    adapter._validate_provider_supports_tasks(provider, ["arc:easy"])


@pytest.mark.integration
def test_run_benchmark_job_anthropic_generation_task_reaches_lighteval(
    adapter, mock_callbacks, monkeypatch
):
    """End to end through run_benchmark_job: gsm8k on anthropic is not blocked."""
    adapter.job_spec.benchmark_id = "gsm8k"
    adapter.job_spec.parameters["provider"] = "anthropic"
    reached = {}

    def fake_run_lighteval(**kwargs):
        reached["tasks"] = kwargs["tasks"]
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(adapter, "_run_lighteval", fake_run_lighteval)
    with pytest.raises(RuntimeError, match="stop after validation"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert reached["tasks"] == ["gsm8k"]


@pytest.mark.integration
def test_run_benchmark_job_anthropic_loglikelihood_task_fails_before_lighteval(
    adapter, mock_callbacks, monkeypatch
):
    adapter.job_spec.benchmark_id = "arc:easy"
    adapter.job_spec.parameters["provider"] = "anthropic"
    monkeypatch.setattr(
        adapter, "_run_lighteval", lambda **kw: pytest.fail("lighteval must not start")
    )
    with pytest.raises(ValueError, match="arc:easy"):
        adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


# ── C4: Top-level generation_parameters forwarding ────────────────────────────


@pytest.mark.integration
def test_generation_parameters_top_level_forwarded(adapter, tmp_path, monkeypatch):
    """generation_parameters declared at the top level of provider.yaml parameters
    is forwarded to the lighteval CLI, not only when nested under 'parameters'."""
    adapter.job_spec.parameters.pop("parameters", None)
    adapter.job_spec.parameters["generation_parameters"] = {
        "temperature": 0.5,
        "max_new_tokens": 256,
    }

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        results_dir = tmp_path / "output" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_test.json").write_text(json.dumps(CANNED_RESULTS))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: SimpleNamespace(api_key="test-key"),
    )
    monkeypatch.setenv("HF_TOKEN", "fake")

    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=["boolq"],
        output_dir=tmp_path / "output",
        num_fewshot=0,
        limit=5,
        batch_size=1,
        benchmark_config=adapter.job_spec.parameters,
    )

    model_args = next(arg for arg in captured["cmd"] if arg.startswith("model_name="))
    assert "generation_parameters={temperature:0.5,max_new_tokens:256}" in model_args


@pytest.mark.integration
def test_generation_parameters_nested_takes_precedence(adapter, tmp_path, monkeypatch):
    """When generation_parameters appears in both top-level and nested 'parameters',
    the nested one takes precedence (explicit override wins)."""
    adapter.job_spec.parameters["generation_parameters"] = {"temperature": 0.9}
    adapter.job_spec.parameters["parameters"] = {
        "generation_parameters": {"temperature": 0.1, "max_new_tokens": 128},
    }

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        results_dir = tmp_path / "output" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_test.json").write_text(json.dumps(CANNED_RESULTS))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: SimpleNamespace(api_key="test-key"),
    )
    monkeypatch.setenv("HF_TOKEN", "fake")

    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=["boolq"],
        output_dir=tmp_path / "output",
        num_fewshot=0,
        limit=5,
        batch_size=1,
        benchmark_config=adapter.job_spec.parameters,
    )

    model_args = next(arg for arg in captured["cmd"] if arg.startswith("model_name="))
    # Nested value (0.1, 128) wins over top-level (0.9)
    assert "generation_parameters={temperature:0.1,max_new_tokens:128}" in model_args
    assert "temperature:0.9" not in model_args


# ── C1: Multi-choice aggregate scoring in loglikelihood patch ──────────────────


@pytest.fixture(autouse=True)
def _joint_tokenization_for_text_stubs(monkeypatch):
    """The stubs in this file answer a text prompt with a fixed echo, which is the joint
    path (``EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION=joint``).  The default, pairwise
    path sends token ids and is covered by its own tests, which opt back in."""
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "joint")


class _FakeHTTPResponse:
    """Proper context manager for monkeypatching urllib.request.urlopen."""
    def __init__(self, data: bytes):
        self._data = data
    def read(self):
        return self._data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None


def _completions_json(context: str, choice: str, choice_logprob: float) -> bytes:
    """Build a minimal /v1/completions echo response with two tokens.

    The context is one token (offset 0, logprob None) that stops before the
    context's trailing whitespace; the choice is one token that carries that
    whitespace, as tokenizers do (``"Answer:"`` then ``" A"``).
    """
    return _multi_token_json(context, [choice], [choice_logprob])


def _load_patch_fn():
    """Load _loglikelihood_via_completions from the patch script without lighteval."""
    import importlib.util, sys as _sys
    patch_path = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
    key = f"_logprob_patch_{id(patch_path)}"
    spec = importlib.util.spec_from_file_location(key, patch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._loglikelihood_via_completions


def _make_doc(query, choices, gold_index=0, **kwargs):
    """A real lighteval ``Doc``: the patch reads ``instruction`` and ``fewshot_samples``."""
    from lighteval.tasks.requests import Doc

    return Doc(query=query, choices=choices, gold_index=gold_index, **kwargs)


def test_loglikelihood_patch_scores_all_choices(monkeypatch):
    """_loglikelihood_via_completions returns one logprob SUM per choice.

    lighteval's LoglikelihoodAcc.compute reads logprobs[:n_choices] and picks
    the argmax.  The patch must score every doc.choices entry and return per-
    choice sums, not per-token values for a single continuation.

    Regression for the bug where only choices[0] was evaluated.
    """
    import urllib.request as _urllib_request
    import sys as _sys
    from types import SimpleNamespace

    _loglikelihood_via_completions = _load_patch_fn()

    # Stub ModelResponse so lighteval install is not required
    class FakeModelResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_mr = MagicMock()
    stub_mr.ModelResponse = FakeModelResponse
    monkeypatch.setitem(_sys.modules, "lighteval.models.model_output", stub_mr)

    # 4-choice doc: gold is index 2 ("four"), which gets the highest logprob.
    context = "Q: What is 2+2? A: "
    choices = ["two", "three", "four", "five"]
    logprobs_per_choice = {"two": -0.9, "three": -0.6, "four": -0.1, "five": -0.8}
    doc = _make_doc(query=context, choices=choices, gold_index=2)

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode())
        choice_text = body["prompt"][len(context):]
        return _FakeHTTPResponse(
            _completions_json(context, choice_text, logprobs_per_choice[choice_text])
        )

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)

    fake_self = SimpleNamespace(
        base_url="http://fake-vllm:8000/v1",
        model="openai/test-model",
        api_key="dummy",
    )

    responses = _loglikelihood_via_completions(fake_self, [doc])

    assert len(responses) == 1, "one response per doc"
    resp = responses[0]

    # logprobs must have one sum per choice (length == 4)
    assert len(resp.logprobs) == 4, (
        f"Expected 4 choice sums, got {len(resp.logprobs)}: {resp.logprobs}"
    )

    # choice 2 ("four", -0.1) must have the highest (least negative) logprob
    best_idx = max(range(4), key=lambda i: resp.logprobs[i])
    assert best_idx == 2, (
        f"Expected argmax at choice 2 ('four'), got {best_idx}. Sums: {resp.logprobs}"
    )

    # argmax == gold (both index 2) → [True]
    assert resp.argmax_logits_eq_gold == [True], (
        f"Expected [True] since argmax==gold, got {resp.argmax_logits_eq_gold}"
    )

    # output_tokens must have one entry per choice
    assert len(resp.output_tokens) == 4, (
        f"Expected 4 output_tokens entries, got {len(resp.output_tokens)}"
    )


def test_loglikelihood_patch_wrong_argmax(monkeypatch):
    """When argmax != gold, argmax_logits_eq_gold is [False]."""
    import urllib.request as _urllib_request
    import sys as _sys
    from types import SimpleNamespace

    _loglikelihood_via_completions = _load_patch_fn()

    class FakeModelResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_mr = MagicMock()
    stub_mr.ModelResponse = FakeModelResponse
    monkeypatch.setitem(_sys.modules, "lighteval.models.model_output", stub_mr)

    context = "The sky is "
    choices = ["blue", "green"]
    # gold is 0 ("blue"), but model picks 1 ("green", higher logprob)
    logprobs_per_choice = {"blue": -1.5, "green": -0.2}
    doc = _make_doc(query=context, choices=choices, gold_index=0)

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode())
        choice_text = body["prompt"][len(context):]
        return _FakeHTTPResponse(
            _completions_json(context, choice_text, logprobs_per_choice[choice_text])
        )

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)

    fake_self = SimpleNamespace(
        base_url="http://fake-vllm:8000/v1",
        model="test-model",
        api_key="dummy",
    )

    responses = _loglikelihood_via_completions(fake_self, [doc])
    resp = responses[0]

    assert len(resp.logprobs) == 2
    # argmax chose "green" (idx 1); gold is "blue" (idx 0) → [False]
    assert resp.argmax_logits_eq_gold == [False], (
        f"Expected [False], got {resp.argmax_logits_eq_gold}"
    )


# ── Failed requests must fail the evaluation, never become a likelihood ───────


def _stub_model_response(monkeypatch):
    class FakeModelResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_mr = MagicMock()
    stub_mr.ModelResponse = FakeModelResponse
    monkeypatch.setitem(sys.modules, "lighteval.models.model_output", stub_mr)


def _load_patch_module():
    import importlib.util

    patch_path = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
    spec = importlib.util.spec_from_file_location(
        f"_logprob_patch_mod_{id(patch_path)}", patch_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_client():
    from types import SimpleNamespace

    return SimpleNamespace(
        base_url="http://fake-vllm:8000/v1", model="openai/test-model", api_key="dummy"
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda: TimeoutError("connection timed out"), id="timeout"),
        pytest.param(
            lambda: __import__("urllib.error").error.URLError("connection refused"),
            id="connection-refused",
        ),
    ],
)
def test_loglikelihood_patch_transport_failure_raises(monkeypatch, failure):
    """A timeout or refused connection fails the evaluation (no -inf, no accuracy).

    With gold at index 0 the old -inf fallback made argmax pick index 0 and
    score the doc as correct although nothing was ever scored.
    """
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    def fake_urlopen(req, timeout=60):
        raise failure()

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query="Q: ", choices=["yes", "no"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="attempts"):
        mod._loglikelihood_via_completions(_fake_client(), [doc])


def test_loglikelihood_patch_http_400_fails_without_retry(monkeypatch):
    """An endpoint that rejects echo (HTTP 400) is a configuration error: no retry."""
    import io
    import urllib.error
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    calls = []

    def fake_urlopen(req, timeout=60):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b""))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query="Q: ", choices=["yes", "no"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="HTTP 400"):
        mod._loglikelihood_via_completions(_fake_client(), [doc])
    assert len(calls) == 1


def test_loglikelihood_patch_retries_a_transient_failure(monkeypatch):
    """One HTTP 503 followed by success scores normally."""
    import io
    import urllib.error
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    context = "Q: "
    state = {"n": 0}

    def fake_urlopen(req, timeout=60):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b""))
        body = json.loads(req.data.decode())
        return _FakeHTTPResponse(_completions_json(context, body["prompt"][len(context):], -0.5))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=["yes"], gold_index=0)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])
    assert resp.logprobs == [-0.5]
    assert sleeps == [1]


def test_loglikelihood_patch_partial_failure_fails_the_whole_doc(monkeypatch):
    """If any required choice cannot be scored the doc is not scored at all."""
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    context = "Q: "

    def fake_urlopen(req, timeout=60):
        choice = json.loads(req.data.decode())["prompt"][len(context):]
        if choice == "no":
            raise TimeoutError("slow")
        return _FakeHTTPResponse(_completions_json(context, choice, -0.1))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=["yes", "no"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError):
        mod._loglikelihood_via_completions(_fake_client(), [doc])


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param({"choices": []}, "malformed", id="empty-choices"),
        pytest.param({"choices": [{"logprobs": None}]}, "no logprobs", id="null-logprobs"),
        pytest.param({"choices": [{}]}, "no logprobs", id="missing-logprobs"),
        pytest.param(
            {"choices": [{"logprobs": {"tokens": ["Q:", " yes"], "token_logprobs": [None, None], "text_offset": [0, 2]}}]},
            "no usable logprob",
            id="null-continuation-logprob",
        ),
        pytest.param(
            {"choices": [{"logprobs": {"tokens": ["Q:"], "token_logprobs": [None], "text_offset": [0]}}]},
            "end before the choice begins",
            id="no-continuation-tokens",
        ),
    ],
)
def test_loglikelihood_patch_unusable_response_raises(monkeypatch, body, match):
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    monkeypatch.setattr(
        _urllib_request,
        "urlopen",
        lambda req, timeout=60: _FakeHTTPResponse(json.dumps(body).encode()),
    )
    doc = _make_doc(query="Q: ", choices=["yes", "no"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match=match):
        mod._loglikelihood_via_completions(_fake_client(), [doc])


def test_loglikelihood_patch_doc_without_choices_raises(monkeypatch):
    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    doc = _make_doc(query="Q: ", choices=[], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="no choices"):
        mod._loglikelihood_via_completions(_fake_client(), [doc])


# ── Real lighteval 0.13.0 client layout and metric consumption ───────────────


def _multi_token_json(context: str, pieces: list[str], logprobs: list[float]) -> bytes:
    """An echo response: one context token, then one token per continuation piece.

    The context token ends before the context's trailing whitespace, which is
    carried by the first continuation piece.
    """
    head = context.rstrip()
    pieces = [context[len(head):] + pieces[0], *pieces[1:]]
    offsets = [0]
    pos = len(head)
    for piece in pieces:
        offsets.append(pos)
        pos += len(piece)
    return json.dumps(
        {
            "choices": [
                {
                    "logprobs": {
                        "tokens": [head, *pieces],
                        "token_logprobs": [None, *logprobs],
                        "text_offset": offsets,
                    }
                }
            ]
        }
    ).encode()


def test_loglikelihood_patch_uses_model_from_the_real_litellm_client(monkeypatch, tmp_path):
    """The model sent to /v1/completions comes from LiteLLMClient.model.

    lighteval 0.13.0 keeps the configured model in ``self.model`` and has no
    ``model_name`` attribute on the client, so reading ``model_name`` sent
    ``"model": ""``.  The stubs used elsewhere in this file hid that.
    """
    import urllib.request as _urllib_request

    pytest.importorskip("lighteval")
    from lighteval.models.endpoints.litellm_model import (
        LiteLLMClient,
        LiteLLMModelConfig,
    )

    client = LiteLLMClient(
        LiteLLMModelConfig(
            model_name="openai/served-model",
            base_url="http://fake-vllm:8000/v1",
            api_key="secret",
            cache_dir=str(tmp_path / "lighteval-cache"),
        )
    )
    assert not hasattr(client, "model_name"), "layout assumption of lighteval 0.13.0"

    mod = _load_patch_module()
    seen = []
    context = "Q: "

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode())
        seen.append((req.full_url, body["model"], req.get_header("Authorization")))
        return _FakeHTTPResponse(_completions_json(context, body["prompt"][len(context):], -0.3))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=["a", "b"], gold_index=0)

    responses = mod._loglikelihood_via_completions(client, [doc])  # real ModelResponse

    assert len(responses) == 1
    assert seen == [("http://fake-vllm:8000/v1/completions", "served-model", "Bearer secret")] * 2


def test_loglikelihood_patch_missing_model_name_raises(monkeypatch):
    """No resolvable model: fail instead of sending an empty "model"."""
    from types import SimpleNamespace as NS

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    doc = _make_doc(query="Q: ", choices=["yes"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="model name"):
        mod._loglikelihood_via_completions(
            NS(base_url="http://fake-vllm:8000/v1", api_key="k"), [doc]
        )


def test_loglikelihood_patch_output_tokens_feed_token_normalisation(monkeypatch):
    """LogProbTokenNorm divides by len(output_tokens[ix]); it must not be empty.

    Empty token lists made lighteval raise ZeroDivisionError.  With real counts
    the normalised argmax differs from the raw one here: "short" wins on the
    raw sum, "long" wins per token, and the gold is "long".
    """
    import urllib.request as _urllib_request

    pytest.importorskip("lighteval")
    from lighteval.metrics.metrics_sample import LoglikelihoodAcc
    from lighteval.metrics.normalizations import LogProbTokenNorm
    from lighteval.tasks.requests import Doc

    mod = _load_patch_module()
    context = "Q: "
    pieces = {"short": (["sh", "ort"], [-1.0, -1.0]), "long": (["lo", "n", "g"], [-0.9, -0.9, -0.9])}

    def fake_urlopen(req, timeout=60):
        choice = json.loads(req.data.decode())["prompt"][len(context):]
        toks, lps = pieces[choice]
        return _FakeHTTPResponse(_multi_token_json(context, toks, lps))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = Doc(query=context, choices=["short", "long"], gold_index=1)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])

    assert [len(t) for t in resp.output_tokens] == [2, 3]
    assert resp.logprobs == pytest.approx([-2.0, -2.7])
    # Raw: -2.0 beats -2.7 -> "short" (wrong).  Per token: -0.9 beats -1.0 -> "long".
    assert LoglikelihoodAcc().compute(doc, resp) == 0
    assert LoglikelihoodAcc(logprob_normalization=LogProbTokenNorm()).compute(doc, resp) == 1


# ── provider.yaml primary metrics must equal the keys the adapter emits ──────
#
# EvalHub resolves ``primary_score.metric`` with an exact lookup against the
# reported ``metric_name``, which the adapter builds as ``<task>.<metric>`` after
# stripping ``|N`` and simplifying parameterised names (pass@k:k=1 -> pass@1).
# The raw metric names below come from lighteval's own task registry, so a
# lighteval upgrade that renames a metric fails here instead of silently
# breaking the primary score.

# Benchmarks whose primary metric cannot be asserted from the registry alone.
# math:* emit a name that embeds a function repr, truthfulqa:mc emits one
# metric object that fans out to several keys, and the group ids expand to
# many tasks. They still declare bare metric names and need a live run to fix.
_PRIMARY_METRIC_KNOWN_GAPS = {
    "math",
    "math:algebra",
    "math:counting_and_probability",
    "truthfulqa:mc",
    "truthfulness",
    "commonsense_reasoning",
    "scientific_reasoning",
    "physical_commonsense",
    "knowledge",
    "mmlu",
    "language_understanding",
}


def _lighteval_task_metric_names() -> dict[str, list[str]]:
    """Map task name -> raw metric names for every task in lighteval's registry."""
    import importlib

    import lighteval.tasks.tasks as tasks_pkg
    from lighteval.tasks.lighteval_task import LightevalTaskConfig

    found: dict[str, list[str]] = {}

    def _collect(module) -> None:
        for attr in dir(module):
            obj = getattr(module, attr)
            configs = []
            if isinstance(obj, LightevalTaskConfig):
                configs = [obj]
            elif isinstance(obj, (list, tuple)) and obj:
                configs = [c for c in obj if isinstance(c, LightevalTaskConfig)]
            for cfg in configs:
                found[cfg.name] = [
                    getattr(m, "metric_name", None) or getattr(m, "name", "")
                    for m in cfg.metrics
                ]

    # The adapter re-implements some benchmarks (custom_tasks.py); their metrics,
    # not the script-based built-ins', are what a job emits.  Keyed by provider id.
    from custom_tasks import TASKS_TABLE
    from main import CUSTOM_TASK_RESULT_NAMES

    # Walk the filesystem: pkgutil skips sub-directories without __init__.py
    # (lighteval keeps lcb/ that way).
    root = Path(tasks_pkg.__path__[0])
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root).with_suffix("")
        name = ".".join((tasks_pkg.__name__, *rel.parts))
        try:
            _collect(importlib.import_module(name))
        except Exception:  # noqa: BLE001 - optional task deps may be absent
            continue
    for cfg in TASKS_TABLE:
        found[CUSTOM_TASK_RESULT_NAMES.get(cfg.name, cfg.name)] = [
            getattr(m, "metric_name", None) or getattr(m, "name", "") for m in cfg.metrics
        ]
    return found


def _provider_benchmarks() -> list[dict]:
    import yaml

    path = Path(__file__).parent.parent / "provider.yaml"
    return yaml.safe_load(path.read_text())["benchmarks"]


@pytest.mark.parametrize(
    ("task", "primary"),
    [
        ("gpqa:diamond", "gpqa:diamond.gpqa_pass@1"),
        ("math_500", "math_500.pass@1"),
        ("aime24", "aime24.pass@1"),
        ("aime25", "aime25.pass@1"),
        ("lcb:codegeneration_v6", "lcb:codegeneration_v6.codegen_pass@1"),
    ],
)
def test_primary_metric_matches_emitted_key_for_reasoning_benchmarks(adapter, task, primary):
    """Result-fixture check for the benchmarks whose lighteval names are parameterised."""
    pytest.importorskip("lighteval")
    raw_names = _lighteval_task_metric_names()[task]

    results = {
        "results": {
            f"{task}|0": {
                **{name: 0.5 for name in raw_names},
                **{f"{name}_stderr": 0.01 for name in raw_names},
            }
        }
    }
    emitted, _ = adapter._extract_evaluation_results(results, task)
    emitted_names = {r.metric_name for r in emitted}

    bench = next(b for b in _provider_benchmarks() if b["id"] == task)
    assert primary in emitted_names
    assert bench["primary_score"]["metric"] == primary
    assert set(bench["metrics"]) == emitted_names


def test_every_resolvable_primary_metric_is_an_emitted_key(adapter):
    """provider.yaml declares only metric names the adapter will actually report."""
    pytest.importorskip("lighteval")
    registry = _lighteval_task_metric_names()

    checked = []
    for bench in _provider_benchmarks():
        task = bench["id"]
        if task in _PRIMARY_METRIC_KNOWN_GAPS or task not in registry:
            continue
        raw_names = registry[task]
        # Jobs report custom tasks under their lighteval name (evalhub_piqa|0).
        raw_task = LightEvalAdapter.CUSTOM_TASK_ALIASES.get(task, task)
        results = {"results": {f"{raw_task}|0": {name: 0.5 for name in raw_names}}}
        emitted, _ = adapter._extract_evaluation_results(results, task)
        emitted_names = {r.metric_name for r in emitted}

        primary = bench["primary_score"]["metric"]
        assert primary in emitted_names, (
            f"{task}: primary_score.metric {primary!r} is not an emitted key; "
            f"adapter emits {sorted(emitted_names)}"
        )
        undeclared = [m for m in bench["metrics"] if m not in emitted_names]
        assert not undeclared, f"{task}: metrics {undeclared} are never emitted"
        checked.append(task)

    # Guard against the registry walk silently finding nothing.
    assert len(checked) >= 15, checked


@pytest.mark.parametrize(
    "task",
    ["evalhub_piqa", "evalhub_truthfulqa_mc"],
)
@pytest.mark.integration
def test_run_lighteval_cmd_loads_custom_tasks(adapter, tmp_path, monkeypatch, task):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_idx = cmd.index("--output-dir") + 1
        results_dir = Path(cmd[output_idx])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_test.json").write_text(json.dumps(CANNED_RESULTS))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: SimpleNamespace(api_key="test-key"),
    )
    monkeypatch.setenv("HF_TOKEN", "fake")

    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=[task],
        output_dir=tmp_path / "output",
        num_fewshot=0,
        limit=5,
        batch_size=1,
        benchmark_config={},
    )

    cmd = captured["cmd"]
    custom_tasks_idx = cmd.index("--custom-tasks")
    assert f"{task}|0" in cmd
    assert Path(cmd[custom_tasks_idx + 1]).name == "custom_tasks.py"
    assert Path(cmd[custom_tasks_idx + 1]).is_file()


def _echo_json(tokens: list[str], logprobs: list[float | None]) -> bytes:
    """An echo response for the given tokens; offsets are their cumulative lengths."""
    offsets, pos = [], 0
    for token in tokens:
        offsets.append(pos)
        pos += len(token)
    return json.dumps(
        {
            "choices": [
                {
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": logprobs,
                        "text_offset": offsets,
                    }
                }
            ]
        }
    ).encode()


def test_loglikelihood_patch_first_answer_token_is_scored_when_context_ends_in_space(
    monkeypatch,
):
    """The boundary sits before the context's trailing whitespace.

    ``"Answer: "`` + ``"Paris is"`` tokenises as ``... "Answer", ":", " Paris",
    " is"``: the first answer token starts one character *before* ``len(context)``.
    Filtering on ``offset >= len(context)`` dropped it, so only `` is`` was
    summed and the winner flipped from Rome to Paris.  lighteval's own
    ``tok_encode_pair`` moves trailing context whitespace to the continuation.
    """
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    context = "Question: capital? Answer: "
    head = ["Question: capital? Answer", ":"]
    answers = {
        "Paris is": ([" Paris", " is"], [-5.0, -0.1]),
        "Rome is": ([" Rome", " is"], [-1.0, -2.0]),
    }

    def fake_urlopen(req, timeout=60):
        choice = json.loads(req.data.decode())["prompt"][len(context):]
        pieces, lps = answers[choice]
        return _FakeHTTPResponse(_echo_json([*head, *pieces], [None, -0.5, *lps]))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=list(answers), gold_index=1)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])

    # Correct: both tokens of each answer are scored; Rome wins (-3.0 > -5.1).
    assert resp.logprobs == pytest.approx([-5.1, -3.0])
    assert resp.argmax_logits_eq_gold == [True]
    assert [len(t) for t in resp.output_tokens] == [2, 2]


def test_loglikelihood_patch_counts_a_token_that_straddles_the_boundary(monkeypatch):
    """A token that starts inside the context and ends in the answer is scored."""
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    context = "Answer:"

    def fake_urlopen(req, timeout=60):
        # "Answer" then ":yes" merged into one token across the boundary.
        return _FakeHTTPResponse(_echo_json(["Answer", ":yes"], [None, -0.7]))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=["yes"], gold_index=0)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])
    assert resp.logprobs == pytest.approx([-0.7])


def test_loglikelihood_patch_scores_the_full_reference_prompt(monkeypatch):
    """System prompt, instruction and few-shot examples are part of the context.

    Scoring ``doc.query`` alone changes the result of every ``num_fewshot > 0``
    run and ignores a configured ``system_prompt``.
    """
    import urllib.request as _urllib_request
    from types import SimpleNamespace

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    sent = []

    def fake_urlopen(req, timeout=60):
        prompt = json.loads(req.data.decode())["prompt"]
        sent.append(prompt)
        return _FakeHTTPResponse(_echo_json([prompt[: -len(" yes")], " yes"], [None, -0.2]))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    shot = _make_doc(query="Q1: sky? A:", choices=[" blue", " red"], gold_index=0)
    doc = _make_doc(
        query="Pick one.\nQ2: grass? A:",
        choices=[" yes"],
        gold_index=0,
        instruction="Pick one.\n",
        fewshot_samples=[shot],
    )
    client = SimpleNamespace(
        base_url="http://fake-vllm:8000/v1",
        model="openai/test-model",
        api_key="dummy",
        config=SimpleNamespace(system_prompt="Be terse."),
    )

    mod._loglikelihood_via_completions(client, [doc])

    # lighteval's PromptManager(use_chat_template=False) layout, then the choice:
    # system prompt, instruction (which itself ends in "\n"), few-shot example with
    # its gold answer, then the query with the instruction prefix removed.
    assert sent == ["Be terse.\n\nPick one.\n\n\nQ1: sky? A: blue\n\nQ2: grass? A: yes"]


def _model_args_for(adapter, tmp_path, monkeypatch, tasks, benchmark_config=None):
    """The ``model_args`` string the adapter hands to the lighteval CLI."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        results_dir = Path(cmd[cmd.index("--output-dir") + 1])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "results_test.json").write_text(json.dumps(CANNED_RESULTS))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "main.resolve_model_credentials", lambda: SimpleNamespace(api_key="k")
    )
    monkeypatch.setenv("HF_TOKEN", "fake")
    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=tasks,
        output_dir=tmp_path / "output",
        num_fewshot=0,
        limit=2,
        batch_size=1,
        benchmark_config=benchmark_config or {},
    )
    return next(arg for arg in captured["cmd"] if arg.startswith("model_name="))


@pytest.mark.integration
@pytest.mark.parametrize(
    "tasks", [["math:algebra"], ["math:counting_and_probability"], ["gsm8k", "math:algebra"]]
)
def test_majority_vote_tasks_get_a_sampling_temperature(adapter, tmp_path, monkeypatch, tasks):
    """maj@4 needs 4 samples per question; lighteval's LiteLLM backend raises at
    temperature=0 ("num_samples > 1 is not supported"), so a default is supplied."""
    model_args = _model_args_for(adapter, tmp_path, monkeypatch, tasks)
    assert "generation_parameters={temperature:0.7}" in model_args


@pytest.mark.integration
@pytest.mark.parametrize("temperature", [0.2, 0])
def test_caller_temperature_is_never_overridden(adapter, tmp_path, monkeypatch, temperature):
    """An explicit temperature wins, including 0 (lighteval then reports its own error)."""
    model_args = _model_args_for(
        adapter, tmp_path, monkeypatch, ["math:algebra"],
        {"generation_parameters": {"temperature": temperature, "max_new_tokens": 64}},
    )
    assert f"temperature:{json.dumps(temperature)}" in model_args
    assert "temperature:0.7" not in model_args


@pytest.mark.integration
def test_other_generation_parameters_are_kept_when_temperature_is_defaulted(
    adapter, tmp_path, monkeypatch
):
    model_args = _model_args_for(
        adapter, tmp_path, monkeypatch, ["math:algebra"],
        {"generation_parameters": {"max_new_tokens": 64}},
    )
    assert "generation_parameters={max_new_tokens:64,temperature:0.7}" in model_args


@pytest.mark.integration
@pytest.mark.parametrize("tasks", [["gsm8k"], ["math_500"], ["arc:easy"], ["evalhub_piqa"]])
def test_single_sample_tasks_keep_lighteval_defaults(adapter, tmp_path, monkeypatch, tasks):
    model_args = _model_args_for(adapter, tmp_path, monkeypatch, tasks)
    assert "generation_parameters" not in model_args


@pytest.mark.integration
def test_preformatted_generation_parameters_are_not_edited(adapter, tmp_path, monkeypatch):
    model_args = _model_args_for(
        adapter, tmp_path, monkeypatch, ["math:algebra"],
        {"generation_parameters": "{max_new_tokens:8}"},
    )
    assert "generation_parameters={max_new_tokens:8}" in model_args
    assert "temperature" not in model_args


def test_litellm_response_cache_is_disabled_so_cache_hits_cannot_crash(monkeypatch, tmp_path):
    """A litellm cache hit must not be able to turn into an empty answer.

    lighteval installs a disk response cache and sends ``caching=True``.  On
    litellm 1.103.9 a hit imports ``litellm.proxy`` modules that need ``orjson``
    and ``soundfile`` (not installed by ``litellm[caching]``), so the repeated
    request raised ``ModuleNotFoundError``; lighteval retried and finally returned
    an empty answer.  The wrapper switches the cache off, so every request goes
    to the model and no proxy-only package is needed.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    pytest.importorskip("lighteval")
    import litellm
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from lighteval.models.endpoints.litellm_model import LiteLLMClient

    served = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            served.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "tiny",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "A"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    # Both globals are restored after the test: _apply_patch replaces them.
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path)))
    monkeypatch.setattr(LiteLLMClient, "loglikelihood", LiteLLMClient.loglikelihood)

    try:
        _load_patch_module()._apply_patch()
        assert litellm.cache is None

        request = {
            "model": "openai/tiny",
            "messages": [{"role": "user", "content": "same prompt twice"}],
            "max_tokens": 5,
            "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "api_key": "dummy",
            "caching": True,
            "timeout": 10,
        }
        for _ in range(2):
            assert litellm.completion(**request).choices[0].message.content == "A"
        assert len(served) == 2, "the repeated request must reach the model, not a cache"
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "leading",
    [[], ["<s>"], ["<s>", "<|begin_of_text|>"]],
    ids=["no-special-token", "bos", "two-special-tokens"],
)
def test_loglikelihood_patch_ignores_special_token_text_ahead_of_the_prompt(monkeypatch, leading):
    """vLLM's ``text_offset`` is the running length of the *decoded* tokens, so a BOS
    token such as ``"<s>"`` shifts every offset by its length.  The shift must not make
    the last context tokens count as part of the answer (reproduced with a BOS-adding
    tokenizer in test_logprob_reference.py)."""
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    context = "Question: capital? Answer: "
    head = [*leading, "Question: capital? Answer", ":"]
    answers = {
        "Paris is": ([" Paris", " is"], [-5.0, -0.1]),
        "Rome is": ([" Rome", " is"], [-1.0, -2.0]),
    }

    def fake_urlopen(req, timeout=60):
        choice = json.loads(req.data.decode())["prompt"][len(context):]
        pieces, lps = answers[choice]
        # A real endpoint gives no logprob for the first token (BOS or not).
        logprobs = [None] * len(leading) + [None, -0.5] if leading else [None, -0.5]
        return _FakeHTTPResponse(_echo_json([*head, *pieces], [*logprobs, *lps]))

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    doc = _make_doc(query=context, choices=list(answers), gold_index=1)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])

    assert resp.logprobs == pytest.approx([-5.1, -3.0])
    assert [len(t) for t in resp.output_tokens] == [2, 2]


def test_loglikelihood_patch_rejects_an_echo_it_cannot_align_with_the_prompt(monkeypatch):
    """Extra echoed text that is not made of whole leading tokens must fail the job.

    Guessing a shift would give a number that looks like a likelihood."""
    import urllib.request as _urllib_request

    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    monkeypatch.setattr(
        _urllib_request,
        "urlopen",
        # Prompt "Q: yes" is 6 characters; the echo is 7, and no leading token is 1 long.
        lambda req, timeout=60: _FakeHTTPResponse(_echo_json(["Q:", " yesX"], [None, -0.3])),
    )
    doc = _make_doc(query="Q: ", choices=["yes"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="not whole leading tokens"):
        mod._loglikelihood_via_completions(_fake_client(), [doc])


def test_memory_cap_is_copied_into_the_image():
    """The Containerfile copies the cap module next to the patch launcher; without it the
    launcher would run lighteval with its grader uncapped."""
    containerfile = (Path(__file__).parent.parent / "Containerfile").read_text()
    assert "COPY lighteval_memory_cap.py" in containerfile
    assert "COPY lighteval_logprob_patch.py" in containerfile


# ── Pairwise tokenization: question and answer tokenized separately ──────────────────


def _toy_vllm(monkeypatch, *, return_ids=True):
    """A fake vLLM completions endpoint that accepts text or token ids.

    Tokens are words with their leading whitespace, so ``"capit" + "al"`` is two tokens
    separately and one token (``"capital"``) as a single string.  A BOS token (id 0) is
    prepended to text prompts unless ``add_special_tokens`` is false, as vLLM does.
    Returns the list of request bodies it served.
    """
    import re
    import urllib.request as _urllib_request

    vocab: dict[str, int] = {}
    served: list[dict] = []

    def encode(text, add_special_tokens=True):
        ids = [vocab.setdefault(tok, len(vocab) + 1) for tok in re.findall(r"\s*\w+|\s*[^\w\s]", text)]
        return [0, *ids] if add_special_tokens else ids

    def fake_urlopen(req, timeout=60):
        body = json.loads(req.data.decode())
        served.append(body)
        prompt = body["prompt"]
        ids = list(prompt) if isinstance(prompt, list) else encode(prompt, body.get("add_special_tokens", True))
        names = {0: "", **{i: tok for tok, i in vocab.items()}}  # BOS has no text here
        pieces = [names[i] for i in ids]
        offsets = [sum(len(piece) for piece in pieces[:i]) for i in range(len(pieces))]
        choice = {
            "logprobs": {
                "tokens": pieces,
                "token_logprobs": [None] + [-0.1 * (1 + (ids[i] + ids[i - 1]) % 5) for i in range(1, len(ids))],
                "text_offset": offsets,
            }
        }
        if return_ids and body.get("return_token_ids"):
            choice["prompt_token_ids"] = ids
        return _FakeHTTPResponse(json.dumps({"choices": [choice]}).encode())

    monkeypatch.setattr(_urllib_request, "urlopen", fake_urlopen)
    return served, encode


def test_pairwise_scores_the_separately_tokenized_ids(monkeypatch):
    """A token that would merge across the boundary stays two tokens, as in the
    ``transformers`` backend: the scored prompt is the context ids followed by the
    choice's own ids, not the ids of the joint string."""
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "pairwise")
    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    served, encode = _toy_vllm(monkeypatch)
    doc = _make_doc(query="Question: capit", choices=["al"], gold_index=0)

    (resp,) = mod._loglikelihood_via_completions(_fake_client(), [doc])

    context_ids = encode("Question: capit")  # with BOS
    answer_ids = encode("al", add_special_tokens=False)
    joint_ids = encode("Question: capital")
    assert joint_ids != context_ids + answer_ids, "the toy tokenizer must merge across the boundary"
    scoring_request = served[-1]
    assert scoring_request["prompt"] == context_ids + answer_ids
    assert "add_special_tokens" not in scoring_request  # ids are sent as they are
    ids = context_ids + answer_ids
    expected = -0.1 * (1 + (ids[-1] + ids[-2]) % 5)
    assert resp.logprobs == pytest.approx([expected])
    assert resp.output_tokens == [[len(context_ids)]]


def test_pairwise_moves_trailing_whitespace_of_the_context_to_the_choice(monkeypatch):
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "pairwise")
    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    served, encode = _toy_vllm(monkeypatch)
    doc = _make_doc(query="Answer: ", choices=["Paris", "Rome"], gold_index=0)

    mod._loglikelihood_via_completions(_fake_client(), [doc])

    texts = [(b["prompt"], b.get("add_special_tokens")) for b in served if isinstance(b["prompt"], str)]
    # Context without its trailing space (special tokens on), then each choice with that
    # space in front (special tokens off): the same split as tok_encode_pair.
    assert texts == [("Answer:", None), (" Paris", False), (" Rome", False)]
    assert served[-2]["prompt"] == encode("Answer:") + encode(" Paris", add_special_tokens=False)


def test_pairwise_needs_token_ids_from_the_endpoint_and_says_how_to_opt_out(monkeypatch):
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "pairwise")
    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    _toy_vllm(monkeypatch, return_ids=False)
    doc = _make_doc(query="Q: ", choices=["yes"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="prompt_token_ids") as exc:
        mod._loglikelihood_via_completions(_fake_client(), [doc])
    assert "EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION=joint" in str(exc.value)


def test_joint_opt_out_sends_the_text_as_one_prompt(monkeypatch):
    mod = _load_patch_module()  # the autouse fixture selects joint
    _stub_model_response(monkeypatch)
    served, _ = _toy_vllm(monkeypatch, return_ids=False)
    doc = _make_doc(query="Question: capit", choices=["al"], gold_index=0)

    mod._loglikelihood_via_completions(_fake_client(), [doc])

    assert [b["prompt"] for b in served] == ["Question: capital"]


def test_unknown_tokenization_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("EVALHUB_LIGHTEVAL_LOGLIKELIHOOD_TOKENIZATION", "sideways")
    mod = _load_patch_module()
    _stub_model_response(monkeypatch)
    doc = _make_doc(query="Q: ", choices=["yes"], gold_index=0)

    with pytest.raises(mod.LoglikelihoodScoringError, match="sideways"):
        mod._loglikelihood_via_completions(_fake_client(), [doc])
