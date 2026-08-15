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
def test_max_samples_from_parameters_forwarded(adapter, monkeypatch, tmp_path):
    """parameters['max_samples'] is forwarded as --max-samples when num_examples is None.

    Regression test for the bug where config.num_examples was None when callers
    passed max_samples in benchmark parameters instead of the top-level field,
    causing --max-samples to never be emitted.
    """
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        # Return enough structure for the caller to find a results file
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

    # num_examples is NOT set; max_samples comes from parameters
    adapter._run_lighteval(
        model_config=adapter.job_spec.model,
        tasks=["gsm8k"],
        output_dir=tmp_path,
        num_fewshot=0,
        limit=adapter.job_spec.num_examples or adapter.job_spec.parameters.get("max_samples"),
        batch_size=1,
        benchmark_config=adapter.job_spec.parameters,
    )

    assert "--max-samples" in captured["cmd"], (
        "--max-samples must appear in the lighteval CLI when max_samples is in parameters"
    )
    idx = captured["cmd"].index("--max-samples")
    assert captured["cmd"][idx + 1] == "5", (
        f"Expected --max-samples 5, got {captured['cmd'][idx + 1]}"
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


# ── L3 fix: TASK_ALIASES remapping ────────────────────────────────────────────

@pytest.mark.integration
def test_task_alias_applied_to_single_benchmark(adapter):
    """TASK_ALIASES remaps benchmark IDs that differ from the lighteval registry."""
    # math:counting_and_probability → math:counting_and_prob
    tasks = adapter._parse_benchmark_tasks("math:counting_and_probability", {})
    assert tasks == ["math:counting_and_prob"], (
        "math:counting_and_probability must be aliased to math:counting_and_prob"
    )


@pytest.mark.integration
def test_truthfulqa_generation_not_aliased(adapter):
    """truthfulqa:generation has no alias — its registry name is version-dependent."""
    tasks = adapter._parse_benchmark_tasks("truthfulqa:generation", {})
    # Pass-through unchanged; the lighteval CLI will raise if unavailable
    assert tasks == ["truthfulqa:generation"]


@pytest.mark.integration
def test_task_alias_applied_in_category(adapter):
    """TASK_ALIASES is applied when expanding a category to individual tasks."""
    tasks = adapter._parse_benchmark_tasks("math", {})
    assert "math:counting_and_prob" in tasks, (
        "math:counting_and_probability should be aliased to math:counting_and_prob"
    )
    assert "math:counting_and_probability" not in tasks
    # SUPPORTED_TASKS["math"] must include math_500 as declared in provider.yaml
    assert "math_500" in tasks, "math_500 must be in the math category suite"


@pytest.mark.integration
def test_knowledge_category_includes_openbookqa(adapter):
    """knowledge suite must include openbookqa as declared in provider.yaml."""
    tasks = adapter._parse_benchmark_tasks("knowledge", {})
    assert "openbookqa" in tasks, "openbookqa must be in the knowledge category suite"


@pytest.mark.integration
def test_truthfulness_category_excludes_generation(adapter):
    """truthfulness suite must not contain truthfulqa:generation (no stable alias)."""
    tasks = adapter._parse_benchmark_tasks("truthfulness", {})
    assert "truthfulqa:generation" not in tasks, (
        "truthfulqa:generation has no stable alias and must not be in the truthfulness suite"
    )
    assert "truthfulqa:mc" in tasks


@pytest.mark.integration
def test_no_alias_for_unknown_benchmark(adapter):
    """Benchmarks not in TASK_ALIASES pass through unchanged."""
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
def test_provider_anthropic_raises_value_error(adapter):
    """provider: anthropic must be rejected before execution.

    Anthropic's API does not implement /v1/completions with echo+logprobs,
    so loglikelihood benchmarks would silently return -inf scores. Fail fast
    at validation time instead.
    """
    adapter.job_spec.parameters["provider"] = "anthropic"
    with pytest.raises(ValueError, match="anthropic"):
        adapter._validate_config(adapter.job_spec)


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
