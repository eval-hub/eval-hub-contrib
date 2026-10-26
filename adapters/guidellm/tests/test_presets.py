"""Tests for the GuideLLM Quick/Full performance benchmark presets."""

import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from evalhub.adapter import JobCallbacks, OCIArtifactResult
from main import (
    FULL_BENCHMARK_ID,
    FULL_PROFILES,
    QUICK_BENCHMARK_ID,
    GuideLLMAdapter,
)


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _spec(adapter: GuideLLMAdapter, benchmark_id: str, parameters=None, num_examples=None):
    spec = adapter.job_spec.model_copy(deep=True)
    spec.benchmark_id = benchmark_id
    spec.parameters = parameters or {}
    if num_examples is not None:
        spec.num_examples = num_examples
    return spec


@pytest.fixture
def adapter(tmp_path: Path) -> GuideLLMAdapter:
    adapter = GuideLLMAdapter(job_spec_path="meta/job.json")
    adapter.results_dir = tmp_path
    return adapter


def test_quick_preset_resolves_to_bounded_sweep(adapter: GuideLLMAdapter, tmp_path):
    spec = _spec(adapter, QUICK_BENCHMARK_ID)

    assert adapter._resolve_run_profiles(spec) == ["sweep"]

    parameters = adapter._parameters_for_profile(spec, "sweep")
    cmd = adapter._build_guidellm_command(
        spec, parameters=parameters, output_dir=tmp_path / "sweep"
    )

    assert _flag_value(cmd, "--profile") == "sweep"
    assert _flag_value(cmd, "--max-seconds") == "30"
    assert _flag_value(cmd, "--max-requests") == "100"


def test_quick_preset_keeps_explicit_parameters(adapter: GuideLLMAdapter, tmp_path):
    spec = _spec(
        adapter,
        QUICK_BENCHMARK_ID,
        parameters={
            "profile": "constant",
            "rate": 2,
            "max_seconds": 5,
            "max_requests": 7,
        },
    )

    assert adapter._resolve_run_profiles(spec) == ["constant"]

    parameters = adapter._parameters_for_profile(spec, "constant")
    cmd = adapter._build_guidellm_command(
        spec, parameters=parameters, output_dir=tmp_path
    )

    assert _flag_value(cmd, "--profile") == "constant"
    assert _flag_value(cmd, "--rate") == "2"
    assert _flag_value(cmd, "--max-seconds") == "5"
    assert _flag_value(cmd, "--max-requests") == "7"


def test_quick_preset_honours_num_examples_over_max_requests(
    adapter: GuideLLMAdapter, tmp_path
):
    spec = _spec(adapter, QUICK_BENCHMARK_ID, num_examples=7)

    parameters = adapter._parameters_for_profile(spec, "sweep")
    cmd = adapter._build_guidellm_command(
        spec, parameters=parameters, output_dir=tmp_path
    )

    assert _flag_value(cmd, "--max-requests") == "7"


def test_full_preset_runs_all_profiles_with_rate_defaults(
    adapter: GuideLLMAdapter, tmp_path
):
    spec = _spec(adapter, FULL_BENCHMARK_ID)

    profiles = adapter._resolve_run_profiles(spec)
    assert profiles == list(FULL_PROFILES)

    commands = {}
    for profile in profiles:
        parameters = adapter._parameters_for_profile(spec, profile)
        commands[profile] = adapter._build_guidellm_command(
            spec, parameters=parameters, output_dir=tmp_path / profile
        )

    assert _flag_value(commands["sweep"], "--profile") == "sweep"
    assert "--rate" not in commands["sweep"]
    assert "--rate" not in commands["throughput"]
    assert _flag_value(commands["concurrent"], "--rate") == "10"
    assert _flag_value(commands["constant"], "--rate") == "5"
    assert _flag_value(commands["poisson"], "--rate") == "5"
    for cmd in commands.values():
        assert _flag_value(cmd, "--max-seconds") == "60"
        assert _flag_value(cmd, "--max-requests") == "300"


def test_full_preset_can_be_limited_to_selected_profiles(adapter: GuideLLMAdapter):
    spec = _spec(
        adapter,
        FULL_BENCHMARK_ID,
        parameters={"profiles": ["sweep", "poisson"]},
    )
    assert adapter._resolve_run_profiles(spec) == ["sweep", "poisson"]

    single = _spec(
        adapter,
        FULL_BENCHMARK_ID,
        parameters={"profile": "concurrent"},
    )
    assert adapter._resolve_run_profiles(single) == ["concurrent"]


def test_unknown_benchmark_keeps_single_profile_behaviour(adapter: GuideLLMAdapter):
    spec = _spec(adapter, "local_quick")

    assert adapter._resolve_run_profiles(spec) == ["sweep"]
    parameters = adapter._parameters_for_profile(spec, "sweep")
    assert "max_seconds" not in parameters
    assert "max_requests" not in parameters


def test_invalid_profile_is_rejected(adapter: GuideLLMAdapter):
    spec = _spec(adapter, FULL_BENCHMARK_ID, parameters={"profiles": ["nope"]})

    with pytest.raises(ValueError, match="Unsupported GuideLLM profile"):
        adapter._resolve_run_profiles(spec)


def test_full_preset_runs_every_profile_and_aggregates_results(
    adapter: GuideLLMAdapter, monkeypatch
):
    spec = _spec(adapter, FULL_BENCHMARK_ID)

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake", reference="fake:latest"
    )

    commands = []

    def fake_run_guidellm(cmd):
        profile = _flag_value(cmd, "--profile")
        rps = 10.0 + FULL_PROFILES.index(profile)
        commands.append(cmd)
        output_dir = Path(_flag_value(cmd, "--output-path"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmarks.json").write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "metrics": {
                                "requests_per_second": {
                                    "successful": {"mean": rps}
                                },
                                "prompt_tokens_per_second": {
                                    "successful": {"mean": 1000.0}
                                },
                                "output_tokens_per_second": {
                                    "successful": {"mean": 500.0}
                                },
                                "time_to_first_token_ms": {
                                    "successful": {"mean": 40.0}
                                },
                                "inter_token_latency_ms": {
                                    "successful": {"mean": 8.0}
                                },
                                "request_totals": {"successful": 20},
                            }
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(adapter, "_run_guidellm", fake_run_guidellm)

    results = adapter.run_benchmark_job(spec, callbacks)

    assert [_flag_value(cmd, "--profile") for cmd in commands] == list(FULL_PROFILES)
    assert results.num_examples_evaluated == 20 * len(FULL_PROFILES)

    metrics = {result.metric_name: result for result in results.results}
    for profile in FULL_PROFILES:
        assert f"{profile}_requests_per_second" in metrics
        assert f"{profile}_mean_ttft_ms" in metrics
        assert metrics[f"{profile}_requests_per_second"].metadata["profile"] == profile

    # Plain metric names carry the cross-profile average so the server's
    # primary-score and pass-criteria definitions keep working.
    assert metrics["requests_per_second"].metadata["aggregate"] is True
    assert metrics["requests_per_second"].metadata["profiles"] == list(FULL_PROFILES)
    assert metrics["requests_per_second"].metric_value == pytest.approx(12.0)
    assert results.overall_score == pytest.approx(12.0)

    assert results.evaluation_metadata["profiles"] == list(FULL_PROFILES)
    assert set(results.evaluation_metadata["profile_metrics"]) == set(FULL_PROFILES)

    # Per-profile output files are persisted as artifacts and to MLflow.
    assert "sweep/benchmarks.json" in {a.path for a in adapter.mlflow_artifacts}
