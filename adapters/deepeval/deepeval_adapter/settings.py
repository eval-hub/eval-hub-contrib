"""Job parameter parsing and validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from evalhub.adapter import JobSpec

from .benchmarks import BENCHMARKS, BenchmarkSpec

DEFAULT_THRESHOLD = 0.5
DEFAULT_DATASET_FORMAT = "csv"
# Reasoning models (DeepSeek-R1, Phi-4) emit long chain-of-thought before the
# first token, so DeepEval's own per-attempt timeout is too short for them.
DEFAULT_PER_ATTEMPT_TIMEOUT_SECONDS = 300.0


def validate_job_spec(config: JobSpec) -> BenchmarkSpec:
    """Check the benchmark id and its required parameters; return its spec."""
    if not config.benchmark_id:
        raise ValueError("benchmark_id is required")

    spec = BENCHMARKS.get(config.benchmark_id)
    if spec is None:
        raise ValueError(
            f"Unsupported benchmark_id: {config.benchmark_id!r}. "
            f"Supported: {', '.join(sorted(BENCHMARKS))}"
        )

    for name in spec.required_params:
        if not config.parameters.get(name):
            raise ValueError(
                f"parameters.{name} is required for benchmark_id={spec.id!r}"
            )
    return spec


@dataclass(frozen=True, slots=True)
class JobSettings:
    """The parts of a JobSpec the adapter acts on, validated and defaulted."""

    spec: BenchmarkSpec
    params: Mapping[str, Any]
    judge_name: str
    judge_url: str
    threshold: float
    dataset_format: str

    @classmethod
    def from_job_spec(cls, config: JobSpec) -> JobSettings:
        spec = validate_job_spec(config)
        params = config.parameters
        model_url = config.model.url.strip().rstrip("/")
        return cls(
            spec=spec,
            params=params,
            judge_name=params.get("eval_model_name") or config.model.name,
            judge_url=params.get("eval_model_url") or model_url,
            threshold=float(params.get("threshold", DEFAULT_THRESHOLD)),
            dataset_format=params.get("dataset_format", DEFAULT_DATASET_FORMAT),
        )


def apply_runtime_tuning(params: Mapping[str, Any]) -> None:
    """Export DeepEval's retry and timeout knobs from job parameters.

    DeepEval reads these from the environment, so they must be set before the
    metric runs.
    """
    timeout = params.get(
        "per_attempt_timeout_seconds", DEFAULT_PER_ATTEMPT_TIMEOUT_SECONDS
    )
    if timeout is not None:
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(
            float(timeout)
        )
    if (retry_max := params.get("retry_max_attempts")) is not None:
        os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(int(retry_max))
    if (retry_cap := params.get("retry_cap_seconds")) is not None:
        os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = str(float(retry_cap))
