"""Turn evaluation output into the artifacts eval-hub stores."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    CapabilityEvalEntry,
    EvalCardMetadata,
    EvaluationResult,
    JobCallbacks,
    JobResults,
    JobSpec,
    OCIArtifactSpec,
    SafetyEvalEntry,
)
from evalhub.adapter.mlflow import MlflowArtifact

logger = logging.getLogger(__name__)


def deepeval_version() -> str:
    try:
        return importlib.metadata.version("deepeval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_eval_card(
    card_entries: Sequence[CapabilityEvalEntry | SafetyEvalEntry],
    params: Mapping[str, Any],
    footnotes: str,
) -> EvalCardMetadata:
    """Build the EvalCard, splitting capability and safety entries."""
    return EvalCardMetadata(
        modalities_input=["text"],
        modalities_output=["text"],
        languages_count=params.get("languages_count", 1),
        languages=params.get("languages", ["en"]),
        capability_evaluations=[
            e for e in card_entries if isinstance(e, CapabilityEvalEntry)
        ],
        safety_evaluations=[e for e in card_entries if isinstance(e, SafetyEvalEntry)],
        developer_footnotes=footnotes,
    )


def build_summary(
    benchmark_id: str,
    overall_score: float | None,
    num_evaluated: int,
    results: Sequence[EvaluationResult],
) -> bytes:
    """Serialise the results summary stored as an OCI / MLflow artifact."""
    return json.dumps(
        {
            "benchmark_id": benchmark_id,
            "overall_score": overall_score,
            "num_examples_evaluated": num_evaluated,
            "results": [
                {"metric_name": r.metric_name, "metric_value": r.metric_value}
                for r in results
            ],
        },
        indent=2,
        default=str,
    ).encode()


def persist_oci_artifact(
    oci_exports: Any,
    config: JobSpec,
    callbacks: JobCallbacks,
    summary: bytes,
    results_dir: Path,
    version: str,
) -> Any:
    """Write the summary to ``results_dir`` and publish it as an OCI artifact."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results_summary.json").write_bytes(summary)

    coords = oci_exports.coordinates.model_copy(deep=True)
    coords.annotations.update(
        {
            "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
            "io.github.eval-hub.benchmark": config.benchmark_id,
            "io.github.eval-hub.model": config.model.name,
            "io.github.eval-hub.job_id": config.id,
            "io.github.eval-hub.framework": "deepeval",
            "io.github.eval-hub.deepeval.version": version,
        }
    )
    artifact = callbacks.create_oci_artifact(
        OCIArtifactSpec(files_path=results_dir, coordinates=coords)
    )
    logger.info("OCI artifact created: %s", artifact.reference)
    return artifact


def mlflow_experiment_name(config: JobSpec) -> str:
    """Experiment from the JobSpec, else ``parameters.mlflow_experiment_name``."""
    name = (config.experiment_name or "").strip()
    if name:
        return name
    raw = config.parameters.get("mlflow_experiment_name")
    return raw.strip() if isinstance(raw, str) else ""


def save_to_mlflow(
    callbacks: JobCallbacks, config: JobSpec, job_results: JobResults, summary: bytes
) -> None:
    """Save the run to MLflow when an experiment is configured.

    A failure is logged, not raised: the evaluation itself already succeeded.
    """
    experiment_name = mlflow_experiment_name(config)
    if not experiment_name:
        return
    try:
        spec = config.model_copy(update={"experiment_name": experiment_name})
        run_id = callbacks.mlflow.save(
            job_results,
            spec,
            artifacts=[
                MlflowArtifact("results_summary.json", summary, "application/json")
            ],
        )
        if run_id:
            job_results.mlflow_run_id = run_id
            logger.info("MLflow run saved: %s (experiment=%s)", run_id, experiment_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow save failed (job still completes): %s", exc)
