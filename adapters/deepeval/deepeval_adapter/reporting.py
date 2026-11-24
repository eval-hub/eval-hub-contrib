"""Turn evaluation output into the artifacts eval-hub stores."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import tempfile
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

from .samples import SAMPLE_FILES

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


SUMMARY_FILE = "results_summary.json"


def write_local_results(
    results_dir: Path, summary: bytes, artifacts: Sequence[MlflowArtifact]
) -> None:
    """Keep this run's files in ``results_dir``, dropping stale sample files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        (results_dir / name).unlink(missing_ok=True)
    (results_dir / SUMMARY_FILE).write_bytes(summary)
    for artifact in artifacts:
        (results_dir / artifact.path).write_bytes(artifact.content)


def persist_oci_artifact(
    oci_exports: Any,
    config: JobSpec,
    callbacks: JobCallbacks,
    files: Mapping[str, bytes],
    version: str,
) -> Any:
    """Publish this run's files as an OCI artifact.

    The files are staged in a temporary directory so unrelated or stale files
    in the results directory are never uploaded.
    """
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
    with tempfile.TemporaryDirectory(prefix="deepeval-oci-") as staging:
        for name, content in files.items():
            (Path(staging) / name).write_bytes(content)
        artifact = callbacks.create_oci_artifact(
            OCIArtifactSpec(files_path=Path(staging), coordinates=coords)
        )
    if artifact is None:
        raise RuntimeError("OCI export returned no artifact reference")
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
    callbacks: JobCallbacks,
    config: JobSpec,
    job_results: JobResults,
    artifacts: Sequence[MlflowArtifact],
    *,
    strict: bool = False,
) -> None:
    """Save the run to MLflow when an experiment is configured.

    A failure is logged, not raised, because the evaluation itself already
    succeeded. With ``strict`` (per-sample export was requested) a failure or a
    missing run ID raises instead, so samples are never silently dropped.
    """
    experiment_name = mlflow_experiment_name(config)
    if not experiment_name:
        return
    try:
        spec = config.model_copy(update={"experiment_name": experiment_name})
        run_id = callbacks.mlflow.save(job_results, spec, artifacts=list(artifacts))
        if strict and not run_id:
            raise RuntimeError("MLflow per-sample export returned no run ID")
        if run_id:
            job_results.mlflow_run_id = run_id
            logger.info("MLflow run saved: %s (experiment=%s)", run_id, experiment_name)
    except Exception as exc:
        if strict:
            raise
        logger.warning("MLflow save failed (job still completes): %s", exc)
