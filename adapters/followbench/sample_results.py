"""Persist native per-sample observations for post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evalhub.adapter import JobCallbacks, JobResults, JobSpec, OCIArtifactSpec
from evalhub.adapter.mlflow import MlflowArtifact


def sample_results_enabled(config: JobSpec) -> bool:
    """Validate the opt-out flag before starting inference."""
    enabled = config.parameters.get("save_sample_results", True)
    if type(enabled) is not bool:
        raise ValueError("parameters.save_sample_results must be a boolean")
    return enabled


def persist_sample_results(
    config: JobSpec,
    callbacks: JobCallbacks,
    result: JobResults,
    records: list[dict] | None,
    metric: str,
) -> bytes:
    """Export enabled samples and preserve MLflow metrics when samples are disabled."""
    identities = [str(row["sample_id"]) for row in records or []]
    if len(set(identities)) != len(identities):
        raise ValueError("Sample results require unique sample IDs")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in records or []
    ).encode("utf-8")
    manifest = json.dumps(
        {"primary_score": {"metric": metric}, "estimand": "mean"}
    ).encode("utf-8")
    oci = config.exports.oci if config.exports else None
    mlflow = getattr(callbacks, "mlflow", None)
    # No destination is a valid evaluation-only job; configured upload failures propagate.
    if records is not None and oci is not None:
        with TemporaryDirectory(prefix="evalhub-samples-") as temporary:
            directory = Path(temporary)
            (directory / "sample_results.jsonl").write_bytes(payload)
            (directory / "manifest.json").write_bytes(manifest)
            result.oci_artifact = callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=directory,
                    coordinates=oci.coordinates.model_copy(deep=True),
                )
            )
    if config.experiment_name:
        if mlflow is None:
            raise RuntimeError("MLflow persistence is configured but unavailable")
        artifacts = []
        if records is not None:
            artifacts = [
                MlflowArtifact("sample_results.jsonl", payload, "application/x-ndjson"),
                MlflowArtifact("manifest.json", manifest, "application/json"),
            ]
        run_id = mlflow.save(result, config, artifacts=artifacts)
        if not run_id:
            raise RuntimeError("MLflow did not confirm result persistence")
        result.mlflow_run_id = run_id
    return payload
