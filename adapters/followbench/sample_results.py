"""Persist native per-sample observations for post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evalhub.adapter import JobCallbacks, JobResults, JobSpec, OCIArtifactSpec
from evalhub.adapter.mlflow import MlflowArtifact


def persist_sample_results(
    config: JobSpec,
    callbacks: JobCallbacks,
    result: JobResults,
    records: list[dict],
    metric: str,
    *,
    save_mlflow: bool = True,
) -> bytes:
    """Synchronously persist complete observations before reporting completion."""
    identities = [str(row["sample_id"]) for row in records]
    if not records or len(set(identities)) != len(identities):
        raise ValueError("Sample results require nonempty, unique sample IDs")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records
    ).encode("utf-8")
    oci = config.exports.oci if config.exports else None
    mlflow = getattr(callbacks, "mlflow", None)
    use_mlflow = save_mlflow and bool(config.experiment_name) and mlflow is not None
    if save_mlflow and oci is None and not use_mlflow:
        raise ValueError(
            "save_sample_results requires OCI exports or MLflow persistence"
        )
    # Keep temporary files until the synchronous OCI upload completes.
    if oci is not None:
        with TemporaryDirectory(prefix="evalhub-samples-") as temporary:
            directory = Path(temporary)
            (directory / "sample_results.jsonl").write_bytes(payload)
            (directory / "manifest.json").write_text(
                json.dumps({"primary_score": {"metric": metric}, "estimand": "mean"}),
                encoding="utf-8",
            )
            result.oci_artifact = callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=directory,
                    coordinates=oci.coordinates.model_copy(deep=True),
                )
            )
    # Publish the run ID only after MLflow confirms persistence.
    if use_mlflow:
        run_id = mlflow.save(
            result,
            config,
            artifacts=[
                MlflowArtifact("sample_results.jsonl", payload, "application/x-ndjson")
            ],
        )
        if not run_id:
            raise RuntimeError("MLflow did not confirm sample-result persistence")
        result.mlflow_run_id = run_id
    return payload
