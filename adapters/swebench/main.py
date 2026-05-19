#!/usr/bin/env python3
"""
SWE-bench Adapter for eval-hub

Evaluates LLM-generated code patches against the SWE-bench benchmark
by orchestrating Kubernetes Jobs.  Each evaluation instance runs as a
self-contained K8s Job using a pre-built SWE-bench container image.

The adapter is a thin orchestration layer:
  1. Loads a predictions JSON file (standard SWE-bench format)
  2. Calls swebench.harness.k8s_evaluation.run_instances_k8s()
  3. Maps the graded results to eval-hub EvaluationResult objects
  4. Reports resolve_rate as the overall_score

Architecture:
    JobSpec (predictions file, registry, namespace)
        -> run_instances_k8s() creates K8s Jobs
        -> each Job applies patch + runs tests in a SWE-bench image
        -> adapter grades results via swebench.harness.grading
        -> JobResults with resolve_rate
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    DefaultCallbacks,
    ErrorInfo,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
    OCIArtifactResult,
)
from evalhub.models.api import OCICoordinates

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_BENCHMARKS = {
    "swebench_verified": "princeton-nlp/SWE-bench_Verified",
    "swebench_lite": "princeton-nlp/SWE-bench_Lite",
    "swebench_full": "princeton-nlp/SWE-bench",
}

_KEY_INSTANCE_ID = "instance_id"
_KEY_MODEL = "model_name_or_path"
_KEY_PREDICTION = "model_patch"


class SWEBenchAdapter(FrameworkAdapter):
    """Adapter for SWE-bench code patch evaluation.

    Orchestrates Kubernetes Jobs to evaluate model-generated patches
    against SWE-bench instances.  Requires pre-built SWE-bench images
    in a container registry accessible from the cluster.
    """

    def run_benchmark_job(
        self, config: JobSpec, callbacks: JobCallbacks
    ) -> JobResults:
        start_time = time.time()
        logger.info("Starting SWE-bench evaluation job: %s", config.id)

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.0,
                message=MessageInfo(
                    message="Initializing SWE-bench evaluation",
                    message_code="initializing",
                ),
            )
        )

        try:
            # -- Phase 1: INITIALIZING --
            benchmark_id = config.benchmark_id
            if benchmark_id not in SUPPORTED_BENCHMARKS:
                msg = (
                    f"Unknown benchmark_id: {benchmark_id}. "
                    f"Supported: {list(SUPPORTED_BENCHMARKS)}"
                )
                raise ValueError(msg)

            dataset_name = SUPPORTED_BENCHMARKS[benchmark_id]
            params = config.parameters or {}

            predictions_path = params.get("predictions_path", "")
            k8s_registry = params.get("k8s_registry", "docker.io/swebench")
            k8s_namespace = params.get("k8s_namespace") or self._current_namespace()
            max_workers = int(params.get("max_workers", 10))
            timeout = int(params.get("timeout_per_instance", 1800))
            split = params.get("split", "test")
            instance_ids = params.get("instance_ids")
            run_id = config.id

            # Load predictions
            predictions = self._load_predictions(predictions_path, dataset_name, split)
            logger.info("Loaded %d predictions", len(predictions))

            # -- Phase 2: LOADING_DATA --
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=MessageInfo(
                        message=f"Loading SWE-bench dataset ({benchmark_id}, {len(predictions)} predictions)",
                        message_code="loading_data",
                    ),
                )
            )

            from swebench.harness.utils import load_swebench_dataset

            dataset = load_swebench_dataset(dataset_name, split=split, instance_ids=instance_ids)
            dataset = [
                inst for inst in dataset
                if inst[_KEY_INSTANCE_ID] in predictions
            ]
            logger.info("Evaluating %d instances", len(dataset))

            if not dataset:
                return self._empty_results(config, start_time)

            # -- Phase 3: RUNNING_EVALUATION --
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.2,
                    message=MessageInfo(
                        message=f"Running evaluation ({len(dataset)} instances, {max_workers} workers)",
                        message_code="running_evaluation",
                    ),
                )
            )

            from swebench.harness.k8s_eval import run_instances_k8s

            results = run_instances_k8s(
                predictions=predictions,
                dataset=dataset,
                run_id=run_id,
                registry=k8s_registry,
                namespace=k8s_namespace,
                max_workers=max_workers,
                timeout=timeout,
                cleanup=True,
            )

            # -- Phase 4: POST_PROCESSING --
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing evaluation results",
                        message_code="post_processing",
                    ),
                )
            )

            evaluation_results, overall_score, summary = self._process_results(
                results, config
            )

            # -- Phase 5: PERSISTING_ARTIFACTS --
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting evaluation artifacts",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            artifact_files = self._save_artifacts(results, summary, config)
            self._persist_oci_artifact(artifact_files, config, callbacks)

            duration = time.time() - start_time

            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name if config.model else "",
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=len(results),
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error("SWE-bench evaluation failed: %s", e, exc_info=True)
            duration = time.time() - start_time

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.0,
                    error=ErrorInfo(
                        message=str(e),
                        message_code="evaluation_failed",
                    ),
                )
            )

            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name if config.model else "",
                results=[],
                overall_score=None,
                num_examples_evaluated=0,
                duration_seconds=duration,
            )

    # -- Helpers ---------------------------------------------------------------

    def _current_namespace() -> str:
        try:
            with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
                return f.read().strip()
        except OSError:
            return "default"

    def _load_predictions(
        self, predictions_path: str, dataset_name: str, split: str
    ) -> dict[str, dict]:
        """Load predictions from a JSON file.

        Supports both dict format (``{instance_id: {...}}``) and list
        format (``[{instance_id: ..., model_patch: ...}, ...]``).

        If ``predictions_path`` is ``"gold"``, loads gold patches from
        the dataset.
        """
        if predictions_path == "gold":
            from swebench.harness.utils import load_swebench_dataset

            dataset = load_swebench_dataset(dataset_name, split=split)
            return {
                inst[_KEY_INSTANCE_ID]: {
                    _KEY_INSTANCE_ID: inst[_KEY_INSTANCE_ID],
                    _KEY_PREDICTION: inst["patch"],
                    _KEY_MODEL: "gold",
                }
                for inst in dataset
            }

        path = Path(predictions_path)
        if not path.exists():
            msg = f"Predictions file not found: {predictions_path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            content = f.read()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try JSONL
            data = [json.loads(line) for line in content.splitlines() if line.strip()]

        if isinstance(data, list):
            return {p[_KEY_INSTANCE_ID]: p for p in data}
        return data

    def _process_results(
        self, results: dict[str, dict], config: JobSpec
    ) -> tuple[list[EvaluationResult], float | None, dict[str, Any]]:
        """Convert raw results to EvaluationResult objects."""
        total = len(results)
        completed = sum(1 for r in results.values() if r.get("completed"))
        resolved = sum(
            1 for r in results.values()
            if r.get("completed") and r.get("resolved")
        )
        errors = sum(1 for r in results.values() if not r.get("completed"))

        resolve_rate = resolved / max(completed, 1)

        evaluation_results = [
            EvaluationResult(
                metric_name="resolve_rate",
                metric_value=resolve_rate,
                metric_type="accuracy",
            ),
            EvaluationResult(
                metric_name="resolved_count",
                metric_value=float(resolved),
                metric_type="count",
            ),
            EvaluationResult(
                metric_name="total_instances",
                metric_value=float(total),
                metric_type="count",
            ),
            EvaluationResult(
                metric_name="completed_count",
                metric_value=float(completed),
                metric_type="count",
            ),
            EvaluationResult(
                metric_name="error_count",
                metric_value=float(errors),
                metric_type="count",
            ),
        ]

        summary = {
            "resolve_rate": resolve_rate,
            "resolved": resolved,
            "completed": completed,
            "total": total,
            "errors": errors,
            "per_instance": {
                iid: {
                    "resolved": r.get("resolved", False),
                    "completed": r.get("completed", False),
                }
                for iid, r in results.items()
            },
        }

        logger.info(
            "Results: %d/%d resolved (%.1f%%), %d errors",
            resolved, total, resolve_rate * 100, errors,
        )

        return evaluation_results, resolve_rate, summary

    def _save_artifacts(
        self, results: dict, summary: dict, config: JobSpec
    ) -> list[Path]:
        """Save result artifacts to local files for OCI persistence."""
        base_path = self._get_results_dir(config)
        base_path.mkdir(parents=True, exist_ok=True)

        summary_path = base_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        results_path = base_path / "results.json"
        results_path.write_text(json.dumps(results, indent=2))

        logger.info("Artifacts saved to %s", base_path)
        return [summary_path, results_path]

    def _get_results_dir(self, config: JobSpec) -> Path:
        """Get the results directory, respecting local_jobs_base_path."""
        if self.local_jobs_base_path:
            return Path(self.local_jobs_base_path) / "results"
        return Path("/tmp/swebench_results") / config.id

    def _persist_oci_artifact(
        self,
        files: list[Path],
        config: JobSpec,
        callbacks: JobCallbacks,
    ) -> None:
        """Create an OCI artifact from result files."""
        if not files:
            return

        try:
            artifact_dir = files[0].parent
            callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=artifact_dir,
                    coordinates=OCICoordinates(
                        oci_host="",
                        oci_repository=f"swebench/{config.id}",
                    ),
                )
            )
            logger.info("OCI artifact created from %s", artifact_dir)
        except Exception as e:
            logger.warning("Failed to create OCI artifact: %s", e)

    def _empty_results(self, config: JobSpec, start_time: float) -> JobResults:
        """Return empty results when no instances to evaluate."""
        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name if config.model else "",
            results=[
                EvaluationResult(
                    metric_name="resolve_rate",
                    metric_value=0.0,
                    metric_type="accuracy",
                ),
            ],
            overall_score=0.0,
            num_examples_evaluated=0,
            duration_seconds=time.time() - start_time,
        )


def main() -> None:
    """Standard eval-hub adapter entry point."""
    try:
        job_spec_path = os.getenv(
            "EVALHUB_JOB_SPEC_PATH", "/meta/job.json"
        )
        adapter = SWEBenchAdapter(job_spec_path=job_spec_path)
        logger.info("Loaded job %s", adapter.job_spec.id)
        logger.info("Benchmark: %s", adapter.job_spec.benchmark_id)

        callbacks = DefaultCallbacks.from_adapter(adapter)

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        callbacks.report_results(results)

        logger.info("Job completed successfully")
        if results.overall_score is not None:
            logger.info("Resolve rate: %.1f%%", results.overall_score * 100)
        logger.info("Instances evaluated: %d", results.num_examples_evaluated)
        logger.info("Duration: %.1fs", results.duration_seconds)

        sys.exit(0)

    except Exception as e:
        logger.error("Fatal error in SWE-bench adapter: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
