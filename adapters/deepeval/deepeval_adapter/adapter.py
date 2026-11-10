"""``DeepEvalAdapter``: sequences the job phases and delegates the work."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from evalhub.adapter import (
    EnvironmentCardMetadata,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
)

from . import reporting
from .benchmarks import BenchmarkSpec, create_metric, find_benchmark
from .cases import build_test_cases
from .data_loading import load_dataset, resolve_data_dir
from .json_correctness import run_json_correctness
from .judge import resolve_judge_model
from .results import compute_overall_score, extract_results
from .settings import JobSettings, apply_runtime_tuning, validate_job_spec

logger = logging.getLogger(__name__)

# Resolves to the adapter directory (/app in the image) when no local jobs path is set.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _status(
    callbacks: JobCallbacks, phase: JobPhase, progress: float | None = None
) -> None:
    update: dict[str, Any] = {"status": JobStatus.RUNNING, "phase": phase}
    if progress is not None:
        update["progress"] = progress
    callbacks.report_status(JobStatusUpdate(**update))


class DeepEvalAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter that runs DeepEval metrics and returns JobResults."""

    def __init__(self, job_spec_path: str | None = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a DeepEval benchmark: load data, run metric, extract results."""
        start_time = time.time()
        logger.info(
            "Starting DeepEval job %s | benchmark=%s | model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )
        try:
            return self._run(config, callbacks, start_time)
        except Exception as exc:
            logger.exception("DeepEval evaluation failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(exc), message_code="evaluation_error"
                    ),
                )
            )
            raise

    def _run(
        self, config: JobSpec, callbacks: JobCallbacks, start_time: float
    ) -> JobResults:
        _status(callbacks, JobPhase.INITIALIZING)

        # Cache the version once per job: it is read for the cards and metadata.
        version = reporting.deepeval_version()
        env_card = EnvironmentCardMetadata.capture(
            framework_name="deepeval",
            framework_version=version,
            extra_packages=["deepeval", "litellm", "pandas"],
        )

        settings = JobSettings.from_job_spec(config)
        spec = settings.spec
        apply_runtime_tuning(settings.params)

        _status(callbacks, JobPhase.LOADING_DATA)
        data_dir = resolve_data_dir(config)
        records = self._load_records(config, settings, data_dir)
        test_cases = build_test_cases(records, spec)
        logger.info(
            "Loaded %d test cases | benchmark=%s | is_conversational=%s",
            len(test_cases),
            spec.id,
            spec.is_conversational,
        )

        _status(callbacks, JobPhase.RUNNING_EVALUATION, progress=0.0)
        evaluation_results, card_entries = self._evaluate(settings, records, test_cases)
        _status(callbacks, JobPhase.RUNNING_EVALUATION, progress=1.0)

        _status(callbacks, JobPhase.POST_PROCESSING)
        overall_score = compute_overall_score(evaluation_results, spec.id)
        # Direct validation scores every record; other benchmarks score only the
        # records that became test cases.
        num_evaluated = len(records) if spec.is_direct else len(test_cases)
        judge_label = (
            "none (direct validation)" if spec.is_direct else settings.judge_name
        )
        eval_card = reporting.build_eval_card(
            card_entries,
            settings.params,
            footnotes=(
                f"DeepEval {version} | benchmark={spec.id} | judge={judge_label} | "
                f"samples={num_evaluated} | threshold={settings.threshold}"
            ),
        )
        summary = reporting.build_summary(
            spec.id, overall_score, num_evaluated, evaluation_results
        )

        oci_artifact = self._persist_oci(config, callbacks, summary, version)
        if oci_artifact:
            env_card.oci_artifact_ref = oci_artifact.reference

        job_results = JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=evaluation_results,
            overall_score=overall_score,
            num_examples_evaluated=num_evaluated,
            duration_seconds=time.time() - start_time,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "framework": "deepeval",
                "framework_version": version,
                "benchmark_id": spec.id,
                "eval_model_name": settings.judge_name,
                "threshold": settings.threshold,
                "dataset_format": settings.dataset_format,
                "data_dir": data_dir,
                "is_conversational": spec.is_conversational,
            },
            oci_artifact=oci_artifact,
            eval_card=eval_card,
            env_card=env_card,
        )
        reporting.save_to_mlflow(callbacks, config, job_results, summary)

        # Do NOT report COMPLETED here: report_results() sends results AND
        # COMPLETED in one atomic event, and an early COMPLETED makes the
        # following report_results() fail with a 409 that drops every metric.
        return job_results

    @staticmethod
    def _load_records(
        config: JobSpec, settings: JobSettings, data_dir: str
    ) -> list[dict[str, Any]]:
        records = load_dataset(data_dir, settings.dataset_format)
        limit = config.num_examples or settings.params.get("num_examples")
        if limit and int(limit) > 0:
            records = records[: int(limit)]
            logger.info("Capped dataset to %d records", len(records))
        return records

    @staticmethod
    def _evaluate(
        settings: JobSettings, records: list[dict[str, Any]], test_cases: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        params = settings.params
        if settings.spec.is_direct:
            return run_json_correctness(records, params)

        judge = resolve_judge_model(settings.judge_name, settings.judge_url)
        metric = create_metric(settings.spec.id, judge, settings.threshold, params)
        raw_results = evaluate(
            test_cases=test_cases,
            metrics=[metric],
            async_config=AsyncConfig(
                run_async=True,
                throttle_value=float(params.get("throttle_value", 0)),
                max_concurrent=int(params.get("max_concurrent", 1)),
            ),
        )
        return extract_results(raw_results, settings.spec.id)

    def _persist_oci(
        self, config: JobSpec, callbacks: JobCallbacks, summary: bytes, version: str
    ) -> Any:
        oci_exports = config.exports.oci if config.exports else None
        if oci_exports is None:
            return None
        _status(callbacks, JobPhase.PERSISTING_ARTIFACTS)
        results_dir = (
            self.local_jobs_base_path / "results"
            if self.local_jobs_base_path is not None
            else _DEFAULT_RESULTS_DIR
        )
        return reporting.persist_oci_artifact(
            oci_exports, config, callbacks, summary, results_dir, version
        )

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        """Provide prompting strategy and dataset metadata for EvalCard downstream."""
        metadata = results.evaluation_metadata or {}
        benchmark_id = metadata.get("benchmark_id", results.benchmark_id)
        try:
            spec = find_benchmark(benchmark_id)
            is_direct = spec is not None and spec.is_direct
            return {
                "zero_shot": results.overall_score,
                "prompting_strategy": "direct-validation"
                if is_direct
                else "zero-shot LLM-as-judge",
                "framework": "deepeval",
                "framework_version": metadata.get(
                    "framework_version", reporting.deepeval_version()
                ),
                "benchmark_id": benchmark_id,
                "eval_model": None if is_direct else metadata.get("eval_model_name"),
                "threshold": metadata.get("threshold"),
                "num_samples": results.num_examples_evaluated,
            }
        except Exception:
            logger.warning("Failed to generate additional_info", exc_info=True)
            return None

    def _validate_config(self, config: JobSpec) -> BenchmarkSpec:
        """Validate the job's benchmark id and required parameters."""
        return validate_job_spec(config)
