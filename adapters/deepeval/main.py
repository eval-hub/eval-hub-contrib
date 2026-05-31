#!/usr/bin/env python3
"""DeepEval adapter for eval-hub.

Loads a JobSpec, reads test data (CSV/JSONL/JSON), builds DeepEval TestCase
objects, runs the appropriate metric via deepeval.evaluate(), then maps
results to evalhub-sdk JobResults.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    SummarizationMetric,
)
from deepeval.test_case import LLMTestCase
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
)

logger = logging.getLogger(__name__)

# Maps benchmark_id to the DeepEval metric class and required test-case fields.
BENCHMARK_METRICS = {
    "deepeval-faithfulness": {
        "class": FaithfulnessMetric,
        "required_columns": ["input", "actual_output", "retrieval_context"],
    },
    "deepeval-relevancy": {
        "class": AnswerRelevancyMetric,
        "required_columns": ["input", "actual_output"],
    },
    "deepeval-hallucination": {
        "class": HallucinationMetric,
        "required_columns": ["input", "actual_output", "context"],
    },
    "deepeval-correctness": {
        "class": GEval,
        "required_columns": ["input", "actual_output", "expected_output"],
    },
    "deepeval-summarization": {
        "class": SummarizationMetric,
        "required_columns": ["input", "actual_output"],
    },
}


def _load_dataset(data_dir: str, fmt: str) -> list[dict[str, Any]]:
    """Load test data from the given directory in the specified format."""
    data_path = Path(data_dir)
    records: list[dict[str, Any]] = []

    if fmt == "csv":
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        for f in csv_files:
            df = pd.read_csv(f)
            records.extend(df.to_dict("records"))
    elif fmt == "jsonl":
        jsonl_files = sorted(data_path.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in {data_dir}")
        for f in jsonl_files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    elif fmt == "json":
        json_files = sorted(data_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        for f in json_files:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
    else:
        raise ValueError(f"Unsupported dataset_format: {fmt!r}. Use csv, jsonl, or json.")

    if not records:
        raise ValueError(f"No records loaded from {data_dir} (format={fmt})")

    logger.info("Loaded %d records from %s (format=%s)", len(records), data_dir, fmt)
    return records


def _build_test_cases(records: list[dict[str, Any]], benchmark_id: str) -> list[LLMTestCase]:
    """Convert raw records into DeepEval LLMTestCase objects."""
    spec = BENCHMARK_METRICS.get(benchmark_id)
    if not spec:
        raise ValueError(f"Unknown benchmark_id: {benchmark_id}")

    test_cases: list[LLMTestCase] = []
    for i, rec in enumerate(records):
        missing = [c for c in spec["required_columns"] if c not in rec or rec[c] is None]
        if missing:
            logger.warning("Skipping record %d: missing columns %s", i, missing)
            continue

        kwargs: dict[str, Any] = {
            "input": str(rec["input"]),
            "actual_output": str(rec["actual_output"]),
        }
        if "expected_output" in rec and rec["expected_output"] is not None:
            kwargs["expected_output"] = str(rec["expected_output"])
        if "retrieval_context" in rec and rec["retrieval_context"] is not None:
            ctx = rec["retrieval_context"]
            kwargs["retrieval_context"] = ctx if isinstance(ctx, list) else [str(ctx)]
        if "context" in rec and rec["context"] is not None:
            ctx = rec["context"]
            kwargs["context"] = ctx if isinstance(ctx, list) else [str(ctx)]

        test_cases.append(LLMTestCase(**kwargs))

    if not test_cases:
        raise ValueError(f"No valid test cases built from {len(records)} records")

    logger.info("Built %d test cases for %s", len(test_cases), benchmark_id)
    return test_cases


def _create_metric(benchmark_id: str, model: str, threshold: float):
    """Instantiate the DeepEval metric for the given benchmark."""
    if benchmark_id == "deepeval-correctness":
        return GEval(
            name="Correctness",
            criteria="Determine if the actual output is factually correct compared to the expected output.",
            evaluation_params=["input", "actual_output", "expected_output"],
            model=model,
            threshold=threshold,
        )

    spec = BENCHMARK_METRICS.get(benchmark_id)
    if not spec:
        raise ValueError(f"Unknown benchmark_id: {benchmark_id}")

    return spec["class"](model=model, threshold=threshold)


def _resolve_data_dir(config: JobSpec) -> str:
    """Find the directory containing test data, checking standard mount paths first."""
    for candidate in ("/test_data", "/data"):
        p = Path(candidate)
        if p.is_dir() and any(p.iterdir()):
            logger.info("Using data from %s", candidate)
            return candidate

    data_dir = config.parameters.get("data_dir")
    if data_dir and Path(data_dir).is_dir():
        logger.info("Using data_dir from parameters: %s", data_dir)
        return data_dir

    raise ValueError(
        "No input data found: mount data under /test_data or /data, "
        "or set parameters.data_dir"
    )


class DeepEvalAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter that runs DeepEval metrics and returns JobResults."""

    def __init__(self, job_spec_path: Optional[str] = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a DeepEval benchmark: load data, run metric, extract results."""
        start_time = time.time()
        logger.info("Starting DeepEval job %s for benchmark %s", config.id, config.benchmark_id)

        try:
            # --- Phase: INITIALIZING ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing DeepEval evaluation",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)
            benchmark_id = config.benchmark_id
            model_name = config.parameters["eval_model_name"]
            threshold = float(config.parameters.get("threshold", 0.5))
            dataset_format = config.parameters.get("dataset_format", "csv")

            # --- Phase: LOADING_DATA ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.2,
                    message=MessageInfo(
                        message="Loading test dataset",
                        message_code="loading_data",
                    ),
                )
            )

            data_dir = _resolve_data_dir(config)
            records = _load_dataset(data_dir, dataset_format)
            test_cases = _build_test_cases(records, benchmark_id)

            # --- Phase: RUNNING_EVALUATION ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.4,
                    message=MessageInfo(
                        message=f"Running DeepEval {benchmark_id} on {len(test_cases)} test cases",
                        message_code="running_evaluation",
                    ),
                )
            )

            metric = _create_metric(benchmark_id, model_name, threshold)
            eval_results = evaluate(test_cases=test_cases, metrics=[metric])

            # --- Phase: POST_PROCESSING ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing DeepEval results",
                        message_code="post_processing",
                    ),
                )
            )

            evaluation_results = self._extract_results(eval_results, benchmark_id)
            overall_score = self._compute_overall_score(evaluation_results, benchmark_id)

            # --- Phase: PERSISTING_ARTIFACTS ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting DeepEval artifacts",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            duration = time.time() - start_time
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=len(test_cases),
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "deepeval",
                    "benchmark_id": benchmark_id,
                    "eval_model_name": model_name,
                    "threshold": threshold,
                    "dataset_format": dataset_format,
                    "data_dir": data_dir,
                },
            )

        except Exception as exc:
            logger.exception("DeepEval evaluation failed")
            error_msg = str(exc)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(
                        message=error_msg,
                        message_code="failed",
                    ),
                    error=ErrorInfo(
                        message=error_msg,
                        message_code="evaluation_error",
                    ),
                    error_details={
                        "exception_type": type(exc).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )
            raise

    def _validate_config(self, config: JobSpec) -> None:
        """Validate required configuration fields."""
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if config.benchmark_id not in BENCHMARK_METRICS:
            raise ValueError(
                f"Unsupported benchmark_id: {config.benchmark_id!r}. "
                f"Supported: {', '.join(BENCHMARK_METRICS)}"
            )

        if "eval_model_name" not in config.parameters:
            raise ValueError("eval_model_name is required in parameters")

    def _extract_results(
        self, eval_results: Any, benchmark_id: str
    ) -> list[EvaluationResult]:
        """Map DeepEval evaluation output to EvaluationResult objects."""
        results: list[EvaluationResult] = []
        scores: list[float] = []

        for i, test_result in enumerate(eval_results.test_results):
            for metric_result in test_result.metrics_data:
                score = metric_result.score if metric_result.score is not None else 0.0
                scores.append(score)
                results.append(
                    EvaluationResult(
                        metric_name=f"case_{i}.{metric_result.name}",
                        metric_value=round(score, 6),
                        metric_type="float",
                        metadata={
                            "success": metric_result.success,
                            "reason": metric_result.reason or "",
                        },
                    )
                )

        # Aggregate metrics by benchmark type
        if scores:
            mean_score = sum(scores) / len(scores)
        else:
            mean_score = 0.0

        if benchmark_id == "deepeval-faithfulness":
            results.append(
                EvaluationResult(
                    metric_name="faithfulness_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
            results.append(
                EvaluationResult(
                    metric_name="claims_count",
                    metric_value=len(scores),
                    metric_type="int",
                )
            )
            results.append(
                EvaluationResult(
                    metric_name="supported_claims_count",
                    metric_value=sum(1 for s in scores if s >= 0.5),
                    metric_type="int",
                )
            )
        elif benchmark_id == "deepeval-relevancy":
            results.append(
                EvaluationResult(
                    metric_name="relevancy_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "deepeval-hallucination":
            results.append(
                EvaluationResult(
                    metric_name="hallucination_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
            results.append(
                EvaluationResult(
                    metric_name="hallucination_detected",
                    metric_value=1 if mean_score > 0.5 else 0,
                    metric_type="int",
                )
            )
        elif benchmark_id == "deepeval-correctness":
            results.append(
                EvaluationResult(
                    metric_name="correctness_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "deepeval-summarization":
            results.append(
                EvaluationResult(
                    metric_name="summarization_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )

        logger.info("Extracted %d metrics from DeepEval results", len(results))
        return results

    def _compute_overall_score(
        self, results: list[EvaluationResult], benchmark_id: str
    ) -> Optional[float]:
        """Compute overall score as the primary aggregate metric for the benchmark."""
        primary_metric = {
            "deepeval-faithfulness": "faithfulness_score",
            "deepeval-relevancy": "relevancy_score",
            "deepeval-hallucination": "hallucination_score",
            "deepeval-correctness": "correctness_score",
            "deepeval-summarization": "summarization_score",
        }.get(benchmark_id)

        if primary_metric:
            for r in results:
                if r.metric_name == primary_metric:
                    return r.metric_value
        return None


def main() -> None:
    """Load JobSpec, run DeepEvalAdapter, emit JobResults via DefaultCallbacks."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = DeepEvalAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)

        logger.info(
            "Done %s score=%s n=%s %.2fs",
            results.id,
            results.overall_score,
            results.num_examples_evaluated,
            results.duration_seconds,
        )
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error("Job spec not found: %s (set EVALHUB_JOB_SPEC_PATH)", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
