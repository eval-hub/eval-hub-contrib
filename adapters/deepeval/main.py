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
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ConversationCompletenessMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    SummarizationMetric,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, SingleTurnParams, Turn
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
from evalhub.adapter.auth import resolve_model_credentials

logger = logging.getLogger(__name__)

# Maps benchmark_id to the DeepEval metric class and required test-case fields (single-turn).
BENCHMARK_METRICS = {
    "faithfulness": {
        "class": FaithfulnessMetric,
        "required_columns": ["input", "actual_output", "retrieval_context"],
    },
    "relevancy": {
        "class": AnswerRelevancyMetric,
        "required_columns": ["input", "actual_output"],
    },
    "hallucination": {
        "class": HallucinationMetric,
        "required_columns": ["input", "actual_output", "context"],
    },
    "correctness": {
        "class": GEval,
        "required_columns": ["input", "actual_output", "expected_output"],
    },
    "summarization": {
        "class": SummarizationMetric,
        "required_columns": ["input", "actual_output"],
    },
}

# Maps benchmark_id to the DeepEval metric class and required fields (multi-turn).
CONVERSATIONAL_BENCHMARK_METRICS = {
    "conversation-completeness": {
        "class": ConversationCompletenessMetric,
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario", "expected_outcome"],
    },
    "role-adherence": {
        "class": RoleAdherenceMetric,
        "required_columns": ["turns", "chatbot_role"],
        "optional_columns": ["scenario"],
    },
    "knowledge-retention": {
        "class": KnowledgeRetentionMetric,
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario"],
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


def _build_conversational_test_cases(
    records: list[dict[str, Any]], benchmark_id: str
) -> list[ConversationalTestCase]:
    """Convert raw records into DeepEval ConversationalTestCase objects.

    Each record must have a ``turns`` field: either a list of dicts (JSONL/JSON)
    or a JSON-encoded string (CSV). Each turn dict requires ``role`` and ``content``.
    Optional top-level fields ``chatbot_role``, ``scenario``, and ``expected_outcome``
    are forwarded to the test case when present.
    """
    spec = CONVERSATIONAL_BENCHMARK_METRICS.get(benchmark_id)
    if not spec:
        raise ValueError(f"Unknown conversational benchmark_id: {benchmark_id}")

    test_cases: list[ConversationalTestCase] = []
    for i, rec in enumerate(records):
        missing = [c for c in spec["required_columns"] if c not in rec or rec[c] is None]
        if missing:
            logger.warning("Skipping record %d: missing columns %s", i, missing)
            continue

        raw_turns = rec["turns"]
        if isinstance(raw_turns, str):
            raw_turns = json.loads(raw_turns)

        turns = [Turn(role=t["role"], content=t["content"]) for t in raw_turns]

        kwargs: dict[str, Any] = {"turns": turns}
        for field in ("chatbot_role", "scenario", "expected_outcome"):
            val = rec.get(field)
            if val:
                kwargs[field] = str(val)

        test_cases.append(ConversationalTestCase(**kwargs))

    if not test_cases:
        raise ValueError(f"No valid conversational test cases built from {len(records)} records")

    logger.info("Built %d conversational test cases for %s", len(test_cases), benchmark_id)
    return test_cases


def _resolve_judge_model(judge_name: str, judge_url: str) -> Any:
    """Return a GPTModel pointed at an OpenAI-compatible endpoint.

    Credentials are resolved via the EvalHub SDK (mounted secret file or env var)
    so the adapter never reads the key name directly.
    JSON mode is forced to ensure small local models return parseable output.
    """
    from deepeval.models.llms import GPTModel

    creds = resolve_model_credentials()
    api_key = creds.api_key
    if not api_key:
        auth_value = creds.auth_headers.get("Authorization", "")
        if auth_value.startswith("Bearer "):
            api_key = auth_value.removeprefix("Bearer ").strip()

    url = judge_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return GPTModel(
        model=judge_name,
        base_url=url,
        api_key=api_key or "EMPTY",
        generation_kwargs={"response_format": {"type": "json_object"}},
    )


def _create_metric(benchmark_id: str, model: Any, threshold: float):
    """Instantiate the DeepEval metric for the given benchmark."""
    if benchmark_id == "correctness":
        return GEval(
            name="Correctness",
            criteria="Determine if the actual output is factually correct compared to the expected output.",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            model=model,
            threshold=threshold,
        )

    spec = BENCHMARK_METRICS.get(benchmark_id) or CONVERSATIONAL_BENCHMARK_METRICS.get(benchmark_id)
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
            model_url = config.model.url.strip().rstrip("/")
            model_name = config.model.name
            judge_name = config.parameters.get("eval_model_name") or model_name
            judge_url = config.parameters.get("eval_model_url") or model_url
            threshold = float(config.parameters.get("threshold", 0.5))
            dataset_format = config.parameters.get("dataset_format", "csv")

            # Allow callers to tune retry/timeout behaviour; deepeval reads these from the environment.
            # Default is 300s to accommodate reasoning models (e.g. DeepSeek-R1) that emit long
            # chain-of-thought sequences before the first response token.
            per_attempt_timeout = config.parameters.get("per_attempt_timeout_seconds", 300.0)
            if per_attempt_timeout is not None:
                os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(float(per_attempt_timeout))
            retry_max_attempts = config.parameters.get("retry_max_attempts")
            if retry_max_attempts is not None:
                os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(int(retry_max_attempts))
            retry_cap_seconds = config.parameters.get("retry_cap_seconds")
            if retry_cap_seconds is not None:
                os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = str(float(retry_cap_seconds))

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
            is_conversational = benchmark_id in CONVERSATIONAL_BENCHMARK_METRICS
            if is_conversational:
                test_cases = _build_conversational_test_cases(records, benchmark_id)
            else:
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

            judge = _resolve_judge_model(judge_name, judge_url)
            metric = _create_metric(benchmark_id, judge, threshold)
            throttle_value = float(config.parameters.get("throttle_value", 0))
            max_concurrent = int(config.parameters.get("max_concurrent", 1))
            eval_results = evaluate(
                test_cases=test_cases,
                metrics=[metric],
                async_config=AsyncConfig(
                    run_async=True,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                ),
            )

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

        all_benchmarks = {**BENCHMARK_METRICS, **CONVERSATIONAL_BENCHMARK_METRICS}
        if config.benchmark_id not in all_benchmarks:
            raise ValueError(
                f"Unsupported benchmark_id: {config.benchmark_id!r}. "
                f"Supported: {', '.join(all_benchmarks)}"
            )

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

        if benchmark_id == "faithfulness":
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
        elif benchmark_id == "relevancy":
            results.append(
                EvaluationResult(
                    metric_name="relevancy_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "hallucination":
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
        elif benchmark_id == "correctness":
            results.append(
                EvaluationResult(
                    metric_name="correctness_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "summarization":
            results.append(
                EvaluationResult(
                    metric_name="summarization_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "conversation-completeness":
            results.append(
                EvaluationResult(
                    metric_name="conversation_completeness_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "role-adherence":
            results.append(
                EvaluationResult(
                    metric_name="role_adherence_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )
        elif benchmark_id == "knowledge-retention":
            results.append(
                EvaluationResult(
                    metric_name="knowledge_retention_score",
                    metric_value=round(mean_score, 6),
                    metric_type="float",
                )
            )

        logger.info("Extracted %d metrics from DeepEval results", len(results))
        return results

    def _compute_overall_score(
        self, results: list[EvaluationResult], benchmark_id: str
    ) -> Optional[float]:  # type: ignore[override]
        """Compute overall score as the primary aggregate metric for the benchmark."""
        primary_metric = {
            "faithfulness": "faithfulness_score",
            "relevancy": "relevancy_score",
            "hallucination": "hallucination_score",
            "correctness": "correctness_score",
            "summarization": "summarization_score",
            "conversation-completeness": "conversation_completeness_score",
            "role-adherence": "role_adherence_score",
            "knowledge-retention": "knowledge_retention_score",
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
