#!/usr/bin/env python3
"""DeepEval adapter for eval-hub.

Loads a JobSpec, reads test data (CSV/JSONL/JSON), builds DeepEval TestCase
objects, runs the appropriate metric via deepeval.evaluate(), then maps
results to evalhub-sdk JobResults including EvalCard and EnvironmentCard.

Supported benchmarks
---------------------
RAG evaluation:
  faithfulness          FaithfulnessMetric   — claims grounded in retrieval context
  relevancy             AnswerRelevancyMetric — response relevance to the query
  hallucination         HallucinationMetric  — unsupported factual claims
  correctness           GEval (fixed)        — factual accuracy vs expected output
  summarization         SummarizationMetric  — source coverage and alignment

Custom LLM-as-judge:
  geval                 GEval (configurable) — user-defined criteria via parameters.criteria
  conversational-geval  ConversationalGEval  — GEval for multi-turn conversations
  dag                   DAGMetric            — directed-graph evaluation criteria

Agentic:
  task-completion       TaskCompletionMetric  — binary task completion
  tool-correctness      ToolCorrectnessMetric — tool name + argument accuracy
  argument-correctness  ArgumentCorrectnessMetric — argument accuracy for tool calls

Safety:
  bias                  BiasMetric          — demographic / political bias detection
  toxicity              ToxicityMetric      — harmful content detection
  pii-leakage           PIILeakageMetric    — personally identifiable info exposure
  non-advice            NonAdviceMetric     — unsolicited medical/legal/financial advice
  misuse                MisuseMetric        — misuse pattern detection
  role-violation        RoleViolationMetric — out-of-role behaviour

Multi-turn conversational:
  conversation-completeness   ConversationCompletenessMetric
  conversation-relevancy      ConversationRelevancyMetric
  role-adherence              RoleAdherenceMetric
  knowledge-retention         KnowledgeRetentionMetric

Format:
  json-correctness      JsonCorrectnessMetric — JSON schema validation (no LLM judge)
"""

from __future__ import annotations

import importlib.metadata
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
    ArgumentCorrectnessMetric,
    BiasMetric,
    ConversationCompletenessMetric,
    ConversationRelevancyMetric,
    DAGMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    JsonCorrectnessMetric,
    KnowledgeRetentionMetric,
    MisuseMetric,
    NonAdviceMetric,
    PIILeakageMetric,
    RoleAdherenceMetric,
    RoleViolationMetric,
    SummarizationMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ToxicityMetric,
)
from deepeval.metrics import GEval as ConversationalGEval  # same class; role set by test_case type
from deepeval.test_case import ConversationalTestCase, LLMTestCase, SingleTurnParams, Turn
from evalhub.adapter import (
    CapabilityEvalEntry,
    DefaultCallbacks,
    EnvironmentCardMetadata,
    EvalCardMetadata,
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
    SafetyEvalEntry,
)
from evalhub.adapter.auth import resolve_model_credentials
from evalhub.adapter.mlflow import MlflowArtifact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark → metric class + required / optional dataset columns
# ---------------------------------------------------------------------------

# Single-turn benchmarks (LLMTestCase)
SINGLE_TURN_BENCHMARKS: dict[str, dict[str, Any]] = {
    # ── RAG ────────────────────────────────────────────────────────────────
    "faithfulness": {
        "class": FaithfulnessMetric,
        "required_columns": ["input", "actual_output", "retrieval_context"],
        "category": "rag",
        "ability": "rag",
    },
    "relevancy": {
        "class": AnswerRelevancyMetric,
        "required_columns": ["input", "actual_output"],
        "category": "rag",
        "ability": "rag",
    },
    "hallucination": {
        "class": HallucinationMetric,
        "required_columns": ["input", "actual_output", "context"],
        "category": "rag",
        "ability": "rag",
    },
    "correctness": {
        "class": GEval,          # fixed-criteria GEval
        "required_columns": ["input", "actual_output", "expected_output"],
        "category": "accuracy",
        "ability": "accuracy",
    },
    "summarization": {
        "class": SummarizationMetric,
        "required_columns": ["input", "actual_output"],
        "category": "nlp",
        "ability": "summarization",
    },
    # ── Custom LLM-as-judge ────────────────────────────────────────────────
    "geval": {
        "class": GEval,          # user-configured at runtime via parameters.criteria
        "required_columns": ["input", "actual_output"],
        "category": "custom",
        "ability": "custom",
    },
    "dag": {
        "class": DAGMetric,
        "required_columns": ["input", "actual_output"],
        "category": "custom",
        "ability": "custom",
    },
    # ── Agentic ───────────────────────────────────────────────────────────
    "task-completion": {
        "class": TaskCompletionMetric,
        "required_columns": ["input", "actual_output"],
        "category": "agent",
        "ability": "agent",
    },
    "tool-correctness": {
        "class": ToolCorrectnessMetric,
        "required_columns": ["input", "actual_output", "tools_called"],
        "category": "agent",
        "ability": "agent",
    },
    "argument-correctness": {
        "class": ArgumentCorrectnessMetric,
        "required_columns": ["input", "actual_output", "tools_called"],
        "category": "agent",
        "ability": "agent",
    },
    # ── Safety ────────────────────────────────────────────────────────────
    "bias": {
        "class": BiasMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "fairness",
    },
    "toxicity": {
        "class": ToxicityMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "safety",
    },
    "pii-leakage": {
        "class": PIILeakageMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "privacy",
    },
    "non-advice": {
        "class": NonAdviceMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "safety",
    },
    "misuse": {
        "class": MisuseMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "safety",
    },
    "role-violation": {
        "class": RoleViolationMetric,
        "required_columns": ["input", "actual_output"],
        "category": "safety",
        "ability": "safety",
    },
    # ── Format ────────────────────────────────────────────────────────────
    "json-correctness": {
        "class": JsonCorrectnessMetric,
        "required_columns": ["input", "actual_output"],
        "category": "format",
        "ability": "format",
    },
}

# Multi-turn benchmarks (ConversationalTestCase)
CONVERSATIONAL_BENCHMARKS: dict[str, dict[str, Any]] = {
    "conversation-completeness": {
        "class": ConversationCompletenessMetric,
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario", "expected_outcome"],
        "category": "multi-turn",
        "ability": "multi-turn",
    },
    "conversation-relevancy": {
        "class": ConversationRelevancyMetric,
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario"],
        "category": "multi-turn",
        "ability": "multi-turn",
    },
    "role-adherence": {
        "class": RoleAdherenceMetric,
        "required_columns": ["turns", "chatbot_role"],
        "optional_columns": ["scenario"],
        "category": "multi-turn",
        "ability": "multi-turn",
    },
    "knowledge-retention": {
        "class": KnowledgeRetentionMetric,
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario"],
        "category": "multi-turn",
        "ability": "multi-turn",
    },
    "conversational-geval": {
        "class": ConversationalGEval,  # user-configured via parameters.criteria
        "required_columns": ["turns"],
        "optional_columns": ["chatbot_role", "scenario"],
        "category": "custom",
        "ability": "custom",
    },
}

# Safety benchmarks that use SafetyEvalEntry (not CapabilityEvalEntry)
_SAFETY_BENCHMARKS = frozenset({
    "bias", "toxicity", "pii-leakage", "non-advice", "misuse", "role-violation",
})

# Primary aggregate metric per benchmark_id
_PRIMARY_METRIC: dict[str, str] = {
    "faithfulness": "faithfulness_score",
    "relevancy": "relevancy_score",
    "hallucination": "hallucination_score",
    "correctness": "correctness_score",
    "summarization": "summarization_score",
    "geval": "geval_score",
    "dag": "dag_score",
    "task-completion": "task_completion_score",
    "tool-correctness": "tool_correctness_score",
    "argument-correctness": "argument_correctness_score",
    "bias": "bias_score",
    "toxicity": "toxicity_score",
    "pii-leakage": "pii_leakage_score",
    "non-advice": "non_advice_score",
    "misuse": "misuse_score",
    "role-violation": "role_violation_score",
    "json-correctness": "json_correctness_score",
    "conversation-completeness": "conversation_completeness_score",
    "conversation-relevancy": "conversation_relevancy_score",
    "role-adherence": "role_adherence_score",
    "knowledge-retention": "knowledge_retention_score",
    "conversational-geval": "conversational_geval_score",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test case builders
# ---------------------------------------------------------------------------

def _coerce_list(value: Any) -> list[str]:
    """Parse a string-encoded or native list into list[str]."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


def _build_single_turn_test_cases(
    records: list[dict[str, Any]], benchmark_id: str
) -> list[LLMTestCase]:
    """Convert raw records into DeepEval LLMTestCase objects."""
    spec = SINGLE_TURN_BENCHMARKS[benchmark_id]
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
            kwargs["retrieval_context"] = _coerce_list(rec["retrieval_context"])
        if "context" in rec and rec["context"] is not None:
            kwargs["context"] = _coerce_list(rec["context"])

        # Agentic tool calls: expect a list of {name, input_parameters} dicts
        if "tools_called" in rec and rec["tools_called"] is not None:
            raw = rec["tools_called"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            try:
                from deepeval.test_case import ToolCall
                kwargs["tools_called"] = [
                    ToolCall(
                        name=t.get("name", ""),
                        input_parameters=t.get("input_parameters", {}),
                        output=t.get("output"),
                    )
                    for t in raw
                ]
            except (ImportError, Exception) as exc:
                logger.warning("ToolCall import failed, passing raw: %s", exc)
                kwargs["tools_called"] = raw

        if "expected_tools" in rec and rec["expected_tools"] is not None:
            raw = rec["expected_tools"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            try:
                from deepeval.test_case import ToolCall
                kwargs["expected_tools"] = [
                    ToolCall(name=t.get("name", ""), input_parameters=t.get("input_parameters", {}))
                    for t in raw
                ]
            except (ImportError, Exception) as exc:
                logger.warning("ToolCall import for expected_tools failed: %s", exc)

        test_cases.append(LLMTestCase(**kwargs))

    if not test_cases:
        raise ValueError(f"No valid test cases built from {len(records)} records")

    logger.info("Built %d single-turn test cases for %s", len(test_cases), benchmark_id)
    return test_cases


def _build_conversational_test_cases(
    records: list[dict[str, Any]], benchmark_id: str
) -> list[ConversationalTestCase]:
    """Convert raw records into DeepEval ConversationalTestCase objects."""
    spec = CONVERSATIONAL_BENCHMARKS[benchmark_id]
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


# ---------------------------------------------------------------------------
# Judge model resolution
# ---------------------------------------------------------------------------

def _resolve_judge_model(judge_name: str, judge_url: str) -> Any:
    """Return a GPTModel pointed at an OpenAI-compatible endpoint.

    Forces JSON mode so small local models return parseable output.
    Credentials are resolved via the EvalHub SDK (mounted secret or env var).
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


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------

def _create_metric(benchmark_id: str, model: Any, threshold: float, params: dict[str, Any]) -> Any:
    """Instantiate the DeepEval metric for the given benchmark."""

    # ── Fixed-criteria GEval (correctness) ─────────────────────────────────
    if benchmark_id == "correctness":
        return GEval(
            name="Correctness",
            criteria=(
                "Determine if the actual output is factually correct "
                "compared to the expected output."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=model,
            threshold=threshold,
        )

    # ── User-configurable GEval ────────────────────────────────────────────
    if benchmark_id == "geval":
        criteria = params.get("criteria")
        if not criteria:
            raise ValueError("parameters.criteria is required for the geval benchmark")
        raw_params = params.get("evaluation_params", ["INPUT", "ACTUAL_OUTPUT"])
        if isinstance(raw_params, str):
            raw_params = json.loads(raw_params)
        eval_params = [SingleTurnParams[p.upper()] for p in raw_params]
        return GEval(
            name=params.get("geval_name", "CustomEval"),
            criteria=criteria,
            evaluation_params=eval_params,
            model=model,
            threshold=threshold,
        )

    # ── Conversational GEval ───────────────────────────────────────────────
    if benchmark_id == "conversational-geval":
        criteria = params.get("criteria")
        if not criteria:
            raise ValueError("parameters.criteria is required for the conversational-geval benchmark")
        return GEval(
            name=params.get("geval_name", "ConversationalEval"),
            criteria=criteria,
            model=model,
            threshold=threshold,
        )

    # ── DAGMetric ─────────────────────────────────────────────────────────
    if benchmark_id == "dag":
        dag_json = params.get("dag_criteria_json")
        if not dag_json:
            raise ValueError(
                "parameters.dag_criteria_json is required for the dag benchmark. "
                "Provide a JSON-encoded DAGMetric criteria graph."
            )
        if isinstance(dag_json, str):
            dag_json = json.loads(dag_json)
        return DAGMetric(
            name=params.get("dag_name", "DAGEval"),
            criteria=dag_json,
            model=model,
            threshold=threshold,
        )

    # ── JsonCorrectnessMetric (no LLM judge) ───────────────────────────────
    if benchmark_id == "json-correctness":
        schema = params.get("json_schema")
        if schema and isinstance(schema, str):
            schema = json.loads(schema)
        if schema:
            return JsonCorrectnessMetric(expected_schema=schema, threshold=threshold)
        return JsonCorrectnessMetric(threshold=threshold)

    # ── All other benchmarks: instantiate class from spec ─────────────────
    spec = SINGLE_TURN_BENCHMARKS.get(benchmark_id) or CONVERSATIONAL_BENCHMARKS.get(benchmark_id)
    if not spec:
        raise ValueError(f"Unknown benchmark_id: {benchmark_id!r}")

    cls = spec["class"]
    if benchmark_id == "json-correctness":
        return cls(threshold=threshold)
    # JsonCorrectnessMetric already handled above; all others need model
    return cls(model=model, threshold=threshold)


# ---------------------------------------------------------------------------
# Result extraction + card building
# ---------------------------------------------------------------------------

def _extract_results(
    eval_results: Any,
    benchmark_id: str,
) -> tuple[list[EvaluationResult], list[CapabilityEvalEntry | SafetyEvalEntry]]:
    """Map DeepEval output to (EvaluationResult list, card entry list)."""
    results: list[EvaluationResult] = []
    scores: list[float] = []
    card_entries: list[CapabilityEvalEntry | SafetyEvalEntry] = []
    is_safety = benchmark_id in _SAFETY_BENCHMARKS

    for i, test_result in enumerate(eval_results.test_results):
        for md in test_result.metrics_data:
            score = md.score if md.score is not None else 0.0
            scores.append(score)
            results.append(
                EvaluationResult(
                    metric_name=f"case_{i}.{md.name}",
                    metric_value=round(score, 6),
                    metric_type="float",
                    num_samples=1,
                    metadata={
                        "success": md.success,
                        "reason": md.reason or "",
                    },
                )
            )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    primary_key = _PRIMARY_METRIC.get(benchmark_id, f"{benchmark_id}_score")

    results.append(
        EvaluationResult(
            metric_name=primary_key,
            metric_value=round(mean_score, 6),
            metric_type="float",
            num_samples=len(scores),
        )
    )

    # Supplementary aggregate metrics per benchmark
    if benchmark_id == "faithfulness":
        results.append(EvaluationResult(metric_name="claims_count", metric_value=len(scores), metric_type="int"))
        results.append(EvaluationResult(
            metric_name="supported_claims_count",
            metric_value=sum(1 for s in scores if s >= 0.5),
            metric_type="int",
        ))
    elif benchmark_id == "hallucination":
        results.append(EvaluationResult(
            metric_name="hallucination_detected",
            metric_value=1 if mean_score > 0.5 else 0,
            metric_type="int",
        ))
    elif benchmark_id == "pii-leakage":
        results.append(EvaluationResult(
            metric_name="pii_detected",
            metric_value=1 if mean_score > 0.5 else 0,
            metric_type="int",
        ))
    elif benchmark_id == "json-correctness":
        results.append(EvaluationResult(
            metric_name="schema_valid",
            metric_value=1 if mean_score >= 1.0 else 0,
            metric_type="int",
        ))

    # EvalCard entry
    spec = SINGLE_TURN_BENCHMARKS.get(benchmark_id) or CONVERSATIONAL_BENCHMARKS.get(benchmark_id) or {}
    ability = spec.get("ability", benchmark_id)

    if is_safety:
        card_entries.append(SafetyEvalEntry(
            feature=ability,
            benchmark=f"deepeval/{benchmark_id}",
            metric=primary_key,
            zero_shot=round(mean_score, 4),
        ))
    else:
        card_entries.append(CapabilityEvalEntry(
            ability=ability,
            benchmark=f"deepeval/{benchmark_id}",
            metric=primary_key,
            zero_shot=round(mean_score, 4),
        ))

    return results, card_entries


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DeepEvalAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter that runs DeepEval metrics and returns JobResults."""

    def __init__(self, job_spec_path: Optional[str] = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a DeepEval benchmark: load data, run metric, extract results."""
        start_time = time.time()
        logger.info(
            "Starting DeepEval job %s | benchmark=%s | model=%s",
            config.id, config.benchmark_id, config.model.name,
        )

        try:
            # ── Phase: INITIALIZING ──────────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
            )

            # Capture environment at the earliest possible moment
            env_card = EnvironmentCardMetadata.capture(
                framework_name="deepeval",
                framework_version=_deepeval_version(),
                extra_packages=["deepeval", "litellm", "pandas"],
            )

            self._validate_config(config)
            benchmark_id = config.benchmark_id
            params = config.parameters
            model_url = config.model.url.strip().rstrip("/")
            model_name = config.model.name
            judge_name = params.get("eval_model_name") or model_name
            judge_url = params.get("eval_model_url") or model_url
            threshold = float(params.get("threshold", 0.5))
            dataset_format = params.get("dataset_format", "csv")

            # Tune DeepEval retry / timeout via environment variables.
            # Default 300s accommodates reasoning models (DeepSeek-R1, Phi-4)
            # that emit long chain-of-thought before the first token.
            per_attempt_timeout = params.get("per_attempt_timeout_seconds", 300.0)
            if per_attempt_timeout is not None:
                os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(float(per_attempt_timeout))
            retry_max = params.get("retry_max_attempts")
            if retry_max is not None:
                os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(int(retry_max))
            retry_cap = params.get("retry_cap_seconds")
            if retry_cap is not None:
                os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = str(float(retry_cap))

            # ── Phase: LOADING_DATA ──────────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
            )

            data_dir = _resolve_data_dir(config)
            records = _load_dataset(data_dir, dataset_format)

            is_conversational = benchmark_id in CONVERSATIONAL_BENCHMARKS
            if is_conversational:
                test_cases = _build_conversational_test_cases(records, benchmark_id)
            else:
                test_cases = _build_single_turn_test_cases(records, benchmark_id)

            logger.info(
                "Loaded %d test cases | benchmark=%s | is_conversational=%s",
                len(test_cases), benchmark_id, is_conversational,
            )

            # ── Phase: RUNNING_EVALUATION ────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.0,
                )
            )

            # json-correctness has no LLM judge
            if benchmark_id == "json-correctness":
                judge = None
            else:
                judge = _resolve_judge_model(judge_name, judge_url)

            metric = _create_metric(benchmark_id, judge, threshold, params)
            throttle_value = float(params.get("throttle_value", 0))
            max_concurrent = int(params.get("max_concurrent", 1))

            raw_results = evaluate(
                test_cases=test_cases,
                metrics=[metric],
                async_config=AsyncConfig(
                    run_async=True,
                    throttle_value=throttle_value,
                    max_concurrent=max_concurrent,
                ),
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=1.0,
                )
            )

            # ── Phase: POST_PROCESSING ───────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
            )

            evaluation_results, card_entries = _extract_results(raw_results, benchmark_id)
            overall_score = _compute_overall_score(evaluation_results, benchmark_id)

            # Build EvalCard — separate capability and safety entries
            capability_entries = [e for e in card_entries if isinstance(e, CapabilityEvalEntry)]
            safety_entries = [e for e in card_entries if isinstance(e, SafetyEvalEntry)]

            eval_card = EvalCardMetadata(
                modalities_input=["text"],
                modalities_output=["text"],
                languages_count=params.get("languages_count", 1),
                languages=params.get("languages", ["en"]),
                capability_evaluations=capability_entries,
                safety_evaluations=safety_entries,
                developer_footnotes=(
                    f"DeepEval {_deepeval_version()} | "
                    f"benchmark={benchmark_id} | "
                    f"judge={judge_name} | "
                    f"samples={len(test_cases)} | "
                    f"threshold={threshold}"
                ),
            )

            # Build results summary for OCI artifact
            summary_bytes = json.dumps(
                {
                    "benchmark_id": benchmark_id,
                    "overall_score": overall_score,
                    "num_examples_evaluated": len(test_cases),
                    "results": [
                        {"metric_name": r.metric_name, "metric_value": r.metric_value}
                        for r in evaluation_results
                    ],
                },
                indent=2,
                default=str,
            ).encode()

            # ── Phase: PERSISTING_ARTIFACTS ─────────────────────────────────
            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None:
                callbacks.report_status(
                    JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
                )
                if self.local_jobs_base_path is not None:
                    results_dir = self.local_jobs_base_path / "results"
                else:
                    results_dir = Path(__file__).parent / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "results_summary.json").write_bytes(summary_bytes)

                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update({
                    "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                    "io.github.eval-hub.benchmark": config.benchmark_id,
                    "io.github.eval-hub.model": config.model.name,
                    "io.github.eval-hub.job_id": config.id,
                    "io.github.eval-hub.framework": "deepeval",
                    "io.github.eval-hub.deepeval.version": _deepeval_version(),
                })
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(files_path=results_dir, coordinates=coords)
                )
                logger.info("OCI artifact created: %s", oci_artifact.reference)

                if oci_artifact:
                    env_card.oci_artifact_ref = oci_artifact.reference

            duration = time.time() - start_time

            job_results = JobResults(
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
                    "framework_version": _deepeval_version(),
                    "benchmark_id": benchmark_id,
                    "eval_model_name": judge_name,
                    "threshold": threshold,
                    "dataset_format": dataset_format,
                    "data_dir": data_dir,
                    "is_conversational": is_conversational,
                },
                oci_artifact=oci_artifact,
                eval_card=eval_card,
                env_card=env_card,
            )

            # MLflow save
            experiment_name = (config.experiment_name or "").strip()
            if not experiment_name:
                raw = params.get("mlflow_experiment_name")
                if isinstance(raw, str) and raw.strip():
                    experiment_name = raw.strip()

            if experiment_name:
                try:
                    spec = config.model_copy(update={"experiment_name": experiment_name})
                    rid = callbacks.mlflow.save(
                        job_results,
                        spec,
                        artifacts=[
                            MlflowArtifact("results_summary.json", summary_bytes, "application/json")
                        ],
                    )
                    if rid:
                        job_results.mlflow_run_id = rid
                        logger.info("MLflow run saved: %s (experiment=%s)", rid, experiment_name)
                except Exception as exc:
                    logger.warning("MLflow save failed (job still completes): %s", exc)

            # Do NOT call report_status(COMPLETED) here — report_results() sends
            # results AND COMPLETED in one atomic event. Calling COMPLETED early
            # causes a 409 on the subsequent report_results(), dropping all metrics.
            return job_results

        except Exception as exc:
            logger.exception("DeepEval evaluation failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(exc),
                        message_code="evaluation_error",
                    ),
                )
            )
            raise

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        """Provide prompting strategy and dataset metadata for EvalCard downstream."""
        metadata = results.evaluation_metadata or {}
        benchmark_id = metadata.get("benchmark_id", results.benchmark_id)
        try:
            return {
                "zero_shot": results.overall_score,
                "prompting_strategy": "zero-shot LLM-as-judge",
                "framework": "deepeval",
                "framework_version": metadata.get("framework_version", _deepeval_version()),
                "benchmark_id": benchmark_id,
                "eval_model": metadata.get("eval_model_name"),
                "threshold": metadata.get("threshold"),
                "num_samples": results.num_examples_evaluated,
            }
        except Exception:
            logger.warning("Failed to generate additional_info", exc_info=True)
            return None

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_config(self, config: JobSpec) -> None:
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        all_benchmarks = {**SINGLE_TURN_BENCHMARKS, **CONVERSATIONAL_BENCHMARKS}
        if config.benchmark_id not in all_benchmarks:
            raise ValueError(
                f"Unsupported benchmark_id: {config.benchmark_id!r}. "
                f"Supported: {', '.join(sorted(all_benchmarks))}"
            )

        # Validate that required parameters for parameterized metrics are present
        if config.benchmark_id in ("geval", "conversational-geval"):
            if not config.parameters.get("criteria"):
                raise ValueError(
                    f"parameters.criteria is required for benchmark_id={config.benchmark_id!r}"
                )
        if config.benchmark_id == "dag":
            if not config.parameters.get("dag_criteria_json"):
                raise ValueError(
                    "parameters.dag_criteria_json is required for benchmark_id='dag'"
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_overall_score(
    results: list[EvaluationResult], benchmark_id: str
) -> Optional[float]:
    primary = _PRIMARY_METRIC.get(benchmark_id)
    if primary:
        for r in results:
            if r.metric_name == primary:
                return r.metric_value
    return None


# Backward-compatible alias — existing tests import _build_test_cases
_build_test_cases = _build_single_turn_test_cases


def _deepeval_version() -> str:
    try:
        return importlib.metadata.version("deepeval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

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
            "Job %s | benchmark=%s | model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)

        logger.info(
            "Done %s | score=%s | n=%s | %.2fs",
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
