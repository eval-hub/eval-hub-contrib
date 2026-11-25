"""Benchmark registry: the single source of truth for what each benchmark is.

Adding a benchmark means adding one ``BenchmarkSpec`` to ``BENCHMARKS`` (and its
entry to ``provider.yaml``; ``tests/test_benchmarks.py`` checks the two agree).
Dispatch on benchmark id elsewhere in the adapter reads these specs instead of
comparing ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BiasMetric,
    ConversationCompletenessMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    KnowledgeRetentionMetric,
    PIILeakageMetric,
    RoleAdherenceMetric,
    SummarizationMetric,
    ToolCorrectnessMetric,
    ToxicityMetric,
    TurnRelevancyMetric,
)

from . import metrics, supplementary
from .metrics import MetricFactory, plain_metric
from .supplementary import ExtraMetrics


class Kind(StrEnum):
    """Shape of one dataset record."""

    SINGLE_TURN = "single_turn"
    CONVERSATIONAL = "conversational"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Everything the adapter needs to know about one benchmark."""

    id: str
    kind: Kind
    category: str
    ability: str
    required_columns: tuple[str, ...]
    # None means the benchmark is validated directly and uses no DeepEval metric.
    metric_factory: MetricFactory | None
    optional_columns: tuple[str, ...] = ()
    # Job parameters that must be present, checked before any data is loaded.
    required_params: tuple[str, ...] = ()
    extra_metrics: ExtraMetrics | None = None

    @property
    def primary_metric(self) -> str:
        """Name of the aggregate metric reported as the benchmark score."""
        return f"{self.id.replace('-', '_')}_score"

    @property
    def is_safety(self) -> bool:
        return self.category == "safety"

    @property
    def is_conversational(self) -> bool:
        return self.kind is Kind.CONVERSATIONAL

    @property
    def is_direct(self) -> bool:
        """True when records are validated directly, with no judge model."""
        return self.metric_factory is None


def _single_turn(
    benchmark_id: str,
    category: str,
    ability: str,
    required_columns: tuple[str, ...],
    metric_factory: MetricFactory | None,
    **options: Any,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=benchmark_id,
        kind=Kind.SINGLE_TURN,
        category=category,
        ability=ability,
        required_columns=required_columns,
        metric_factory=metric_factory,
        **options,
    )


def _conversational(
    benchmark_id: str,
    category: str,
    metric_factory: MetricFactory,
    required_columns: tuple[str, ...] = ("turns",),
    **options: Any,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        id=benchmark_id,
        kind=Kind.CONVERSATIONAL,
        category=category,
        ability=category,
        required_columns=required_columns,
        metric_factory=metric_factory,
        **options,
    )


_IO = ("input", "actual_output")
_TURN_CONTEXT = ("chatbot_role", "scenario")

_SPECS: tuple[BenchmarkSpec, ...] = (
    # ── RAG ────────────────────────────────────────────────────────────────
    _single_turn(
        "faithfulness",
        "rag",
        "rag",
        (*_IO, "retrieval_context"),
        plain_metric(FaithfulnessMetric),
        extra_metrics=supplementary.faithfulness_counts,
    ),
    _single_turn("relevancy", "rag", "rag", _IO, plain_metric(AnswerRelevancyMetric)),
    _single_turn(
        "hallucination",
        "rag",
        "rag",
        (*_IO, "context"),
        plain_metric(HallucinationMetric),
        extra_metrics=supplementary.hallucination_flag,
    ),
    _single_turn(
        "correctness",
        "accuracy",
        "accuracy",
        (*_IO, "expected_output"),
        metrics.correctness_metric,
    ),
    _single_turn(
        "summarization", "nlp", "summarization", _IO, plain_metric(SummarizationMetric)
    ),
    # ── Custom LLM-as-judge ────────────────────────────────────────────────
    _single_turn(
        "geval",
        "custom",
        "custom",
        _IO,
        metrics.geval_metric,
        required_params=("criteria",),
    ),
    _single_turn(
        "dag",
        "custom",
        "custom",
        _IO,
        metrics.dag_metric,
        required_params=("dag_criteria_json",),
    ),
    # ── Agentic ────────────────────────────────────────────────────────────
    _single_turn(
        "task-completion", "agent", "agent", _IO, metrics.task_completion_metric
    ),
    # ToolCorrectnessMetric lists expected_tools in its required params; a record
    # without it raises MissingTestCaseParamsError and fails the job.
    _single_turn(
        "tool-correctness",
        "agent",
        "agent",
        (*_IO, "tools_called", "expected_tools"),
        plain_metric(ToolCorrectnessMetric),
    ),
    _single_turn(
        "argument-correctness",
        "agent",
        "agent",
        (*_IO, "tools_called"),
        plain_metric(ArgumentCorrectnessMetric),
    ),
    # ── Safety ─────────────────────────────────────────────────────────────
    _single_turn("bias", "safety", "fairness", _IO, plain_metric(BiasMetric)),
    _single_turn("toxicity", "safety", "safety", _IO, plain_metric(ToxicityMetric)),
    _single_turn(
        "pii-leakage",
        "safety",
        "privacy",
        _IO,
        plain_metric(PIILeakageMetric),
        extra_metrics=supplementary.pii_flag,
    ),
    _single_turn("non-advice", "safety", "safety", _IO, metrics.non_advice_metric),
    _single_turn("misuse", "safety", "safety", _IO, metrics.misuse_metric),
    _single_turn(
        "role-violation",
        "safety",
        "safety",
        _IO,
        metrics.role_violation_metric,
        required_params=("role",),
    ),
    # ── Format ─────────────────────────────────────────────────────────────
    # JsonCorrectnessMetric needs a Pydantic BaseModel class, not a dict, so this
    # benchmark uses json.loads() + jsonschema directly (json_correctness.py).
    _single_turn("json-correctness", "format", "format", _IO, None),
    # ── Multi-turn conversational ──────────────────────────────────────────
    _conversational(
        "conversation-completeness",
        "multi-turn",
        plain_metric(ConversationCompletenessMetric),
        optional_columns=(*_TURN_CONTEXT, "expected_outcome"),
    ),
    _conversational(
        "conversation-relevancy",
        "multi-turn",
        plain_metric(TurnRelevancyMetric),
        optional_columns=_TURN_CONTEXT,
    ),
    _conversational(
        "role-adherence",
        "multi-turn",
        plain_metric(RoleAdherenceMetric),
        required_columns=("turns", "chatbot_role"),
        optional_columns=("scenario",),
    ),
    _conversational(
        "knowledge-retention",
        "multi-turn",
        plain_metric(KnowledgeRetentionMetric),
        optional_columns=_TURN_CONTEXT,
    ),
    _conversational(
        "conversational-geval",
        "custom",
        metrics.conversational_geval_metric,
        optional_columns=_TURN_CONTEXT,
        required_params=("criteria",),
    ),
)

BENCHMARKS: Mapping[str, BenchmarkSpec] = {spec.id: spec for spec in _SPECS}
assert len(BENCHMARKS) == len(_SPECS), "duplicate benchmark id in _SPECS"


def find_benchmark(benchmark_id: str) -> BenchmarkSpec | None:
    """Return the spec for ``benchmark_id``, or ``None`` if it is not registered."""
    return BENCHMARKS.get(benchmark_id)


def get_benchmark(benchmark_id: str) -> BenchmarkSpec:
    """Return the spec for ``benchmark_id`` or raise ``ValueError``."""
    spec = find_benchmark(benchmark_id)
    if spec is None:
        raise ValueError(f"Unknown benchmark_id: {benchmark_id!r}")
    return spec


def create_metric(
    benchmark_id: str, model: Any, threshold: float, params: Mapping[str, Any]
) -> Any:
    """Instantiate the DeepEval metric for a benchmark.

    Returns ``None`` for direct-validation benchmarks (``json-correctness``),
    which the job runner scores without a metric.
    """
    spec = get_benchmark(benchmark_id)
    if spec.metric_factory is None:
        return None
    return spec.metric_factory(model, threshold, params)
