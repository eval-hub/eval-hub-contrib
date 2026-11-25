"""DeepEval metric factories.

Every factory takes ``(model, threshold, params)`` and returns a configured
metric, so the benchmark registry can hold one callable per benchmark instead of
the job runner branching on benchmark ids.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from deepeval.metrics import (
    ConversationalGEval,
    DAGMetric,
    GEval,
    MisuseMetric,
    NonAdviceMetric,
    RoleViolationMetric,
    TaskCompletionMetric,
)
from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph, VerdictNode
from deepeval.test_case import SingleTurnParams
from deepeval.test_case.conversational_test_case import MultiTurnParams

MetricFactory = Callable[[Any, float, Mapping[str, Any]], Any]


def _json_param(value: Any) -> Any:
    """Decode a JSON-encoded job parameter; pass native values through."""
    return json.loads(value) if isinstance(value, str) else value


def plain_metric(metric_cls: type) -> MetricFactory:
    """Factory for metrics that take only ``model`` and ``threshold``."""

    def factory(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
        return metric_cls(model=model, threshold=threshold)

    return factory


def correctness_metric(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
    """Fixed-criteria GEval comparing the output with the expected output."""
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


def geval_metric(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
    """GEval with user-supplied criteria and evaluation parameters."""
    criteria = params.get("criteria")
    if not criteria:
        raise ValueError("parameters.criteria is required for the geval benchmark")
    raw_params = _json_param(
        params.get("evaluation_params", ["INPUT", "ACTUAL_OUTPUT"])
    )
    return GEval(
        name=params.get("geval_name", "CustomEval"),
        criteria=criteria,
        evaluation_params=[SingleTurnParams[p.upper()] for p in raw_params],
        model=model,
        threshold=threshold,
    )


def conversational_geval_metric(
    model: Any, threshold: float, params: Mapping[str, Any]
) -> Any:
    """ConversationalGEval with user-supplied criteria; unknown params are dropped."""
    criteria = params.get("criteria")
    if not criteria:
        raise ValueError(
            "parameters.criteria is required for the conversational-geval benchmark"
        )
    raw_params = _json_param(params.get("evaluation_params", ["ROLE", "CONTENT"]))
    eval_params = [
        MultiTurnParams[p.upper()]
        for p in raw_params
        if p.upper() in MultiTurnParams.__members__
    ]
    return ConversationalGEval(
        name=params.get("geval_name", "ConversationalEval"),
        criteria=criteria,
        evaluation_params=eval_params or None,
        model=model,
        threshold=threshold,
    )


def build_dag_from_json(dag_spec: Mapping[str, Any]) -> DeepAcyclicGraph:
    """Convert a serialisable ``dag_criteria_json`` structure into a graph.

    Supported formats::

        {"root_nodes": [{"type": "BinaryJudgementNode", "criteria": "...",
                         "children": [{"type": "VerdictNode", "verdict": true,
                                       "score": 1.0}]}]}
        {"criteria": "Is the response accurate?"}   # one BinaryJudgementNode

    DeepEval requires a BinaryJudgementNode to have exactly one True and one
    False VerdictNode child and scores verdicts on a 0 to 10 scale, so the
    shorthand scores a pass 10 and a fail 0.
    """

    def build_node(spec: Mapping[str, Any]) -> Any:
        node_type = spec.get("type", "BinaryJudgementNode")
        if node_type == "VerdictNode":
            return VerdictNode(
                verdict=bool(spec.get("verdict", True)),
                score=float(spec.get("score", 1.0)),
            )
        if node_type == "BinaryJudgementNode":
            children = [build_node(c) for c in spec.get("children", [])]
            return BinaryJudgementNode(
                criteria=spec.get("criteria", ""),
                children=children or None,
            )
        raise ValueError(f"Unsupported DAG node type: {node_type!r}")

    if "root_nodes" in dag_spec:
        root_nodes = [build_node(n) for n in dag_spec["root_nodes"]]
    elif "criteria" in dag_spec:
        root_nodes = [
            BinaryJudgementNode(
                criteria=dag_spec["criteria"],
                children=[
                    VerdictNode(verdict=True, score=10),
                    VerdictNode(verdict=False, score=0),
                ],
            )
        ]
    else:
        raise ValueError(
            "dag_criteria_json must contain 'root_nodes' (list of node specs) "
            "or 'criteria' (shorthand for a single BinaryJudgementNode)"
        )

    return DeepAcyclicGraph(root_nodes=root_nodes)


def dag_metric(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
    """DAGMetric built from the JSON-encoded ``dag_criteria_json`` parameter."""
    dag_json = params.get("dag_criteria_json")
    if not dag_json:
        raise ValueError(
            "parameters.dag_criteria_json is required for the dag benchmark. "
            "Provide a JSON-encoded DAGMetric criteria graph."
        )
    return DAGMetric(
        name=params.get("dag_name", "DAGEval"),
        dag=build_dag_from_json(_json_param(dag_json)),
        model=model,
        threshold=threshold,
    )


def non_advice_metric(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
    """NonAdviceMetric over the configured advice types."""
    advice_types = _json_param(
        params.get("advice_types", ["medical", "legal", "financial"])
    )
    return NonAdviceMetric(advice_types=advice_types, model=model, threshold=threshold)


def misuse_metric(model: Any, threshold: float, params: Mapping[str, Any]) -> Any:
    """MisuseMetric for the configured domain."""
    return MisuseMetric(
        domain=params.get("domain", "general"), model=model, threshold=threshold
    )


def role_violation_metric(
    model: Any, threshold: float, params: Mapping[str, Any]
) -> Any:
    """RoleViolationMetric for the configured role."""
    return RoleViolationMetric(
        model=model, threshold=threshold, role=params.get("role")
    )


def task_completion_metric(
    model: Any, threshold: float, params: Mapping[str, Any]
) -> Any:
    """TaskCompletionMetric with an optional task description."""
    task = params.get("task_description") or params.get("task")
    return TaskCompletionMetric(model=model, threshold=threshold, task=task)
