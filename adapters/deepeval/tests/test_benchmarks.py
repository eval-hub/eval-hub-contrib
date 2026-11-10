"""Benchmark registry, metric factories and the provider.yaml contract."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from deepeval import metrics as deepeval_metrics
from deepeval.test_case.conversational_test_case import MultiTurnParams

from deepeval_adapter import metrics, supplementary
from deepeval_adapter.benchmarks import (
    BENCHMARKS,
    Kind,
    create_metric,
    find_benchmark,
    get_benchmark,
)

# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------


def test_registry_keys_match_spec_ids():
    assert all(key == spec.id for key, spec in BENCHMARKS.items())


def test_every_benchmark_requires_input_columns():
    for spec in BENCHMARKS.values():
        expected = "turns" if spec.kind is Kind.CONVERSATIONAL else "input"
        assert expected in spec.required_columns, spec.id


def test_only_json_correctness_is_direct():
    assert {s.id for s in BENCHMARKS.values() if s.is_direct} == {"json-correctness"}


def test_primary_metric_is_derived_from_the_id():
    assert BENCHMARKS["pii-leakage"].primary_metric == "pii_leakage_score"
    assert (
        BENCHMARKS["conversational-geval"].primary_metric
        == "conversational_geval_score"
    )


def test_safety_benchmarks_are_those_with_the_safety_category():
    safety = {s.id for s in BENCHMARKS.values() if s.is_safety}
    assert safety == {
        "bias",
        "toxicity",
        "pii-leakage",
        "non-advice",
        "misuse",
        "role-violation",
    }


def test_find_and_get_benchmark():
    assert find_benchmark("bias") is BENCHMARKS["bias"]
    assert find_benchmark("nope") is None
    with pytest.raises(ValueError, match="Unknown benchmark_id: 'nope'"):
        get_benchmark("nope")


# ---------------------------------------------------------------------------
# Metric factories
# ---------------------------------------------------------------------------


def test_create_metric_is_none_for_direct_benchmarks(stub_judge):
    assert create_metric("json-correctness", stub_judge, 0.5, {}) is None


def test_create_metric_rejects_unknown_benchmark(stub_judge):
    with pytest.raises(ValueError, match="Unknown benchmark_id"):
        create_metric("nope", stub_judge, 0.5, {})


@pytest.mark.parametrize(
    "benchmark_id,expected_cls",
    [
        ("faithfulness", deepeval_metrics.FaithfulnessMetric),
        ("bias", deepeval_metrics.BiasMetric),
        ("tool-correctness", deepeval_metrics.ToolCorrectnessMetric),
        ("conversation-completeness", deepeval_metrics.ConversationCompletenessMetric),
        ("correctness", deepeval_metrics.GEval),
    ],
)
def test_create_metric_builds_the_registered_class(
    benchmark_id, expected_cls, stub_judge
):
    metric = create_metric(benchmark_id, stub_judge, 0.7, {})
    assert isinstance(metric, expected_cls)
    assert metric.threshold == 0.7


def test_geval_requires_criteria(stub_judge):
    with pytest.raises(ValueError, match="parameters.criteria is required"):
        create_metric("geval", stub_judge, 0.5, {})


def test_geval_accepts_json_encoded_evaluation_params(stub_judge):
    metric = create_metric(
        "geval",
        stub_judge,
        0.5,
        {"criteria": "Be concise", "evaluation_params": '["INPUT", "ACTUAL_OUTPUT"]'},
    )
    assert len(metric.evaluation_params) == 2


def test_conversational_geval_drops_unknown_evaluation_params(stub_judge):
    metric = create_metric(
        "conversational-geval",
        stub_judge,
        0.5,
        {"criteria": "Stay polite", "evaluation_params": ["ROLE", "NOT_A_PARAM"]},
    )
    # DeepEval keeps CONTENT alongside the requested ROLE; the unknown name is dropped.
    assert MultiTurnParams.ROLE in metric.evaluation_params


def test_non_advice_decodes_json_advice_types(stub_judge):
    metric = create_metric("non-advice", stub_judge, 0.5, {"advice_types": '["legal"]'})
    assert metric.advice_types == ["legal"]


def test_misuse_defaults_to_the_general_domain(stub_judge):
    assert create_metric("misuse", stub_judge, 0.5, {}).domain == "general"


def test_dag_requires_criteria_json(stub_judge):
    with pytest.raises(ValueError, match="dag_criteria_json is required"):
        create_metric("dag", stub_judge, 0.5, {})


def test_dag_accepts_the_criteria_shorthand(stub_judge):
    metric = create_metric(
        "dag", stub_judge, 0.5, {"dag_criteria_json": '{"criteria": "Is it correct?"}'}
    )
    assert isinstance(metric, deepeval_metrics.DAGMetric)


def test_dag_shorthand_has_one_pass_and_one_fail_verdict():
    """DeepEval rejects a BinaryJudgementNode without exactly one of each."""
    graph = metrics.build_dag_from_json({"criteria": "Is it correct?"})
    verdicts = {c.verdict: c.score for c in graph.root_nodes[0].children}
    assert verdicts == {True: 10, False: 0}


def test_build_dag_rejects_unknown_node_types():
    with pytest.raises(ValueError, match="Unsupported DAG node type"):
        metrics.build_dag_from_json({"root_nodes": [{"type": "Mystery"}]})


def test_build_dag_requires_nodes_or_criteria():
    with pytest.raises(ValueError, match="root_nodes"):
        metrics.build_dag_from_json({})


# ---------------------------------------------------------------------------
# Supplementary metrics
# ---------------------------------------------------------------------------


def _by_name(results):
    return {r.metric_name: r.metric_value for r in results}


def test_faithfulness_counts():
    out = _by_name(supplementary.faithfulness_counts([0.9, 0.2, 0.5], 0.53))
    assert out == {"cases_count": 3, "passing_cases_count": 2}


@pytest.mark.parametrize(
    "flag,name",
    [
        (supplementary.hallucination_flag, "hallucination_detected"),
        (supplementary.pii_flag, "pii_detected"),
    ],
)
def test_detection_flags_fire_below_the_pass_score(flag, name):
    assert _by_name(flag([0.1], 0.1))[name] == 1
    assert _by_name(flag([0.5], 0.5))[name] == 0
    assert _by_name(flag([0.9], 0.9))[name] == 0


# ---------------------------------------------------------------------------
# provider.yaml contract and score direction
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_provider_declares_primary_score_for_every_benchmark():
    """Each benchmark names its primary metric and a direction.

    The server reads primary_score.lower_is_better to decide whether a value
    passes its threshold, so a missing entry makes a bad run look good.
    """
    import yaml

    provider_path = Path(__file__).parent.parent / "provider.yaml"
    provider = yaml.safe_load(provider_path.read_text())
    benchmarks = {b["id"]: b for b in provider["benchmarks"]}
    assert set(benchmarks) == set(BENCHMARKS)
    for benchmark_id, entry in benchmarks.items():
        primary = entry["primary_score"]
        assert primary["metric"] == BENCHMARKS[benchmark_id].primary_metric, (
            benchmark_id
        )
        assert primary["lower_is_better"] is False, benchmark_id
        assert entry["metrics"][0] == primary["metric"], benchmark_id


@pytest.mark.integration
@pytest.mark.parametrize(
    "metric_name",
    ["BiasMetric", "ToxicityMetric", "MisuseMetric", "PIILeakageMetric"],
)
def test_deepeval_safety_scores_are_higher_is_better(metric_name, stub_judge):
    """Canary for the deepeval pin: no verdict flagged means score 1.0 and a pass.

    deepeval 4.2.0 reversed bias, toxicity and misuse (and hallucination) from
    lower-is-better. provider.yaml declares lower_is_better: false, so a bump
    that flips the direction back must fail here first.
    """
    from deepeval import metrics

    extra = {"domain": "general"} if metric_name == "MisuseMetric" else {}
    metric = getattr(metrics, metric_name)(model=stub_judge, threshold=0.5, **extra)
    verdict = SimpleNamespace(verdict="no")
    metric.verdicts = [verdict, verdict]
    metric.score = metric._calculate_score()
    metric.error = None
    assert metric.score == 1.0
    assert metric.is_successful() is True
