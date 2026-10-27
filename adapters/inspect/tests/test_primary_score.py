"""HLE/BFCL representative-score regressions; all individual metrics remain."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from evalhub.adapter import EvaluationResult

import main
from evalhub.models.api import PrimaryScore

from _results import compute_overall_score, extract_results, resolve_primary_score
from main import InspectAdapter


CASES = [
    ("inspect/hle", "hle/regex_judge", "hle/accuracy", 0,
     {"hle/stderr": 0, "hle/cerr": 0.8, "hle/unscored": 0}),
    ("inspect/bfcl", "bfcl_scorer", "accuracy", 0.8,
     {"stderr": 0.1, "simple_python_acc": 0.6, "simple_python_sterr": 0.1}),
]


def metric(name, value):
    return EvaluationResult(metric_name=name, metric_value=value, metric_type="float", num_samples=3)


@pytest.mark.parametrize("bid,scorer,key,value,extras", CASES)
def test_accuracy_selected_without_changing_individual_metrics(bid, scorer, key, value, extras):
    log = {"results": {"total_samples": 3, "scores": [{
        "name": scorer, "scorer": scorer,
        "metrics": {k: {"value": v} for k, v in {key: value, **extras}.items()},
    }]}}
    results, _, _ = extract_results(log, bid, "standard")
    assert compute_overall_score(results, "standard", bid) == value
    assert {r.metric_name: r.metric_value for r in results} == {
        f"{scorer}/{k}": v for k, v in {key: value, **extras}.items()
    }


@pytest.mark.parametrize("bid,scorer,key,value,extras", CASES)
@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "inf", "negative_inf"])
def test_invalid_accuracy_does_not_fall_back_to_other_metrics(bid, scorer, key, value, extras, problem):
    results = [metric(f"{scorer}/{k}", v) for k, v in extras.items()]
    if problem == "duplicate":
        results += [metric(f"{scorer}/{key}", value), metric(f"{scorer}/{key}", value)]
    elif problem != "missing":
        invalid = {"nan": float("nan"), "inf": float("inf"), "negative_inf": -float("inf")}
        results.append(metric(f"{scorer}/{key}", invalid[problem]))
    assert compute_overall_score(results, "standard", bid) is None


@pytest.mark.integration
@pytest.mark.parametrize("bid,scorer,key,value,extras", CASES)
@pytest.mark.parametrize("missing", [False, True])
def test_job_overall_and_confirmed_zero_shot_follow_accuracy(
    monkeypatch, job_spec_path, tmp_path, bid, scorer, key, value, extras, missing,
):
    raw = json.loads(Path(job_spec_path).read_text())
    raw["benchmark_id"] = bid
    raw.pop("exports", None)
    test_spec = tmp_path / "job.json"
    test_spec.write_text(json.dumps(raw))
    metrics = extras if missing else {key: value, **extras}
    log_file = tmp_path / "result.json"
    # Model a saved log with confirmed zero-shot metadata, not a task default.
    log_file.write_text(json.dumps({"status": "success",
        "eval": {"task_args": {"fewshot": 0}}, "results": {
        "total_samples": 3, "scores": [{
            "name": scorer, "scorer": scorer,
            "metrics": {k: {"value": v} for k, v in metrics.items()},
        }],
    }}))
    adapter = InspectAdapter(job_spec_path=str(test_spec))
    monkeypatch.setattr(main, "run_inspect", lambda *_: log_file)
    callbacks = MagicMock()
    callbacks.mlflow.save.return_value = None
    result = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    assert {r.metric_name: r.metric_value for r in result.results} == {
        f"{scorer}/{k}": v for k, v in metrics.items()
    }
    if missing:
        assert result.overall_score is None
        assert "zero_shot" not in result.additional_info
        assert result.primary_score is None
    else:
        assert result.overall_score == value
        assert result.additional_info["zero_shot"] == value
        assert result.primary_score is not None
        assert result.primary_score.metric == f"{scorer}/{key}"


def test_other_benchmark_aggregation_is_unchanged():
    results = [metric("strong_reject_scorer/jailbreak_rate", 0.1),
               metric("strong_reject_scorer/strong_reject_metric", 0.5)]
    assert compute_overall_score(results, "standard", "inspect/strong-reject") == 0.3


@pytest.mark.parametrize("bid", ["inspect/petri-sycophancy", "inspect/bloom-custom"])
def test_petri_bloom_primary_selection_is_unchanged(bid):
    results = [metric("concerning/mean", 3.2), metric("admirable/mean", 8)]
    assert compute_overall_score(results, "petri" if "petri" in bid else "bloom", bid) == 3.2


@pytest.mark.parametrize("bid", ["inspect/petri-sycophancy", "inspect/bloom-custom"])
def test_petri_bloom_missing_primary_retains_legacy_fallback(bid):
    results = [metric("a/mean", 2), metric("b/mean", 4), metric("a/stderr", 0.1)]
    assert compute_overall_score(results, "petri" if "petri" in bid else "bloom", bid) == 3


def test_resolve_primary_score_custom_unique_accuracy():
    """inspect/custom: pick the unique */accuracy scorer key for the event."""
    results = [
        metric("choice/accuracy", 0.2),
        metric("choice/stderr", 0.01),
    ]
    ps = resolve_primary_score(
        results,
        mode="standard",
        benchmark_id="inspect/custom",
        job_primary=PrimaryScore(metric="accuracy/accuracy"),
    )
    assert ps is not None
    assert ps.metric == "choice/accuracy"


def test_resolve_primary_score_prefers_job_when_present():
    results = [
        metric("choice/accuracy", 0.2),
        metric("match/accuracy", 0.9),
    ]
    ps = resolve_primary_score(
        results,
        mode="standard",
        benchmark_id="inspect/custom",
        job_primary=PrimaryScore(metric="match/accuracy"),
    )
    assert ps is not None
    assert ps.metric == "match/accuracy"


def test_resolve_primary_score_bfcl():
    results = [
        metric("bfcl_scorer/accuracy", 0.8),
        metric("bfcl_scorer/stderr", 0.1),
    ]
    ps = resolve_primary_score(
        results,
        mode="standard",
        benchmark_id="inspect/bfcl",
        job_primary=None,
    )
    assert ps is not None
    assert ps.metric == "bfcl_scorer/accuracy"


def test_resolve_primary_score_bfcl_missing_accuracy_does_not_use_secondary():
    """Missing bfcl_scorer/accuracy must not pick simple_python_acc as primary."""
    results = [
        metric("bfcl_scorer/stderr", 0.1),
        metric("bfcl_scorer/simple_python_acc", 0.6),
        metric("bfcl_scorer/simple_python_sterr", 0.1),
    ]
    assert (
        resolve_primary_score(
            results,
            mode="standard",
            benchmark_id="inspect/bfcl",
            job_primary=None,
        )
        is None
    )


@pytest.mark.parametrize("mode,bid", [("petri", "inspect/petri-escalation"), ("bloom", "inspect/bloom-custom")])
def test_resolve_primary_score_petri_bloom_lower_is_better(mode, bid):
    """Petri/Bloom fallback must mark concerning/mean as lower-is-better."""
    from _benchmarks import PETRI_BLOOM_PRIMARY

    results = [
        metric(PETRI_BLOOM_PRIMARY.metric, 0.8),
        metric("other/mean", 0.1),
    ]
    ps = resolve_primary_score(results, mode=mode, benchmark_id=bid, job_primary=None)
    assert ps is not None
    assert ps.metric == PETRI_BLOOM_PRIMARY.metric
    assert ps.lower_is_better is True


def test_resolve_primary_score_ambiguous_custom_returns_none():
    """Multiple non-accuracy scores: no unique primary without job primary_score."""
    results = [
        metric("choice/f1", 0.2),
        metric("match/em", 0.9),
    ]
    assert (
        resolve_primary_score(
            results,
            mode="standard",
            benchmark_id="inspect/custom",
            job_primary=None,
        )
        is None
    )
