"""HLE/BFCL representative-score regressions; all individual metrics remain."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from evalhub.adapter import EvaluationResult

import main
from _results import compute_overall_score, extract_results
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
    else:
        assert result.overall_score == value
        assert result.additional_info["zero_shot"] == value


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
