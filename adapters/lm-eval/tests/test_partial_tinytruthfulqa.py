"""Preserve sample execution without inventing a partial tinyTruthfulQA score."""

import pytest


def run_tiny(adapter, mock_callbacks, monkeypatch, count, score, requested=None):
    adapter.job_spec.benchmark_id = "tinyTruthfulQA"
    adapter.job_spec.num_examples = count if requested is None else requested
    adapter.job_spec.parameters["tokenizer"] = "/test_data/tokenizer"
    raw = {
        "results": {"tinyTruthfulQA": {"acc,none": score}},
        "n-samples": {"tinyTruthfulQA": {"original": 100, "effective": count}},
        "n-shot": {"tinyTruthfulQA": 0},
    }
    monkeypatch.setattr(adapter, "_run_lmeval", lambda *args: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: raw)
    return adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)


@pytest.mark.integration
@pytest.mark.parametrize("count", [1, 99, 101])
@pytest.mark.parametrize("score", [None, 0.25])
def test_partial_tinytruthfulqa_is_unscored(
    adapter, mock_callbacks, monkeypatch, caplog, count, score
):
    caplog.set_level("INFO", logger="main")
    result = run_tiny(adapter, mock_callbacks, monkeypatch, count, score)
    assert result.results == []
    assert result.overall_score is None
    assert result.num_examples_evaluated == count
    assert result.additional_info["num_examples_evaluated"] == count
    assert result.additional_info["aggregation"]["status"] == "skipped"
    assert result.additional_info["aggregation"]["required_examples"] == 100
    assert "100" in result.additional_info["aggregation"]["reason"]
    assert not result.eval_card.capability_evaluations
    assert "score=unscored" in caplog.text


@pytest.mark.integration
def test_complete_tinytruthfulqa_keeps_official_score(
    adapter, mock_callbacks, monkeypatch
):
    result = run_tiny(adapter, mock_callbacks, monkeypatch, 100, 0.75)
    assert result.overall_score == pytest.approx(0.75)
    assert result.num_examples_evaluated == 100
    assert result.additional_info is None
    assert result.eval_card.capability_evaluations


@pytest.mark.integration
def test_tinytruthfulqa_without_outcomes_fails(adapter, mock_callbacks, monkeypatch):
    with pytest.raises(RuntimeError, match="no completed sample outcomes"):
        run_tiny(adapter, mock_callbacks, monkeypatch, 0, None)


@pytest.mark.integration
def test_actual_outcomes_determine_partial_aggregation(
    adapter, mock_callbacks, monkeypatch
):
    result = run_tiny(adapter, mock_callbacks, monkeypatch, 1, None, requested=100)
    assert result.num_examples_evaluated == 1
    assert result.additional_info["aggregation"]["status"] == "skipped"


@pytest.mark.integration
def test_missing_completed_count_is_not_replaced_by_original_size(
    adapter, mock_callbacks, monkeypatch
):
    with pytest.raises(TypeError, match="missing a valid completed sample count"):
        run_tiny(adapter, mock_callbacks, monkeypatch, None, None)


@pytest.mark.integration
def test_partial_tinytruthfulqa_preserves_other_task_results(
    adapter, mock_callbacks, monkeypatch
):
    adapter.job_spec.benchmark_id = "lm-eval/custom"
    adapter.job_spec.parameters.update(
        task="tinyTruthfulQA,arc_easy", tokenizer="/test_data/tokenizer"
    )
    raw = {
        "results": {
            "tinyTruthfulQA": {"acc,none": None},
            "arc_easy": {"acc,none": 0.5},
        },
        "n-samples": {
            "tinyTruthfulQA": {"original": 100, "effective": 1},
            "arc_easy": {"original": 100, "effective": 2},
        },
    }
    monkeypatch.setattr(adapter, "_run_lmeval", lambda *args: None)
    monkeypatch.setattr(adapter, "_parse_results", lambda _: raw)
    result = adapter.run_benchmark_job(adapter.job_spec, mock_callbacks)
    assert result.overall_score == 0.5
    assert result.num_examples_evaluated == 3
    assert all((r.metadata or {}).get("task") == "arc_easy" for r in result.results)
    assert result.additional_info["aggregation"]["task"] == "tinyTruthfulQA"
