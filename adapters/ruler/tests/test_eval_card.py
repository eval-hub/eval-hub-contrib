"""EvalCard scores must describe the prompting used by the RULER generators."""

import pytest

from evalhub.adapter import EvaluationResult
from main import BENCHMARK_TO_TASKS, RulerAdapter


@pytest.fixture
def adapter():
    return RulerAdapter.__new__(RulerAdapter)


def task_result(adapter, task_id, value):
    return EvaluationResult(
        metric_name=f"{task_id}.overall",
        metric_value=value,
        metadata={"task_id": task_id, "category": adapter._get_category(task_id)},
    )


@pytest.mark.parametrize(
    "benchmark_id,task_id",
    [(benchmark, tasks[0]) for benchmark, tasks in BENCHMARK_TO_TASKS.items()],
)
def test_all_supported_tasks_report_their_prompting(adapter, benchmark_id, task_id):
    card = adapter._build_eval_card(
        benchmark_id, [task_result(adapter, task_id, 0.75)], [4096],
    )
    entries = card.model_dump(exclude_none=True)["capability_evaluations"]
    assert len(entries) == 1
    entry = entries[0]
    if task_id in ("cwe", "vt"):
        assert "zero_shot" not in entry
        assert entry["alt_prompting"] == 0.75
        assert entry["alt_prompting_description"] == "1-shot prompting (worked examples)"
    else:
        assert entry["zero_shot"] == 0.75
        assert "alt_prompting" not in entry
        assert "alt_prompting_description" not in entry
    assert "All tasks use zero-shot" not in card.developer_footnotes


def test_mixed_aggregation_keeps_zero_shot_and_one_shot_separate(adapter):
    # Score real prediction lists, then check their serialized EvalCard.
    results = adapter._evaluate_predictions({
        "cwe": {4096: [{"pred": "apple", "outputs": ["apple", "banana"]}]},
        "fwe": {4096: [{"pred": "apple banana", "outputs": ["apple", "banana"]}]},
        "vt": {4096: [{"pred": "A", "outputs": ["A", "B", "C", "D"]}]},
    })
    before = [result.model_dump() for result in results]
    card = adapter._build_eval_card("common-words-extraction", results, [4096])
    entries = card.model_dump(exclude_none=True)["capability_evaluations"]
    aggregation = [entry for entry in entries if entry["ability"] == "Long-context aggregation"]
    assert len(aggregation) == 2
    zero_shot = next(entry for entry in aggregation if "zero_shot" in entry)
    few_shot = next(entry for entry in aggregation if "alt_prompting" in entry)
    assert zero_shot["zero_shot"] == 1.0
    assert "alt_prompting" not in zero_shot
    assert few_shot["alt_prompting"] == 0.5
    assert "zero_shot" not in few_shot
    assert [result.model_dump() for result in results] == before
    assert adapter._compute_overall_score(results) == pytest.approx((0.5 + 1.0 + 0.25) / 3)


def test_zero_shot_tasks_in_same_category_still_average(adapter):
    results = [task_result(adapter, "qa_1", 0.2), task_result(adapter, "qa_2", 0.8)]
    card = adapter._build_eval_card("qa-squad", results, [4096, 8192])
    entries = card.model_dump(exclude_none=True)["capability_evaluations"]
    assert len(entries) == 1
    assert entries[0]["zero_shot"] == 0.5
    assert "alt_prompting" not in entries[0]


@pytest.mark.parametrize("num_shots", [0, 3])
def test_cwe_follows_configured_example_count(adapter, monkeypatch, num_shots):
    config = adapter._load_task_config("cwe")
    config = {**config, "args": {**config["args"], "num_fewshot": num_shots}}
    monkeypatch.setattr(adapter, "_load_task_config", lambda _: config)
    card = adapter._build_eval_card(
        "common-words-extraction", [task_result(adapter, "cwe", 0.7)], [4096],
    )
    entry = card.model_dump(exclude_none=True)["capability_evaluations"][0]
    if num_shots == 0:
        assert entry["zero_shot"] == 0.7
        assert "alt_prompting" not in entry
    else:
        assert "zero_shot" not in entry
        assert entry["alt_prompting"] == 0.7
        assert entry["alt_prompting_description"] == "3-shot prompting (worked examples)"


@pytest.mark.parametrize("task_id", ["cwe", "vt"])
def test_zero_valued_few_shot_score_is_retained(adapter, task_id):
    card = adapter._build_eval_card(task_id, [task_result(adapter, task_id, 0.0)], [4096])
    entry = card.model_dump(exclude_none=True)["capability_evaluations"][0]
    assert entry["alt_prompting"] == 0.0
    assert "zero_shot" not in entry


def test_task_name_can_be_recovered_from_metric_key(adapter):
    result = EvaluationResult(metric_name="vt.overall", metric_value=0.4)
    card = adapter._build_eval_card("variable-tracking", [result], [4096])
    entry = card.model_dump(exclude_none=True)["capability_evaluations"][0]
    assert entry["ability"] == "Long-context variable tracking"
    assert entry["alt_prompting"] == 0.4
    assert "zero_shot" not in entry


def test_context_scores_are_not_counted_again_in_card(adapter):
    results = [
        task_result(adapter, "cwe", 0.25),
        EvaluationResult(metric_name="cwe.ctx_4096.score", metric_value=1.0),
    ]
    card = adapter._build_eval_card("common-words-extraction", results, [4096])
    entry = card.model_dump(exclude_none=True)["capability_evaluations"][0]
    assert entry["alt_prompting"] == 0.25
    assert "zero_shot" not in entry
