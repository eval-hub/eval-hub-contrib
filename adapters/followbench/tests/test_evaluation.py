from pathlib import Path

import pytest

from _evaluation import (
    ScoredConstraint,
    compute_metrics,
    load_examples,
    parse_judge_result,
)


DATA_DIR = Path(__file__).parents[1] / "data"


def test_load_examples():
    examples = load_examples(DATA_DIR)

    assert len(examples) == 944
    assert {example.category for example in examples} == {
        "content",
        "example",
        "format",
        "mixed",
        "situation",
        "style",
    }


def test_load_examples_have_required_fields():
    examples = load_examples(DATA_DIR)

    for example in examples:
        assert example.example_id > 0
        assert example.category
        assert example.source
        assert example.level >= 0

        if example.level > 0:
            assert example.instruction


def test_parse_single_judge_result():
    assert parse_judge_result("YES", level=1) == (1, 1.0)
    assert parse_judge_result("NO", level=1) == (0, 0.0)


def test_parse_multiple_judge_results():
    assert parse_judge_result("['YES', 'NO']", level=2) == (0, 0.5)
    assert parse_judge_result("['YES', 'YES']", level=2) == (1, 1.0)
    assert parse_judge_result("[YES, NO]", level=2) == (0, 0.5)


def test_parse_judge_result_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_judge_result("['YES']", level=2)


def test_parse_judge_result_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_judge_result("['YES', 'INVALID']", level=2)


def test_compute_metrics():
    results = [
        ScoredConstraint(1, 1, True, 1.0),
        ScoredConstraint(1, 2, False, 0.5),
        ScoredConstraint(1, 3, True, 1.0),
        ScoredConstraint(2, 1, True, 1.0),
        ScoredConstraint(2, 2, True, 1.0),
    ]

    metrics = compute_metrics(results)

    assert metrics["hsr"] == pytest.approx(0.8)
    assert metrics["ssr"] == pytest.approx(0.9)
    assert metrics["csl"] == pytest.approx(1.5)
    assert metrics["n_evaluated"] == pytest.approx(5.0)


def test_compute_metrics_keeps_categories_as_distinct_groups():
    results = [
        *[
            ScoredConstraint(1, level, True, 1.0, group_id="content:1")
            for level in range(1, 6)
        ],
        ScoredConstraint(1, 1, True, 1.0, group_id="style:1"),
    ]

    metrics = compute_metrics(results)

    assert metrics["csl"] == pytest.approx(3.0)
