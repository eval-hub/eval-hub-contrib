from rules import (
    check_match,
    evaluate_rule_constraint,
    rule_evaluation_BBH_logical,
    rule_evaluation_CONLL2003,
    rule_evaluation_E2E,
    rule_evaluation_gigaword,
)


def test_rule_evaluation_e2e():
    assert rule_evaluation_E2E("expected", "expected", 1)
    assert not rule_evaluation_E2E("wrong", "expected", 1)


def test_rule_evaluation_conll2003():
    assert rule_evaluation_CONLL2003("cat", "['cat', 'dog']", 1)
    assert not rule_evaluation_CONLL2003("bird", "['cat', 'dog']", 1)


def test_rule_evaluation_bbh_logical():
    assert rule_evaluation_BBH_logical("The answer is (A)", "(A)", 1)
    assert not rule_evaluation_BBH_logical("The answer is (B)", "(A)", 1)


def test_rule_evaluation_gigaword():
    response = "one two three four five six seven eight"

    assert rule_evaluation_gigaword(response, "", 1)


def test_check_match():
    assert check_match("{{answer}}", "correct answer")
    assert not check_match("{{answer}}", "")


def test_evaluate_rule_constraint_returns_none_for_judge_sources():
    result = evaluate_rule_constraint(
        source="COGNAC",
        generation="answer",
        target="answer",
        level=1,
        example_id=1,
    )

    assert result is None