"""Rendered prompts must match the pinned NVIDIA RULER protocol."""

import pytest

from main import RulerAdapter


# Official templates from NVIDIA/RULER commit
# c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, licensed under Apache 2.0:
# https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/data/synthetic/constants.py
@pytest.mark.parametrize(
    "task_id,query,expected",
    [
        (
            "cwe",
            "",
            "Below is a numbered list of words. In these words, some appear more often "
            "than others. Memorize the ones that appear most often.\n"
            "1. alpha 2. beta\n"
            "Question: What are the 10 most common words in the above list?",
        ),
        (
            "fwe",
            "",
            "Read the following coded text and track the frequency of each coded word. "
            "Find the three most frequently appeared coded words. 1. alpha 2. beta\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?",
        ),
        (
            "qa_1",
            "test question",
            "Answer the question based on the given documents. Only give me the answer "
            "and do not output any other words.\n\n"
            "The following are given documents.\n\n"
            "1. alpha 2. beta\n\n"
            "Answer the question based on the given documents. Only give me the answer "
            "and do not output any other words.\n\n"
            "Question: test question",
        ),
        (
            "qa_2",
            "test question",
            "Answer the question based on the given documents. Only give me the answer "
            "and do not output any other words.\n\n"
            "The following are given documents.\n\n"
            "1. alpha 2. beta\n\n"
            "Answer the question based on the given documents. Only give me the answer "
            "and do not output any other words.\n\n"
            "Question: test question",
        ),
    ],
    ids=["cwe", "fwe", "qa-squad", "qa-hotpotqa"],
)
def test_rendered_prompt_matches_official_protocol(task_id, query, expected):
    adapter = RulerAdapter.__new__(RulerAdapter)
    template = adapter._load_task_config(task_id)["template"]
    prompt = template.format(context="1. alpha 2. beta", query=query, num_cw=10)
    assert prompt == expected
