"""Offline, read-only non-root image import/scorer smoke."""

import sys

sys.path.insert(0, "/app")

import pandas as pd
from datasets import Dataset
from evalhub.adapter import JobSpec

from _evaluation import build_prompt, make_evaluator, score_answer
from main import DataBenchAdapter

row = {
    "dataset": "001_Test",
    "question": "How many rows?",
    "type": "number",
    "answer": "100",
    "sample_answer": "2",
}
qa = Dataset.from_list([row])
assert score_answer(make_evaluator(qa), "2", row, True)
assert not score_answer(make_evaluator(qa), "2", row, False)
assert "A\n1\n2" in build_prompt(row, pd.DataFrame({"A": [1, 2]}), 10000)
assert JobSpec.from_file("/meta/job.json").benchmark_id == "databench_lite"
assert DataBenchAdapter is not None
print("Offline non-root DataBench imports and official scoring passed")
