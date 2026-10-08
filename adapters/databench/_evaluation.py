"""DataBench direct-answer table QA with the official comparison function."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from databench_eval import Evaluator
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

DATASET = "cardiffnlp/databench"
REVISION = "e75d53add267d2f9cfa32efd65ad77f0807adfad"
BENCHMARKS = {"databench": False, "databench_lite": True}


def load_questions(
    limit: int | None, table_id: str | None, token: str | None
) -> Dataset:
    qa = load_dataset(DATASET, "qa", split="train", revision=REVISION, token=token)
    if table_id:
        qa = qa.filter(lambda row: row["dataset"] == table_id)
    if limit is not None:
        qa = qa.select(range(min(limit, len(qa))))
    if not len(qa):
        raise ValueError("No DataBench questions were selected")
    return qa


def load_table(table_id: str, lite: bool, token: str | None) -> pd.DataFrame:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", table_id):
        raise ValueError("Invalid DataBench table ID")
    filename = f"data/{table_id}/{'sample' if lite else 'all'}.parquet"
    path = hf_hub_download(
        DATASET, filename, repo_type="dataset", revision=REVISION, token=token
    )
    return pd.read_parquet(path)


def build_prompt(row: dict[str, Any], table: pd.DataFrame, max_chars: int) -> str:
    # Keep the entire selected table. Never silently replace the full-table task
    # with a truncated-table evaluation or use columns_used (gold annotation).
    csv = table.to_csv(index=False)
    prompt = (
        "Answer the question using only the CSV table below. Return only the answer, "
        "without explanations or code. For booleans return True or False; for numbers "
        "return a number; for categories return the value; for lists return a list of values.\n"
        f"Answer type: {row['type']}\nCSV table:\n{csv}\nQuestion: {row['question']}\nAnswer:"
    )
    if len(prompt) > max_chars:
        raise ValueError(
            f"Table {row['dataset']} prompt exceeds max_prompt_chars={max_chars}; "
            "increase the limit for a compatible endpoint or select databench_lite"
        )
    return prompt


def make_evaluator(qa: Dataset) -> Evaluator:
    return Evaluator(qa=qa)


def score_answer(
    evaluator: Evaluator, response: str, row: dict[str, Any], lite: bool
) -> bool:
    truth = row["sample_answer" if lite else "answer"]
    return bool(evaluator.default_compare(response, truth, row["type"]))
