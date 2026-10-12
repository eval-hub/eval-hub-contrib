"""Per-example exports contain answers and scores, not raw benchmark records."""

import json

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from sample_results import save_sample_results


def _write_detail(root, task, row):
    path = root / "details" / "model" / f"details_{task}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist([row]), path)


def test_exports_answers_and_scores_without_raw_benchmark_data(tmp_path):
    _write_detail(
        tmp_path,
        "gpqa",
        {
            "doc": {
                "query": "Question with options",
                "task_name": "gpqa:diamond",
                "specific": {"private_tests": ["hidden input and output"]},
                "gold_index": 2,
            },
            "model_response": {"text": ["Answer: B"]},
            "metric": {"gpqa_pass@1": 0},
        },
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    target, count = save_sample_results(tmp_path, results_dir, "gpqa")

    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert count == 1
    assert records == [{
        "sample_index": 0,
        "task": "gpqa:diamond",
        "answers": ["Answer: B"],
        "metric": {"gpqa_pass@1": 0},
    }]
    assert "hidden input and output" not in target.read_text()
    assert "Question with options" not in target.read_text()
    assert "gold_index" not in target.read_text()


def test_missing_details_fail_explicitly(tmp_path):
    with pytest.raises(RuntimeError, match="no sample Parquet files"):
        save_sample_results(tmp_path, tmp_path, "gpqa:diamond")
