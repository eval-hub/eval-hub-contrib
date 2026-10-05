"""Export selected per-example answers and scores from LightEval details."""

from __future__ import annotations

import json
from pathlib import Path


def save_sample_results(
    source_dir: Path,
    results_dir: Path,
    benchmark_id: str,
) -> tuple[Path, int]:
    """Write only sample identity, task, model answers, and metrics to JSONL."""
    import pyarrow.parquet as parquet

    detail_files = sorted((source_dir / "details").rglob("*.parquet"))
    if not detail_files:
        raise RuntimeError("LightEval --save-details produced no sample Parquet files")

    target = results_dir / "sample_results.jsonl"
    count = 0
    with target.open("w", encoding="utf-8") as stream:
        for detail_file in detail_files:
            for row in parquet.read_table(detail_file).to_pylist():
                doc = row.get("doc") or {}
                response = row.get("model_response") or {}
                answers = response.get("text") or []
                if isinstance(answers, str):
                    answers = [answers]
                record = {
                    "sample_index": count,
                    "task": doc.get("task_name") or benchmark_id,
                    "answers": answers,
                    "metric": row.get("metric"),
                }
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                count += 1

    if count == 0:
        target.unlink()
        raise RuntimeError("LightEval detail Parquet files contained no examples")
    return target, count
