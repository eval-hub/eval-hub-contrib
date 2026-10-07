"""Retained per-example scores use the real upstream RULER scorers."""
import json
import shutil

import pytest
import main


@pytest.mark.parametrize("task_type,expected", [
    ("variable_tracking", [0.5, 1.0, 0.0]),
    ("niah", [0.5, 1.0, 0.0]),
    ("common_words_extraction", [0.5, 1.0, 0.0]),
    ("freq_words_extraction", [0.5, 1.0, 0.0]),
    ("qa", [1.0, 1.0, 0.0]),
])
def test_saved_diagnostics_survive_cleanup(tmp_path, monkeypatch, task_type, expected):
    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    monkeypatch.setattr(type(adapter), "local_jobs_base_path", property(lambda self: tmp_path / "job"))
    monkeypatch.setattr(adapter, "_load_task_config", lambda task: {"task": task_type})
    work_dir = tmp_path / "temporary"
    work_dir.mkdir()
    samples = [
        {"index": 0, "input": "Find α and B", "outputs": ["α", "B"], "pred": "α"},
        {"index": 1, "input": "Find α and B", "outputs": ["α", "B"], "pred": "α b"},
        {"index": 2, "input": "Find α and B", "outputs": ["α", "B"], "pred": ""},
    ]
    raw = {"vt": {4096: samples, 8192: samples}}
    evaluated = adapter._evaluate_predictions(raw)
    files = adapter._save_results("job1", "variable-tracking", "model", evaluated, work_dir, raw)
    shutil.rmtree(work_dir)
    assert {p.name for p in files} == {"summary.csv", "results.json", "samples.jsonl"}
    assert all(p.is_file() for p in files)
    assert len({p.parent for p in files}) == 1  # OCI exports this directory.
    rows = [json.loads(line) for line in (files[0].parent / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert [r["score"] for r in rows[:3]] == expected
    assert rows[0]["reference_matches"] == [True, False]
    for i, row in enumerate(rows):
        assert row["task_id"] == "vt"
        assert row["context_length"] == (4096 if i < 3 else 8192)
        assert row["metric_name"] == f"vt.ctx_{row['context_length']}.score"
        assert row["scorer"] == ("string_match_part" if task_type == "qa" else "string_match_all")
        for key, value in samples[i % 3].items():
            assert row[key] == value
    aggregate = next(r.metric_value for r in evaluated if r.metric_name == "vt.ctx_4096.score")
    assert sum(expected) / 3 == pytest.approx(aggregate, abs=0.0001)
