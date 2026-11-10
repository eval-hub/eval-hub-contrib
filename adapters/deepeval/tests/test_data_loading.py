"""Dataset discovery and reading."""

import json
from types import SimpleNamespace

import pytest

from deepeval_adapter.data_loading import load_dataset, resolve_data_dir


def test_csv_files_are_read_in_name_order(tmp_path):
    (tmp_path / "b.csv").write_text("input,actual_output\nq2,a2\n")
    (tmp_path / "a.csv").write_text("input,actual_output\nq1,a1\n")
    records = load_dataset(str(tmp_path), "csv")
    assert [r["input"] for r in records] == ["q1", "q2"]


def test_jsonl_skips_blank_lines(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"input": "q1"}\n\n  \n{"input": "q2"}\n')
    assert [r["input"] for r in load_dataset(str(tmp_path), "jsonl")] == ["q1", "q2"]


def test_json_accepts_a_list_or_a_single_object(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"input": "q1"}, {"input": "q2"}]))
    (tmp_path / "b.json").write_text(json.dumps({"input": "q3"}))
    assert len(load_dataset(str(tmp_path), "json")) == 3


def test_unsupported_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset_format: 'xml'"):
        load_dataset(str(tmp_path), "xml")


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "json"])
def test_missing_files_name_the_format(tmp_path, fmt):
    with pytest.raises(ValueError, match=f"No {fmt.upper()} files found"):
        load_dataset(str(tmp_path), fmt)


def test_empty_files_yield_no_records(tmp_path):
    (tmp_path / "d.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="No records loaded"):
        load_dataset(str(tmp_path), "jsonl")


def test_data_dir_parameter_is_used_when_no_mount_exists(tmp_path):
    config = SimpleNamespace(parameters={"data_dir": str(tmp_path)})
    # /test_data and /data are absent on a development machine.
    assert resolve_data_dir(config) == str(tmp_path)
