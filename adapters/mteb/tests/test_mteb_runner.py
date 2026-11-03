"""Tests for sample limiting and batch-size forwarding in the MTEB runner."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict

import mteb_runner


class FakeTask:
    def __init__(self, dataset, name="FakeTask", eval_splits=None):
        self.dataset = dataset
        self.metadata = SimpleNamespace(
            name=name,
            eval_splits=eval_splits or ["test"],
        )

    def load_data(self):
        pass


def test_limit_task_caps_evaluation_split_and_preserves_training_data():
    task = FakeTask(
        DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["train-1", "train-2"]}),
                "test": Dataset.from_dict({"text": ["test-1", "test-2", "test-3"]}),
            }
        )
    )

    count = mteb_runner._limit_task(task, 2)

    assert count == 2
    assert len(task.dataset["train"]) == 2
    assert len(task.dataset["test"]) == 2


def test_limit_retrieval_task_keeps_corpus_and_matching_relevance_data():
    corpus = Dataset.from_dict({"id": ["d1", "d2", "d3"], "text": ["a", "b", "c"]})
    queries = Dataset.from_dict(
        {"id": ["q1", "q2", "q3"], "text": ["one", "two", "three"]}
    )
    task = FakeTask(
        {
            "default": {
                "test": {
                    "corpus": corpus,
                    "queries": queries,
                    "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}},
                    "top_ranked": {"q1": ["d1"], "q2": ["d2"], "q3": ["d3"]},
                }
            }
        },
        name="FakeRetrieval",
    )

    count = mteb_runner._limit_task(task, 1)
    split = task.dataset["default"]["test"]

    assert count == 1
    assert len(split["queries"]) == 1
    assert split["queries"][0]["id"] == "q1"
    assert split["relevant_docs"] == {"q1": {"d1": 1}}
    assert split["top_ranked"] == {"q1": ["d1"]}
    assert len(split["corpus"]) == 3


def test_limit_task_reports_unsupported_data_structure():
    task = FakeTask({"test": ["not", "a", "dataset"]}, name="UnsupportedTask")

    with pytest.raises(ValueError, match="UnsupportedTask.*unsupported data structure"):
        mteb_runner._limit_task(task, 1)


def test_run_forwards_batch_size_and_writes_effective_sampling_metadata(
    tmp_path, monkeypatch
):
    task = FakeTask(
        DatasetDict(
            {
                "test": Dataset.from_dict({"text": ["one", "two", "three"]}),
            }
        )
    )
    evaluate = Mock()
    monkeypatch.setattr(mteb_runner.mteb, "get_tasks", lambda **kwargs: [task])
    monkeypatch.setattr(mteb_runner.mteb, "get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(mteb_runner.mteb, "ResultCache", lambda **kwargs: object())
    monkeypatch.setattr(mteb_runner.mteb, "evaluate", evaluate)

    output_dir = tmp_path / "results"
    output_dir.mkdir()
    config_path = tmp_path / "runner.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "tasks": ["FakeTask"],
                "num_examples": 2,
                "batch_size": 7,
                "output_dir": str(output_dir),
                "verbosity": 0,
            }
        ),
        encoding="utf-8",
    )

    mteb_runner.run(config_path)

    assert evaluate.call_args.kwargs["encode_kwargs"] == {"batch_size": 7}
    sampling_info = json.loads(
        (output_dir / "evalhub_sampling_info.json").read_text(encoding="utf-8")
    )
    assert sampling_info == {
        "num_examples_by_task": {"FakeTask": 2},
        "num_examples_evaluated": 2,
    }
