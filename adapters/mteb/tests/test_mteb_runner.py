"""Tests for sample limiting and batch-size forwarding in the MTEB runner."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mteb_runner


class Dataset:
    """Small stand-in for a Hugging Face Dataset used by unit tests."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    @classmethod
    def from_dict(cls, columns):
        row_count = len(next(iter(columns.values()), []))
        return cls(
            [
                {column: values[index] for column, values in columns.items()}
                for index in range(row_count)
            ]
        )

    def select(self, indices):
        return Dataset([self.rows[index] for index in indices])

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        return [row[key] for row in self.rows]

    def __len__(self):
        return len(self.rows)


class DatasetDict(dict):
    """Mapping stand-in for the Hugging Face DatasetDict."""


class FakeTask:
    def __init__(self, dataset, name="FakeTask", eval_splits=None, task_type=None):
        self.dataset = dataset
        self.metadata = SimpleNamespace(
            name=name,
            eval_splits=eval_splits or ["test"],
            type=task_type,
        )
        self._source_dataset = dataset
        self.loaded = False

    def load_data(self):
        if not self.loaded:
            self.dataset = self._source_dataset
        self.loaded = True

    def unload_data(self):
        self.dataset = None
        self.loaded = False


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


def test_limit_task_supports_subset_then_split_dataset_layout():
    task = FakeTask(
        {
            "subset-a": DatasetDict(
                {"test": Dataset.from_dict({"text": ["a", "b", "c"]})}
            )
        },
        name="NestedTask",
    )

    count = mteb_runner._limit_task(task, 2)

    assert count == 2
    assert len(task.dataset["subset-a"]["test"]) == 2


def test_limit_task_errors_when_metadata_evaluation_split_is_missing():
    task = FakeTask(
        {
            "complete-subset": DatasetDict(
                {"test": Dataset.from_dict({"text": ["a", "b"]})}
            ),
            "missing-subset": DatasetDict(
                {"validation": Dataset.from_dict({"text": ["c"]})}
            ),
        },
        name="MissingSplitTask",
        eval_splits=["test"],
    )

    with pytest.raises(ValueError, match="MissingSplitTask.*validation"):
        mteb_runner._limit_task(task, 1)


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
                    "relevant_docs": {
                        "q1": {"d1": 1},
                        "q2": {"d2": 1},
                        "q3": {"d3": 1},
                    },
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


def test_limit_retrieval_task_samples_only_queries_with_qrels():
    queries = Dataset.from_dict(
        {"id": ["unjudged", "q1", "q2"], "text": ["x", "one", "two"]}
    )
    task = FakeTask(
        {
            "default": {
                "test": {
                    "corpus": Dataset.from_dict({"id": ["d1"], "text": ["doc"]}),
                    "queries": queries,
                    "relevant_docs": {"q1": {"d1": 1}, "q2": {"d1": 1}},
                }
            }
        },
        name="PartialQrelsRetrieval",
    )

    count = mteb_runner._limit_task(task, 1)
    split = task.dataset["default"]["test"]

    assert count == 1
    assert split["queries"][0]["id"] == "q1"
    assert set(split["relevant_docs"]) == {"q1"}


def test_limit_task_reports_unsupported_data_structure():
    task = FakeTask({"test": ["not", "a", "dataset"]}, name="UnsupportedTask")

    with pytest.raises(ValueError, match="UnsupportedTask.*unsupported data structure"):
        mteb_runner._limit_task(task, 1)


def test_sts_task_rejects_one_example_with_actionable_error():
    task = FakeTask(
        DatasetDict({"test": Dataset.from_dict({"text": ["one", "two"]})}),
        name="STSBenchmark",
        task_type="STS",
    )

    with pytest.raises(
        ValueError,
        match="STSBenchmark.*requires num_examples >= 2.*correlation metrics",
    ):
        mteb_runner._limit_task(task, 1)

    assert not task.loaded


def test_run_forwards_batch_size_and_writes_effective_sampling_metadata(
    tmp_path, monkeypatch
):
    tasks = [
        FakeTask(
            DatasetDict(
                {
                    "test": Dataset.from_dict({"text": ["one", "two", "three"]}),
                }
            )
        ),
        FakeTask(
            DatasetDict({"test": Dataset.from_dict({"text": ["four", "five"]})}),
            name="SecondTask",
        ),
    ]
    evaluate_kwargs = {}

    def evaluate(_model, received_tasks, **_kwargs):
        evaluate_kwargs.update(_kwargs)
        for current_task in received_tasks:
            assert all(not task.loaded for task in tasks)
            current_task.load_data()
            assert current_task.loaded
            assert all(not task.loaded for task in tasks if task is not current_task)
            current_task.unload_data()

    model_kwargs = {}

    def get_model(*_args, **kwargs):
        model_kwargs.update(kwargs)
        return object()

    fake_mteb = SimpleNamespace(
        get_tasks=lambda **kwargs: tasks,
        get_model=get_model,
        ResultCache=lambda **kwargs: object(),
        evaluate=evaluate,
    )
    monkeypatch.setitem(sys.modules, "mteb", fake_mteb)

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
                "device": "cuda:0",
                "output_dir": str(output_dir),
                "verbosity": 0,
            }
        ),
        encoding="utf-8",
    )

    mteb_runner.run(config_path)

    assert evaluate_kwargs["encode_kwargs"] == {"batch_size": 7}
    assert model_kwargs["device"] == "cuda:0"
    sampling_info = json.loads(
        (output_dir / "evalhub_sampling_info.json").read_text(encoding="utf-8")
    )
    assert sampling_info == {
        "num_examples_by_task": {"FakeTask": 2, "SecondTask": 2},
        "total_examples_evaluated": 4,
        "num_tasks_evaluated": 2,
    }
    assert all(not task.loaded for task in tasks)
