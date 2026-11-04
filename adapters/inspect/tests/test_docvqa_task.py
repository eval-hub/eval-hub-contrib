"""Tests for the EvalHub DocVQA task's early sample selection."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _load_task_module():
    """Import the task module with lightweight stand-ins for runtime packages."""
    inspect_ai = ModuleType("inspect_ai")
    inspect_ai.__path__ = []
    inspect_ai.Task = lambda **kwargs: kwargs
    inspect_ai.task = lambda function: function

    inspect_dataset = ModuleType("inspect_ai.dataset")
    inspect_dataset.MemoryDataset = lambda **kwargs: kwargs

    datasets = ModuleType("datasets")
    datasets.Image = lambda decode=True: SimpleNamespace(decode=decode)

    inspect_evals = ModuleType("inspect_evals")
    inspect_evals.__path__ = []
    inspect_evals_utils = ModuleType("inspect_evals.utils")
    inspect_evals_utils.__path__ = []
    inspect_evals_huggingface = ModuleType("inspect_evals.utils.huggingface")
    inspect_evals_huggingface.load_dataset = None
    docvqa_package = ModuleType("inspect_evals.docvqa")
    docvqa_package.__path__ = []
    docvqa = ModuleType("inspect_evals.docvqa.docvqa")
    docvqa.DATASET_PATH = "lmms-lab-encoder/DocVQA"
    docvqa.DOCVQA_DATASET_REVISION = "pinned-revision"
    docvqa.EVAL_VERSION = SimpleNamespace(
        comparability_version="test", to_metadata=lambda: {}
    )
    docvqa.docvqa_scorer = lambda: "scorer"
    docvqa.docvqa_solver = lambda: "solver"
    docvqa.record_to_sample = lambda record: record

    modules = {
        "inspect_ai": inspect_ai,
        "inspect_ai.dataset": inspect_dataset,
        "datasets": datasets,
        "inspect_evals": inspect_evals,
        "inspect_evals.utils": inspect_evals_utils,
        "inspect_evals.utils.huggingface": inspect_evals_huggingface,
        "inspect_evals.docvqa": docvqa_package,
        "inspect_evals.docvqa.docvqa": docvqa,
    }
    path = Path(__file__).resolve().parents[1] / "evalhub_docvqa.py"
    spec = importlib.util.spec_from_file_location("evalhub_docvqa_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


evalhub_docvqa = _load_task_module()


class TrackingDataset:
    """Small Hugging Face Dataset stand-in that records materialized rows."""

    def __init__(self, records):
        self.records = records
        self.selected = records
        self.operations = []

    def __len__(self):
        return len(self.selected)

    def cast_column(self, name, feature):
        self.operations.append(("cast_column", name, feature.decode))
        return self

    def shuffle(self, seed=None):
        self.operations.append(("shuffle", seed))
        return self

    def select(self, indices):
        indices = list(indices)
        self.operations.append(("select", indices))
        self.selected = [self.selected[index] for index in indices]
        return self

    def __iter__(self):
        self.operations.append("iterate")
        return iter(self.selected)


@pytest.fixture
def task_constructors(monkeypatch):
    """Replace Inspect constructors so tests focus on dataset access ordering."""
    monkeypatch.setattr(evalhub_docvqa, "Task", lambda **kwargs: kwargs)
    monkeypatch.setattr(evalhub_docvqa, "MemoryDataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(evalhub_docvqa, "docvqa_solver", lambda: "solver")
    monkeypatch.setattr(evalhub_docvqa, "docvqa_scorer", lambda: "scorer")
    monkeypatch.setattr(
        evalhub_docvqa,
        "EVAL_VERSION",
        SimpleNamespace(comparability_version="test", to_metadata=lambda: {}),
    )


def test_docvqa_selects_capped_rows_before_converting_records(monkeypatch, task_constructors):
    dataset = TrackingDataset([{"id": i} for i in range(5_349)])
    converted = []
    calls = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(evalhub_docvqa, "load_dataset", load_dataset)
    monkeypatch.setattr(
        evalhub_docvqa,
        "record_to_sample",
        lambda record: converted.append(record["id"]) or record,
    )

    result = evalhub_docvqa.build_docvqa_task(num_examples=1)

    assert dataset.operations == [
        ("cast_column", "image", False),
        ("shuffle", None),
        ("select", [0]),
        "iterate",
    ]
    assert converted == [0]
    assert result["dataset"]["samples"] == [{"id": 0}]
    assert calls == [{
        "path": "lmms-lab-encoder/DocVQA",
        "name": "DocVQA",
        "split": "validation",
        "revision": "pinned-revision",
    }]


def test_docvqa_caps_at_dataset_length(monkeypatch, task_constructors):
    dataset = TrackingDataset([{"id": i} for i in range(2)])
    monkeypatch.setattr(evalhub_docvqa, "load_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(evalhub_docvqa, "record_to_sample", lambda record: record)

    result = evalhub_docvqa.build_docvqa_task(num_examples=10, shuffle=False)

    assert dataset.operations == [("cast_column", "image", False), ("select", [0, 1]), "iterate"]
    assert result["dataset"]["samples"] == [{"id": 0}, {"id": 1}]


@pytest.mark.parametrize("num_examples", [0, -1])
def test_docvqa_rejects_non_positive_sample_limit(
    monkeypatch, task_constructors, num_examples
):
    dataset = TrackingDataset([{"id": 0}])
    monkeypatch.setattr(evalhub_docvqa, "load_dataset", lambda **kwargs: dataset)

    with pytest.raises(ValueError, match="num_examples must be a positive integer"):
        evalhub_docvqa.build_docvqa_task(num_examples=num_examples, shuffle=False)

    assert dataset.operations == []
