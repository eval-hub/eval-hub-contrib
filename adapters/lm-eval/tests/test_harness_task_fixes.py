"""Exercise the patched task code from the actual pinned Harness package."""

import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.apply_harness_fixes import apply_fixes


def test_unreviewed_version_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="review or remove"):
        apply_fixes(tmp_path, "0.4.14")
    assert not list(tmp_path.iterdir())


def test_changed_dependency_source_is_rejected(tmp_path):
    path = tmp_path / "tasks/careqa/utils_open.py"
    path.parent.mkdir(parents=True)
    path.write_text("unexpected upstream change\n")
    with pytest.raises(RuntimeError, match="Unexpected Harness source"):
        apply_fixes(tmp_path, "0.4.13")
    assert path.read_text() == "unexpected upstream change\n"


def task_file(relative):
    return Path(distribution("lm_eval").locate_file("lm_eval/tasks/" + relative))


def task_module(relative):
    path = task_file(relative)
    spec = importlib.util.spec_from_file_location("task_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.lmeval
def test_task_patch_is_idempotent(tmp_path, lmeval):
    for name in ["careqa/utils_open.py", "tinyBenchmarks/agg_functions.py"]:
        path = tmp_path / "tasks" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(task_file(name).read_bytes())
    apply_fixes(tmp_path, "0.4.13")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.py")}
    apply_fixes(tmp_path, "0.4.13")
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.py")}


@pytest.mark.lmeval
def test_careqa_perplexity_yaml_needs_no_generation_metrics(monkeypatch, lmeval):
    monkeypatch.setitem(sys.modules, "evaluate", None)
    module = task_module("careqa/utils_open.py")
    assert module.doc_to_text({"question": "question"}) == "question"
    assert module.doc_to_target({"answer": "answer"}) == "answer"
    from lm_eval.tasks._yaml_loader import load_yaml

    config = load_yaml(
        task_file("careqa/careqa_open_perplexity.yaml"), resolve_func=True
    )
    assert config["output_type"] == "loglikelihood_rolling"


@pytest.mark.lmeval
def test_careqa_generation_loads_original_metrics_once(monkeypatch, lmeval):
    calls = []
    scores = {
        "bleu": {"bleu": 0.5},
        "rouge": {"rouge1": 0.6, "rouge2": 0.7, "rougeL": 0.8},
        "bertscore": {"f1": [0.9]},
        "bleurt": {"scores": [0.4]},
    }

    def load(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return SimpleNamespace(compute=lambda **kw: scores[name])

    monkeypatch.setitem(sys.modules, "evaluate", SimpleNamespace(load=load))
    module = task_module("careqa/utils_open.py")
    assert calls == []
    expected = {
        "bleu": 0.5,
        "rouge1": 0.6,
        "rouge2": 0.7,
        "rougeL": 0.8,
        "bert_score": 0.9,
        "bleurt": 0.4,
    }
    assert module.doc_eval(["answer"], ["answer"]) == expected
    assert module.doc_eval(["answer"], ["answer"]) == expected
    assert [c[0] for c in calls] == ["bleu", "rouge", "bertscore", "bleurt"]
    assert calls[-1] == ("bleurt", ("bleurt-base-512",), {"module_type": "metric"})


@pytest.mark.lmeval
def test_careqa_generation_reports_actual_missing_metric(monkeypatch, lmeval):
    def load(name, *args, **kwargs):
        if name == "bleurt":
            raise ImportError("BLEURT unavailable")
        return object()

    monkeypatch.setitem(sys.modules, "evaluate", SimpleNamespace(load=load))
    module = task_module("careqa/utils_open.py")
    with pytest.raises(RuntimeError, match="metric 'bleurt'") as caught:
        module.doc_eval(["answer"], ["answer"])
    assert isinstance(caught.value.__cause__, ImportError)


@pytest.mark.lmeval
@pytest.mark.parametrize("count", [0, 1, 99, 101])
def test_partial_tinytruthfulqa_does_not_call_estimator(monkeypatch, lmeval, count):
    monkeypatch.setitem(sys.modules, "tinyBenchmarks", None)
    module = task_module("tinyBenchmarks/agg_functions.py")
    assert module.agg_gpirt_truthfulqa([1.0] * count) is None


@pytest.mark.lmeval
def test_full_tinytruthfulqa_and_other_tasks_keep_estimator(monkeypatch, lmeval):
    calls = []

    def evaluate(items, benchmark):
        calls.append((list(items), benchmark))
        return {benchmark: {"gpirt": 0.75}}

    monkeypatch.setitem(
        sys.modules, "tinyBenchmarks", SimpleNamespace(evaluate=evaluate)
    )
    module = task_module("tinyBenchmarks/agg_functions.py")
    anchors = [float(i % 2) for i in range(100)]
    assert module.agg_gpirt_truthfulqa(anchors) == 0.75
    assert calls == [(anchors, "truthfulqa")]
    # The 100-anchor guard applies only to tinyTruthfulQA.
    assert module.agg_gpirt_arc([1.0]) == 0.75
    assert calls[-1] == ([1.0], "arc")


@pytest.mark.lmeval
def test_real_tinytruthfulqa_yaml_uses_partial_guard(monkeypatch, lmeval):
    monkeypatch.setitem(sys.modules, "tinyBenchmarks", None)
    from lm_eval.tasks._yaml_loader import load_yaml

    config = load_yaml(
        task_file("tinyBenchmarks/tinyTruthfulQA_mc2.yaml"), resolve_func=True
    )
    aggregation = config["metric_list"][0]["aggregation"]
    assert aggregation([1.0]) is None


@pytest.mark.lmeval
@pytest.mark.parametrize("count", [1, 100])
def test_actual_cli_writes_tinytruthfulqa_results(tmp_path, lmeval, count):
    """Run the real CLI with local records, dummy inference and blocked sockets."""
    import yaml

    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"question": f"Question {i}", "choices": ["A", "B"], "gold": 0})
            for i in range(100)
        )
    )
    task = tmp_path / "tiny.yaml"
    config = {
        "task": "tinyTruthfulQA",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(dataset)}},
        "test_split": "test",
        "output_type": "multiple_choice",
        "doc_to_text": "{{question}}",
        "doc_to_choice": "choices",
        "doc_to_target": "gold",
    }
    task.write_text(
        yaml.safe_dump(config)
        + "metric_list:\n  - metric: acc\n"
        + "    aggregation: !function lm_eval.tasks.tinyBenchmarks.agg_functions.agg_gpirt_truthfulqa\n"
        + "    higher_is_better: true\n"
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import socket\n"
        "def blocked(*args, **kwargs):\n"
        "    raise RuntimeError('Network disabled in CLI regression test')\n"
        "socket.socket.connect = blocked\n"
        "socket.create_connection = blocked\n"
    )
    # Verify the official-estimator call path, not its statistical formula.
    (tmp_path / "tinyBenchmarks.py").write_text(
        "def evaluate(items, benchmark):\n"
        "    assert len(items) == 100\n"
        "    return {benchmark: {'gpirt': 0.75}}\n"
    )
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(tmp_path),
        HF_HOME=str(tmp_path / "hf"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "dummy",
            "--tasks",
            str(task),
            "--limit",
            str(count),
            "--num_fewshot",
            "0",
            "--output_path",
            str(output),
            "--log_samples",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from _results import find_results_file, parse_results_file

    raw = parse_results_file(find_results_file(output))
    assert raw["n-samples"]["tinyTruthfulQA"]["effective"] == count
    assert raw["results"]["tinyTruthfulQA"]["acc,none"] == (
        None if count == 1 else 0.75
    )
