"""Consistency checks for provider.yaml's benchmark catalog."""

from pathlib import Path

import pytest
import yaml

_PROVIDER = Path(__file__).resolve().parent.parent / "provider.yaml"
_BENCHMARKS = yaml.safe_load(_PROVIDER.read_text())["benchmarks"]


@pytest.mark.parametrize("benchmark", _BENCHMARKS, ids=lambda b: b["id"])
def test_primary_score_is_one_of_the_declared_metrics(benchmark):
    primary = (benchmark.get("primary_score") or {}).get("metric")
    if primary is None:
        pytest.skip("no primary_score declared")
    assert primary in benchmark["metrics"], (
        f"{benchmark['id']}: primary_score.metric {primary!r} is not in metrics {benchmark['metrics']}"
    )


# Metric names are "<scorer>/<metric>" as written in the Inspect log, not a generic
# "accuracy/accuracy". These benchmarks use inspect-ai's choice() / the verify() scorers.
@pytest.mark.parametrize(
    "benchmark_id,scorer",
    [
        ("inspect/mmlu", "choice"),
        ("inspect/mmlu-pro", "choice"),
        ("inspect/gpqa", "choice"),
        ("inspect/arc", "choice"),
        ("inspect/hellaswag", "choice"),
        ("inspect/winogrande", "choice"),
        ("inspect/truthfulqa", "choice"),
        ("inspect/wmdp", "choice"),
        ("inspect/gsm8k", "match"),
        ("inspect/humaneval", "verify"),
        ("inspect/mbpp", "verify"),
        ("inspect/bigcodebench", "verify"),
    ],
)
def test_metric_names_use_the_runtime_scorer_name(benchmark_id, scorer):
    benchmark = next(b for b in _BENCHMARKS if b["id"] == benchmark_id)
    assert all(m.startswith(f"{scorer}/") for m in benchmark["metrics"]), benchmark["metrics"]
    assert benchmark["primary_score"]["metric"].startswith(f"{scorer}/")
