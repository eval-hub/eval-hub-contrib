"""Integration tests for the RAGAS adapter.

Verifies adapter plumbing with a single monkeypatch point — _run_ragas
returns a mock result object, so no model endpoint is needed.
"""

import yaml
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

from evalhub.adapter import JobCallbacks, JobPhase, OCIArtifactResult
from main import METRIC_MAPPING, RagasAdapter, _run_collections_evaluation

_PROVIDER_YAML = Path(__file__).resolve().parent.parent / "provider.yaml"


def _provider_metric_names() -> set[str]:
    """Return every metric name declared across all benchmarks in provider.yaml."""
    data = yaml.safe_load(_PROVIDER_YAML.read_text())
    names = set()
    for benchmark in data.get("benchmarks", []):
        names.update(benchmark.get("metrics", []))
    return names


def test_metric_mapping_covers_provider_yaml():
    """Every metric declared in provider.yaml must be resolvable via METRIC_MAPPING,
    and each mapping entry's .name attribute must match its key."""
    provider_metrics = _provider_metric_names()
    assert provider_metrics, "provider.yaml has no metrics — test is vacuous"

    missing = provider_metrics - METRIC_MAPPING.keys()
    assert not missing, f"Metrics in provider.yaml missing from METRIC_MAPPING: {missing}"

    for name in provider_metrics:
        assert METRIC_MAPPING[name].name == name, (
            f"METRIC_MAPPING[{name!r}].name == {METRIC_MAPPING[name].name!r}; key and .name have drifted"
        )


def _make_mock_ragas_result(metric_names, n_rows=5):
    """Create a mock ragas EvaluationResult with to_pandas()."""
    data = {name: [0.8 + i * 0.01 for i in range(n_rows)] for name in metric_names}
    df = pd.DataFrame(data)
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = df
    return mock_result


@pytest.mark.integration
def test_ragas_happy_path(monkeypatch, tmp_path):
    """Full run_benchmark_job with mocked _run_ragas returning canned results."""
    adapter = RagasAdapter(job_spec_path="meta/job.json")

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake",
        reference="fake:latest",
    )

    metric_names = ["answer_relevancy", "context_precision", "faithfulness", "context_recall"]
    mock_result = _make_mock_ragas_result(metric_names)

    monkeypatch.setattr(adapter, "_run_ragas", lambda **kwargs: mock_result)

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input": "What is AI?", "response": "Artificial Intelligence", "retrieved_contexts": ["AI is..."], "reference": "AI stands for..."}\n'
        '{"user_input": "What is ML?", "response": "Machine Learning", "retrieved_contexts": ["ML is..."], "reference": "ML stands for..."}\n'
    )
    monkeypatch.setattr(
        "main._resolve_data_path",
        lambda config: dataset_file,
    )

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.id == adapter.job_spec.id
    assert results.benchmark_id == adapter.job_spec.benchmark_id
    assert results.benchmark_index == adapter.job_spec.benchmark_index
    assert results.model_name == adapter.job_spec.model.name
    assert results.duration_seconds >= 0

    assert len(results.results) == 4
    assert any(r.metric_name == "answer_relevancy" for r in results.results)
    assert any(r.metric_name == "faithfulness" for r in results.results)

    assert results.overall_score is not None
    assert results.num_examples_evaluated == 5

    phases = [c.args[0].phase for c in callbacks.report_status.call_args_list]
    assert JobPhase.INITIALIZING in phases
    assert JobPhase.LOADING_DATA in phases
    assert JobPhase.RUNNING_EVALUATION in phases
    assert JobPhase.POST_PROCESSING in phases

    meta = results.evaluation_metadata
    assert "ragas_version" in meta, "Eval Card missing ragas_version"
    assert meta["ragas_version"], "ragas_version must not be empty"
    assert "judge_llm" in meta, "Eval Card missing judge_llm"
    assert meta["judge_llm"] == adapter.job_spec.model.name
    assert "embedding_model" in meta, "Eval Card missing embedding_model"


@pytest.mark.integration
def test_run_collections_evaluation_builds_metric_columns():
    """Collections evaluation should score each row via abatch_score()."""
    from ragas import EvaluationDataset

    records = [
        {
            "user_input": "What is AI?",
            "response": "Artificial Intelligence",
            "retrieved_contexts": ["AI is..."],
            "reference": "AI stands for...",
        },
        {
            "user_input": "What is ML?",
            "response": "Machine Learning",
            "retrieved_contexts": ["ML is..."],
            "reference": "ML stands for...",
        },
    ]
    eval_dataset = EvaluationDataset.from_list(records)

    class FakeMetric:
        def __init__(self, name: str, fields: list[str], values: list[float]):
            self.name = name
            self._fields = fields
            self._values = values
            self.batch_calls = 0

        async def ascore(self, user_input: str = "", response: str = "", reference: str = "", retrieved_contexts: list[str] | None = None):
            raise NotImplementedError

        async def abatch_score(self, inputs):
            self.batch_calls += 1
            assert inputs == [
                {field: record[field] for field in self._fields}
                for record in records
            ]
            return [type("MetricResult", (), {"value": value})() for value in self._values]

    fake_metrics = {
        "answer_relevancy": FakeMetric(
            "answer_relevancy",
            ["user_input", "response"],
            [0.9, 0.8],
        ),
        "faithfulness": FakeMetric(
            "faithfulness",
            ["user_input", "response", "retrieved_contexts"],
            [0.7, 0.6],
        ),
    }

    def metric_input_fields(metric):
        return metric._fields

    def factory(metric_name):
        return lambda _llm, _emb: fake_metrics[metric_name]

    from types import SimpleNamespace

    patched_defs = []
    for name in ["answer_relevancy", "faithfulness"]:
        metric_def = SimpleNamespace(name=name)
        metric_def.factory = factory(name)
        patched_defs.append(metric_def)

    import main as main_module

    original_fields = main_module._metric_input_fields
    main_module._metric_input_fields = metric_input_fields
    try:
        result = _run_collections_evaluation(
            eval_dataset=eval_dataset,
            metric_defs=patched_defs,
            llm=object(),
            embeddings=object(),
        )
    finally:
        main_module._metric_input_fields = original_fields
    result_df = result.to_pandas()

    assert list(result_df["answer_relevancy"]) == [0.9, 0.8]
    assert list(result_df["faithfulness"]) == [0.7, 0.6]
    assert fake_metrics["answer_relevancy"].batch_calls == 1
    assert fake_metrics["faithfulness"].batch_calls == 1


@pytest.mark.integration
def test_run_collections_evaluation_respects_max_workers():
    """abatch_score should receive ordered chunks bounded by RunConfig.max_workers."""
    from ragas import EvaluationDataset
    from ragas.run_config import RunConfig

    records = [
        {
            "user_input": "What is AI?",
            "response": "Artificial Intelligence",
            "retrieved_contexts": ["AI is..."],
            "reference": "AI stands for...",
        },
        {
            "user_input": "What is ML?",
            "response": "Machine Learning",
            "retrieved_contexts": ["ML is..."],
            "reference": "ML stands for...",
        },
        {
            "user_input": "What is DL?",
            "response": "Deep Learning",
            "retrieved_contexts": ["DL is..."],
            "reference": "DL stands for...",
        },
    ]
    eval_dataset = EvaluationDataset.from_list(records)

    class FakeMetric:
        def __init__(self, name: str, fields: list[str], values: list[float]):
            self.name = name
            self._fields = fields
            self._values = values
            self.batch_calls = []

        async def ascore(self, user_input: str = "", response: str = ""):
            raise NotImplementedError

        async def abatch_score(self, inputs):
            self.batch_calls.append(inputs)
            start = sum(len(call) for call in self.batch_calls) - len(inputs)
            chunk_values = self._values[start : start + len(inputs)]
            return [type("MetricResult", (), {"value": value})() for value in chunk_values]

    fake_metric = FakeMetric(
        "answer_relevancy",
        ["user_input", "response"],
        [0.9, 0.8, 0.7],
    )

    from types import SimpleNamespace

    metric_def = SimpleNamespace(name="answer_relevancy")
    metric_def.factory = lambda _llm, _emb: fake_metric

    import main as main_module

    original_fields = main_module._metric_input_fields
    main_module._metric_input_fields = lambda metric: metric._fields
    try:
        result = _run_collections_evaluation(
            eval_dataset=eval_dataset,
            metric_defs=[metric_def],
            llm=object(),
            embeddings=object(),
            run_config=RunConfig(max_workers=2),
        )
    finally:
        main_module._metric_input_fields = original_fields

    result_df = result.to_pandas()
    assert list(result_df["answer_relevancy"]) == [0.9, 0.8, 0.7]
    assert len(fake_metric.batch_calls) == 2
    assert len(fake_metric.batch_calls[0]) == 2
    assert len(fake_metric.batch_calls[1]) == 1


@pytest.mark.integration
def test_run_collections_evaluation_uses_single_event_loop_for_chunks(monkeypatch):
    """Multiple chunks must share one asyncio.run() so async HTTP clients stay bound."""
    import asyncio
    from ragas import EvaluationDataset
    from ragas.run_config import RunConfig

    records = [
        {
            "user_input": f"Question {idx}",
            "response": f"Answer {idx}",
            "retrieved_contexts": [f"Context {idx}"],
            "reference": f"Reference {idx}",
        }
        for idx in range(5)
    ]
    eval_dataset = EvaluationDataset.from_list(records)

    class FakeAsyncClient:
        def __init__(self):
            self.loop_ids: list[int] = []

        async def request(self, *_args, **_kwargs):
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return {"score": 1.0}

    client = FakeAsyncClient()

    class FakeMetric:
        name = "answer_relevancy"

        def __init__(self):
            self.batch_calls = []

        async def ascore(self, user_input: str = "", response: str = ""):
            await client.request()
            return type("MetricResult", (), {"value": 0.5})()

        async def abatch_score(self, inputs):
            self.batch_calls.append(inputs)
            start = sum(len(call) for call in self.batch_calls) - len(inputs)
            results = await asyncio.gather(
                *[
                    self.ascore(**input_dict)
                    for input_dict in inputs
                ]
            )
            return [
                type("MetricResult", (), {"value": float(start + idx + 1)})()
                for idx, _result in enumerate(results)
            ]

    fake_metric = FakeMetric()
    asyncio_run_calls = []

    def track_asyncio_run(coro):
        asyncio_run_calls.append(coro)
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    from types import SimpleNamespace

    metric_def = SimpleNamespace(name="answer_relevancy")
    metric_def.factory = lambda _llm, _emb: fake_metric

    import main as main_module

    original_fields = main_module._metric_input_fields
    original_asyncio_run = main_module.asyncio.run
    main_module._metric_input_fields = lambda metric: ["user_input", "response"]
    monkeypatch.setattr(main_module.asyncio, "run", track_asyncio_run)
    try:
        result = _run_collections_evaluation(
            eval_dataset=eval_dataset,
            metric_defs=[metric_def],
            llm=object(),
            embeddings=object(),
            run_config=RunConfig(max_workers=2),
        )
    finally:
        main_module._metric_input_fields = original_fields
        monkeypatch.setattr(main_module.asyncio, "run", original_asyncio_run)

    result_df = result.to_pandas()
    assert list(result_df["answer_relevancy"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(fake_metric.batch_calls) == 3
    assert [len(chunk) for chunk in fake_metric.batch_calls] == [2, 2, 1]
    assert len(asyncio_run_calls) == 1
    assert len(client.loop_ids) == 5
    assert len(set(client.loop_ids)) == 1


@pytest.mark.integration
def test_run_collections_evaluation_rejects_incomplete_records():
    """Validation must check every record, not only the first row."""
    from ragas import EvaluationDataset

    records = [
        {
            "user_input": "What is AI?",
            "response": "Artificial Intelligence",
            "retrieved_contexts": ["AI is..."],
            "reference": "AI stands for...",
        },
        {
            "user_input": "What is ML?",
            "response": "Machine Learning",
            "retrieved_contexts": ["ML is..."],
        },
    ]
    eval_dataset = EvaluationDataset.from_list(records)

    class FakeMetric:
        name = "faithfulness"

        async def ascore(
            self,
            user_input: str = "",
            response: str = "",
            reference: str = "",
            retrieved_contexts: list[str] | None = None,
        ):
            raise NotImplementedError

        async def abatch_score(self, inputs):
            raise AssertionError("abatch_score must not run when validation fails")

    from types import SimpleNamespace

    metric_def = SimpleNamespace(name="faithfulness")
    metric_def.factory = lambda _llm, _emb: FakeMetric()

    import main as main_module

    original_fields = main_module._metric_input_fields
    main_module._metric_input_fields = lambda _metric: [
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
    ]
    try:
        with pytest.raises(ValueError, match=r"record\[1\] missing \['reference'\]"):
            _run_collections_evaluation(
                eval_dataset=eval_dataset,
                metric_defs=[metric_def],
                llm=object(),
                embeddings=object(),
            )
    finally:
        main_module._metric_input_fields = original_fields


@pytest.mark.integration
def test_mode_suffixed_metric_columns(monkeypatch, tmp_path):
    """Metrics whose result column carries a mode suffix (e.g.
    factual_correctness(mode=f1)) are still extracted and reported under the
    requested metric name."""
    adapter = RagasAdapter(job_spec_path="meta/job.json")
    adapter.job_spec.parameters["metrics"] = ["faithfulness", "factual_correctness"]

    callbacks = create_autospec(JobCallbacks)
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:fake",
        reference="fake:latest",
    )

    mock_result = _make_mock_ragas_result(
        ["faithfulness", "factual_correctness(mode=f1)"]
    )
    monkeypatch.setattr(adapter, "_run_ragas", lambda **kwargs: mock_result)

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input": "What is AI?", "response": "Artificial Intelligence", "retrieved_contexts": ["AI is..."], "reference": "AI stands for..."}\n'
    )
    monkeypatch.setattr("main._resolve_data_path", lambda config: dataset_file)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    reported = {r.metric_name for r in results.results}
    assert reported == {"faithfulness", "factual_correctness"}
    assert results.overall_score is not None
