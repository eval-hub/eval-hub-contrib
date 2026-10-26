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
from main import (
    METRIC_MAPPING,
    RagasAdapter,
    _openai_credentials,
    _run_collections_evaluation,
)

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


def test_provider_declares_separate_judge_parameters():
    data = yaml.safe_load(_PROVIDER_YAML.read_text())
    parameters = {parameter["name"] for parameter in data.get("parameters", [])}
    assert {"judge_model", "judge_url", "judge_api_key"} <= parameters


def test_external_judge_does_not_resolve_model_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    def fail_if_called():
        raise AssertionError("model credentials must not be resolved for an external judge")

    monkeypatch.setattr("main.resolve_model_credentials", fail_if_called)

    assert _openai_credentials(
        "https://api.openai.com/v1", use_model_credentials=False
    ) == ("https://api.openai.com/v1", "env-key")


def test_model_endpoint_still_uses_mounted_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: type("Credentials", (), {"api_key": "mounted-key"})(),
    )

    assert _openai_credentials("http://localhost:8000") == (
        "http://localhost:8000/v1",
        "mounted-key",
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
def test_separate_judge_is_used_for_llm_metrics(monkeypatch, tmp_path):
    adapter = RagasAdapter(job_spec_path="meta/job.json")
    adapter.job_spec.parameters.update(
        {
            "judge_model": "gpt-4o-mini",
            "judge_url": "https://api.openai.com/v1",
            "judge_api_key": "test-key",
        }
    )
    callbacks = create_autospec(JobCallbacks)
    created = {}

    def fake_create_llm(base_url, model_id, **kwargs):
        created.update(base_url=base_url, model_id=model_id, kwargs=kwargs)
        return object()

    monkeypatch.setattr("main._create_ragas_llm", fake_create_llm)
    monkeypatch.setattr("main._create_ragas_embeddings", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        adapter,
        "_run_ragas",
        lambda **kwargs: _make_mock_ragas_result(["answer_relevancy"], n_rows=1),
    )
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input": "What is AI?", "response": "Artificial Intelligence", "retrieved_contexts": ["AI is..."], "reference": "AI stands for..."}\n'
    )
    monkeypatch.setattr("main._resolve_data_path", lambda config: dataset_file)

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert created == {
        "base_url": "https://api.openai.com/v1",
        "model_id": "gpt-4o-mini",
        "kwargs": {
            "max_tokens": 512,
            "temperature": 0.1,
            "api_key": "test-key",
            "use_model_credentials": False,
        },
    }
    assert results.evaluation_metadata["judge_llm"] == "gpt-4o-mini"
    assert results.evaluation_metadata["judge_url"] == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    "embedding_url, expected_url, use_model_credentials",
    [
        ("http://granite-embedding:8080/v1/", "http://granite-embedding:8080/v1", False),
        ("http://127.0.0.1:8001/v1", "http://127.0.0.1:8001/v1", False),
        ("http://localhost:8082/v1/", "http://localhost:8082/v1", True),
        (None, "http://localhost:8082/v1", True),
    ],
)
def test_standalone_embedding_endpoint_routing(
    monkeypatch, tmp_path, embedding_url, expected_url, use_model_credentials
):
    monkeypatch.setenv("EVALHUB_MODE", "local")
    adapter = RagasAdapter(job_spec_path="meta/job.json")
    adapter.job_spec.model.url = "http://localhost:8082/v1"
    adapter.job_spec.parameters["embedding_model"] = "granite-embedding"
    adapter.job_spec.parameters["embedding_url"] = embedding_url
    callbacks = create_autospec(JobCallbacks)
    create_embeddings = MagicMock(return_value=object())
    monkeypatch.setattr("main._create_ragas_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr("main._create_ragas_embeddings", create_embeddings)
    monkeypatch.setattr(
        adapter,
        "_run_ragas",
        lambda **kwargs: _make_mock_ragas_result(["answer_relevancy"], n_rows=1),
    )
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input":"What is AI?","response":"Artificial Intelligence",'
        '"retrieved_contexts":["AI is..."],"reference":"AI stands for..."}\n'
    )
    monkeypatch.setattr("main._resolve_data_path", lambda config: dataset_file)

    adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert create_embeddings.call_args.args == (expected_url, "granite-embedding")
    assert create_embeddings.call_args.kwargs["use_model_credentials"] is use_model_credentials


def test_standalone_embeddings_do_not_use_model_credentials(monkeypatch):
    from main import _create_ragas_embeddings

    monkeypatch.setenv("OPENAI_API_KEY", "embedding-key")

    def fail_if_called():
        raise AssertionError("embedding endpoint must not receive chat credentials")

    monkeypatch.setattr("main.resolve_model_credentials", fail_if_called)
    factory = MagicMock()
    monkeypatch.setattr("ragas.embeddings.base.embedding_factory", factory)

    _create_ragas_embeddings(
        "http://granite-embedding:8080/v1",
        "granite-embedding",
        use_model_credentials=False,
    )

    client = factory.call_args.kwargs["client"]
    assert str(client.base_url) == "http://granite-embedding:8080/v1/"
    assert client.api_key == "embedding-key"


def test_k8s_judge_and_embeddings_use_projected_reference_tokens(monkeypatch, tmp_path):
    from main import _async_openai_client

    auth_dir = tmp_path / "model-auth"
    auth_dir.mkdir()
    sidecar_url = "http://localhost:8082/v1"
    for name, value in {
        "api-key": "api-key:ref",
        "judge_api-key": "judge_api-key:ref",
        "judge_url": sidecar_url,
        "embedding_api-key": "embedding_api-key:ref",
        "embedding_url": sidecar_url,
    }.items():
        (auth_dir / name).write_text(value)
    monkeypatch.setattr("evalhub.adapter.auth._MODEL_AUTH_DIR", auth_dir)
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")

    adapter = RagasAdapter(job_spec_path="meta/job.json")
    adapter.job_spec.model.url = sidecar_url
    adapter.job_spec.parameters = {
        "judge_model": "judge-model",
        "embedding_model": "granite-embedding",
        "metrics": ["answer_relevancy"],
    }
    llm_factory = MagicMock(return_value=object())
    embedding_factory = MagicMock(return_value=object())
    monkeypatch.setattr("ragas.llms.llm_factory", llm_factory)
    monkeypatch.setattr("ragas.embeddings.base.embedding_factory", embedding_factory)
    monkeypatch.setattr(
        adapter, "_run_ragas",
        lambda **kwargs: _make_mock_ragas_result(["answer_relevancy"], n_rows=1),
    )
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text(
        '{"user_input":"What is AI?","response":"Artificial Intelligence",'
        '"retrieved_contexts":["AI is..."],"reference":"AI stands for..."}\n'
    )
    monkeypatch.setattr("main._resolve_data_path", lambda config: dataset_file)

    adapter.run_benchmark_job(adapter.job_spec, create_autospec(JobCallbacks))

    primary_client = _async_openai_client(sidecar_url)
    judge_client = llm_factory.call_args.kwargs["client"]
    embedding_client = embedding_factory.call_args.kwargs["client"]
    assert llm_factory.call_args.args == ("judge-model",)
    assert embedding_factory.call_args.kwargs["model"] == "granite-embedding"
    # All three destinations have the same localhost URL. Only the reference
    # token selects the upstream and its credential in the sidecar.
    for client, ref in [
        (primary_client, "api-key:ref"),
        (judge_client, "judge_api-key:ref"),
        (embedding_client, "embedding_api-key:ref"),
    ]:
        assert str(client.base_url) == sidecar_url + "/"
        assert client.api_key == ref
        assert client.auth_headers["Authorization"] == f"Bearer {ref}"


@pytest.mark.parametrize("role", ["judge", "embedding"])
@pytest.mark.parametrize("separate_path", [False, True])
def test_role_connection_reads_projected_credentials(monkeypatch, role, separate_path):
    from main import _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    model_url = "http://localhost:8082/v1"
    role_url = f"http://localhost:8082/{role}/v1" if separate_path else model_url
    projected = {f"{role}_api-key": f"{role}_api-key:ref", f"{role}_url": role_url}
    monkeypatch.setattr("main.read_model_auth_key", projected.get)

    url, key, _ = _resolve_role_connection(role, model_url)

    assert url == role_url
    assert key == f"{role}_api-key:ref"


@pytest.mark.parametrize("role", ["judge", "embedding"])
def test_role_url_requires_matching_credential(monkeypatch, role):
    from main import _resolve_role_connection

    monkeypatch.setattr(
        "main.read_model_auth_key", {f"{role}_url": "http://localhost:8082/v1"}.get
    )

    with pytest.raises(ValueError, match=f"{role}_url.*{role}_api-key"):
        _resolve_role_connection(role, "http://localhost:8082/v1")


@pytest.mark.parametrize("role", ["judge", "embedding"])
def test_role_credential_without_url_uses_model_endpoint(monkeypatch, role):
    from main import _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr(
        "main.read_model_auth_key", {f"{role}_api-key": f"{role}_api-key:ref"}.get
    )

    url, key, _ = _resolve_role_connection(role, "http://localhost:8082/v1")

    assert url == "http://localhost:8082/v1"
    assert key == f"{role}_api-key:ref"


@pytest.mark.parametrize("role", ["judge", "embedding"])
def test_k8s_role_cannot_bypass_sidecar(monkeypatch, role):
    from main import _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr("main.read_model_auth_key", lambda key: None)

    with pytest.raises(ValueError, match=f"{role}_url.*model.auth.secret_ref"):
        _resolve_role_connection(
            role, "http://localhost:8082/v1", url_override="https://external.example/v1"
        )


def test_k8s_explicit_judge_reference_remains_supported(monkeypatch):
    from main import _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr("main.read_model_auth_key", lambda key: None)

    url, key, _ = _resolve_role_connection(
        "judge", "http://localhost:8082/v1", api_key_override="custom_api-key:ref"
    )

    assert url == "http://localhost:8082/v1"
    assert key == "custom_api-key:ref"


def test_k8s_judge_rejects_literal_key(monkeypatch):
    from main import _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setattr("main.read_model_auth_key", lambda key: None)

    with pytest.raises(ValueError, match="judge_api-key.*model.auth.secret_ref"):
        _resolve_role_connection(
            "judge", "http://localhost:8082/v1", api_key_override="literal-key"
        )


@pytest.mark.parametrize("role", ["judge", "embedding"])
def test_k8s_role_without_projected_entries_uses_primary_reference(monkeypatch, role):
    from main import _openai_credentials, _resolve_role_connection

    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-env-key")
    monkeypatch.setattr("main.read_model_auth_key", lambda key: None)
    monkeypatch.setattr(
        "main.resolve_model_credentials",
        lambda: type("Credentials", (), {"api_key": "api-key:ref"})(),
    )
    url, key, use_model_credentials = _resolve_role_connection(
        role, "http://localhost:8082/v1",
        url_override="http://localhost:8082/alternate/v1",
    )

    assert _openai_credentials(url, key, use_model_credentials=use_model_credentials) == (
        "http://localhost:8082/alternate/v1", "api-key:ref"
    )


def test_http_failure_reports_request_url_without_credentials(monkeypatch, tmp_path):
    import httpx
    from openai import NotFoundError
    from evalhub.adapter import JobStatus

    adapter = RagasAdapter(job_spec_path="meta/job.json")
    callbacks = create_autospec(JobCallbacks)
    monkeypatch.setattr("main._create_ragas_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr("main._create_ragas_embeddings", lambda *args, **kwargs: object())
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"user_input":"hello"}\n')
    monkeypatch.setattr("main._resolve_data_path", lambda config: dataset_file)
    error = NotFoundError(
        "Not Found",
        response=httpx.Response(
            404,
            request=httpx.Request(
                "POST", "https://user:password@chat.example/v1/embeddings?api_key=secret#fragment"
            ),
        ),
        body={"detail": "Not Found"},
    )
    monkeypatch.setattr(adapter, "_run_ragas", MagicMock(side_effect=error))

    with pytest.raises(NotFoundError) as raised:
        adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert raised.value is error
    status = callbacks.report_status.call_args.args[0]
    assert status.status == JobStatus.FAILED
    assert "POST https://chat.example/v1/embeddings" in status.error_message.message
    for sensitive in ("user", "password", "api_key", "secret", "fragment"):
        assert sensitive not in status.error_message.message


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
