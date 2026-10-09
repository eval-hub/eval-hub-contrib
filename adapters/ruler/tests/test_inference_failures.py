"""Inference failures must not contaminate benchmark scores."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from evalhub.adapter import JobPhase, JobSpec, JobStatus

import main


@pytest.fixture
def adapter():
    return main.RulerAdapter.__new__(main.RulerAdapter)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "validation.jsonl"
    samples = [
        {"index": index, "input": f"Question {index}", "outputs": ["alpha"]}
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    return path


def response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def infer(adapter, dataset, pred_file, client, callbacks=None):
    return adapter._run_api_inference(
        model_name="test-model",
        data_file=dataset,
        pred_file=pred_file,
        tokens_to_generate=128,
        batch_size=1,
        api_client=client,
        callbacks=callbacks or MagicMock(),
        total_samples_hint=1,
    )


@pytest.mark.parametrize(
    "failure", [400, 401, 403, 404, 422, 429, 500, "connection", "timeout"]
)
def test_failed_request_stops_inference_without_writing_predictions(
    adapter, dataset, tmp_path, failure
):
    request = httpx.Request("POST", "http://model.invalid/v1/chat/completions")
    if failure == "connection":
        error = openai.APIConnectionError(request=request)
    elif failure == "timeout":
        error = openai.APITimeoutError(request=request)
    else:
        error_class = {
            400: openai.BadRequestError,
            401: openai.AuthenticationError,
            403: openai.PermissionDeniedError,
            404: openai.NotFoundError,
            422: openai.UnprocessableEntityError,
            429: openai.RateLimitError,
            500: openai.InternalServerError,
        }[failure]
        error = error_class(
            "Request failed",
            response=httpx.Response(failure, request=request),
            body=None,
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        response("alpha"), error, response("alpha")
    ]
    pred_file = tmp_path / "vt_4096.jsonl"

    with pytest.raises(
        RuntimeError, match="vt_4096, sample 1 on model test-model"
    ) as caught:
        infer(adapter, dataset, pred_file, client)

    assert caught.value.__cause__ is error
    assert type(error).__name__ in str(caught.value)
    if isinstance(failure, int):
        assert f"HTTP {failure}" in str(caught.value)
    if failure in (401, 403):
        assert "model.auth.secret_ref" in str(caught.value)
    assert client.chat.completions.create.call_count == 2
    assert not pred_file.exists()


def test_failed_job_reports_failed_without_scoring_or_exporting(
    adapter, dataset, monkeypatch
):
    request = httpx.Request("POST", "http://model.invalid/v1/chat/completions")
    error = openai.BadRequestError(
        "Invalid request", response=httpx.Response(400, request=request), body=None
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        response("alpha"), error, response("alpha")
    ]
    callbacks = MagicMock()

    def generate_data(**kwargs):
        path = (
            kwargs["data_dir"] / str(kwargs["context_length"])
            / kwargs["task_id"] / "validation.jsonl"
        )
        path.parent.mkdir(parents=True)
        path.write_text(dataset.read_text())

    monkeypatch.setattr(adapter, "_verify_tokenizer", MagicMock())
    monkeypatch.setattr(adapter, "_generate_task_data", generate_data)
    monkeypatch.setattr(adapter, "_make_api_client", MagicMock(return_value=client))
    score = MagicMock()
    save = MagicMock()
    monkeypatch.setattr(adapter, "_evaluate_predictions", score)
    monkeypatch.setattr(adapter, "_save_results", save)
    config = JobSpec(
        id="inference-failure-test",
        provider_id="ruler",
        benchmark_id="variable-tracking",
        benchmark_index=0,
        model={"name": "test-model", "url": "http://model.invalid/v1"},
        num_examples=3,
        parameters={"context_lengths": [4096]},
        callback_url="http://callback.invalid",
    )

    with pytest.raises(RuntimeError, match="sample 1"):
        adapter.run_benchmark_job(config, callbacks)

    updates = [call.args[0] for call in callbacks.report_status.call_args_list]
    assert updates[-1].status == JobStatus.FAILED
    assert "HTTP 400" in updates[-1].error.message
    assert not any(
        update.phase in (JobPhase.POST_PROCESSING, JobPhase.PERSISTING_ARTIFACTS)
        for update in updates
    )
    assert client.chat.completions.create.call_count == 2
    score.assert_not_called()
    save.assert_not_called()
    callbacks.create_oci_artifact.assert_not_called()


def test_successful_empty_and_wrong_answers_remain_in_score(adapter, dataset, tmp_path):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        response(None), response("wrong"), response("alpha")
    ]
    pred_file = tmp_path / "vt_4096.jsonl"

    predictions = infer(adapter, dataset, pred_file, client)
    assert [prediction["pred"] for prediction in predictions] == [
        "", "wrong", "alpha"
    ]
    saved = [json.loads(line) for line in pred_file.read_text().splitlines()]
    assert saved == predictions

    scores = adapter._evaluate_predictions({"vt": {4096: predictions}})
    assert {score.metric_name: score.metric_value for score in scores} == {
        "vt.ctx_4096.score": pytest.approx(0.3333),
        "vt.overall": pytest.approx(0.3333),
    }
    assert all(score.num_samples == 3 for score in scores)


@pytest.mark.parametrize(
    "statuses, expected_calls",
    [([400], 1), ([429, 429, 429], 3), ([503, 200], 2)],
)
def test_sdk_retries_complete_before_adapter_handles_failure(
    adapter, dataset, tmp_path, monkeypatch, statuses, expected_calls
):
    calls = []

    def handle_request(request):
        calls.append(request)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "Test failure"}})
        return httpx.Response(200, json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "alpha"},
                "finish_reason": "stop",
            }],
        })

    # Exercise default SDK retries without backoff delays or network access.
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _: None)
    single_sample = tmp_path / "single.jsonl"
    single_sample.write_text(dataset.read_text().splitlines()[0] + "\n")
    pred_file = tmp_path / "vt_4096.jsonl"
    with openai.OpenAI(
        base_url="http://model.invalid/v1",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
    ) as client:
        if statuses[-1] == 200:
            predictions = infer(adapter, single_sample, pred_file, client)
            assert predictions[0]["pred"] == "alpha"
            assert pred_file.exists()
        else:
            with pytest.raises(RuntimeError, match=f"HTTP {statuses[-1]}"):
                infer(adapter, single_sample, pred_file, client)
            assert not pred_file.exists()

    assert len(calls) == expected_calls
