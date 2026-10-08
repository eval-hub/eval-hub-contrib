import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from evalhub.adapter import JobSpec

import main
from _evaluation import build_prompt, load_table, make_evaluator, score_answer


@pytest.mark.parametrize(
    "semantic,answer,truth,expected",
    [
        ("boolean", "yes", "True", True),
        ("boolean", "False", "True", False),
        ("number", "23.123", "23.129", True),
        ("category", "London", "london", False),
        ("list[category]", "['B', 'A']", "['A', 'B']", True),
        ("list[number]", "[2, 1]", "[1, 2]", True),
    ],
)
def test_official_comparison(semantic, answer, truth, expected):
    row = {"type": semantic, "answer": truth, "sample_answer": truth}
    evaluator = make_evaluator(Dataset.from_list([row]))
    assert score_answer(evaluator, answer, row, False) is expected


def test_lite_uses_sample_answer():
    row = {"type": "number", "answer": "100", "sample_answer": "5"}
    evaluator = make_evaluator(Dataset.from_list([row]))
    assert score_answer(evaluator, "5", row, True)
    assert not score_answer(evaluator, "5", row, False)


def test_prompt_preserves_rows_columns_and_ignores_gold_column_hint():
    row = {
        "dataset": "001_Test",
        "question": "How many?",
        "type": "number",
        "columns_used": "secret",
    }
    table = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    prompt = build_prompt(row, table, 10000)
    assert "A,B\n1,3\n2,4" in prompt
    assert "secret" not in prompt
    with pytest.raises(ValueError, match="exceeds"):
        build_prompt(row, table, 10)


@pytest.mark.parametrize("lite,filename", [(True, "sample"), (False, "all")])
def test_table_loader_pins_revision_and_variant(monkeypatch, tmp_path, lite, filename):
    path = tmp_path / "table.parquet"
    pd.DataFrame({"A": [1]}).to_parquet(path)
    download = MagicMock(return_value=str(path))
    monkeypatch.setattr("_evaluation.hf_hub_download", download)
    assert len(load_table("001_Test", lite, None)) == 1
    assert download.call_args.args[1] == f"data/001_Test/{filename}.parquet"
    assert download.call_args.kwargs["revision"] == main.REVISION


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "all"])
def test_invalid_limit(value):
    with pytest.raises((ValueError, TypeError)):
        main.positive_int(value, "num_examples")


@pytest.fixture
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            answer = "2" if len(requests) == 1 else "9"
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}/v1", requests
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


@pytest.mark.parametrize("with_export", [False, True])
def test_adapter_http_generation_scoring_and_diagnostics(
    monkeypatch, tmp_path, server, with_export
):
    url, requests = server
    qa = Dataset.from_list(
        [
            {
                "dataset": "001_Test",
                "question": "Count?",
                "type": "number",
                "answer": "100",
                "sample_answer": "2",
            },
            {
                "dataset": "001_Test",
                "question": "Count again?",
                "type": "number",
                "answer": "100",
                "sample_answer": "2",
            },
        ]
    )
    monkeypatch.setattr(main, "load_questions", lambda *args: qa)
    tables = MagicMock(return_value=pd.DataFrame({"A": [1, 2]}))
    monkeypatch.setattr(main, "load_table", tables)
    monkeypatch.setattr(main, "read_model_auth_key", lambda name: None)
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: None)
    monkeypatch.setattr(main.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    config = JobSpec.model_validate(
        {
            "id": "test",
            "provider_id": "databench",
            "callback_url": "http://localhost:8081",
            "benchmark_id": "databench_lite",
            "benchmark_index": 0,
            "model": {"url": url, "name": "test"},
            "parameters": {"num_examples": 2},
        }
    )
    callbacks = MagicMock()
    def check_status_timing(status):
        if status.phase == main.JobPhase.INITIALIZING:
            assert not tables.called
            assert not requests
        elif status.phase == main.JobPhase.RUNNING_EVALUATION and status.progress == 0:
            assert not requests
        elif status.phase == main.JobPhase.POST_PROCESSING:
            assert len(requests) == 2
            assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 2
            assert not (tmp_path / "results.json").exists()
        elif status.phase == main.JobPhase.PERSISTING_ARTIFACTS:
            assert (tmp_path / "results.json").exists()

    callbacks.report_status.side_effect = check_status_timing
    if with_export:
        from evalhub.adapter import OCIArtifactResult

        config = JobSpec.model_validate(
            {
                **config.model_dump(),
                "exports": {
                    "oci": {
                        "coordinates": {
                            "oci_host": "quay.io",
                            "oci_repository": "test/results",
                            "oci_tag": "smoke",
                        }
                    }
                },
            }
        )
        callbacks.create_oci_artifact.return_value = OCIArtifactResult(
            digest="sha256:test", reference="quay.io/test/results@sha256:test"
        )
    adapter = object.__new__(main.DataBenchAdapter)
    result = adapter.run_benchmark_job(config, callbacks)
    phases = [call.args[0].phase for call in callbacks.report_status.call_args_list]
    assert phases == [
        main.JobPhase.INITIALIZING,
        main.JobPhase.LOADING_DATA,
        main.JobPhase.RUNNING_EVALUATION,
        main.JobPhase.RUNNING_EVALUATION,
        main.JobPhase.RUNNING_EVALUATION,
        main.JobPhase.POST_PROCESSING,
        main.JobPhase.PERSISTING_ARTIFACTS,
    ]
    if with_export:
        assert result.oci_artifact.reference == "quay.io/test/results@sha256:test"
        assert callbacks.create_oci_artifact.call_args.args[0].files_path == tmp_path
    assert result.overall_score == 0.5
    assert result.num_examples_evaluated == 2
    assert result.results[0].metric_name == "accuracy"
    assert tables.call_count == 1
    assert len(requests) == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "samples.jsonl").read_text().splitlines()
    ]
    assert [record["correct"] for record in records] == [True, False]
    assert records[0]["expected_answer"] == "2"
    assert records[0]["model_answer"] == "2"
    assert (
        json.loads((tmp_path / "results.json").read_text())["dataset_revision"]
        == main.REVISION
    )


def test_api_failure_is_not_a_zero_score(monkeypatch, tmp_path):
    qa = Dataset.from_list(
        [
            {
                "dataset": "001_Test",
                "question": "Q",
                "type": "number",
                "answer": "1",
                "sample_answer": "1",
            }
        ]
    )
    monkeypatch.setattr(main, "load_questions", lambda *args: qa)
    monkeypatch.setattr(main, "load_table", lambda *args: pd.DataFrame({"A": [1]}))
    monkeypatch.setattr(main, "read_model_auth_key", lambda name: None)
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: None)
    monkeypatch.setattr(main.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    client = MagicMock()
    client.__enter__.return_value.chat.completions.create.side_effect = RuntimeError(
        "API failed"
    )
    monkeypatch.setattr(main.openai, "OpenAI", lambda **kwargs: client)
    config = JobSpec.model_validate(
        {
            "id": "test",
            "provider_id": "databench",
            "callback_url": "http://localhost:8081",
            "benchmark_id": "databench",
            "benchmark_index": 0,
            "model": {"url": "http://localhost/v1", "name": "test"},
            "parameters": {},
        }
    )
    with pytest.raises(RuntimeError, match="API failed"):
        object.__new__(main.DataBenchAdapter).run_benchmark_job(config, MagicMock())
    assert not (tmp_path / "results.json").exists()


def test_csv_quotes_strings_and_preserves_missing_cells():
    row = {"dataset": "001_Test", "question": "Q", "type": "category"}
    table = pd.DataFrame({"text": ["a,b", "two\nlines"], "value": [None, "x"]})
    prompt = build_prompt(row, table, 10000)
    assert 'text,value\n"a,b",\n"two\nlines",x\n' in prompt


@pytest.mark.parametrize(
    "semantic,answer",
    [("number", "not a number"), ("boolean", "maybe"), ("list[number]", "invalid")],
)
def test_invalid_nonempty_answers_follow_official_checker(semantic, answer):
    row = {"type": semantic, "answer": "1", "sample_answer": "1"}
    assert not score_answer(
        make_evaluator(Dataset.from_list([row])), answer, row, False
    )


@pytest.mark.parametrize("content", [None, "", "   "])
def test_missing_answer_fails_job(monkeypatch, tmp_path, content):
    qa = Dataset.from_list(
        [
            {
                "dataset": "001_Test",
                "question": "Q",
                "type": "number",
                "answer": "1",
                "sample_answer": "1",
            }
        ]
    )
    monkeypatch.setattr(main, "load_questions", lambda *args: qa)
    monkeypatch.setattr(main, "load_table", lambda *args: pd.DataFrame({"A": [1]}))
    monkeypatch.setattr(main, "read_model_auth_key", lambda name: None)
    monkeypatch.setattr(main, "resolve_model_credentials", lambda: None)
    monkeypatch.setattr(main.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    client = MagicMock()
    client.__enter__.return_value.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    monkeypatch.setattr(main.openai, "OpenAI", lambda **kwargs: client)
    config = JobSpec.model_validate(
        {
            "id": "test",
            "provider_id": "databench",
            "benchmark_id": "databench",
            "benchmark_index": 0,
            "callback_url": "http://localhost:8081",
            "model": {"url": "http://localhost/v1", "name": "test"},
            "parameters": {},
        }
    )
    with pytest.raises(RuntimeError, match="no answer"):
        object.__new__(main.DataBenchAdapter).run_benchmark_job(config, MagicMock())


def test_entrypoint_reports_result_via_callbacks(monkeypatch):
    adapter = MagicMock()
    callbacks = MagicMock()
    monkeypatch.setattr(main, "configure_telemetry", lambda: None)
    monkeypatch.setattr(main, "DataBenchAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(
        main.DefaultCallbacks, "from_adapter", lambda instance: callbacks
    )
    main.main()
    callbacks.report_results.assert_called_once_with(
        adapter.run_benchmark_job.return_value
    )


def test_entrypoint_reports_failed_status(monkeypatch):
    adapter = MagicMock()
    adapter.run_benchmark_job.side_effect = RuntimeError("failed")
    callbacks = MagicMock()
    monkeypatch.setattr(main, "configure_telemetry", lambda: None)
    monkeypatch.setattr(main, "DataBenchAdapter", lambda **kwargs: adapter)
    monkeypatch.setattr(
        main.DefaultCallbacks, "from_adapter", lambda instance: callbacks
    )
    with pytest.raises(RuntimeError):
        main.main()
    assert callbacks.report_status.call_args.args[0].status == main.JobStatus.FAILED
    callbacks.report_results.assert_not_called()
