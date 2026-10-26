"""Verify legacy judge parameters with real OpenAI HTTP requests."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import main as adapter


@pytest.fixture
def judge_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
            })
            body = json.dumps({
                "id": "legacy-judge-response",
                "object": "chat.completion",
                "created": 0,
                "model": "legacy-judge-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("explicit_key", ["legacy-literal-key", None])
def test_k8s_legacy_judge_parameters_reach_direct_endpoint(
    monkeypatch, judge_endpoint, explicit_key
):
    endpoint, requests = judge_endpoint
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-env-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    # An unused, incomplete Secret default must not invalidate a legacy override.
    monkeypatch.setattr(
        adapter, "read_model_auth_key",
        {"judge_url": "http://localhost:8082/unused-judge/v1"}.get,
    )

    def fail_if_called():
        raise AssertionError("direct judge must not borrow the target credential")

    monkeypatch.setattr(adapter, "resolve_model_credentials", fail_if_called)
    url, key, use_primary = adapter._resolve_role_connection(
        "judge", "http://localhost:8082/v1",
        url_override=endpoint, api_key_override=explicit_key,
    )

    async def run():
        async with adapter._async_openai_client(
            url, key, use_model_credentials=use_primary
        ) as client:
            client.max_retries = 0
            client.timeout = 2
            return await client.chat.completions.create(
                model="legacy-judge-model",
                messages=[{"role": "user", "content": "Reply OK."}],
            )

    response = asyncio.run(run())

    assert response.choices[0].message.content == "OK"
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["authorization"] == f"Bearer {explicit_key or 'legacy-env-key'}"
    assert requests[0]["body"]["model"] == "legacy-judge-model"
