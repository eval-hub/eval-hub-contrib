"""Local integration test for the LightEval adapter against a real OpenAI-compatible endpoint.

Requires:
- An OpenAI-compatible server running at http://localhost:11434 with a model loaded
  (e.g. Ollama, vLLM-CPU, or any server exposing /v1/models and /v1/chat/completions)
- The adapter venv with all dependencies installed

The test spins up a mock eval-hub sidecar HTTP server, runs the full adapter
pipeline, and verifies that the sidecar receives the expected lifecycle events
and final metrics.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from evalhub.adapter import DefaultCallbacks
from main import LightEvalAdapter


# Must be a non-thinking model.  Thinking models (e.g. qwen3) wrap output in
# <think>…</think> tags.  At least Ollama's OpenAI-compatible API moves all
# thinking tokens into a separate "reasoning" field, leaving "content" empty.
# LightEval reads "content", gets an empty string, and every metric evaluates
# to zero.  This behaviour may be Ollama-specific, but since both Ollama and
# vLLM are actively used serving runtimes we restrict to models that are known
# to work correctly across both.
_NON_THINKING_MODELS = {"qwen2.5:0.5b", "Qwen/Qwen2.5-0.5B-Instruct"}


def _server_available():
    """Check if the server is reachable and serves a known non-thinking model."""
    try:
        resp = requests.get("http://localhost:11434/v1/models", timeout=3)
        ids = {m["id"] for m in resp.json().get("data", [])}
        return bool(ids & _NON_THINKING_MODELS)
    except Exception:
        return False


def _server_model():
    """Return the first available non-thinking model name from the server."""
    resp = requests.get("http://localhost:11434/v1/models", timeout=3)
    ids = [m["id"] for m in resp.json()["data"]]
    for mid in ids:
        if mid in _NON_THINKING_MODELS:
            return mid
    raise RuntimeError(f"No non-thinking model found on server; available: {ids}")


# ── mock sidecar ──────────────────────────────────────────────────────


class _SidecarHandler(BaseHTTPRequestHandler):
    """Collects POSTed events into the server's ``events`` list."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        self.server.events.append({"path": self.path, "body": body})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence request logs


@pytest.fixture()
def mock_sidecar():
    """Start a mock sidecar on a free port; yield (url, events); shut down."""
    server = HTTPServer(("127.0.0.1", 0), _SidecarHandler)
    server.events = []
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", server.events
    server.shutdown()


# ── test ──────────────────────────────────────────────────────────────


@pytest.mark.local
@pytest.mark.openai_endpoint
@pytest.mark.skipif(not _server_available(), reason="OpenAI-compatible server not running on localhost:11434")
def test_lighteval_local_openai(tmp_path, mock_sidecar):
    """Run a minimal LightEval benchmark against an OpenAI-compatible endpoint and verify sidecar events.

    Uses gsm8k (extractive_match metric) with a non-thinking model so that
    the generative endpoint actually produces scorable content.  Thinking
    models (e.g. qwen3) place all reasoning tokens in a separate field,
    leaving "content" empty and every metric at zero.
    """
    sidecar_url, events = mock_sidecar
    model_name = _server_model()

    job_spec = {
        "id": "lighteval-test-local",
        "provider_id": "lighteval",
        "benchmark_id": "gsm8k",
        "benchmark_index": 0,
        "model": {
            "url": "http://localhost:11434/v1",
            "name": model_name,
        },
        "num_examples": 20,
        "parameters": {
            "provider": "endpoint",
            "batch_size": 1,
            "num_few_shot": 0,
        },
        "callback_url": sidecar_url,
        "timeout_seconds": 120,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job_spec))

    # Run the adapter
    adapter = LightEvalAdapter(job_spec_path=str(job_path))
    callbacks = DefaultCallbacks.from_adapter(adapter)
    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    callbacks.report_results(results)

    # ── assert adapter results ────────────────────────────────────────
    assert results.id == "lighteval-test-local"
    assert results.benchmark_id == "gsm8k"
    assert results.model_name == model_name
    assert results.duration_seconds > 0

    # Should have extracted at least one metric (extractive_match for gsm8k)
    assert len(results.results) > 0
    metric_names = {r.metric_name for r in results.results}
    assert any("extractive_match" in name for name in metric_names), (
        f"Expected an 'extractive_match' metric, got {metric_names}"
    )

    # The non-thinking model should produce a positive score
    em_results = [r for r in results.results if "extractive_match" in r.metric_name]
    for r in em_results:
        assert r.metric_value > 0, f"{r.metric_name} should be > 0, got {r.metric_value}"
    assert results.overall_score is not None, "overall_score should be set for gsm8k"
    assert results.overall_score > 0, f"overall_score should be > 0, got {results.overall_score}"

    # ── assert sidecar received expected events ───────────────────────
    assert len(events) >= 2, f"Expected at least 2 sidecar events, got {len(events)}"

    # All events should hit the correct path
    expected_path = "/api/v1/evaluations/jobs/lighteval-test-local/events"
    for ev in events:
        assert ev["path"] == expected_path

    # Extract states from the events
    states = [
        ev["body"]["benchmark_status_event"]["state"]
        for ev in events
        if "benchmark_status_event" in ev["body"]
    ]
    assert "running" in states, f"Expected 'running' state in events, got {states}"
    assert states[-1] == "completed", f"Last event should be 'completed', got {states[-1]}"

    # The final completed event should carry metrics
    final_event = events[-1]["body"]["benchmark_status_event"]
    assert "metrics" in final_event, "Completed event must include metrics"
    assert len(final_event["metrics"]) > 0
