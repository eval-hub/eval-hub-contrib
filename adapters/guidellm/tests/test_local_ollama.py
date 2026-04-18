"""Local integration test for the GuideLLM adapter against a real Ollama instance.

Requires:
- Ollama running at http://localhost:11434 with a model loaded
- The adapter venv with all dependencies installed

The test spins up a mock eval-hub sidecar HTTP server, runs the full adapter
pipeline, and verifies that the sidecar receives the expected lifecycle events
and final metrics.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from evalhub.adapter import DefaultCallbacks
from main import GuideLLMAdapter


def _ollama_available():
    """Check if Ollama is reachable and has at least one model."""
    try:
        resp = requests.get("http://localhost:11434/v1/models", timeout=3)
        data = resp.json()
        return bool(data.get("data"))
    except Exception:
        return False


def _ollama_model():
    """Return the first available Ollama model name."""
    resp = requests.get("http://localhost:11434/v1/models", timeout=3)
    return resp.json()["data"][0]["id"]


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
@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running on localhost:11434")
def test_guidellm_local_ollama(tmp_path, mock_sidecar):
    """Run a minimal GuideLLM benchmark against Ollama and verify sidecar events."""
    sidecar_url, events = mock_sidecar
    model_name = _ollama_model()

    # Write a minimal job spec pointing at the mock sidecar
    job_spec = {
        "id": "guidellm-test-local",
        "provider_id": "guidellm",
        "benchmark_id": "local_quick",
        "benchmark_index": 0,
        "model": {
            "url": "http://localhost:11434/v1",
            "name": model_name,
        },
        "parameters": {
            "profile": "synchronous",
            "max_requests": 2,
            "data": "prompt_tokens=30,output_tokens=10",
            "request_type": "chat_completions",
            "warmup": "0",
            "detect_saturation": False,
            "backend_kwargs": {"validate_backend": False},
        },
        "callback_url": sidecar_url,
        "timeout_seconds": 120,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job_spec))

    # Run the adapter
    adapter = GuideLLMAdapter(job_spec_path=str(job_path))
    callbacks = DefaultCallbacks.from_adapter(adapter)
    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    callbacks.report_results(results)

    # ── assert adapter results ────────────────────────────────────────
    assert results.id == "guidellm-test-local"
    assert results.benchmark_id == "local_quick"
    assert results.model_name == model_name
    assert results.duration_seconds > 0

    # Expect at least some throughput metrics
    metric_names = {r.metric_name for r in results.results}
    assert "requests_per_second" in metric_names

    # ── assert sidecar received expected events ───────────────────────
    assert len(events) >= 2, f"Expected at least 2 sidecar events, got {len(events)}"

    # All events should hit the correct path
    expected_path = f"/api/v1/evaluations/jobs/guidellm-test-local/events"
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
    assert "requests_per_second" in final_event["metrics"]
    assert final_event["metrics"]["requests_per_second"] > 0
