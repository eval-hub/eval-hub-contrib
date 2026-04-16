"""Local integration test for the MTEB adapter with a real HuggingFace model.

Downloads sentence-transformers/all-MiniLM-L6-v2 (~80 MB, cached after first
run) and evaluates it on STSBenchmark.  No external service is needed besides
network access for the initial model/dataset download.

The test spins up a mock eval-hub sidecar HTTP server, runs the full adapter
pipeline, and verifies that the sidecar receives the expected lifecycle events
and final metrics with a meaningful score.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evalhub.adapter import DefaultCallbacks
from main import MTEBAdapter

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
def test_mteb_local_stsbenchmark(tmp_path, mock_sidecar):
    """Run STSBenchmark with all-MiniLM-L6-v2 and verify sidecar events."""
    sidecar_url, events = mock_sidecar

    # Write a minimal job spec pointing at the mock sidecar
    job_spec = {
        "id": "mteb-test-local",
        "provider_id": "mteb",
        "benchmark_id": "STSBenchmark",
        "benchmark_index": 0,
        "model": {
            "url": "local://sentence-transformers",
            "name": HF_MODEL,
        },
        "parameters": {
            "languages": ["eng"],
            "batch_size": 32,
            "verbosity": 2,
            "overwrite_results": True,
        },
        "callback_url": sidecar_url,
        "timeout_seconds": 300,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job_spec))

    # Run the adapter
    adapter = MTEBAdapter(job_spec_path=str(job_path))
    callbacks = DefaultCallbacks.from_adapter(adapter)
    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    callbacks.report_results(results)

    # ── assert adapter results ────────────────────────────────────────
    assert results.id == "mteb-test-local"
    assert results.benchmark_id == "STSBenchmark"
    assert results.model_name == HF_MODEL
    assert results.duration_seconds > 0

    # Should have extracted main_score and correlation metrics
    assert len(results.results) > 0
    metric_names = {r.metric_name for r in results.results}
    assert any("main_score" in name for name in metric_names), (
        f"Expected a 'main_score' metric, got {metric_names}"
    )

    # all-MiniLM-L6-v2 reliably scores > 0.5 on STSBenchmark
    assert results.overall_score is not None, "overall_score should be set"
    assert results.overall_score > 0.5, (
        f"Expected overall_score > 0.5, got {results.overall_score}"
    )

    # ── assert sidecar received expected events ───────────────────────
    assert len(events) >= 2, f"Expected at least 2 sidecar events, got {len(events)}"

    # All events should hit the correct path
    expected_path = "/api/v1/evaluations/jobs/mteb-test-local/events"
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

    # The final completed event should carry metrics with a strong score
    final_event = events[-1]["body"]["benchmark_status_event"]
    assert "metrics" in final_event, "Completed event must include metrics"
    main_score_metrics = {
        k: v for k, v in final_event["metrics"].items() if "main_score" in k
    }
    assert len(main_score_metrics) > 0, "Completed event should have main_score metrics"
    for name, value in main_score_metrics.items():
        assert value > 0.75, f"{name} should be > 0.75, got {value}"
