"""Tests for the fail-fast network pre-flight (_preflight.py)."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import _preflight as pf


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve 206 for all paths except /missing (404)."""
        self.send_response(404 if self.path == "/missing" else 206)
        self.end_headers()

    def do_HEAD(self):
        """Reset the connection — models a proxy or WAF that drops HEAD but serves GET."""
        self.connection.shutdown(socket.SHUT_RDWR)

    def log_message(self, format, *args):
        """Suppress server-side request logging during tests."""


@pytest.fixture
def http_server():
    """Start a local HTTP server backed by _Handler; yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def silent_server():
    """Yield an http URL that accepts TCP connections but never responds — simulates a blackholed host."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()


@pytest.fixture
def closed_port_url():
    """Yield an http URL whose port is immediately closed so connections are refused."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


# -- parameter validation -----------------------------------------------------------------

def test_normalize_mode_accepts_known_values_case_insensitively():
    """normalize_mode strips whitespace and lowercases the value before comparing."""
    assert pf.normalize_mode(" WARN ") == "warn"


@pytest.mark.parametrize("bad", ["", "maybe", None, 1])
def test_normalize_mode_rejects_unknown_and_names_parameter(bad):
    """normalize_mode raises ValueError naming the parameter for any unrecognised mode."""
    with pytest.raises(ValueError, match="preflight must be one of"):
        pf.normalize_mode(bad)


@pytest.mark.parametrize("bad", [0, -1, "abc", None])
def test_normalize_timeout_rejects_non_positive(bad):
    """normalize_timeout raises ValueError naming the parameter for non-positive or non-numeric input."""
    with pytest.raises(ValueError, match="preflight_timeout_s"):
        pf.normalize_timeout(bad)


def test_normalize_hosts_accepts_list_and_comma_string():
    """normalize_hosts accepts a comma-separated string, a list, or None."""
    assert pf.normalize_hosts("https://a.example, https://b.example") == ["https://a.example", "https://b.example"]
    assert pf.normalize_hosts(["https://a.example"]) == ["https://a.example"]
    assert pf.normalize_hosts(None) == []


def test_normalize_hosts_rejects_non_http_scheme():
    """normalize_hosts raises ValueError when an entry does not use http(s)."""
    with pytest.raises(ValueError, match="http"):
        pf.normalize_hosts(["ftp://a.example"])


# -- which hosts a task needs --------------------------------------------------------------

def test_hf_task_probes_default_hub():
    """An inspect_evals/ task with no overrides probes the default HuggingFace endpoint."""
    assert pf.required_urls("inspect_evals/mmlu_pro", {}) == ["https://huggingface.co"]


def test_hf_endpoint_override_is_probed_instead():
    """HF_ENDPOINT in the job env replaces the default huggingface.co probe target."""
    urls = pf.required_urls("inspect_evals/mmlu_pro", {"HF_ENDPOINT": "https://hf-mirror.internal"})
    assert urls == ["https://hf-mirror.internal"]


@pytest.mark.parametrize("flag", ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"])
def test_offline_mode_skips_hub_probe(flag):
    """Setting HF_HUB_OFFLINE or HF_DATASETS_OFFLINE to 1 suppresses the HF probe."""
    assert pf.required_urls("inspect_evals/mmlu_pro", {flag: "1"}) == []


@pytest.mark.parametrize("task", ["inspect_petri/audit", "petri_bloom/bloom_audit"])
def test_petri_and_bloom_never_probe(task):
    """Petri and Bloom tasks talk only to model endpoints; no dataset host is probed."""
    assert pf.required_urls(task, {}) == []


def test_custom_task_file_is_not_probed_by_default():
    """A bare file path is not matched by any known prefix; no hosts are probed."""
    assert pf.required_urls("/work/my_task.py", {}) == []


def test_bfcl_needs_github_not_huggingface(tmp_path):
    """BFCL downloads from GitHub, not HuggingFace; HF must never be probed for it."""
    env = {"INSPECT_EVALS_CACHE_DIR": str(tmp_path)}
    assert pf.required_urls("inspect_evals/bfcl", env) == ["https://github.com"]
    (tmp_path / "BFCL").mkdir()
    (tmp_path / "BFCL" / "BFCL_v4_simple_python.json").write_text("{}")
    assert pf.required_urls("inspect_evals/bfcl", env) == []


def test_run_preflight_returns_within_deadline_when_dns_stalls(monkeypatch):
    """run_preflight must not block beyond preflight_timeout_s even if DNS stalls."""
    import time as _time

    def _slow_probe(url, timeout_s, env):
        """Sleep for 10× timeout_s to simulate a DNS-stalled probe."""
        _time.sleep(timeout_s * 10)
        return (True, "ok", timeout_s * 10)

    monkeypatch.setattr(pf, "probe", _slow_probe)
    env = {pf.MODE_ENV: "warn", pf.TIMEOUT_ENV: "0.2"}
    start = _time.monotonic()
    pf.run_preflight("inspect_evals/mmlu_pro", env)
    elapsed = _time.monotonic() - start
    assert elapsed < 2.0, f"preflight blocked for {elapsed:.1f}s — DNS deadline not enforced"


def test_user_hosts_are_added_and_deduplicated():
    """INSPECT_PREFLIGHT_HOSTS entries are appended and duplicates with the auto-selected set are removed."""
    env = {pf.HOSTS_ENV: "https://huggingface.co,https://corp.example"}
    assert pf.required_urls("inspect_evals/gsm8k", env) == ["https://huggingface.co", "https://corp.example"]


# -- the probe itself -------------------------------------------------------------------------

def test_probe_reachable(http_server):
    """probe returns ok=True and detail='ok' when the server responds successfully."""
    ok, detail, _ = pf.probe(http_server, 2, {})
    assert ok and detail == "ok"


def test_probe_sends_a_ranged_get_not_head(http_server):
    """A server that resets HEAD must still count as reachable (the real download is a GET)."""
    ok, detail, _ = pf.probe(http_server, 2, {})
    assert ok, detail


def test_probe_http_error_counts_as_reachable(http_server):
    """Any HTTP response (including 4xx) counts as reachable — only connect errors count as unreachable."""
    ok, detail, _ = pf.probe(http_server + "/missing", 2, {})
    assert ok and detail == "HTTP 404"


def test_probe_refused_connection_is_unreachable(closed_port_url):
    """A refused connection is reported as unreachable."""
    ok, _, _ = pf.probe(closed_port_url, 2, {})
    assert not ok


def test_probe_silent_host_times_out_quickly(silent_server):
    """A host that accepts the TCP connection but never responds is reported as timed out."""
    ok, detail, elapsed = pf.probe(silent_server, 0.3, {})
    assert not ok
    assert "no response within 0.3s" in detail
    assert elapsed < 3


def test_probe_uses_the_subprocess_proxy_not_ours(closed_port_url):
    """HTTPS_PROXY in the job env must be honored: a dead proxy makes the target unreachable."""
    ok, _, _ = pf.probe("http://target.invalid", 1, {"HTTP_PROXY": closed_port_url})
    assert not ok


def test_probe_no_proxy_bypasses_proxy(http_server, closed_port_url):
    """NO_PROXY bypasses a dead proxy so the target is reached directly."""
    env = {"HTTP_PROXY": closed_port_url, "NO_PROXY": "127.0.0.1"}
    ok, _, _ = pf.probe(http_server, 2, env)
    assert ok


# -- run_preflight modes ----------------------------------------------------------------------

def _env(url, **extra):
    """Build a minimal env dict pointing HF_ENDPOINT at url with a 0.3s probe timeout."""
    return {"HF_ENDPOINT": url, pf.TIMEOUT_ENV: "0.3", **extra}


def test_fail_mode_raises_with_actionable_message(silent_server):
    """In fail mode, PreflightError names the unreachable host and the available remedies."""
    with pytest.raises(pf.PreflightError) as exc:
        pf.run_preflight("inspect_evals/mmlu_pro", _env(silent_server))
    msg = str(exc.value)
    assert silent_server in msg
    assert "test_data_ref" in msg and "HF_ENDPOINT" in msg and "preflight" in msg


def test_fail_is_the_default_mode(closed_port_url):
    """With no explicit mode set, an unreachable host raises PreflightError."""
    with pytest.raises(pf.PreflightError):
        pf.run_preflight("inspect_evals/mmlu_pro", _env(closed_port_url))


def test_warn_mode_logs_and_continues(closed_port_url, caplog):
    """In warn mode, an unreachable host logs a warning and does not raise."""
    with caplog.at_level("WARNING"):
        pf.run_preflight("inspect_evals/mmlu_pro", _env(closed_port_url, **{pf.MODE_ENV: "warn"}))
    assert "continuing because" in caplog.text


def test_off_mode_skips_probe_entirely(closed_port_url):
    """off mode returns immediately without probing any host."""
    pf.run_preflight("inspect_evals/mmlu_pro", _env(closed_port_url, **{pf.MODE_ENV: "off"}))


def test_reachable_host_passes(http_server):
    """run_preflight completes without raising when all required hosts are reachable."""
    pf.run_preflight("inspect_evals/mmlu_pro", _env(http_server))


def test_invalid_env_mode_names_the_variable():
    """An invalid INSPECT_PREFLIGHT value raises ValueError naming the env var."""
    with pytest.raises(ValueError, match=pf.MODE_ENV):
        pf.run_preflight("inspect_evals/mmlu_pro", {pf.MODE_ENV: "sometimes"})
