"""Exercise the real OpenAI client against local, privately signed HTTPS."""

import asyncio
import json
import ssl
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import trustme
from openai import APIConnectionError

import main as adapter


@contextmanager
def embedding_server(ca, hostname="localhost"):
    """Run a local HTTPS server that returns a test embedding."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Return an OpenAI-compatible embedding response."""
            assert self.path == "/v1/embeddings"
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Suppress HTTP server logging during tests."""
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    ca.issue_cert(hostname).configure_cert(context)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request_embedding(url):
    """Request one embedding from the HTTPS test endpoint."""

    async def run():
        """Send the asynchronous embedding request."""
        async with adapter._async_openai_client(url, api_key="test") as client:
            client.max_retries = 0
            client.timeout = 2
            return await client.embeddings.create(model="test", input=["hello"])
    return asyncio.run(run())


@pytest.fixture
def service_ca(monkeypatch, tmp_path):
    """Mount a temporary CA while isolating the test from proxy settings."""
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    path = tmp_path / "service-ca.crt"
    monkeypatch.setattr(adapter, "SERVICE_CA_PATH", path)
    ca = trustme.CA()
    ca.cert_pem.write_to_path(path)
    return ca


def test_mounted_service_ca_trusted(service_ca):
    """Trust a server certificate issued by the mounted service CA."""
    with embedding_server(service_ca) as url:
        assert request_embedding(url).data[0].embedding == [0.1, 0.2]


@pytest.mark.parametrize("configured_bundle", [False, True])
def test_existing_trust_preserved(service_ca, monkeypatch, tmp_path, configured_bundle):
    """Both the previous bundle and mounted CA must remain trusted."""
    previous_ca = trustme.CA()
    bundle = tmp_path / "previous-bundle.pem"
    previous_ca.cert_pem.write_to_path(bundle)
    if configured_bundle:
        monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    else:
        monkeypatch.setattr("certifi.where", lambda: str(bundle))
    for ca in (previous_ca, service_ca):
        with embedding_server(ca) as url:
            assert request_embedding(url).data[0].embedding == [0.1, 0.2]


@pytest.mark.parametrize("failure", ["untrusted", "hostname", "missing_mount"])
def test_invalid_certificates_rejected(service_ca, monkeypatch, tmp_path, failure):
    """Reject untrusted, hostname-mismatched, and unmounted certificates."""
    ca = trustme.CA() if failure == "untrusted" else service_ca
    hostname = "other.example" if failure == "hostname" else "localhost"
    if failure == "missing_mount":
        monkeypatch.setattr(adapter, "SERVICE_CA_PATH", tmp_path / "missing.crt")
    with embedding_server(ca, hostname) as url:
        with pytest.raises(APIConnectionError) as exc:
            request_embedding(url)
        # HTTPX wraps TLS errors; OpenSSL and macOS truststore use different text.
        assert "certificate" in str(exc.value.__cause__).lower()


def test_malformed_mounted_ca_fails_closed(service_ca):
    """Reject a malformed mounted CA instead of disabling verification."""
    adapter.SERVICE_CA_PATH.write_text("not a certificate")
    with pytest.raises(ssl.SSLError):
        adapter._async_openai_client("https://localhost", api_key="test")
