"""Authenticated-by-sidecar API access and reliable benchmark event delivery."""

from __future__ import annotations

import os
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from evalhub.adapter import JobCallbacks, JobPhase, JobResults, JobSpec, JobStatusUpdate

NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def _tenant() -> str:
    """Resolve the tenant namespace projected into the adapter pod."""
    try:
        tenant = NAMESPACE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        tenant = ""
    if not tenant:
        tenant = os.getenv("EVALHUB_TENANT", "").strip()
    if not tenant:
        tenant = os.getenv("MLFLOW_WORKSPACE", "").strip()
    if not tenant:
        raise RuntimeError(
            "Eval Hub requests require the projected pod namespace or EVALHUB_TENANT"
        )
    return tenant


def tls_context() -> ssl.SSLContext:
    """Use the deployment CA without disabling certificate verification."""
    return ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE") or None)


class Sidecar:
    def __init__(self, base_url: str, *, client: httpx.Client | None = None):
        """Connect to the job-local sidecar or use an injected HTTP client."""
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("callback_url must be the HTTP(S) sidecar base URL")
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(verify=tls_context(), timeout=60)
        self._owns_client = client is None

    def get(self, path: str, **kwargs: Any) -> dict:
        """Fetch a tenant-scoped JSON object without following redirects."""
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-Tenant"] = _tenant()
        response = self.client.get(
            self.base_url + path, headers=headers, follow_redirects=False, **kwargs
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected an object from sidecar endpoint {path}")
        return data

    def event(self, job_id: str, event: dict) -> None:
        """Retry transient delivery failures before reporting the event as failed."""
        url = f"{self.base_url}/api/v1/evaluations/jobs/{quote(job_id, safe='')}/events"
        for attempt in range(3):
            try:
                response = self.client.post(
                    url,
                    json={"benchmark_status_event": event},
                    headers={"X-Tenant": _tenant()},
                    follow_redirects=False,
                )
                response.raise_for_status()
                return
            except httpx.TransportError:
                if attempt == 2:
                    raise
            except httpx.HTTPStatusError as exc:
                if attempt == 2 or exc.response.status_code < 500:
                    raise
            time.sleep(0.2 * 2**attempt)

    def close(self) -> None:
        """Close the HTTP client only when this instance created it."""
        if self._owns_client:
            self.client.close()


class PostProcessorCallbacks(JobCallbacks):
    """Use the normal /events protocol, with no synthetic evaluation metrics."""

    def __init__(self, spec: JobSpec, sidecar: Sidecar):
        """Bind job identity and start time to the sidecar event transport."""
        self.spec = spec
        self.sidecar = sidecar
        self.started_at = datetime.now(UTC)

    def _event(self, status: str) -> dict:
        """Build the fields common to every benchmark status event."""
        return {
            "id": self.spec.benchmark_id,
            "provider_id": self.spec.provider_id,
            "benchmark_index": self.spec.benchmark_index,
            "status": status,
            "started_at": self.started_at.isoformat(),
        }

    def report_status(self, update: JobStatusUpdate) -> None:
        """Send a running or failed benchmark status event."""
        status = getattr(update.status, "value", update.status)
        event = self._event(status)
        if update.phase is not None:
            event["phase"] = getattr(update.phase, "value", update.phase)
        error = update.error_message
        if error is not None:
            event["error_message"] = error.model_dump(exclude_none=True)
            event["error_message"]["message_origin"] = "adapter"
        if status == "failed":
            now = datetime.now(UTC)
            event["completed_at"] = now.isoformat()
            event["duration_seconds"] = (now - self.started_at).total_seconds()
        self.sidecar.event(self.spec.id, event)

    def report_results(self, results: JobResults) -> None:
        """Send the completed event with post-processing outputs."""
        event = self._event("completed")
        event.update(
            phase=JobPhase.COMPLETED.value,
            completed_at=results.completed_at.isoformat(),
            duration_seconds=results.duration_seconds,
            additional_info=results.additional_info or {},
        )
        self.sidecar.event(self.spec.id, event)

    def create_oci_artifact(self, spec: Any) -> Any:
        """Reject artifact creation because this adapter emits event data only."""
        raise NotImplementedError("Post-processing results are returned through additional_info")
