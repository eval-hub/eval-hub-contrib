"""Resolve local auth from the invocation spec, never a second on-disk spec."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from evalhub.adapter import JobSpec
from evalhub.adapter.auth import ModelCredentials, read_model_auth_key

from input_collection import read_bounded


def resolve_job_credentials(spec: JobSpec) -> ModelCredentials:
    if os.getenv("EVALHUB_MODE") == "k8s":
        # Raw cluster credentials remain exclusively sidecar-owned.
        return ModelCredentials(api_key="api-key:ref" if spec.model.auth else None)
    if spec.model.auth is None:
        return ModelCredentials()
    root = _local_auth_root(spec)
    ca_path = root / "ca_cert"
    key = read_job_auth_key(spec, "api-key")
    if ca_path.exists() or ca_path.is_symlink():
        read_bounded(ca_path, 1024 * 1024)
    return ModelCredentials(
        api_key=key or None, ca_cert_path=ca_path if ca_path.exists() else None
    )


def _local_auth_root(spec: JobSpec) -> Path:
    ref = urlparse(spec.model.auth.secret_ref)
    if (
        ref.scheme != "file"
        or ref.netloc
        or not ref.path.startswith("/")
        or ref.query
        or ref.fragment
    ):
        raise ValueError(
            "Local model.auth.secret_ref must be an absolute file:/// directory URL"
        )
    root = Path(unquote(ref.path))
    if any(path.is_symlink() for path in (root, *root.parents)) or not root.is_dir():
        raise ValueError(
            "Local model auth must be a regular directory without symlinks"
        )
    return root


def read_job_auth_key(spec: JobSpec, key: str) -> str | None:
    """Read projected cluster references or the configured local auth directory."""
    if os.getenv("EVALHUB_MODE") == "k8s":
        return read_model_auth_key(key)
    if spec.model.auth is None:
        return None
    path = _local_auth_root(spec) / key
    if path.exists() or path.is_symlink():
        return read_bounded(path, 16 * 1024).decode().strip() or None
    return None
