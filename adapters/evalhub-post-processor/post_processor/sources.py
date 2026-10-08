"""Materialize API data references into isolated local directories.

Secrets and PVCs are supplied by the runtime as read-only mounts. The adapter
does not need Kubernetes API privileges and never treats a Secret name as a key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

from .config import source_type, text_value
from .data import within
from .transport import Sidecar


def copy_data(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise ValueError("Referenced data mount or sub_path does not exist")
    files = [source] if source.is_file() else sorted(source.rglob("*"))
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.is_symlink():
            raise ValueError("Data sources must not contain symbolic links")
        if path.is_file():
            name = path.name if source.is_file() else str(path.relative_to(source))
            target = within(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return destination


class Sources:
    def __init__(self, sidecar: Sidecar, *, exports: dict | None = None):
        self.sidecar = sidecar
        self.exports = exports or {}

    def secret(self, name: str) -> dict[str, str]:
        name = text_value(name, "secret_ref")
        if "/" in name or name in {".", ".."}:
            raise ValueError("secret_ref must name a Secret, not a filesystem path")
        root = Path(
            os.getenv("EVALHUB_POST_PROCESSOR_SECRET_ROOT", "/var/run/secrets/post-processor")
        )
        directory = root / name
        if not directory.is_dir():
            raise ValueError(
                f"Secret {name!r} must be projected under EVALHUB_POST_PROCESSOR_SECRET_ROOT"
            )
        # Kubernetes Secret projection uses symlinks; only the trusted mount root is read here.
        return {p.name: p.read_text().strip() for p in directory.iterdir() if p.is_file()}

    def download(self, ref: dict, destination: Path) -> Path:
        kind = source_type(ref)
        if kind == "eval_job":
            raise ValueError("eval_job must first be resolved into individual benchmark artifacts")
        destination.mkdir(parents=True, exist_ok=True)
        if kind == "oci":
            from .oci import download_oci

            return download_oci(ref[kind], destination, self)
        return getattr(self, f"_{kind}")(ref[kind], destination)

    def _pvc(self, ref: dict, destination: Path) -> Path:
        claim = text_value(ref.get("claim_name"), "pvc.claim_name")
        mounts = json.loads(os.getenv("EVALHUB_POST_PROCESSOR_PVC_MOUNTS", "{}"))
        if claim not in mounts:
            raise ValueError(f"PVC {claim!r} needs a path in EVALHUB_POST_PROCESSOR_PVC_MOUNTS")
        root = Path(mounts[claim])
        return copy_data(within(root, ref.get("sub_path", "")), destination)

    def _s3(self, ref: dict, destination: Path) -> Path:
        import boto3
        from botocore.exceptions import ClientError

        bucket = text_value(ref.get("bucket"), "s3.bucket")
        key = ref.get("key", "").strip("/")
        creds = self.secret(ref["secret_ref"]) if ref.get("secret_ref") else {}
        if creds:
            for required in (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
                "AWS_S3_ENDPOINT",
            ):
                if not creds.get(required):
                    raise ValueError(f"S3 Secret is missing {required}")
        client = boto3.client(
            "s3",
            aws_access_key_id=creds.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=creds.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=creds.get("AWS_SESSION_TOKEN"),
            region_name=creds.get("AWS_DEFAULT_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            endpoint_url=creds.get("AWS_S3_ENDPOINT") or os.getenv("AWS_S3_ENDPOINT"),
        )
        try:
            keys = []
            if key:
                try:
                    client.head_object(Bucket=bucket, Key=key)
                    keys = [key]
                except ClientError as exc:
                    if str(exc.response.get("Error", {}).get("Code")) not in {
                        "404",
                        "NoSuchKey",
                        "NotFound",
                    }:
                        raise
            prefix = key + "/" if key else ""
            if not keys:
                for page in client.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=prefix
                ):
                    keys.extend(
                        item["Key"]
                        for item in page.get("Contents", [])
                        if not item["Key"].endswith("/")
                    )
            if not keys:
                raise ValueError("S3 reference contains no objects")
            for item in keys:
                relative = Path(item).name if item == key else item[len(prefix) :]
                path = within(destination, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, item, str(path))
        finally:
            client.close()
        return destination

    def _hf(self, ref: dict, destination: Path) -> Path:
        from huggingface_hub import snapshot_download

        repo = text_value(ref.get("repo_id"), "hf.repo_id")
        creds = self.secret(ref["secret_ref"]) if ref.get("secret_ref") else {}
        if ref.get("secret_ref") and not creds.get("token"):
            raise ValueError("Hugging Face Secret must contain token")
        snapshot = destination / "snapshot"
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=ref.get("revision"),
            token=creds.get("token") or os.getenv("HF_TOKEN") or False,
            local_dir=snapshot,
        )
        selected = within(snapshot, ref.get("sub_path", ""))
        return copy_data(selected, destination / "data")

    def _git(self, ref: dict, destination: Path) -> Path:
        url = text_value(ref.get("url"), "git.url")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("git.url must be an HTTP(S) URL without embedded credentials")
        revision = text_value(ref.get("ref"), "git.ref")
        if revision.startswith("-"):
            raise ValueError("git.ref must not be an option")
        environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        if ref.get("secret_ref"):
            if parsed.scheme != "https":
                raise ValueError("Private git sources require HTTPS")
            creds = self.secret(ref["secret_ref"])
            if not creds.get("username") or not creds.get("password"):
                raise ValueError("Git Secret must contain username and password")
            askpass = destination / "askpass.sh"
            askpass.write_text(
                '#!/bin/sh\ncase "$1" in *Username*) printf "%s" "$PP_GIT_USERNAME";; *) printf "%s" "$PP_GIT_PASSWORD";; esac\n'
            )
            askpass.chmod(0o700)
            environment.update(
                GIT_ASKPASS=str(askpass),
                PP_GIT_USERNAME=creds["username"],
                PP_GIT_PASSWORD=creds["password"],
            )
        repo = destination / "repository"
        try:
            for args in (
                ["git", "init", "--quiet", str(repo)],
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "credential.helper=",
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "--",
                    url,
                    revision,
                ],
                ["git", "-C", str(repo), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            ):
                subprocess.run(args, env=environment, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError(
                "Git download failed; check the repository, revision, and projected credentials"
            ) from exc
        # Do not copy git internals into a calibration data directory.
        selected = within(repo, ref.get("sub_path", ""))
        if selected == repo:
            shutil.rmtree(repo / ".git")
        return copy_data(selected, destination / "data")

    def _mlflow(self, ref: dict, destination: Path) -> Path:
        run_id = text_value(ref.get("run_id"), "mlflow.run_id")
        requested = ref.get("artifact_path", "")
        within(destination, requested)  # Validate before putting the path in any HTTP request.
        base = self.sidecar.base_url
        headers = {}
        if os.getenv("MLFLOW_WORKSPACE"):
            headers["X-MLFLOW-WORKSPACE"] = os.environ["MLFLOW_WORKSPACE"]
        client = self.sidecar.client

        def get(path: str, params: dict) -> dict:
            response = client.get(
                base + path, params=params, headers=headers, follow_redirects=False
            )
            response.raise_for_status()
            return response.json()

        info = get("/api/2.0/mlflow/runs/get", {"run_id": run_id})["run"]["info"]
        uri = text_value(info.get("artifact_uri"), "MLflow run artifact_uri")
        anchor = "/api/2.0/mlflow-artifacts/artifacts/"
        if uri.startswith("mlflow-artifacts:"):
            run_root = urlsplit(uri).path.lstrip("/")
            if run_root.startswith("workspaces/"):
                _, workspace, run_root = run_root.split("/", 2)
                headers["X-MLFLOW-WORKSPACE"] = workspace
        elif anchor in uri:
            run_root = uri.split(anchor, 1)[1].strip("/")
        else:
            # The upstream artifacts proxy serves the run root at this path even
            # when the tracking server records a file/S3 backend artifact URI.
            run_root = f"{info['experiment_id']}/{run_id}/artifacts"
        within(destination, run_root)

        def entries(path: str) -> list[dict]:
            page_token = None
            found = []
            seen_tokens = set()
            while True:
                params = {"run_id": run_id, "path": path}
                if page_token:
                    params["page_token"] = page_token
                result = get("/api/2.0/mlflow/artifacts/list", params)
                found.extend(result.get("files", []))
                page_token = result.get("next_page_token")
                if not page_token:
                    return found
                if page_token in seen_tokens:
                    raise ValueError("MLflow artifact listing returned a repeated page token")
                seen_tokens.add(page_token)

        queue = entries("")
        files = []
        visited = set()
        while queue:
            item = queue.pop(0)
            path = text_value(item.get("path"), "MLflow artifact path")
            within(destination, path)
            relevant = (
                not requested
                or path == requested
                or path.startswith(requested.rstrip("/") + "/")
                or requested.startswith(path.rstrip("/") + "/")
            )
            if not relevant:
                continue
            if path in visited:
                raise ValueError("MLflow artifact listing contains duplicate or cyclic paths")
            visited.add(path)
            if item.get("is_dir"):
                queue.extend(entries(path))
            else:
                files.append(path)
        # logs_path can be an adapter-local path. Treat it as a hint only if it
        # identifies actual artifacts in the run; never open it on this machine.
        hint = ref.get("path_hint")
        if not requested and isinstance(hint, str) and hint:
            for candidate in (hint.lstrip("/"), Path(hint).name):
                selected = [
                    p for p in files if p == candidate or p.startswith(candidate.rstrip("/") + "/")
                ]
                if selected:
                    files = selected
                    break
        if not files:
            raise ValueError("MLflow run has no artifacts at the requested path")
        for path in files:
            target = within(destination, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with client.stream(
                "GET",
                base + anchor + quote(run_root + "/" + path, safe="/"),
                headers=headers,
                follow_redirects=False,
            ) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
        return destination
