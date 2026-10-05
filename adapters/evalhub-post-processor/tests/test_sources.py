import hashlib
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

from post_processor.oci import extract_layer
from post_processor.sources import Sources
from post_processor.transport import Sidecar


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def archive(name="samples.json", content=b'[{"sample_id":"a","prediction":0.8}]', *, link=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        else:
            member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    return output.getvalue()


def test_mlflow_uses_only_sidecar_even_with_tracking_url_and_redirect_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "upstream-secret")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "sidecar"
        assert "authorization" not in request.headers
        if request.url.path.endswith("/runs/get"):
            return httpx.Response(
                200,
                json={
                    "run": {
                        "info": {
                            "artifact_uri": "mlflow-artifacts:/workspaces/team/1/run/artifacts"
                        }
                    }
                },
            )
        assert request.headers["X-MLFLOW-WORKSPACE"] == "team"
        if request.url.path.endswith("/artifacts/list"):
            if "page_token" not in request.url.params:
                return httpx.Response(
                    200, json={"files": [{"path": "a.json"}], "next_page_token": "page2"}
                )
            return httpx.Response(200, json={"files": [{"path": "b.json"}]})
        assert request.url.path.startswith("/api/2.0/mlflow-artifacts/artifacts/1/run/artifacts/")
        return httpx.Response(200, content=b"[]")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        sources = Sources(Sidecar("http://sidecar", client=client))
        data = sources.download(
            {"mlflow": {"run_id": "run", "artifact_path": ""}}, tmp_path / "out"
        )
    assert sorted(p.name for p in data.iterdir()) == ["a.json", "b.json"]
    assert len(requests) == 5


@pytest.mark.parametrize("stage", ["metadata", "artifact"])
def test_mlflow_refuses_upstream_redirects(tmp_path, stage):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "sidecar"
        if request.url.path.endswith("/runs/get") and stage != "metadata":
            return httpx.Response(
                200, json={"run": {"info": {"artifact_uri": "mlflow-artifacts:/1/run/artifacts"}}}
            )
        if request.url.path.endswith("/artifacts/list"):
            return httpx.Response(200, json={"files": [{"path": "samples.json"}]})
        return httpx.Response(307, headers={"location": "https://upstream.example/data"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(httpx.HTTPStatusError):
            Sources(Sidecar("http://sidecar", client=client)).download(
                {"mlflow": {"run_id": "run"}}, tmp_path / "out"
            )
    assert all(r.url.host == "sidecar" for r in requests)


@pytest.mark.parametrize("corrupt", [None, "manifest", "blob", "size", "redirect"])
def test_oci_downloads_verified_layers_only_through_sidecar(tmp_path, corrupt):
    blob = archive()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": digest(blob),
                    "size": len(blob) + (corrupt == "size"),
                }
            ],
        }
    ).encode()
    coordinates = {"oci_host": "quay.io", "oci_repository": "org/results"}
    reference = {"oci": {"coordinates": coordinates, "digest": digest(manifest)}}
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "sidecar"
        assert "authorization" not in request.headers
        if "/manifests/" in request.url.path:
            return httpx.Response(200, content=manifest + (b" " if corrupt == "manifest" else b""))
        assert "/v2/org/results/blobs/sha256:" in request.url.path
        if corrupt == "redirect":
            return httpx.Response(307, headers={"location": "https://registry.example/blob"})
        return httpx.Response(200, content=blob + (b"wrong" if corrupt == "blob" else b""))

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        sources = Sources(
            Sidecar("http://sidecar", client=client),
            exports={"oci": {"coordinates": {**coordinates, "oci_host": "https://quay.io/"}}},
        )
        if corrupt:
            with pytest.raises((ValueError, httpx.HTTPStatusError)):
                sources.download(reference, tmp_path / "out")
        else:
            output = sources.download(reference, tmp_path / "out")
            assert json.loads((output / "samples.json").read_text())[0]["prediction"] == 0.8
    assert all(r.url.host == "sidecar" for r in requests)


@pytest.mark.parametrize(
    "configuration",
    [{}, {"oci": {"coordinates": {"oci_host": "other.example", "oci_repository": "org/results"}}}],
)
def test_oci_requires_matching_sidecar_configuration(tmp_path, configuration):
    def handler(request):
        pytest.fail("Mismatched OCI configuration must fail before making requests")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        sources = Sources(Sidecar("http://sidecar", client=client), exports=configuration)
        with pytest.raises(ValueError, match="sidecar"):
            sources.download(
                {
                    "oci": {
                        "coordinates": {
                            "oci_host": "quay.io",
                            "oci_repository": "org/results",
                            "oci_tag": "latest",
                        }
                    }
                },
                tmp_path / "out",
            )


@pytest.mark.parametrize("name,link", [("../escape", False), ("/absolute", False), ("link", True)])
def test_oci_archive_cannot_escape(tmp_path, name, link):
    blob = tmp_path / "layer"
    blob.write_bytes(archive(name, link=link))
    with pytest.raises(ValueError):
        extract_layer(
            blob,
            tmp_path / "data",
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"},
            set(),
        )


def test_pvc_requires_declared_mount_and_rejects_traversal(tmp_path, monkeypatch):
    source = tmp_path / "mount"
    source.mkdir()
    (source / "samples.json").write_text("[]")
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_PVC_MOUNTS", json.dumps({"data": str(source)}))
    with httpx.Client() as client:
        sources = Sources(Sidecar("http://sidecar", client=client))
        result = sources.download({"pvc": {"claim_name": "data"}}, tmp_path / "out")
        assert (result / "samples.json").read_text() == "[]"
        with pytest.raises(ValueError, match="needs a path"):
            sources.download({"pvc": {"claim_name": "missing"}}, tmp_path / "bad")
        with pytest.raises(ValueError, match="escapes"):
            sources.download(
                {"pvc": {"claim_name": "data", "sub_path": "../escape"}}, tmp_path / "bad"
            )


def test_s3_prefix_download_and_projected_credentials(tmp_path, monkeypatch):
    import boto3
    from botocore.exceptions import ClientError

    secret = tmp_path / "secrets" / "s3-access"
    secret.mkdir(parents=True)
    credentials = {
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_S3_ENDPOINT": "https://s3.example",
    }
    for name, value in credentials.items():
        (secret / name).write_text(value)
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_SECRET_ROOT", str(secret.parent))
    calls = []

    class Client:
        def head_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, **kwargs):
            assert kwargs == {"Bucket": "bucket", "Prefix": "scores/"}
            yield {"Contents": [{"Key": "scores/a.json"}]}
            yield {"Contents": [{"Key": "scores/b.json"}]}

        def download_file(self, bucket, key, path):
            calls.append((bucket, key))
            Path(path).write_text("[]")

        def close(self):
            calls.append("closed")

    def factory(service, **kwargs):
        assert service == "s3"
        assert kwargs["endpoint_url"] == credentials["AWS_S3_ENDPOINT"]
        assert kwargs["aws_secret_access_key"] == "secret"
        return Client()

    monkeypatch.setattr(boto3, "client", factory)
    with httpx.Client() as client:
        sources = Sources(Sidecar("http://sidecar", client=client))
        result = sources.download(
            {"s3": {"bucket": "bucket", "key": "scores", "secret_ref": "s3-access"}},
            tmp_path / "out",
        )
    assert sorted(p.name for p in result.iterdir()) == ["a.json", "b.json"]
    assert calls[-1] == "closed"


def test_hf_dataset_revision_and_subpath(tmp_path, monkeypatch):
    import huggingface_hub

    def snapshot(**kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["repo_id"] == "org/calibration"
        assert kwargs["revision"] == "commit123"
        path = kwargs["local_dir"] / "subset"
        path.mkdir(parents=True)
        (path / "labels.json").write_text("[]")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with httpx.Client() as client:
        sources = Sources(Sidecar("http://sidecar", client=client))
        result = sources.download(
            {"hf": {"repo_id": "org/calibration", "revision": "commit123", "sub_path": "subset"}},
            tmp_path / "out",
        )
    assert (result / "labels.json").read_text() == "[]"


def test_git_revision_and_projected_credentials(tmp_path, monkeypatch):
    import subprocess

    secret = tmp_path / "secrets" / "git-access"
    secret.mkdir(parents=True)
    (secret / "username").write_text("reader")
    (secret / "password").write_text("private-token")
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_SECRET_ROOT", str(secret.parent))
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        assert kwargs["env"]["PP_GIT_PASSWORD"] == "private-token"
        assert "private-token" not in " ".join(args)
        if args[1] == "init":
            repo = Path(args[-1])
            (repo / ".git").mkdir(parents=True)
            (repo / "labels.json").write_text("[]")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with httpx.Client() as client:
        sources = Sources(Sidecar("http://sidecar", client=client))
        output = sources.download(
            {
                "git": {
                    "url": "https://git.example/data.git",
                    "ref": "commit123",
                    "secret_ref": "git-access",
                }
            },
            tmp_path / "out",
        )
    assert commands[1][-2:] == ["https://git.example/data.git", "commit123"]
    assert (output / "labels.json").read_text() == "[]"
    assert not (output / ".git").exists()
