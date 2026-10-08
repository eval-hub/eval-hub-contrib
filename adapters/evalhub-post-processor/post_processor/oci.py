"""OCI downloads through the sidecar, with verified and safe layer extraction."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from .config import object_value, text_value
from .data import within

if TYPE_CHECKING:
    from .sources import Sources

DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
MAX_EXPANDED_BYTES = 10 * 1024**3


def extract_layer(blob: Path, destination: Path, descriptor: dict, used: set[Path]) -> None:
    media_type = descriptor.get("mediaType", "")
    if "tar" in media_type:
        size = 0
        with tarfile.open(blob, "r:*") as archive:
            for member in archive:
                target = within(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError("OCI archives must contain only regular files and directories")
                size += member.size
                if size > MAX_EXPANDED_BYTES:
                    raise ValueError("OCI layer exceeds the expanded data size limit")
                if target in used:
                    raise ValueError("OCI artifact contains duplicate file paths")
                used.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    else:
        title = descriptor.get("annotations", {}).get("org.opencontainers.image.title")
        if not title:
            raise ValueError("Non-archive OCI layers require an org.opencontainers.image.title")
        target = within(destination, title)
        if target in used:
            raise ValueError("OCI artifact contains duplicate file paths")
        used.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(blob, target)


def download_oci(ref: dict, destination: Path, sources: Sources) -> Path:
    coords = object_value(ref.get("coordinates"), "oci.coordinates")
    host = text_value(coords.get("oci_host"), "oci.coordinates.oci_host")
    repository = text_value(coords.get("oci_repository"), "oci.coordinates.oci_repository")
    within(destination, repository)
    configured = (sources.exports.get("oci") or {}).get("coordinates") or {}

    def normalize_host(value: str) -> str:
        return value.removeprefix("https://").removeprefix("http://").rstrip("/").lower()

    if not configured.get("oci_host") or not configured.get("oci_repository"):
        raise ValueError(
            "OCI downloads require the runtime to configure the sidecar using job.exports.oci.coordinates"
        )
    if (
        normalize_host(host) != normalize_host(configured["oci_host"])
        or repository != configured["oci_repository"]
    ):
        raise ValueError(
            "Requested OCI registry/repository does not match the sidecar's job.exports.oci.coordinates"
        )
    reference = ref.get("digest") or coords.get("oci_tag")
    if not reference:
        raise ValueError("OCI results require an immutable digest or explicit tag")
    if ref.get("digest") and not DIGEST.fullmatch(ref["digest"]):
        raise ValueError("OCI digest must be a sha256 digest")
    if not ref.get("digest") and not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}", reference):
        raise ValueError("OCI tag is invalid")
    requested = ref.get("artifact_path", "")
    within(destination, requested)
    # The runtime configures the sidecar's registry target and credentials.
    # Never authenticate to the registry here or follow redirects to upstream hosts.
    base = sources.sidecar.base_url + "/v2/" + quote(repository, safe="/")
    client = sources.sidecar.client
    data_dir = destination / "data"
    data_dir.mkdir()
    used: set[Path] = set()
    response = client.get(
        base + "/manifests/" + quote(reference, safe=":"),
        headers={
            "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
        },
        follow_redirects=False,
    )
    response.raise_for_status()
    if (
        ref.get("digest")
        and "sha256:" + hashlib.sha256(response.content).hexdigest() != ref["digest"]
    ):
        raise ValueError("OCI manifest digest verification failed")
    manifest = response.json()
    if not manifest.get("layers"):
        raise ValueError("OCI artifact manifest contains no layers")
    for index, layer in enumerate(manifest["layers"]):
        digest = layer.get("digest", "")
        if not DIGEST.fullmatch(digest):
            raise ValueError("Unsupported OCI layer digest")
        blob = destination / f"layer-{index}"
        checksum = hashlib.sha256()
        size = 0
        with client.stream(
            "GET", base + "/blobs/" + quote(digest, safe=":"), follow_redirects=False
        ) as response:
            response.raise_for_status()
            with blob.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_EXPANDED_BYTES:
                        raise ValueError("OCI layer exceeds the download size limit")
                    checksum.update(chunk)
                    stream.write(chunk)
        if "sha256:" + checksum.hexdigest() != digest or size != layer.get("size"):
            raise ValueError("OCI layer digest or size verification failed")
        extract_layer(blob, data_dir, layer, used)
        blob.unlink()
    selected = within(data_dir, requested)
    if not selected.exists() or not used:
        raise ValueError("Requested OCI artifact_path does not exist or contains no data")
    return selected
