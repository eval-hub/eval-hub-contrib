"""Prepare packaged task assets at image build time, before switching to non-root."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

# The 0.26.0 wheel omits this archive and its downloader uses a nonexistent
# unprefixed release tag. Keep the data tied to the exact installed release.
INSPECT_EVALS_VERSION = "0.26.0"
SOURCE_COMMIT = "cac63f6d942ff63c9a0ac1185a026123e62970f4"
ARCHIVE_PATH = "src/inspect_evals/gdm_self_proliferation/data/sp08/secrets.zip"
ARCHIVE_URL = (
    f"https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/"
    f"{SOURCE_COMMIT}/{ARCHIVE_PATH}"
)
ARCHIVE_SHA256 = "dbd1876c6516c4c08d2d004e8092de2f345490dc27056559436a3796aed8cd80"


def verify_archive(path: Path, expected_sha256: str) -> None:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"Task archive checksum mismatch: {path.name}")


def download_archive(url: str, target: Path, expected_sha256: str) -> None:
    """Publish a downloaded asset only after its full contents pass verification."""
    if target.exists():
        verify_archive(target, expected_sha256)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        verify_archive(temporary, expected_sha256)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_self_proliferation(archive: Path | None = None) -> None:
    installed = version("inspect-evals")
    if installed != INSPECT_EVALS_VERSION:
        raise RuntimeError(
            f"Task assets target inspect-evals {INSPECT_EVALS_VERSION}, got {installed}; "
            "update the source commit and archive checksum with the dependency pin"
        )
    from inspect_evals.gdm_self_proliferation import data_setup

    source = (
        archive.resolve(strict=True).as_uri() if archive is not None else ARCHIVE_URL
    )
    download_archive(
        source, data_setup.DATA_DIR / "sp08" / "secrets.zip", ARCHIVE_SHA256
    )
    data_setup.ensure_self_proliferation_data_ready()
    if data_setup._needs_setup(data_setup.DATA_DIR):
        raise RuntimeError("Self-proliferation task data preparation is incomplete")
    print(f"Prepared self-proliferation task data for inspect-evals {installed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="Use a pre-downloaded, verified archive"
    )
    prepare_self_proliferation(parser.parse_args().archive)
