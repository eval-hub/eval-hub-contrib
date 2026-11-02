"""Prepare only the RULER data assets required by a job in a writable cache."""
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)
ESSAY_FILE = "PaulGrahamEssays.json"
MAX_ASSET_BYTES = 200 * 1024 * 1024

ASSETS = {
    "english_words.json": (
        "https://media.githubusercontent.com/media/NVIDIA/RULER/"
        "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/"
        "scripts/data/synthetic/json/english_words.json",
        "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    ),
    "squad.json": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
    "hotpotqa.json": (
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
        "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
        "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    ),
}


def materialize_asset(path: Path, url: str, digest: str, timeout: int = 60) -> None:
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        json.loads(path.read_bytes())
        return
    with urlopen(url, timeout=timeout) as response:
        payload = response.read(MAX_ASSET_BYTES + 1)
    if len(payload) > MAX_ASSET_BYTES:
        raise ValueError(f"Download for {path.name} exceeds {MAX_ASSET_BYTES} bytes")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch for {path.name}")
    json.loads(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    temporary.write_bytes(payload)
    temporary.replace(path)


def required_assets(task_configs: list[dict[str, Any]]) -> set[str]:
    """Resolve asset dependencies from the actual RULER generator configuration."""
    required = set()
    for config in task_configs:
        args = config.get("args", {})
        if config.get("task") == "common_words_extraction":
            required.add("english_words.json")
        elif config.get("task") == "qa":
            dataset = args.get("dataset")
            if dataset not in ("squad", "hotpotqa"):
                raise ValueError(f"Unsupported RULER QA dataset: {dataset}")
            required.add(f"{dataset}.json")
        if args.get("type_haystack") == "essay":
            required.add(ESSAY_FILE)
    return required


def _valid_essay(path: Path) -> bool:
    try:
        text = json.loads(path.read_text())["text"]
        return isinstance(text, str) and bool(text.strip())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_staged_asset(name: str, path: Path) -> None:
    """Reject incomplete or incompatible staged inputs before copying them."""
    if not path.is_file():
        raise ValueError(f"missing file {path}")
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError(f"{path} exceeds {MAX_ASSET_BYTES} bytes")

    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if name == ESSAY_FILE:
        text = value.get("text") if isinstance(value, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path} must contain non-empty JSON text in the 'text' field")
        return

    _, expected_digest = ASSETS[name]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_digest:
        raise ValueError(
            f"{path} checksum mismatch: expected sha256:{expected_digest}, got sha256:{digest}"
        )


def stage_assets(
    required: set[str], source_dir: Path, cache_dir: Path,
) -> None:
    """Validate required read-only staged files and copy them to the job cache.

    This path deliberately has no download fallback: missing or corrupt staged
    files are reported together before any input is copied into the cache.
    """
    errors: list[str] = []
    sources: dict[str, Path] = {}
    for name in sorted(required):
        source = source_dir / name
        try:
            resolved_root = source_dir.resolve()
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(resolved_root)
            _validate_staged_asset(name, resolved_source)
            sources[name] = resolved_source
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")

    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"RULER staged assets under {source_dir} are missing or invalid: {details}. "
            "Stage the required files at the root of the test_data_ref bundle."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = cache_dir / name
        temporary = cache_dir / f".{name}.staged"
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Staged %s from %s into the writable RULER cache", name, source)


def materialize_essays(path: Path, scripts_dir: Path, timeout: int) -> None:
    if _valid_essay(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    downloader = scripts_dir / "data" / "synthetic" / "json" / "download_paulgraham_essay.py"
    logger.info("Downloading RULER essay data into %s", path.parent)
    with tempfile.TemporaryDirectory(prefix="essays_", dir=path.parent) as temporary:
        result = subprocess.run(
            [sys.executable, str(downloader)],
            capture_output=True,
            text=True,
            env={**os.environ, "RULER_DATA_DIR": temporary},
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to download RULER essay haystack. "
                "The runtime must have network access to GitHub and Paul Graham.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        downloaded = Path(temporary) / path.name
        if not _valid_essay(downloaded):
            raise RuntimeError("RULER essay haystack must contain non-empty JSON text")
        downloaded.replace(path)
        logger.info("Essay downloader output:\n%s", result.stdout)


def prepare_assets(
    task_configs: list[dict[str, Any]],
    cache_dir: Path,
    scripts_dir: Path,
    timeout: int = 600,
    staged_data_dir: Path | None = None,
) -> None:
    """Prepare task inputs from staged files or, for connected jobs, download them."""
    required = required_assets(task_configs)
    if staged_data_dir is not None:
        stage_assets(required, staged_data_dir, cache_dir)
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in ASSETS.items():
        if name in required:
            materialize_asset(cache_dir / name, url, digest, timeout=min(timeout, 60))
            logger.info("Verified %s: sha256:%s", name, digest)
    if ESSAY_FILE in required:
        materialize_essays(cache_dir / ESSAY_FILE, scripts_dir, timeout)
