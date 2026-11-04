"""Configure Hugging Face caches for LightEval jobs with staged test data.

EvalHub mounts assets referenced by a benchmark's ``test_data_ref`` at
``/test_data``. LightEval reads Hugging Face datasets through ``datasets`` and
the Hub, so these cache paths must point at the staged tree and network access
to the Hub must be disabled for the duration of the job.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path

TEST_DATA_DIR = Path("/test_data")
LIGHTEVAL_TASK_REGISTRY_DIR: Path | None = None


def is_test_data_mount_usable(test_data_root: str | Path = TEST_DATA_DIR) -> bool:
    """Return whether EvalHub mounted staged data, including an empty mount.

    ``test_data_ref`` is consumed by EvalHub and is not serialized into the
    adapter's JobSpec. The mount itself is therefore the reliable signal. A
    nonempty directory is also accepted for local tests and development runs.
    """
    root = Path(test_data_root)
    try:
        return root.is_mount() or (
            root.is_dir()
            and any(not entry.name.startswith(".") for entry in root.iterdir())
        )
    except OSError:
        return False


def _has_hf_cache_layout(test_data_root: str | Path) -> bool:
    """Return whether staged assets contain a populated Hub or datasets cache."""
    root = Path(test_data_root)
    for cache_name in ("hub", "datasets"):
        cache_dir = root / cache_name
        try:
            if cache_dir.is_dir() and any(
                not entry.name.startswith(".") and not entry.name.endswith(".lock")
                for entry in cache_dir.iterdir()
            ):
                return True
        except OSError:
            continue
    return False


def configure_hf_offline_environment(
    test_data_root: str | Path,
    env: MutableMapping[str, str] | None = None,
    writable_datasets_cache_dir: str | Path | None = None,
) -> Path:
    """Use the staged HF cache tree and disable Hub downloads.

    When ``env`` is omitted, configure the current process so framework imports
    and the LightEval subprocess use the same cache and offline settings. The
    datasets cache is mirrored into a writable directory with symlinked data
    files so Hugging Face can create lock files without copying staged assets.
    """
    root = Path(test_data_root)
    target_env = os.environ if env is None else env
    datasets_cache = _prepare_writable_datasets_cache(
        root,
        writable_datasets_cache_dir,
    )
    target_env.update(
        {
            "HF_HOME": str(root),
            "HF_HUB_CACHE": str(root / "hub"),
            "HF_DATASETS_CACHE": str(datasets_cache),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_EVALUATE_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    staged_nltk_data = root / "nltk_data"
    if staged_nltk_data.is_dir():
        target_env["NLTK_DATA"] = str(staged_nltk_data)
    return datasets_cache


def ensure_nltk_offline_resources(test_data_root: str | Path) -> None:
    """Check tokenizer data before LightEval imports IFEval and tries downloads.

    The staged NLTK directory must provide both resources; do not fall back to
    image-level data or attempt a download for a disconnected job.
    """
    staged_nltk_data = Path(test_data_root) / "nltk_data"
    if not staged_nltk_data.is_dir():
        raise RuntimeError(
            "LightEval requires staged NLTK punkt and punkt_tab under "
            f"{staged_nltk_data / 'tokenizers'} in this benchmark's test_data_ref "
            "bundle. Offline jobs will not download NLTK data."
        )

    try:
        # PunktTokenizer reads punkt_tab, but IFEval also checks for punkt.
        # NLTK's find() accepts empty directories, so require its English model.
        punkt_model = staged_nltk_data / "tokenizers" / "punkt" / "english.pickle"
        if not punkt_model.is_file() or punkt_model.stat().st_size == 0:
            raise LookupError(f"missing or empty staged NLTK model: {punkt_model}")

        import nltk
        from nltk.tokenize.punkt import PunktTokenizer

        staged_path = str(staged_nltk_data)
        original_paths = nltk.data.path
        try:
            # sent_tokenize() can reuse a cached tokenizer or image-level data.
            # A fresh instance must load the staged punkt_tab files alone.
            nltk.data.path = [staged_path]
            sentences = PunktTokenizer("english").tokenize(
                "First sentence. Second sentence."
            )
            if sentences != ["First sentence.", "Second sentence."]:
                raise ValueError(
                    "staged NLTK sentence tokenizer returned unexpected results"
                )
        finally:
            nltk.data.path = original_paths

        if staged_path not in nltk.data.path:
            nltk.data.path.insert(0, staged_path)
    except (LookupError, OSError, ValueError) as exc:
        raise RuntimeError(
            "LightEval requires NLTK punkt and punkt_tab before loading its task "
            "registry. Stage both resources under "
            f"{Path(test_data_root) / 'nltk_data' / 'tokenizers'} in this benchmark's "
            "test_data_ref bundle. "
            "Offline jobs will not download NLTK data."
        ) from exc


def _prepare_writable_datasets_cache(
    test_data_root: Path,
    writable_datasets_cache_dir: str | Path | None,
) -> Path:
    """Mirror dataset-cache directories while symlinking staged files in place."""
    staged_root = test_data_root.resolve()
    staged_cache = test_data_root / "datasets"
    cache_dir = (
        Path(writable_datasets_cache_dir)
        if writable_datasets_cache_dir is not None
        else Path(tempfile.mkdtemp(prefix="lighteval-datasets-cache-"))
    )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_dir.resolve().is_relative_to(staged_root):
            raise RuntimeError(
                f"Writable datasets cache {cache_dir} must be outside staged "
                f"test-data directory {test_data_root}."
            )
        if any(cache_dir.iterdir()):
            raise RuntimeError(
                f"Writable datasets cache directory {cache_dir} must be empty."
            )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create writable Hugging Face datasets cache at {cache_dir}: {exc}"
        ) from exc

    def mirror(source: Path, destination: Path, active_directories: set[Path]) -> None:
        if source.name.endswith(".lock"):
            return
        try:
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(staged_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Staged datasets cache entry {source} is missing or points outside "
                f"{test_data_root}."
            ) from exc

        if resolved_source.is_dir():
            if resolved_source in active_directories:
                raise RuntimeError(
                    f"Staged datasets cache contains a directory cycle at {source}."
                )
            destination.mkdir(parents=True, exist_ok=True)
            next_active = {*active_directories, resolved_source}
            for child in resolved_source.iterdir():
                mirror(child, destination / child.name, next_active)
            return

        if not resolved_source.is_file():
            raise RuntimeError(f"Unsupported entry in staged datasets cache: {source}.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(resolved_source)

    if staged_cache.is_dir():
        try:
            for entry in staged_cache.iterdir():
                mirror(entry, cache_dir / entry.name, set())
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Cannot prepare writable Hugging Face datasets cache "
                f"at {cache_dir}: {exc}"
            ) from exc

    return cache_dir


def ensure_lighteval_registry_assets(
    test_data_root: str | Path,
    task_registry_dir: str | Path | None = None,
) -> None:
    """Seed LightEval's eagerly imported TinyBenchmarks registry asset offline.

    LightEval 0.13.0 downloads this auxiliary file with ``requests`` while
    importing every task config. That request bypasses the HF offline flags,
    even when the selected task does not use TinyBenchmarks.
    """
    root = Path(test_data_root)
    destination_dir = (
        Path(task_registry_dir)
        if task_registry_dir is not None
        else LIGHTEVAL_TASK_REGISTRY_DIR or _default_lighteval_task_registry_dir()
    )
    destination = destination_dir / "tinyBenchmarks.pkl"
    staged_asset = root / "lighteval" / "tinyBenchmarks.pkl"

    if staged_asset.is_file() and staged_asset.stat().st_size > 0:
        if not destination_dir.is_dir():
            raise RuntimeError(
                f"LightEval task registry directory {destination_dir} does not exist."
            )
        shutil.copyfile(staged_asset, destination)
        return

    if destination.is_file() and destination.stat().st_size > 0:
        return

    raise RuntimeError(
        "LightEval's task registry requires tinyBenchmarks.pkl even when the selected "
        "benchmark does not use TinyBenchmarks. Stage this file at "
        f"{root / 'lighteval' / 'tinyBenchmarks.pkl'} to prevent a runtime GitHub download."
    )


def _default_lighteval_task_registry_dir() -> Path:
    """Return the installed LightEval task-module directory without importing its registry."""
    spec = importlib.util.find_spec("lighteval")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "LightEval is not installed; cannot locate its task registry assets."
        )
    return Path(spec.origin).parent / "tasks" / "tasks"


def build_hf_offline_environment(
    test_data_root: str | Path = TEST_DATA_DIR,
    base_env: Mapping[str, str] | None = None,
    writable_datasets_cache_dir: str | Path | None = None,
) -> dict[str, str] | None:
    """Build an offline environment when EvalHub mounted staged test data.

    ``test_data_ref`` is not part of the adapter JobSpec, so use the volume
    mount as the signal. Jobs without a test-data mount return ``None`` and keep
    the existing online behavior. A mounted tree must contain the Hugging Face
    cache layout LightEval consumes; raw repository files staged with the
    ``hf`` source are not a compatible cache.
    """
    if not is_test_data_mount_usable(test_data_root):
        return None

    if not _has_hf_cache_layout(test_data_root):
        raise RuntimeError(
            f"Staged data at {test_data_root} does not contain a populated Hugging Face "
            "cache under hub/ or datasets/. LightEval requires a prepared Hugging Face "
            "cache; the test_data_ref.hf source stages raw repository files and is not "
            "compatible with this cache layout. Stage the cache tree with S3, PVC, or Git."
        )

    env = dict(os.environ if base_env is None else base_env)
    configure_hf_offline_environment(test_data_root, env, writable_datasets_cache_dir)
    return env
