"""Fail-fast network pre-flight for Inspect jobs.

Many inspect-evals tasks download their dataset at runtime (HuggingFace Hub, or GitHub for
BFCL). When the pod cannot reach that host, the download does not fail: the HF client and
inspect-evals retry with exponential backoff, and git has no timeout at all. The job then
sits silent for many minutes. This module probes the hosts a task needs before ``inspect``
starts, so an unreachable host becomes a clear error (or warning) within seconds.

Overrides (job parameter wins over the environment variable of the same meaning):

==========================  ============================  ==============================
parameters key              environment variable          meaning
==========================  ============================  ==============================
``preflight``               ``INSPECT_PREFLIGHT``         ``fail`` (default), ``warn``, ``off``
``preflight_timeout_s``     ``INSPECT_PREFLIGHT_TIMEOUT_S``  per-host timeout, default 10
``preflight_hosts``         ``INSPECT_PREFLIGHT_HOSTS``   extra URLs to probe (list / comma-separated)
==========================  ============================  ==============================

The probe is skipped in offline mode (``HF_HUB_OFFLINE=1``) and for Petri/Bloom, which only
talk to model endpoints.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import re
import runpy
import ssl
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MODE_ENV = "INSPECT_PREFLIGHT"
TIMEOUT_ENV = "INSPECT_PREFLIGHT_TIMEOUT_S"
HOSTS_ENV = "INSPECT_PREFLIGHT_HOSTS"
INSPECT_EVALS_CACHE_DIR_ENV = "INSPECT_EVALS_CACHE_DIR"

MODES = ("fail", "warn", "off")
DEFAULT_MODE = "fail"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_BFCL_TASK = "inspect_evals/bfcl"
_BFCL_PREREQS_TASK = "inspect_evals/bfcl_prereqs"

# Task specs that never download a dataset from the Hub or GitHub.
_NO_PROBE_PREFIXES = ("inspect_petri/", "petri_bloom/")
# Task specs whose dataset comes from the HuggingFace Hub.
_HF_TASK_PREFIXES = ("inspect_evals/",)
# Exact task specs that download from GitHub (not HuggingFace) even when the
# spec falls under an _HF_TASK_PREFIXES namespace.
_GITHUB_ONLY_TASKS = frozenset({_BFCL_TASK, _BFCL_PREREQS_TASK})

# Source files required by the BFCL categories in the pinned inspect-evals release.
# Keep these aligned with inspect_evals.bfcl.backends.loader.CLASS_FILE_MAPPING and
# inspect_evals.bfcl.data.MULTI_TURN_FUNC_DOC_FILE_MAPPING.
_BFCL_MULTI_TURN_ASSETS = {
    "func_source_code": frozenset(
        {
            "gorilla_file_system.py",
            "math_api.py",
            "message_api.py",
            "posting_api.py",
            "ticket_api.py",
            "trading_bot.py",
            "travel_booking.py",
            "vehicle_control.py",
            "long_context.py",
        }
    ),
    "multi_turn_func_doc": frozenset(
        {
            "gorilla_file_system.json",
            "math_api.json",
            "message_api.json",
            "posting_api.json",
            "ticket_api.json",
            "trading_bot.json",
            "travel_booking.json",
            "vehicle_control.json",
        }
    ),
}
_BFCL_AGENTIC_ASSETS = {
    "web_search_base": (
        frozenset({"web_search.py"}),
        frozenset({"web_search.json"}),
    ),
    "web_search_no_snippet": (
        frozenset({"web_search.py"}),
        frozenset({"web_search.json"}),
    ),
    "memory_kv": (
        frozenset({"memory_kv.py", "memory_api_metaclass.py"}),
        frozenset({"memory_kv.json"}),
    ),
    "memory_vector": (
        frozenset({"memory_vector.py", "memory_api_metaclass.py"}),
        frozenset({"memory_vector.json"}),
    ),
    "memory_rec_sum": (
        frozenset({"memory_rec_sum.py", "memory_api_metaclass.py"}),
        frozenset({"memory_rec_sum.json"}),
    ),
}
_BFCL_CATEGORY_ALIASES = {
    "memory": ("memory_kv", "memory_vector", "memory_rec_sum"),
    "web_search": ("web_search_base", "web_search_no_snippet"),
}
# The loader tolerates this dependency module being absent. Memory categories list
# it explicitly above, so only those selections require the staged file.
_BFCL_OPTIONAL_LOADER_MODULES = frozenset({"memory_api_metaclass.py"})
_BFCL_EAGER_LOADER_MODULES = (
    _BFCL_MULTI_TURN_ASSETS["func_source_code"]
    | frozenset().union(*(assets[0] for assets in _BFCL_AGENTIC_ASSETS.values()))
) - _BFCL_OPTIONAL_LOADER_MODULES


class PreflightError(RuntimeError):
    """A host the task needs is unreachable and ``preflight`` is ``fail``."""


def normalize_mode(value: Any, name: str = "preflight") -> str:
    """Return a validated mode string, naming the parameter on failure."""
    mode = str(value).strip().lower()
    if mode not in MODES:
        raise ValueError(f"{name} must be one of {', '.join(MODES)} (got {value!r})")
    return mode


def normalize_timeout(value: Any, name: str = "preflight_timeout_s") -> float:
    """Return a positive timeout in seconds, naming the parameter on failure."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number of seconds (got {value!r})") from None
    if seconds <= 0:
        raise ValueError(f"{name} must be a positive number of seconds (got {value!r})")
    return seconds


def normalize_hosts(value: Any, name: str = "preflight_hosts") -> list[str]:
    """Accept a list or a comma-separated string of http(s) URLs."""
    if value is None or value == "":
        return []
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{name} must be a list of URLs or a comma-separated string (got {value!r})")
    urls = [str(i).strip() for i in items if str(i).strip()]
    for url in urls:
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"{name} entries must start with http:// or https:// (got {url!r})")
    return urls


@lru_cache(maxsize=1)
def _bfcl_category_manifest() -> dict[str, Any] | None:
    """Load Inspect-Evals' lightweight category manifest without importing its task package."""
    try:
        spec = importlib.util.find_spec("inspect_evals")
        if spec is None or not spec.submodule_search_locations:
            return None
        manifest_path = (
            Path(next(iter(spec.submodule_search_locations)))
            / "bfcl"
            / "utils"
            / "task_categories.py"
        )
        if not manifest_path.is_file():
            return None
        categories = runpy.run_path(str(manifest_path)).get("CATEGORIES")
        return categories if isinstance(categories, dict) else None
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _bfcl_selected_categories(
    task_args: dict[str, Any] | None, categories: dict[str, Any]
) -> set[str] | None:
    """Resolve the selected BFCL categories, or return None when they are unknown."""
    requested = (task_args or {}).get("categories")
    if requested is None:
        # inspect-evals 0.26.0 defaults to V1/V2/V3 categories (excluding the
        # unimplemented REST category and format_sensitivity meta-index).
        defaults = {
            name
            for name, category in categories.items()
            if getattr(category, "category_version", None) in {"v1", "v2", "v3"}
            and name not in {"rest", "format_sensitivity"}
        }
        return defaults or None

    if isinstance(requested, str):
        try:
            parsed = ast.literal_eval(requested)
        except (SyntaxError, ValueError):
            parsed = requested
        if isinstance(parsed, str):
            names = [part.strip() for part in parsed.split(",") if part.strip()]
        elif isinstance(parsed, (list, tuple, set)):
            names = [str(part).strip() for part in parsed if str(part).strip()]
        else:
            return None
    elif isinstance(requested, (list, tuple, set)):
        names = [str(part).strip() for part in requested if str(part).strip()]
    else:
        return None

    if not names:
        return None
    selected: set[str] = set()
    for name in names:
        selected.update(_BFCL_CATEGORY_ALIASES.get(name, (name,)))
    if not selected.issubset(categories):
        # An unknown category should never make a partial cache look complete.
        return None
    return selected


def _bfcl_required_runtime_assets(
    selected_categories: set[str] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return backend modules and function docs used by the selected categories."""
    if selected_categories is None:
        return (
            _BFCL_EAGER_LOADER_MODULES,
            frozenset().union(
                _BFCL_MULTI_TURN_ASSETS["multi_turn_func_doc"],
                *(assets[1] for assets in _BFCL_AGENTIC_ASSETS.values()),
            ),
        )

    modules: set[str] = set()
    docs: set[str] = set()
    # Inspect-Evals 0.26.0 setup_backend_classes iterates every entry in
    # CLASS_FILE_MAPPING when any V3/V4 category is selected. Missing an
    # unselected module still raises FileNotFoundError during that eager load.
    if any(
        category.startswith("multi_turn_") or category in _BFCL_AGENTIC_ASSETS
        for category in selected_categories
    ):
        modules.update(_BFCL_EAGER_LOADER_MODULES)
    for category in selected_categories:
        if category.startswith("multi_turn_"):
            modules.update(_BFCL_MULTI_TURN_ASSETS["func_source_code"])
            docs.update(_BFCL_MULTI_TURN_ASSETS["multi_turn_func_doc"])
        elif category in _BFCL_AGENTIC_ASSETS:
            category_modules, category_docs = _BFCL_AGENTIC_ASSETS[category]
            modules.update(category_modules)
            docs.update(category_docs)
    return frozenset(modules), frozenset(docs)


def _bfcl_snapshot_present(
    bfcl_dir: Path, selected_categories: set[str], task_args: dict[str, Any] | None
) -> bool:
    """Return whether a requested memory snapshot exists in the cache."""
    if not any(category.startswith("memory_") for category in selected_categories):
        return True

    snapshot_id = (task_args or {}).get("snapshot_id")
    if snapshot_id is None or not str(snapshot_id).strip():
        return False

    # Match Inspect-Evals' sanitize_snapshot_id while keeping the result a single
    # safe path component under model_results.
    sanitized_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(snapshot_id))
    if sanitized_id in {"", ".", ".."}:
        return False
    snapshot_dir = bfcl_dir / "model_results" / sanitized_id
    try:
        return snapshot_dir.is_dir() and (snapshot_dir / "agentic").is_dir()
    except OSError:
        return False


def _bfcl_data_present(
    env: dict[str, str],
    task_args: dict[str, Any] | None = None,
    *,
    require_snapshot: bool = False,
) -> bool:
    """Return True when the selected BFCL data and runtime assets are staged."""
    cache = env.get(INSPECT_EVALS_CACHE_DIR_ENV)
    if not cache:
        return False

    bfcl_dir = Path(cache) / "BFCL"
    categories = _bfcl_category_manifest()
    if not categories:
        return False

    selected_categories = _bfcl_selected_categories(task_args, categories)
    if selected_categories is None:
        return False

    # Inspect-Evals uses this file as a download canary. It must be present even
    # when the requested category is a subset, otherwise the task downloads the
    # full data tree from GitHub.
    sentinel = categories.get("simple_python")
    if sentinel is None or not (bfcl_dir / sentinel.filename).is_file():
        return False

    # Once the canary is present, Inspect-Evals only loads selected categories.
    for category_name in selected_categories:
        category = categories[category_name]
        if not (bfcl_dir / category.filename).is_file():
            return False
        if category.has_ground_truth_file and not (
            bfcl_dir / category.ground_truth_filename
        ).is_file():
            return False

    if require_snapshot and not _bfcl_snapshot_present(
        bfcl_dir, selected_categories, task_args
    ):
        return False

    required_modules, required_docs = _bfcl_required_runtime_assets(selected_categories)
    if not required_modules and not required_docs:
        return True

    function_sources = bfcl_dir / "func_source_code"
    function_docs = bfcl_dir / "multi_turn_func_doc"
    try:
        return (
            function_sources.is_dir()
            and function_docs.is_dir()
            and all((function_sources / name).is_file() for name in required_modules)
            and all((function_docs / name).is_file() for name in required_docs)
        )
    except OSError:
        return False


def _bfcl_prereqs_requires_github(
    env: dict[str, str],
) -> bool:
    """Return whether bfcl_prereqs will attempt a GitHub download."""
    cache = env.get(INSPECT_EVALS_CACHE_DIR_ENV)
    if not cache:
        return True

    bfcl_dir = Path(cache) / "BFCL"
    categories = _bfcl_category_manifest()
    sentinel = categories.get("simple_python") if categories else None
    if sentinel is None or not (bfcl_dir / sentinel.filename).is_file():
        # bfcl_prereqs downloads the data directory only when this canary is absent.
        return True

    # Unlike the conversation data, backend sources are downloaded whenever this
    # directory is absent. Missing conversation files with an existing canary
    # cause a local cache error, not another download.
    function_sources = bfcl_dir / "func_source_code"
    return not function_sources.exists()


def configure_staged_bfcl_cache(
    task_spec: str,
    env: dict[str, str],
    staged_data_dir: str | Path = "/test_data",
    task_args: dict[str, Any] | None = None,
) -> bool:
    """Point Inspect-Evals at staged BFCL assets when the offline cache is complete."""
    if task_spec != _BFCL_TASK or _bfcl_data_present(
        env, task_args, require_snapshot=True
    ):
        return False

    staged_env = dict(env)
    staged_env[INSPECT_EVALS_CACHE_DIR_ENV] = str(staged_data_dir)
    if not _bfcl_data_present(staged_env, task_args, require_snapshot=True):
        return False

    env[INSPECT_EVALS_CACHE_DIR_ENV] = str(staged_data_dir)
    logger.info(
        "Using staged Inspect-Evals BFCL data and function sources from %s",
        staged_data_dir,
    )
    return True


def required_urls(
    task_spec: str, env: dict[str, str], task_args: dict[str, Any] | None = None
) -> list[str]:
    """URLs the task is known to need at runtime, plus any user-supplied hosts."""
    urls: list[str] = []
    if not task_spec.startswith(_NO_PROBE_PREFIXES):
        hub_offline = env.get("HF_HUB_OFFLINE") == "1" or env.get("HF_DATASETS_OFFLINE") == "1"
        task_key = task_spec.lower()
        is_hf_task = (
            task_key.startswith(_HF_TASK_PREFIXES)
            and task_key not in _GITHUB_ONLY_TASKS
        )
        if is_hf_task and not hub_offline:
            urls.append(env.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT)
        if task_key == _BFCL_TASK and not _bfcl_data_present(env, task_args):
            urls.append("https://github.com")
        elif task_key == _BFCL_PREREQS_TASK and _bfcl_prereqs_requires_github(env):
            urls.append("https://github.com")
    urls += normalize_hosts(env.get(HOSTS_ENV), HOSTS_ENV)
    return list(dict.fromkeys(urls))


def _proxies(env: dict[str, str]) -> dict[str, str]:
    """Proxy settings from the subprocess env (not this process's), as urllib expects them."""
    out: dict[str, str] = {}
    for key, value in env.items():
        k = key.lower()
        if k.endswith("_proxy") and value:
            out[k[: -len("_proxy")]] = value
    return out


def probe(url: str, timeout_s: float, env: dict[str, str]) -> tuple[bool, str, float]:
    """Ranged ``GET`` of ``url``. Any HTTP answer counts as reachable; only connect/timeout errors fail.

    ``GET`` with ``Range: bytes=0-0`` follows the same path as the real dataset download
    (some proxies and WAFs drop or reset ``HEAD``) while the response body is never read.

    A TLS verification error is reported as reachable: the network path works, and the real
    download may use a different CA bundle than this probe.
    """
    proxies = _proxies(env)
    host = urlparse(url).hostname or ""
    handlers: list[urllib.request.BaseHandler] = []
    if proxies and not urllib.request.proxy_bypass_environment(host, proxies):  # type: ignore[attr-defined]
        handlers.append(urllib.request.ProxyHandler(proxies))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
    started = time.monotonic()
    try:
        with opener.open(request, timeout=timeout_s):
            pass
        return True, "ok", time.monotonic() - started
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code}", time.monotonic() - started
    except (urllib.error.URLError, OSError) as e:
        elapsed = time.monotonic() - started
        reason = getattr(e, "reason", e)
        if isinstance(reason, ssl.SSLError):
            return True, f"reachable, TLS verification failed ({reason})", elapsed
        if isinstance(reason, TimeoutError) or isinstance(e, TimeoutError):
            return False, f"no response within {timeout_s:g}s", elapsed
        return False, str(reason), elapsed


def run_preflight(
    task_spec: str, env: dict[str, str], task_args: dict[str, Any] | None = None
) -> None:
    """Probe the hosts ``task_spec`` needs; raise :class:`PreflightError` in ``fail`` mode."""
    mode = normalize_mode(env.get(MODE_ENV) or DEFAULT_MODE, MODE_ENV)
    if mode == "off":
        logger.info("Pre-flight network check disabled (%s=off)", MODE_ENV)
        return
    timeout_s = normalize_timeout(env.get(TIMEOUT_ENV) or DEFAULT_TIMEOUT_S, TIMEOUT_ENV)
    urls = required_urls(task_spec, env, task_args)
    if not urls:
        return

    # Run probes in parallel with daemon threads so stalled DNS workers do not delay
    # interpreter exit.  opener.open(timeout=...) does not bound getaddrinfo; the
    # wall-clock deadline enforced by joining with a remaining-time budget covers it.
    out: dict[str, tuple[bool, str, float]] = {}
    threads = []
    for u in urls:
        t = threading.Thread(
            target=lambda u=u: out.__setitem__(u, probe(u, timeout_s, env)),
            daemon=True,
        )
        t.start()
        threads.append(t)
    deadline = time.monotonic() + timeout_s
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    results: list[tuple[str, bool, str, float]] = [
        (u, *out[u]) if u in out
        else (u, False, f"no response within {timeout_s:g}s (DNS or connection)", timeout_s)
        for u in urls
    ]

    unreachable = []
    for url, ok, detail, elapsed in results:
        if ok:
            logger.info("Pre-flight: %s reachable (%s, %.1fs)", url, detail, elapsed)
        else:
            logger.warning("Pre-flight: %s UNREACHABLE (%s, %.1fs)", url, detail, elapsed)
            unreachable.append((url, detail))
    if not unreachable:
        return

    hosts = "; ".join(f"{u} ({d})" for u, d in unreachable)
    message = (
        f"Pre-flight check: cannot reach {hosts} for task {task_spec!r}. "
        "This task downloads its dataset at runtime, so without access it would stall in "
        "retry/backoff with no output. Fix one of: stage the dataset with test_data_ref "
        "(s3, pvc or git) so the job runs offline; point HF_ENDPOINT at a reachable mirror; "
        "or fix egress/proxy settings for the pod. To run anyway, set parameters.preflight "
        f"to 'warn' or 'off' (env {MODE_ENV})."
    )
    if mode == "fail":
        raise PreflightError(message)
    logger.warning("%s (continuing because %s=warn)", message, MODE_ENV)
