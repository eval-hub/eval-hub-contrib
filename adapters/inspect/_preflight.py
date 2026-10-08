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

import logging
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MODE_ENV = "INSPECT_PREFLIGHT"
TIMEOUT_ENV = "INSPECT_PREFLIGHT_TIMEOUT_S"
HOSTS_ENV = "INSPECT_PREFLIGHT_HOSTS"

MODES = ("fail", "warn", "off")
DEFAULT_MODE = "fail"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

# Task specs that never download a dataset from the Hub or GitHub.
_NO_PROBE_PREFIXES = ("inspect_petri/", "petri_bloom/")
# Task specs whose dataset comes from the HuggingFace Hub.
_HF_TASK_PREFIXES = ("inspect_evals/", "evals/")


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


def _bfcl_data_present(env: dict[str, str]) -> bool:
    cache = env.get("INSPECT_EVALS_CACHE_DIR")
    if not cache:
        return False
    return any((Path(cache) / "BFCL").glob("BFCL_v*_*.json"))


def required_urls(task_spec: str, env: dict[str, str]) -> list[str]:
    """URLs the task is known to need at runtime, plus any user-supplied hosts."""
    urls: list[str] = []
    if not task_spec.startswith(_NO_PROBE_PREFIXES):
        hub_offline = env.get("HF_HUB_OFFLINE") == "1" or env.get("HF_DATASETS_OFFLINE") == "1"
        if task_spec.startswith(_HF_TASK_PREFIXES) and not hub_offline:
            urls.append(env.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT)
        if "bfcl" in task_spec and not _bfcl_data_present(env):
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


def run_preflight(task_spec: str, env: dict[str, str]) -> None:
    """Probe the hosts ``task_spec`` needs; raise :class:`PreflightError` in ``fail`` mode."""
    mode = normalize_mode(env.get(MODE_ENV) or DEFAULT_MODE, MODE_ENV)
    if mode == "off":
        logger.info("Pre-flight network check disabled (%s=off)", MODE_ENV)
        return
    timeout_s = normalize_timeout(env.get(TIMEOUT_ENV) or DEFAULT_TIMEOUT_S, TIMEOUT_ENV)
    urls = required_urls(task_spec, env)
    if not urls:
        return

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        results = list(pool.map(lambda u: (u, *probe(u, timeout_s, env)), urls))

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
