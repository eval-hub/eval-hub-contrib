"""HuggingFace Hub token resolution for Inspect subprocess environments."""

import logging
import os
import time

from evalhub.adapter.auth import read_model_auth_key

logger = logging.getLogger(__name__)

_HF_MOUNT_KEY = "hf-token"
_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_K8S_WAIT_TIMEOUT_S = 60.0
_DEFAULT_POLL_INTERVAL_S = 0.5


def _default_wait_timeout_s() -> float:
    """Wait for projected ``hf-token`` only in EvalHub Kubernetes job pods."""
    mode = os.environ.get("EVALHUB_MODE", "").strip().lower()
    if mode == "k8s":
        return _K8S_WAIT_TIMEOUT_S
    return 0.0


def _is_sidecar_ref_placeholder(value: str) -> bool:
    """True when the value is an EvalHub sidecar ref token, not a real credential."""
    return value.strip().endswith(":ref")


def _valid_hf_token(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if _is_sidecar_ref_placeholder(cleaned):
        return None
    return cleaned


def _read_hf_token_from_mount() -> str | None:
    return _valid_hf_token(read_model_auth_key(_HF_MOUNT_KEY))


def _read_hf_token_from_env(env: dict[str, str]) -> str | None:
    for key in _ENV_KEYS:
        raw = env.get(key)
        if raw is None:
            continue
        token = _valid_hf_token(raw)
        if token:
            return token
        if raw.strip():
            logger.warning(
                "Ignoring %s in subprocess environment (empty or sidecar ref placeholder)",
                key,
            )
    return None


def resolve_hf_token(
    env: dict[str, str],
    *,
    wait_timeout_s: float | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> tuple[str | None, str]:
    """Resolve a HuggingFace token for Hub dataset access.

    Prefers a real ``hf-token`` file under ``/var/run/secrets/model`` (with optional
    retry while the projected volume appears). Falls back to ``HF_TOKEN`` or
    ``HUGGING_FACE_HUB_TOKEN`` in ``env`` when they contain a non-ref value.

    Returns ``(token, source)`` where ``source`` is ``"mount"``, ``"environment"``,
    or ``""`` when unresolved.
    """
    timeout = wait_timeout_s if wait_timeout_s is not None else _default_wait_timeout_s()
    deadline = time.monotonic() + max(timeout, 0.0)
    attempt = 0

    while True:
        attempt += 1
        mount_token = _read_hf_token_from_mount()
        if mount_token:
            if attempt > 1:
                logger.info(
                    "Resolved HuggingFace token from mounted secret after %d attempt(s)",
                    attempt,
                )
            return mount_token, "mount"

        env_token = _read_hf_token_from_env(env)
        if env_token:
            return env_token, "environment"

        if timeout <= 0 or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    if timeout > 0:
        logger.warning(
            "HuggingFace token not found after %.0fs (checked mount key %r and %s)",
            timeout,
            _HF_MOUNT_KEY,
            ", ".join(_ENV_KEYS),
        )
    return None, ""


def apply_hf_hub_auth(
    env: dict[str, str],
    *,
    wait_timeout_s: float | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Set ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN`` on ``env`` when a token resolves."""
    timeout = wait_timeout_s if wait_timeout_s is not None else _default_wait_timeout_s()
    token, source = resolve_hf_token(
        env,
        wait_timeout_s=timeout,
        poll_interval_s=poll_interval_s,
    )
    for key in _ENV_KEYS:
        env.pop(key, None)

    if not token:
        return

    env["HF_TOKEN"] = token
    env["HUGGING_FACE_HUB_TOKEN"] = token
    logger.info("Injected HuggingFace Hub authentication from %s", source)


def refresh_hf_hub_auth(env: dict[str, str]) -> None:
    """Re-resolve HF auth immediately before a subprocess (no long wait)."""
    apply_hf_hub_auth(env, wait_timeout_s=0)
