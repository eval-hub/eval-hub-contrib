"""HuggingFace Hub token resolution for Inspect subprocess environments."""

import logging
import os
import time
from pathlib import Path

from evalhub.adapter.auth import read_model_auth_key

logger = logging.getLogger(__name__)

_HF_MOUNT_KEY = "hf-token"
_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_MODEL_MOUNT_DIR = Path("/var/run/secrets/model")
# Grace period used only when the model-auth mount exists but is still empty.
_K8S_EMPTY_MOUNT_GRACE_S = 5.0
_WAIT_ENV = "INSPECT_HF_TOKEN_WAIT_S"
_DEFAULT_POLL_INTERVAL_S = 0.5


def _default_wait_timeout_s() -> float:
    """Seconds to poll for the projected ``hf-token`` before giving up.

    The model-auth mount is a Kubernetes projected volume that exists only when the job
    sets ``model.auth.secret_ref``, and kubelet writes all of its files at once before
    the container starts. So in EvalHub k8s job pods:

    * no mount directory: no model-auth secret is configured, nothing can appear: no wait
    * mount has files: it is fully populated, a missing ``hf-token`` is final: no wait
    * mount exists but is empty: ambiguous (a secret with none of the optional keys looks
      the same as one not yet populated): a short grace period

    ``INSPECT_HF_TOKEN_WAIT_S`` overrides this (seconds; 0 disables the wait).
    """
    override = os.environ.get(_WAIT_ENV, "").strip()
    if override:
        try:
            return max(float(override), 0.0)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r (expected seconds)", _WAIT_ENV, override)

    if os.environ.get("EVALHUB_MODE", "").strip().lower() != "k8s":
        return 0.0
    try:
        if not _MODEL_MOUNT_DIR.is_dir():
            return 0.0
        populated = any(not entry.name.startswith(".") for entry in _MODEL_MOUNT_DIR.iterdir())
    except OSError:
        return _K8S_EMPTY_MOUNT_GRACE_S
    return 0.0 if populated else _K8S_EMPTY_MOUNT_GRACE_S


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
    else:
        logger.debug(
            "No HuggingFace token found (checked mount key %r and %s)",
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
