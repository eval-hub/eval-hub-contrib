"""Model-role connections come exclusively from the projected model Secret."""

from typing import Any

from evalhub.adapter.auth import read_model_auth_key


def without_connection_options(value: Any) -> Any:
    """Copy model options without inline credentials or endpoint overrides."""
    # Inspect consumes model_args directly, so ignoring top-level connection
    # fields would still let nested credentials/URLs override Secret routing.
    # Copy options recursively without those overrides, retaining generation
    # settings so legacy jobs continue. Secret resolution happens separately.
    if isinstance(value, dict):
        return {
            key: without_connection_options(child)
            for key, child in value.items()
            if key not in {"api_key", "api-key", "api_base", "base_url", "url"}
            and not key.endswith(("_api_key", "_api-key", "_base_url", "_url"))
        }
    if isinstance(value, list):
        return [without_connection_options(child) for child in value]
    return value


def role_connection(role: str, model_url: str) -> tuple[str, str]:
    key = read_model_auth_key(f"{role}_api-key")
    url = read_model_auth_key(f"{role}_url")
    if url and not key:
        raise ValueError(f"{role}_url in model.auth.secret_ref requires {role}_api-key")
    url = (url or model_url or "").strip().rstrip("/")
    if not url:
        raise ValueError(
            f"{role}_url in model.auth.secret_ref or model.url is required"
        )
    return url, key or read_model_auth_key("api-key") or "local"
