"""Client selection and model spec building for the Inspect AI adapter."""

import json
from typing import Any

from _benchmarks import _CLIENT_ANTHROPIC, _CLIENT_OLLAMA, _CLIENT_OPENAI_COMPAT


def _is_ollama_endpoint(url: str | None) -> bool:
    """Return True if the URL points to an Ollama server (default port 11434)."""
    if not url:
        return False
    return ":11434" in url


def select_client(env: dict[str, str], endpoint_url: str | None = None) -> str:
    """Select the Inspect AI client routing key from available credentials."""
    if endpoint_url:
        if _is_ollama_endpoint(endpoint_url):
            return _CLIENT_OLLAMA
        return _CLIENT_OPENAI_COMPAT
    if env.get("OLLAMA_BASE_URL"):
        return _CLIENT_OLLAMA
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_BASE_URL"):
        return _CLIENT_ANTHROPIC
    if env.get("OPENAI_BASE_URL"):
        if _is_ollama_endpoint(env["OPENAI_BASE_URL"]):
            return _CLIENT_OLLAMA
        return _CLIENT_OPENAI_COMPAT
    if env.get("OPENAI_API_KEY"):
        return _CLIENT_OPENAI_COMPAT
    raise ValueError(
        "No API credentials found. "
        "Set OPENAI_BASE_URL or OPENAI_API_KEY for OpenAI-compatible endpoints "
        "(vLLM, OpenRouter), OLLAMA_BASE_URL for Ollama, "
        "or ANTHROPIC_API_KEY for the Anthropic API."
    )


def route_model(model_name: str, client: str) -> str:
    return f"{client}/{model_name}"


def build_role_spec(
    model_name: str,
    global_env: dict[str, str],
    role_base_url: str | None = None,
    role_api_key: str | None = None,
    role_anthropic_base_url: str | None = None,
    role_anthropic_api_key: str | None = None,
) -> str:
    """Build the Inspect AI model spec for one role.

    Returns a bare ``client/model`` string when global env is sufficient,
    or an inline JSON dict when per-role endpoint overrides are present.
    Each role can independently target a different provider or server.
    """
    if role_anthropic_base_url:
        cfg: dict[str, Any] = {
            "model": route_model(model_name, _CLIENT_ANTHROPIC),
            "base_url": role_anthropic_base_url,
        }
        if role_anthropic_api_key or role_api_key:
            cfg["api_key"] = role_anthropic_api_key or role_api_key
        return json.dumps(cfg)

    if role_base_url:
        cfg = {
            "model": route_model(model_name, _CLIENT_OPENAI_COMPAT),
            "base_url": role_base_url,
        }
        if role_api_key:
            cfg["api_key"] = role_api_key
        return json.dumps(cfg)

    if role_api_key:
        client = select_client(global_env)
        return json.dumps({"model": route_model(model_name, client), "api_key": role_api_key})

    if role_anthropic_api_key:
        return json.dumps({
            "model": route_model(model_name, _CLIENT_ANTHROPIC),
            "api_key": role_anthropic_api_key,
        })

    return route_model(model_name, select_client(global_env))


def target_model_spec(
    model_name: str,
    model_url: str | None,
    params: dict[str, Any],
    env: dict[str, str],
) -> str:
    return build_role_spec(
        model_name=model_name,
        global_env=env,
        role_base_url=params.get("target_base_url") or model_url or None,
        role_api_key=params.get("target_api_key") or None,
        role_anthropic_base_url=params.get("target_anthropic_base_url") or None,
        role_anthropic_api_key=params.get("target_anthropic_api_key") or None,
    )


def role_model_spec(
    model_name: str,
    role: str,
    params: dict[str, Any],
    env: dict[str, str],
) -> str:
    return build_role_spec(
        model_name=model_name,
        global_env=env,
        role_base_url=params.get(f"{role}_base_url"),
        role_api_key=params.get(f"{role}_api_key"),
        role_anthropic_base_url=params.get(f"{role}_anthropic_base_url"),
        role_anthropic_api_key=params.get(f"{role}_anthropic_api_key"),
    )
