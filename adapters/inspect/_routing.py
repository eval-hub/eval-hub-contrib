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
        role_client = _CLIENT_OLLAMA if _is_ollama_endpoint(role_base_url) else _CLIENT_OPENAI_COMPAT
        cfg: dict[str, Any] = {
            "model": route_model(model_name, role_client),
            "base_url": role_base_url,
        }
        if role_client == _CLIENT_OPENAI_COMPAT:
            cfg["model_args"] = {"responses_api": False}
        if role_api_key:
            cfg["api_key"] = role_api_key
        return json.dumps(cfg)

    client = select_client(global_env)
    model_str = route_model(model_name, client)

    if role_api_key:
        cfg: dict[str, Any] = {"model": model_str, "api_key": role_api_key}
        if client == _CLIENT_OPENAI_COMPAT:
            cfg["model_args"] = {"responses_api": False}
        return json.dumps(cfg)

    if role_anthropic_api_key:
        return json.dumps({
            "model": route_model(model_name, _CLIENT_ANTHROPIC),
            "api_key": role_anthropic_api_key,
        })

    if client == _CLIENT_OPENAI_COMPAT:
        # Disable the Responses API for all OpenAI-compat roles. vLLM and other
        # compat servers don't implement POST /responses/input_tokens, which
        # inspect_ai calls during compaction when responses_api is True.
        return json.dumps({"model": model_str, "model_args": {"responses_api": False}})

    return model_str


def target_model_spec(
    model_name: str,
    model_url: str | None,
    params: dict[str, Any],
    env: dict[str, str],
) -> str:
    """Build the target role model spec.

    Always returns a JSON dict for OpenAI-compat clients so that
    ``model_args: {"responses_api": false}`` can be injected explicitly.
    vLLM and other compat servers do not implement the Responses API token-
    counting endpoint (POST /responses/input_tokens), which inspect_ai calls
    during compaction if responses_api is True or inferred True.
    Passing it as False here is belt-and-suspenders alongside the version pin.

    This spec must be passed via --model-role CLI flag (not env var) because
    JSON dicts with commas break inspect_ai's env var role parser.
    """
    # Do NOT fall back to model_url here. build_env already sets OPENAI_BASE_URL
    # from config.model.url, so the provider picks it up via env var. Including
    # base_url in the JSON role spec requires a newer inspect_ai that explicitly
    # pops it from the role dict before GenerateConfig validation.
    role_base_url = params.get("target_base_url") or None
    role_api_key = params.get("target_api_key") or None
    role_anthropic_base_url = params.get("target_anthropic_base_url") or None
    role_anthropic_api_key = params.get("target_anthropic_api_key") or None

    if role_anthropic_base_url:
        cfg: dict[str, Any] = {
            "model": route_model(model_name, _CLIENT_ANTHROPIC),
            "base_url": role_anthropic_base_url,
        }
        if role_anthropic_api_key or role_api_key:
            cfg["api_key"] = role_anthropic_api_key or role_api_key
        return json.dumps(cfg)

    # Determine client: explicit base URL takes precedence over global env
    if role_base_url:
        client = _CLIENT_OLLAMA if _is_ollama_endpoint(role_base_url) else _CLIENT_OPENAI_COMPAT
    else:
        client = select_client(env, endpoint_url=model_url)

    model_str = route_model(model_name, client)

    if client == _CLIENT_OPENAI_COMPAT:
        cfg = {"model": model_str, "model_args": {"responses_api": False}}
        if role_base_url:
            cfg["base_url"] = role_base_url
        if role_api_key:
            cfg["api_key"] = role_api_key
        return json.dumps(cfg)

    # Ollama or Anthropic fallback (no model_args injection needed)
    return build_role_spec(
        model_name=model_name,
        global_env=env,
        role_base_url=role_base_url,
        role_api_key=role_api_key,
        role_anthropic_api_key=role_anthropic_api_key,
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
