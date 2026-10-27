"""Model-role connections come exclusively from the projected model Secret."""

from typing import Any

from evalhub.adapter.auth import read_model_auth_key


def validate_connection_parameters(parameters: dict[str, Any]) -> None:
    """Reject connection overrides before passing parameters to Inspect."""

    def visit(values: dict, prefix: str) -> None:
        for key, value in values.items():
            if key in {"api_key", "anthropic_api_key", "base_url"} or key.endswith(
                ("_api_key", "_api-key", "_base_url", "_url")
            ):
                raise ValueError(
                    f"{prefix}.{key} is no longer supported for model connections; "
                    "use <role>_api-key and <role>_url in model.auth.secret_ref"
                )
            if isinstance(value, dict):
                visit(value, f"{prefix}.{key}")

    # task_args may include unrelated dataset URLs. Only inspect model configuration.
    visit({k: v for k, v in parameters.items() if k != "task_args"}, "parameters")


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
