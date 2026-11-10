"""Judge model construction."""

from __future__ import annotations

from typing import Any

from evalhub.adapter.auth import resolve_model_credentials


def resolve_judge_model(judge_name: str, judge_url: str) -> Any:
    """Return a GPTModel pointed at an OpenAI-compatible endpoint.

    Forces JSON mode so small local models return parseable output.
    Credentials are resolved via the EvalHub SDK (mounted secret or env var).
    """
    from deepeval.models.llms import GPTModel

    api_key = resolve_model_credentials().api_key

    url = judge_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return GPTModel(
        model=judge_name,
        base_url=url,
        api_key=api_key or "EMPTY",
        generation_kwargs={"response_format": {"type": "json_object"}},
    )
