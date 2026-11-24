"""Judge model construction."""

from __future__ import annotations

from typing import Any

from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials


def resolve_judge_model(judge_name: str, model_url: str) -> Any:
    """Return a GPTModel pointed at an OpenAI-compatible endpoint.

    Both the judge endpoint and credential come from the model auth Secret
    (``judge_url`` and ``judge_api-key``); without them the evaluated model's
    URL and key are reused. Forces JSON mode so small local models return
    parseable output.
    """
    from deepeval.models.llms import GPTModel

    api_key = read_model_auth_key("judge_api-key")
    judge_url = read_model_auth_key("judge_url")
    if judge_url and not api_key:
        raise ValueError("judge_url in model.auth.secret_ref requires judge_api-key")
    api_key = api_key or resolve_model_credentials().api_key

    url = (judge_url or model_url).strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return GPTModel(
        model=judge_name,
        base_url=url,
        api_key=api_key or "EMPTY",
        generation_kwargs={"response_format": {"type": "json_object"}},
    )
