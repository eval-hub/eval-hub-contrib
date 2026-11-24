"""Judge model construction."""

from __future__ import annotations

from typing import Any

from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials


def resolve_judge_model(
    judge_name: str,
    model_url: str,
    *,
    max_tokens: int | None = None,
    structured_outputs: bool = False,
) -> Any:
    """Return a GPTModel pointed at an OpenAI-compatible endpoint.

    Both the judge endpoint and credential come from the model auth Secret
    (``judge_url`` and ``judge_api-key``); without them the evaluated model's
    URL and key are reused.

    ``max_tokens`` bounds each judge completion and is only passed to DeepEval
    when set. ``structured_outputs`` opts a self-hosted endpoint into DeepEval's
    native JSON-schema generation. When the model uses neither that path nor
    DeepEval's JSON mode, ``response_format`` is forced to ``json_object`` so
    small local models return parseable output; otherwise DeepEval supplies it
    once and the adapter does not duplicate it.
    """
    from deepeval.models.llms import GPTModel

    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("parameters.max_tokens must be a positive integer")
    if type(structured_outputs) is not bool:
        raise ValueError("parameters.judge_structured_outputs must be a boolean")

    api_key = read_model_auth_key("judge_api-key")
    judge_url = read_model_auth_key("judge_url")
    if judge_url and not api_key:
        raise ValueError("judge_url in model.auth.secret_ref requires judge_api-key")
    api_key = api_key or resolve_model_credentials().api_key

    url = (judge_url or model_url).strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    class SchemaJudge(GPTModel):
        """Enable DeepEval's native schema path only for explicitly capable endpoints."""

        def supports_structured_outputs(self):
            return True

    model_class = SchemaJudge if structured_outputs else GPTModel
    judge = model_class(
        model=judge_name,
        base_url=url,
        api_key=api_key or "EMPTY",
        # Keep DeepEval's token-limit default unless the job supplies an override.
        generation_kwargs={"max_tokens": max_tokens} if max_tokens is not None else {},
    )
    if not judge.supports_structured_outputs() and not judge.supports_json_mode():
        judge.generation_kwargs["response_format"] = {"type": "json_object"}
    return judge
