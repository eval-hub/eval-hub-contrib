"""
Monkey-patch LiteLLMClient.loglikelihood before starting the lighteval CLI.

LiteLLMClient.loglikelihood raises NotImplementedError in lighteval >=0.13.0
because the LiteLLM library does not expose token log-probabilities through its
completion interface.  This patch bypasses LiteLLM and calls the OpenAI-compatible
/v1/completions endpoint directly with echo=True and logprobs=1, which vLLM and
most OpenAI-compatible servers support.

Usage (called by the adapter instead of bare ``lighteval``):
    python lighteval_logprob_patch.py endpoint litellm <model_args> <tasks> [flags]

The patch:
- Imports lighteval internals and replaces the method before any task runs.
- Falls back gracefully (log-prob = -inf) if the endpoint does not support
  echo/logprobs, so generative-only evaluations are unaffected.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.lighteval_task import Doc


# ---------------------------------------------------------------------------
# loglikelihood implementation via /v1/completions echo + logprobs
# ---------------------------------------------------------------------------

def _loglikelihood_via_completions(self, docs: list["Doc"]) -> list["ModelResponse"]:  # noqa: N802
    """Compute log-likelihood of the continuation using vLLM echo+logprobs.

    POSTs to ``{base_url}/completions`` with:
        max_tokens=0, echo=True, logprobs=1

    Then sums the token log-probs that correspond to the continuation portion
    of the concatenated (context + continuation) prompt.
    """
    from lighteval.models.model_output import ModelResponse  # noqa: PLC0415

    results: list[ModelResponse] = []

    # Resolve the raw base URL (strip any trailing /v1)
    base_url: str = getattr(self, "base_url", "") or getattr(self, "_base_url", "")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    completions_url = f"{base_url}/v1/completions"

    model_name: str = getattr(self, "model_name", "") or getattr(self, "_model_name", "")
    # Strip the openai/ prefix that the adapter adds for LiteLLM routing
    model_name = model_name.removeprefix("openai/")

    api_key: str = getattr(self, "api_key", "") or getattr(self, "_api_key", "") or "dummy"

    logger.info(
        "loglikelihood patch: %d docs via %s model=%s",
        len(docs),
        completions_url,
        model_name,
    )

    for doc in docs:
        # lighteval 0.13.0 Doc uses `query` (not `context`).
        # For MC loglikelihood tasks lighteval creates one Doc per choice;
        # the continuation to evaluate is choices[0] when there is exactly
        # one choice, or '' when choices is empty (greedy tasks — shouldn't
        # reach this path, but guard defensively).
        context: str = getattr(doc, "query", "") or ""
        continuation: str = getattr(doc, "continuation", None) or (
            doc.choices[0] if getattr(doc, "choices", None) else ""
        )
        full_prompt: str = context + continuation
        ctx_char_len: int = len(context)

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }

        req = urllib.request.Request(
            completions_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                data = json.loads(resp.read())

            choice = data["choices"][0]
            logprobs_obj = choice.get("logprobs") or {}
            tokens: list[str] = logprobs_obj.get("tokens", [])
            token_logprobs: list[float | None] = logprobs_obj.get("token_logprobs", [])
            text_offsets: list[int] = logprobs_obj.get("text_offset", [])

            # Collect per-token log-probs for tokens in the continuation portion
            cont_logprobs: list[float] = []
            for i, offset in enumerate(text_offsets):
                lp = token_logprobs[i] if i < len(token_logprobs) else None
                if offset >= ctx_char_len and lp is not None and math.isfinite(lp):
                    cont_logprobs.append(lp)

            # Fall back to -inf when the endpoint returned no continuation logprobs
            if not cont_logprobs:
                cont_logprobs = [float("-inf")]

            n_ctx_tokens = len(tokens) - len(cont_logprobs)

        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("loglikelihood fallback (echo failed): %s", exc)
            cont_logprobs = [float("-inf")]
            n_ctx_tokens = 0

        # ModelResponse.logprobs holds per-token log-probs for the continuation.
        # argmax_logits_eq_gold is set to False; the downstream metric picks the
        # choice with the highest sum(logprobs) across all candidate Docs.
        results.append(
            ModelResponse(
                logprobs=cont_logprobs,
                argmax_logits_eq_gold=[False] * len(cont_logprobs),
                input_tokens=list(range(n_ctx_tokens)),   # placeholder token ids
                output_tokens=[list(range(len(cont_logprobs)))],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Apply the patch before lighteval's CLI runs
# ---------------------------------------------------------------------------

def _apply_patch() -> None:
    try:
        from lighteval.models.endpoints.litellm_model import LiteLLMClient  # noqa: PLC0415

        LiteLLMClient.loglikelihood = _loglikelihood_via_completions  # type: ignore[method-assign]
        logger.info("loglikelihood patch applied to LiteLLMClient")
    except ImportError as exc:
        logger.warning("Could not apply loglikelihood patch: %s", exc)


# ---------------------------------------------------------------------------
# Entry point — re-invoke the real lighteval CLI after patching
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _apply_patch()

    # Forward all CLI arguments to the real lighteval main
    from lighteval.__main__ import app  # noqa: PLC0415

    # sys.argv[0] is this script; lighteval's Typer app reads sys.argv[1:]
    sys.argv[0] = "lighteval"
    app()
