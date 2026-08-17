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

def _score_one_choice(
    completions_url: str,
    model_name: str,
    api_key: str,
    context: str,
    choice: str,
) -> float:
    """POST a single (context + choice) prompt and return the summed continuation log-prob.

    Returns ``float("-inf")`` on any failure (endpoint error, missing logprobs).
    """
    full_prompt = context + choice
    ctx_char_len = len(context)

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
        choice_resp = data["choices"][0]
        logprobs_obj = choice_resp.get("logprobs") or {}
        token_logprobs: list[float | None] = logprobs_obj.get("token_logprobs", [])
        text_offsets: list[int] = logprobs_obj.get("text_offset", [])
        cont_sum = 0.0
        found = False
        for i, offset in enumerate(text_offsets):
            if offset >= ctx_char_len:
                lp = token_logprobs[i] if i < len(token_logprobs) else None
                if lp is not None and math.isfinite(lp):
                    cont_sum += lp
                    found = True
        return cont_sum if found else float("-inf")
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("loglikelihood fallback (echo failed for choice %r): %s", choice[:40], exc)
        return float("-inf")


def _loglikelihood_via_completions(self, docs: list["Doc"]) -> list["ModelResponse"]:  # noqa: N802
    """Compute log-likelihood for all choices of each Doc using vLLM echo+logprobs.

    lighteval's ``LoglikelihoodAcc`` metric expects ``ModelResponse.logprobs`` to
    contain **one summed log-prob per choice** (``logprobs[:n_choices]``), not
    per-token values for a single continuation.  This implementation scores every
    entry in ``doc.choices`` and returns the per-choice sums.

    POSTs to ``{base_url}/completions`` for each (context, choice) pair with:
        max_tokens=0, echo=True, logprobs=1
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
        # lighteval 0.13.0 Doc.choices holds ALL candidate continuations for this
        # sample.  We must score every choice and return per-choice sums so that
        # LoglikelihoodAcc can pick the argmax (logprobs[:n_choices]).
        context: str = getattr(doc, "query", "") or ""
        choices: list[str] = list(getattr(doc, "choices", None) or [])
        if not choices:
            choices = [""]  # generative fallback — shouldn't reach this path

        choice_sums: list[float] = [
            _score_one_choice(completions_url, model_name, api_key, context, ch)
            for ch in choices
        ]

        # Determine argmax and compare to gold.
        best_idx = max(range(len(choice_sums)), key=lambda i: choice_sums[i])
        gold_index = getattr(doc, "gold_index", 0)
        gold_ixs: list[int] = gold_index if isinstance(gold_index, list) else [gold_index]
        argmax_eq_gold = [best_idx in gold_ixs]

        results.append(
            ModelResponse(
                # logprobs[:n_choices] — one sum per choice, as expected by LoglikelihoodAcc
                logprobs=choice_sums,
                argmax_logits_eq_gold=argmax_eq_gold,
                input_tokens=[],
                # output_tokens[:n_choices] — placeholder lists (no real token IDs available)
                output_tokens=[[] for _ in choices],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Apply the patch before lighteval's CLI runs
# ---------------------------------------------------------------------------

def _apply_patch() -> None:
    try:
        from lighteval.models.endpoints.litellm_model import LiteLLMClient  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("Could not import LiteLLMClient — patch skipped: %s", exc)
        return

    # Guard: only patch when the method still raises NotImplementedError so that
    # a future lighteval release that ships a real implementation is not clobbered.
    import inspect as _inspect
    try:
        src = _inspect.getsource(LiteLLMClient.loglikelihood)
        if "NotImplementedError" not in src:
            logger.info(
                "loglikelihood patch skipped — LiteLLMClient already has an implementation"
            )
            return
    except (OSError, TypeError):
        pass  # can't read source (compiled); apply the patch anyway

    LiteLLMClient.loglikelihood = _loglikelihood_via_completions  # type: ignore[method-assign]
    logger.info("loglikelihood patch applied to LiteLLMClient")


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
