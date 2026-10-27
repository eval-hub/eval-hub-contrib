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
- Fails the evaluation when any choice cannot be scored (HTTP error, timeout,
  missing ``logprobs``, no continuation tokens).  A failed request is never
  turned into a likelihood: scoring it as ``-inf`` lets the argmax fall back to
  choice 0, which can be the gold answer, and the job would report an accuracy
  for a model that was never actually evaluated.  Generative-only evaluations
  never call ``loglikelihood`` and are unaffected.
- Scores the same text lighteval's reference backends (transformers, vLLM) score:
  ``PromptManager(use_chat_template=False).prepare_prompt(doc)``, i.e. the system
  prompt, the task instruction, the few-shot examples and the query, joined by
  blank lines, followed by the choice.  Scoring ``doc.query`` alone silently drops
  every few-shot example and changes the accuracy of ``num_fewshot > 0`` runs.
- Splits context from continuation the way lighteval's ``tok_encode_pair`` does:
  whitespace at the end of the context belongs to the continuation, and a token
  that straddles the boundary counts toward the continuation.  Splitting at
  ``len(context)`` instead drops the first answer token whenever the context ends
  in a space or newline (``"Answer: "`` + ``"A"`` tokenises as ``" A"``).
- Reports the number of scored continuation tokens per choice in
  ``ModelResponse.output_tokens`` so token-normalised metrics
  (``LogProbTokenNorm``) can divide by it.  The completions API returns token
  strings, not IDs, so the entries are positions in the echoed sequence: only
  their count is meaningful.
- Does not support PMI-normalised metrics (``Doc.unconditioned_query`` is
  ignored); lighteval rejects those with its own assertion.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.lighteval_task import Doc


class LoglikelihoodScoringError(RuntimeError):
    """A (context, choice) pair could not be scored.

    Raised instead of returning a likelihood so that a failed request cannot be
    mistaken for a model preference and produce an accuracy.
    """


#: Attempts per (context, choice) request before the evaluation is failed.
_MAX_ATTEMPTS = 3
#: HTTP statuses worth retrying; anything else (e.g. 400 for an endpoint that
#: rejects ``echo``) will not change on a second attempt.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# loglikelihood implementation via /v1/completions echo + logprobs
# ---------------------------------------------------------------------------

def _build_context(system_prompt: str | None, doc: "Doc") -> str:
    """Return the text lighteval's reference backends condition on for ``doc``.

    ``/v1/completions`` takes raw text, so this is the plain-text prompt
    (``use_chat_template=False``), the same one ``TransformersModel`` and
    ``VLLMModel`` use for log-likelihood scoring.
    """
    from lighteval.tasks.prompt_manager import PromptManager  # noqa: PLC0415

    return PromptManager(
        use_chat_template=False, system_prompt=system_prompt
    ).prepare_prompt(doc)


def _post_completion(req: urllib.request.Request) -> dict:
    """POST ``req`` and return the decoded JSON body, retrying transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS:
                raise LoglikelihoodScoringError(
                    f"{req.full_url} rejected the echo/logprobs request: HTTP {exc.code}"
                ) from exc
            last_exc = exc
        except (OSError, json.JSONDecodeError) as exc:
            # URLError, TimeoutError and connection resets are OSError subclasses.
            last_exc = exc
        if attempt + 1 < _MAX_ATTEMPTS:
            time.sleep(2**attempt)
    raise LoglikelihoodScoringError(
        f"{req.full_url} did not return a usable response after "
        f"{_MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


def _score_one_choice(
    completions_url: str,
    model_name: str,
    api_key: str,
    context: str,
    choice: str,
) -> tuple[float, list[int]]:
    """POST a single (context + choice) prompt and score the continuation.

    Returns ``(summed_logprob, token_positions)`` where ``token_positions`` has
    one entry per scored continuation token.

    Raises ``LoglikelihoodScoringError`` on any failure (endpoint error, timeout,
    missing logprobs, no continuation tokens).  Never returns ``-inf`` for a
    failed request.
    """
    full_prompt = context + choice
    # Trailing whitespace of the context is scored as part of the continuation
    # (lighteval's ``tok_encode_pair``), so the boundary sits before it.
    boundary = len(context.rstrip())

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
    data = _post_completion(req)

    try:
        logprobs_obj = data["choices"][0].get("logprobs")
        if not logprobs_obj:
            raise LoglikelihoodScoringError(
                f"{completions_url} returned no logprobs for choice {choice[:40]!r}; "
                "the endpoint must support echo=True with logprobs"
            )
        token_logprobs: list[float | None] = logprobs_obj["token_logprobs"]
        text_offsets: list[int] = logprobs_obj["text_offset"]
        tokens: list[str] = logprobs_obj.get("tokens") or []
    except (IndexError, KeyError, TypeError, AttributeError) as exc:
        raise LoglikelihoodScoringError(
            f"{completions_url} returned a malformed completions response for "
            f"choice {choice[:40]!r}: {exc!r}"
        ) from exc

    # End of the last echoed token: from its own text when the endpoint sends
    # one, so a truncated echo is detected instead of assumed to cover the prompt.
    if text_offsets and len(tokens) == len(text_offsets):
        last_end = text_offsets[-1] + len(tokens[-1])
    else:
        last_end = len(full_prompt)

    cont_sum = 0.0
    positions: list[int] = []
    for i, offset in enumerate(text_offsets):
        # A token belongs to the continuation if it reaches past the boundary;
        # its end is where the next echoed token starts.
        end = text_offsets[i + 1] if i + 1 < len(text_offsets) else last_end
        if end <= boundary:
            continue
        lp = token_logprobs[i] if i < len(token_logprobs) else None
        if lp is None or not math.isfinite(lp):
            raise LoglikelihoodScoringError(
                f"{completions_url} returned no usable logprob for continuation "
                f"token {i} of choice {choice[:40]!r}"
            )
        cont_sum += lp
        positions.append(i)

    if not positions:
        raise LoglikelihoodScoringError(
            f"No continuation tokens to score for choice {choice[:40]!r}: the "
            "endpoint's echoed tokens end before the choice begins"
        )
    return cont_sum, positions


def _loglikelihood_via_completions(self, docs: list["Doc"]) -> list["ModelResponse"]:  # noqa: N802
    """Compute log-likelihood for all choices of each Doc using vLLM echo+logprobs.

    lighteval's ``LoglikelihoodAcc`` metric expects ``ModelResponse.logprobs`` to
    contain **one summed log-prob per choice** (``logprobs[:n_choices]``), not
    per-token values for a single continuation.  This implementation scores every
    entry in ``doc.choices`` and returns the per-choice sums.

    POSTs to ``{base_url}/completions`` for each (context, choice) pair with:
        max_tokens=0, echo=True, logprobs=1

    Raises ``LoglikelihoodScoringError`` when any choice cannot be scored.
    """
    from lighteval.models.model_output import ModelResponse  # noqa: PLC0415

    results: list[ModelResponse] = []

    # Resolve the raw base URL (strip any trailing /v1)
    base_url: str = getattr(self, "base_url", "") or getattr(self, "_base_url", "")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if not base_url:
        raise LoglikelihoodScoringError(
            "loglikelihood patch: the LiteLLM client has no base_url to score against"
        )
    completions_url = f"{base_url}/v1/completions"

    # LiteLLMClient (lighteval 0.13.0) stores the configured model in
    # ``self.model``; ``model_name`` only exists on its config object.
    model_name: str = (
        getattr(self, "model", "")
        or getattr(self, "model_name", "")
        or getattr(self, "_model_name", "")
    )
    # Strip the openai/ prefix that the adapter adds for LiteLLM routing
    model_name = model_name.removeprefix("openai/")
    if not model_name:
        raise LoglikelihoodScoringError(
            "loglikelihood patch: could not resolve the model name from the "
            "LiteLLM client; refusing to send an empty \"model\" to the endpoint"
        )

    api_key: str = getattr(self, "api_key", "") or getattr(self, "_api_key", "") or "dummy"

    # The system prompt configured for the run (``system_prompt=`` in model_args)
    # is part of the prompt lighteval's reference backends score.
    system_prompt: str | None = getattr(getattr(self, "config", None), "system_prompt", None)
    if system_prompt is None:
        system_prompt = getattr(getattr(self, "prompt_manager", None), "system_prompt", None)

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
        context: str = _build_context(system_prompt, doc)
        choices: list[str] = list(getattr(doc, "choices", None) or [])
        if not choices:
            raise LoglikelihoodScoringError(
                f"loglikelihood patch: doc {getattr(doc, 'task_name', '?')!r} has "
                "no choices to score"
            )

        scored = [
            _score_one_choice(completions_url, model_name, api_key, context, ch)
            for ch in choices
        ]
        choice_sums = [total for total, _ in scored]

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
                # output_tokens[:n_choices] — one list per choice whose length is the
                # number of scored continuation tokens (LogProbTokenNorm divides by it).
                output_tokens=[positions for _, positions in scored],
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
