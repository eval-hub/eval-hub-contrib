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
- Aligns the echoed token offsets with the prompt first.  vLLM's ``text_offset`` is the
  running length of the decoded tokens, so a BOS token (``"<s>"``) shifts every offset
  by its length.  The shift is removed only when whole leading tokens account for it;
  an echo shorter than the prompt, or one that cannot be explained that way, fails the
  job instead of being scored.
- Reports the number of scored continuation tokens per choice in
  ``ModelResponse.output_tokens`` so token-normalised metrics
  (``LogProbTokenNorm``) can divide by it.  The completions API returns token
  strings, not IDs, so the entries are positions in the echoed sequence: only
  their count is meaningful.
- Turns off litellm's disk response cache.  lighteval enables it at import
  (``litellm.cache = Cache(DISK)`` plus ``caching=True`` on every request), and on
  litellm >= 1.10x a cache *hit* imports ``litellm.proxy`` modules that need
  ``orjson`` and ``soundfile``, which ``litellm[caching]`` does not install.  The
  request then raises, lighteval retries and finally returns an empty answer, so
  the benchmark is scored on blank output instead of failing.  A hit needs an
  identical repeated request, so it shows up only on re-runs and duplicated
  prompts, and an evaluation should query the model anyway.
- Caps the memory of the process that runs a model-written solution when
  LiveCodeBench is graded.  lighteval's ``reliability_guard()`` is called without a
  limit (its comment says 4 GB), so a solution that allocates in a loop grows until the
  container is killed: a 175-problem run generated every answer and was then OOMKilled
  at 512 GiB.  The grader is a forked child, so the address space it already inherits is
  not the solution's: the limit is that size plus ``EVALHUB_LIGHTEVAL_CODE_MEMORY_MIB``
  (default 2048, ``0`` turns it off).  A solution that exceeds it scores as failed.
  Applies to the fork start method (the default on Linux before Python 3.14), the only
  one under which lighteval's grader runs.
- Does not support PMI-normalised metrics (``Doc.unconditioned_query`` is
  ignored); lighteval rejects those with its own assertion.
"""

from __future__ import annotations

import json
import logging
import math
import os
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


def _anchor_offsets(
    completions_url: str,
    choice: str,
    full_prompt: str,
    tokens: list[str],
    text_offsets: list[int],
) -> tuple[list[int], int]:
    """Return ``(offsets, end_of_last_token)`` as character positions in ``full_prompt``.

    vLLM builds ``text_offset`` from the running length of each *decoded* token string,
    and decodes the first prompt token on its own, so a tokenizer that prepends a BOS
    token (Llama, Mistral) puts ``"<s>"`` in front: every offset is then larger than
    the character position in the prompt by ``len("<s>")``.  Comparing those offsets
    with the context length would count the last few context tokens as part of the
    answer.

    The shift is the echoed text minus the prompt.  It is only accepted when whole
    leading tokens add up to it exactly; anything else cannot be aligned, and scoring
    a guess would produce a number that looks like a likelihood.  An echo shorter than
    the prompt was cut off.
    """
    if not tokens or len(tokens) != len(text_offsets):
        return text_offsets, len(full_prompt)  # nothing to check against

    echoed_len = text_offsets[-1] + len(tokens[-1])
    if echoed_len < len(full_prompt):
        raise LoglikelihoodScoringError(
            f"{completions_url} echoed {echoed_len} characters for a "
            f"{len(full_prompt)}-character prompt (choice {choice[:40]!r}): the echoed "
            "tokens end before the choice begins"
        )
    shift = echoed_len - len(full_prompt)
    if shift == 0:
        return text_offsets, len(full_prompt)

    covered = 0
    for token in tokens:
        covered += len(token)
        if covered >= shift:
            break
    if covered != shift:
        raise LoglikelihoodScoringError(
            f"{completions_url} echoed {shift} characters more than the prompt (choice "
            f"{choice[:40]!r}) and they are not whole leading tokens, so the token "
            "offsets cannot be aligned with the prompt"
        )
    return [max(0, offset - shift) for offset in text_offsets], len(full_prompt)


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

    text_offsets, last_end = _anchor_offsets(
        completions_url, choice, full_prompt, tokens, text_offsets
    )

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

def _disable_litellm_response_cache() -> None:
    """Drop the disk response cache lighteval installs when it imports litellm.

    Must run after ``lighteval.models.endpoints.litellm_model`` has been imported:
    that import is what sets ``litellm.cache``, and later imports are no-ops.
    """
    import litellm  # noqa: PLC0415

    if getattr(litellm, "cache", None) is not None:
        litellm.cache = None
        logger.info("litellm response cache disabled (cache hits need proxy-only packages)")


_CODE_MEMORY_ENV = "EVALHUB_LIGHTEVAL_CODE_MEMORY_MIB"
_DEFAULT_CODE_MEMORY_MIB = 2048


def _address_space_bytes() -> int | None:
    """Current virtual size of this process (``VmSize``), or None where unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _code_memory_budget_bytes() -> int:
    """Memory a graded solution may add to what its process inherited; 0 means no cap."""
    raw = os.environ.get(_CODE_MEMORY_ENV, "")
    try:
        mib = int(raw) if raw.strip() else _DEFAULT_CODE_MEMORY_MIB
    except ValueError:
        logger.warning("ignoring %s=%r (not an integer)", _CODE_MEMORY_ENV, raw)
        mib = _DEFAULT_CODE_MEMORY_MIB
    return max(mib, 0) * 1024 * 1024


def _cap_code_grader_memory() -> None:
    """Give lighteval's LiveCodeBench grader the memory limit it was meant to have.

    ``run_test`` calls ``reliability_guard()`` with no limit, so a solution that
    allocates without bound is stopped only by the container.  Wrap it to set
    ``RLIMIT_AS`` to the child's inherited size plus the budget.  Only the address
    space is limited: ``RLIMIT_STACK`` stays as it was.
    """
    budget = _code_memory_budget_bytes()
    if budget == 0:
        logger.info("%s=0: LiveCodeBench grader memory is not capped", _CODE_MEMORY_ENV)
        return
    try:
        import resource  # noqa: PLC0415 (reliability_guard hides it from the graded code)

        from lighteval.tasks.tasks.lcb import codegen_metrics  # noqa: PLC0415
    except ImportError as exc:
        logger.info("LiveCodeBench grader memory cap skipped: %s", exc)
        return

    original = codegen_metrics.reliability_guard
    if getattr(original, "_evalhub_capped", False):
        return

    def reliability_guard(maximum_memory_bytes=None):
        if maximum_memory_bytes is None and (inherited := _address_space_bytes()) is not None:
            limit = inherited + budget
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError) as exc:
                logger.warning("could not cap grader memory at %d bytes: %s", limit, exc)
        return original(maximum_memory_bytes)

    reliability_guard._evalhub_capped = True  # type: ignore[attr-defined]
    codegen_metrics.reliability_guard = reliability_guard
    logger.info("LiveCodeBench grader memory capped at +%d MiB per solution", budget // (1024 * 1024))


def _apply_patch() -> None:
    _cap_code_grader_memory()
    try:
        from lighteval.models.endpoints.litellm_model import LiteLLMClient  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("Could not import LiteLLMClient — patch skipped: %s", exc)
        return

    _disable_litellm_response_cache()

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
