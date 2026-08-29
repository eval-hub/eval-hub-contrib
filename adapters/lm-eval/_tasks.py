"""Task registry: benchmark IDs → lm-eval task names and API style requirements.

Benchmarks are split into two scoring classes:
  LOGLIKELIHOOD_TASKS  — scored by comparing log probabilities of each answer
                         token. Requires a completions endpoint with echo+logprob
                         support (vLLM /v1/completions, Ollama compat mode).
                         Chat-only endpoints (OpenRouter, Groq, etc.) cannot
                         serve these requests.
  GENERATION_TASKS     — scored by extracting an answer from generated text.
                         Works with any endpoint, including chat-only.
"""

from __future__ import annotations

# Maps benchmark_id → lm-eval task name(s).
# All tasks here use loglikelihood (multiple_choice or loglikelihood output type).
# Requires: completions endpoint with echo=True and logprobs=1.
LOGLIKELIHOOD_TASKS: dict[str, str | list[str]] = {
    "lm-eval/mmlu": "mmlu",
    "lm-eval/mmlu_pro": "mmlu_pro",
    "lm-eval/arc_easy": "arc_easy",
    "lm-eval/arc_challenge": "arc_challenge",
    "lm-eval/hellaswag": "hellaswag",
    "lm-eval/winogrande": "winogrande",
    "lm-eval/truthfulqa_mc2": "truthfulqa_mc2",
    "lm-eval/boolq": "boolq",
    "lm-eval/piqa": "piqa",
    "lm-eval/openbookqa": "openbookqa",
    "lm-eval/bbh": "bbh",
    # GPQA: the zero-shot diamond variant is the registered task name in lm-eval.
    # The leaderboard group task is "leaderboard_gpqa" (used in the OLL-v2 composite).
    "lm-eval/gpqa": "gpqa_diamond_zeroshot",
    # MuSR lives only under the leaderboard group in lm-eval; there is no bare "musr" group.
    "lm-eval/musr": "leaderboard_musr",
}

# Maps benchmark_id → lm-eval task name(s).
# All tasks here use generate_until output type — works with any endpoint.
GENERATION_TASKS: dict[str, str | list[str]] = {
    "lm-eval/gsm8k": "gsm8k",
    "lm-eval/ifeval": "ifeval",
    "lm-eval/triviaqa": "triviaqa",
    "lm-eval/nq_open": "nq_open",
    # The task is registered as "hendrycks_math500" in lm-eval; "math_500" does not exist.
    "lm-eval/math_500": "hendrycks_math500",
}

# Composite suites (mix of task types).
# Value is the list of lm-eval task names to pass to --tasks.
COMPOSITE_SUITES: dict[str, list[str]] = {
    # Open LLM Leaderboard v2 — uses the "leaderboard_*" group task names that
    # map exactly to the configurations used by the HuggingFace Open LLM Leaderboard.
    # Contains loglikelihood tasks — requires completions endpoint.
    "lm-eval/open-llm-leaderboard-v2": [
        "leaderboard_mmlu_pro",
        "leaderboard_gpqa",
        "leaderboard_math_hard",
        "leaderboard_ifeval",
        "leaderboard_bbh",
        "leaderboard_musr",
    ],
    # Generation-only suite — safe for any endpoint including chat-only.
    "lm-eval/generation-suite": [
        "gsm8k",
        "ifeval",
        "triviaqa",
        "nq_open",
        "math_500",
    ],
}

# Which composite suites contain loglikelihood tasks.
_COMPOSITE_NEEDS_COMPLETIONS: dict[str, bool] = {
    "lm-eval/open-llm-leaderboard-v2": True,
    "lm-eval/generation-suite": False,
}

# Tasks explicitly excluded from this adapter with a human-readable reason.
EXCLUDED_TASKS: dict[str, str] = {
    "lm-eval/humaneval": (
        "HumanEval requires sandboxed Python code execution (exec()) and is not "
        "supported in this adapter. For code execution benchmarks use the "
        "swebench adapter, or request a sandboxed lm-eval variant from your "
        "platform team."
    ),
    "lm-eval/mbpp": (
        "MBPP uses code_eval metrics (Python exec) and is not supported in this "
        "adapter. For code execution benchmarks use the swebench adapter."
    ),
}

_ALL_KNOWN: dict[str, str | list[str]] = {**LOGLIKELIHOOD_TASKS, **GENERATION_TASKS}
_LOGLIKELIHOOD_IDS: frozenset[str] = frozenset(LOGLIKELIHOOD_TASKS)
_GENERATION_IDS: frozenset[str] = frozenset(GENERATION_TASKS)


def resolve_tasks(benchmark_id: str, task_override: str | None) -> str:
    """Return lm-eval task name(s) as a comma-separated string.

    Raises ValueError for excluded or unknown benchmark IDs.
    """
    if benchmark_id in EXCLUDED_TASKS:
        raise ValueError(
            f"Benchmark '{benchmark_id}' is not supported: {EXCLUDED_TASKS[benchmark_id]}"
        )

    if benchmark_id == "lm-eval/custom":
        if not task_override:
            raise ValueError(
                "lm-eval/custom requires a 'task' parameter containing one or more "
                "lm-eval task names (comma-separated) or the path to a custom task YAML."
            )
        return task_override

    if benchmark_id in COMPOSITE_SUITES:
        return ",".join(COMPOSITE_SUITES[benchmark_id])

    if benchmark_id in _ALL_KNOWN:
        val = _ALL_KNOWN[benchmark_id]
        return ",".join(val) if isinstance(val, list) else val

    all_ids = sorted(_ALL_KNOWN) + sorted(COMPOSITE_SUITES) + ["lm-eval/custom"]
    raise ValueError(
        f"Unknown benchmark '{benchmark_id}'.\nSupported benchmarks: {all_ids}"
    )


def requires_completions_endpoint(benchmark_id: str) -> bool:
    """Return True if this benchmark uses loglikelihood scoring."""
    if benchmark_id in _LOGLIKELIHOOD_IDS:
        return True
    if benchmark_id in _COMPOSITE_NEEDS_COMPLETIONS:
        return _COMPOSITE_NEEDS_COMPLETIONS[benchmark_id]
    # Unknown/custom: assume generation — user is responsible for endpoint choice.
    return False


def detect_api_style(params: dict, model_url: str) -> str:
    """Determine whether to use completions or chat endpoint.

    Returns 'completions' or 'chat'.
    Explicit api_style parameter always wins; otherwise inferred from model_url.
    Raises ValueError for unrecognised explicit values (typos like 'Chat' would
    otherwise silently fall through to auto-detection and pick the wrong endpoint).
    """
    explicit = params.get("api_style", "auto")
    if explicit == "chat":
        return "chat"
    if explicit == "completions":
        return "completions"
    if explicit != "auto":
        raise ValueError(
            f"Unknown api_style '{explicit}'. Valid values: 'auto', 'completions', 'chat'."
        )
    # Auto-detect from endpoint URL.
    url = (model_url or "").lower()
    if any(host in url for host in ("openrouter.ai", "groq.com", "together.ai")):
        return "chat"
    return "completions"


def lmeval_model_type(api_style: str) -> str:
    """Return the lm-eval model registry name for this API style."""
    return "local-chat-completions" if api_style == "chat" else "local-completions"


def build_endpoint_url(model_url: str, api_style: str) -> str:
    """Append /completions or /chat/completions to the base model URL.

    Strips any existing endpoint suffix before appending, so a URL that already
    contains /completions or /v1/completions does not get double-appended.
    Handles both 'http://host:8000' and 'http://host:8000/v1' inputs.
    """
    url = model_url.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/completions",
        "/chat/completions",
        "/completions",
    ):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    if not url.endswith("/v1"):
        url += "/v1"
    return url + ("/chat/completions" if api_style == "chat" else "/completions")


def preflight_check(benchmark_id: str, api_style: str) -> None:
    """Raise ValueError if a loglikelihood benchmark is paired with a chat-only endpoint.

    This is caught at job start — before any API calls are made — so the user
    gets a clear error message rather than a cryptic lm-eval runtime failure.
    """
    if api_style != "chat":
        return
    if not requires_completions_endpoint(benchmark_id):
        return

    generation_safe = sorted(
        _GENERATION_IDS | {"lm-eval/generation-suite", "lm-eval/custom"}
    )
    raise ValueError(
        f"Benchmark '{benchmark_id}' uses loglikelihood scoring, which requires a "
        f"completions endpoint that supports echo=True and logprobs (e.g. vLLM "
        f"/v1/completions, Ollama in OpenAI-compat mode).\n\n"
        f"Chat-only endpoints (OpenRouter, Groq, etc.) cannot serve loglikelihood "
        f"requests — the /v1/chat/completions spec does not expose token log-probs "
        f"in the format lm-eval requires.\n\n"
        f"Options:\n"
        f"  1. Point model.url at a vLLM or Ollama endpoint and omit api_style "
        f"(auto-detection will pick 'completions').\n"
        f"  2. Switch to a generation-safe benchmark: {generation_safe}"
    )
