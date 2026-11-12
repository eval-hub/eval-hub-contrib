"""Task registry: benchmark IDs -> lm-eval task names and API style requirements.

Everything the adapter knows about a benchmark comes from one generated table,
``_benchmarks.py`` (see ``scripts/generate_benchmarks.py``), which is read from
the pinned lm-eval's own task configs. The predicates below only look things up
in it, so they cannot disagree with each other or with the framework.

Benchmarks fall into two scoring classes:
  loglikelihood  scored by comparing log probabilities of each answer token.
                 Needs a completions endpoint with echo+logprob support (vLLM
                 /v1/completions, Ollama compat mode). Chat-only endpoints
                 (OpenRouter, Groq, etc.) cannot serve these requests.
  generation     scored by extracting an answer from generated text. Works with
                 any endpoint, including chat-only.

``lm-eval/custom`` and tasks absent from the table (user-authored YAML) are
assumed to be generation: the user is responsible for the endpoint choice.
lm-eval itself refuses an ``unsafe_code`` task unless the confirmation flag is
passed, so an unknown task never runs model-generated code by accident.
"""

from __future__ import annotations

from _benchmarks import BENCHMARKS, TASKS, TaskSpec

# Composite suites: the value is the list of lm-eval task names for --tasks.
# Which tasks make up a suite is a product decision, so this stays hand-written;
# whether a suite needs completions is derived from its members.
COMPOSITE_SUITES: dict[str, list[str]] = {
    # Open LLM Leaderboard v2: the "leaderboard_*" group tasks that map to the
    # configurations the HuggingFace Open LLM Leaderboard uses.
    "lm-eval/open-llm-leaderboard-v2": [
        "leaderboard_mmlu_pro",
        "leaderboard_gpqa",
        "leaderboard_math_hard",
        "leaderboard_ifeval",
        "leaderboard_bbh",
        "leaderboard_musr",
    ],
    # Generation-only suite: safe for any endpoint including chat-only.
    "lm-eval/generation-suite": [
        "gsm8k",
        "ifeval",
        "triviaqa",
        "nq_open",
        # Registered as "hendrycks_math500" in lm-eval 0.4.12 ("math_500" is the
        # benchmark ID, not a task name).
        "hendrycks_math500",
    ],
}


# Benchmark IDs the server's ``lm_evaluation_harness`` provider lists but this
# adapter deliberately does not offer, with the reason shown to the user. They are
# kept out of provider.yaml, so ``resolve_tasks`` is the only place that can say why.
_PILE_REASON = (
    "it loads EleutherAI/pile, a dataset script whose data host (the-eye.eu) no "
    "longer serves the files (the original Pile was taken down), so it cannot run "
    "with any datasets version. For perplexity on Pile text, run pile_10k "
    "(NeelNanda/pile-10k) through lm-eval/custom with task=pile_10k"
)
_DATASET_SCRIPT_REASON = (
    "it loads a dataset script from the Hugging Face Hub, which datasets>=4 no "
    "longer supports. This adapter pins the datasets release the AIPCC CPU channel "
    "publishes (5.1.0), so the script loader is gone. Run it through a build that "
    "ships datasets<4 if you need this benchmark"
)
_LMEVAL_VERSION_REASON = (
    "it needs lm-eval 0.4.13 or later, and this adapter pins 0.4.12, the highest "
    "release the AIPCC CPU channel publishes"
)
UNSUPPORTED_BENCHMARKS: dict[str, str] = {
    f"pile_{subset}": _PILE_REASON
    for subset in (
        "arxiv",
        "freelaw",
        "hackernews",
        "openwebtext2",
        "ubuntu-irc",
        "youtubesubtitles",
    )
}
UNSUPPORTED_BENCHMARKS |= {
    benchmark_id: _DATASET_SCRIPT_REASON
    for benchmark_id in (
        "aclue_ancient_chinese_culture",
        "cmmlu_college_mathematics",
        "cmmlu_high_school_mathematics",
        "cmmlu_professional_medicine",
        "cmmlu_traditional_chinese_medicine",
        "ethics_cm",
        "qasper_freeform",
    )
}
UNSUPPORTED_BENCHMARKS["cruxeval_output"] = _LMEVAL_VERSION_REASON


def task_for(benchmark_id: str) -> str | None:
    """Return the lm-eval task for a benchmark ID, or None if it is not in the table.

    Accepts the curated ``lm-eval/arc_easy`` form, the bare ID the server's
    ``lm_evaluation_harness`` provider uses (``arc_easy``), and a bare ID under
    this adapter's prefix (``lm-eval/arc_challenge`` for a bare-only benchmark).
    """
    if benchmark_id in BENCHMARKS:
        return BENCHMARKS[benchmark_id]
    return BENCHMARKS.get(benchmark_id.removeprefix("lm-eval/"))


def _spec(benchmark_id: str) -> TaskSpec | None:
    task = task_for(benchmark_id)
    return TASKS.get(task) if task else None


def is_rolling(benchmark_id: str) -> bool:
    """Return True for perplexity (loglikelihood_rolling) benchmarks."""
    spec = _spec(benchmark_id)
    return bool(spec and spec.rolling)


def is_code_execution(benchmark_id: str) -> bool:
    """Return True if the benchmark executes model-generated code."""
    spec = _spec(benchmark_id)
    return bool(spec and spec.unsafe_code)


def tasks_include_code_execution(tasks: str) -> bool:
    """Return True if a comma-separated lm-eval task list names an unsafe-code task.

    ``is_code_execution`` only sees benchmark IDs; ``lm-eval/custom`` carries its
    tasks in the ``task`` parameter, and lm-eval still demands the confirmation.
    """
    for name in (t.strip() for t in tasks.split(",")):
        spec = TASKS.get(name)
        if spec and spec.unsafe_code:
            return True
    return False


def primary_row(benchmark_id: str, tasks: str) -> str | None:
    """Name of the lm-eval results row that carries the benchmark's own score.

    For a single-task benchmark this is the task itself, except for a tag
    (``winogender``), which has no row of its own and whose ``<tag>_all`` leaf is
    the overall one. A multi-task suite has no single primary row.
    """
    spec = _spec(benchmark_id)
    if spec is not None:
        return spec.primary_row
    return tasks if "," not in tasks else None


def resolve_tasks(
    benchmark_id: str,
    task_override: str | None,
    allow_code_execution: bool = False,
) -> str:
    """Return lm-eval task name(s) as a comma-separated string.

    Raises ValueError for unknown benchmark IDs, and for code-execution
    benchmarks unless ``allow_code_execution`` is set.
    """
    if is_code_execution(benchmark_id) and not allow_code_execution:
        raise ValueError(
            f"Benchmark '{benchmark_id}' executes model-generated Python code "
            "(sandboxed Python code execution). It is disabled by default; set the "
            "job parameter allow_code_execution=true to run it inside the adapter pod."
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

    task = task_for(benchmark_id)
    if task is not None:
        return task

    unsupported = UNSUPPORTED_BENCHMARKS.get(benchmark_id.removeprefix("lm-eval/"))
    if unsupported:
        raise ValueError(f"Benchmark '{benchmark_id}' is not offered: {unsupported}.")

    curated = sorted(i for i in BENCHMARKS if i.startswith("lm-eval/"))
    suites = sorted(COMPOSITE_SUITES) + ["lm-eval/custom"]
    raise ValueError(
        f"Unknown benchmark '{benchmark_id}'.\n"
        f"Curated benchmarks: {curated}\nSuites and custom: {suites}\n"
        f"Or any of the other {len(BENCHMARKS) - len(curated)} lm-eval task IDs "
        "listed in this adapter's provider.yaml."
    )


def requires_completions_endpoint(benchmark_id: str) -> bool:
    """Return True if this benchmark uses loglikelihood scoring."""
    if benchmark_id in COMPOSITE_SUITES:
        members = (TASKS.get(t) for t in COMPOSITE_SUITES[benchmark_id])
        return any(m is not None and m.scoring == "loglikelihood" for m in members)
    spec = _spec(benchmark_id)
    return bool(spec and spec.scoring == "loglikelihood")


def generation_safe_ids(curated_only: bool = False) -> list[str]:
    """Benchmark IDs that work on a chat-only endpoint, sorted.

    ``curated_only`` keeps the ``lm-eval/*`` IDs (short enough for an error message).
    """
    ids = {i for i in BENCHMARKS if not requires_completions_endpoint(i)}
    ids |= {s for s in COMPOSITE_SUITES if not requires_completions_endpoint(s)}
    ids.add("lm-eval/custom")
    return sorted(i for i in ids if i.startswith("lm-eval/") or not curated_only)


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


def resolve_chat_template_mode(
    api_style: str,
    apply_chat_template: bool,
    tokenizer: str | None,
    tokenizer_backend: str,
) -> tuple[str, bool]:
    """Return ``(tokenizer_backend, tokenized_requests)`` for the model args.

    On a chat endpoint the server renders the template, so strings are sent as
    before. On a completions endpoint lm_eval 0.4.12 only renders the Hugging
    Face template when ``tokenizer_backend=huggingface`` and
    ``tokenized_requests=True``; otherwise it hands the endpoint the chat
    history serialised as JSON. That is not a usable prompt, so reject it here
    instead of running a job whose scores reflect it.
    """
    if not apply_chat_template or api_style != "completions":
        return tokenizer_backend, False

    backend = (tokenizer_backend or "None").strip()
    if not tokenizer:
        raise ValueError(
            "apply_chat_template=true on a completions endpoint needs the "
            "'tokenizer' parameter (a HuggingFace model ID or path) so the chat "
            "template can be rendered locally. Without it lm-eval would send the "
            "chat history as a JSON string. Set 'tokenizer', or use api_style=chat "
            "so the server applies the template."
        )
    if backend.lower() not in ("none", "huggingface"):
        raise ValueError(
            f"apply_chat_template=true on a completions endpoint needs "
            f"tokenizer_backend=huggingface, got {backend!r}: only a Hugging Face "
            "tokenizer can render the chat template."
        )
    return "huggingface", True


def resolve_tokenizer_backend(
    benchmark_id: str, tokenizer: str | None, tokenizer_backend: str
) -> str:
    """Return the ``tokenizer_backend`` to pass to lm-eval.

    lm-eval ignores a ``tokenizer`` when the backend is "None", and its API model
    asserts "Tokenizer is required for loglikelihood tasks" without one, so:

    * a ``tokenizer`` with no backend selects ``huggingface`` (the only backend
      that takes a tokenizer name);
    * a loglikelihood benchmark (including perplexity) with no tokenizer at all is
      rejected here, at job start, instead of failing inside lm-eval after the
      dataset has loaded.
    """
    backend = (tokenizer_backend or "None").strip()
    if backend.lower() == "none":
        if tokenizer:
            return "huggingface"
        if requires_completions_endpoint(benchmark_id):
            raise ValueError(
                f"Benchmark '{benchmark_id}' uses loglikelihood scoring, which needs a "
                "tokenizer to compute context lengths. Set the 'tokenizer' parameter "
                "to a HuggingFace model ID or local path (e.g. 'Qwen/Qwen3-8B'), or set "
                "tokenizer_backend=remote to use the inference server's own tokenizer."
            )
    return backend


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

    generation_safe = generation_safe_ids(curated_only=True)
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
