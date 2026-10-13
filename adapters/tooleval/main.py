#!/usr/bin/env python3
"""ToolEval adapter for eval-hub.

ToolBench-style categories (G1/G2/G3):
  - single-tool  — one tool call
  - multi-tool   — multiple tools; short multi-call plan
  - multi-step   — iterative select → /virtual → observe → next step

MUT traffic uses the EvalHub sidecar (job model.url + api-key:ref).
Judge uses judge_api-key:ref + judge_url when present (same model secret).

Metrics: pass_rate (Solved/Unsure/Unsolved), win_rate (WIN/LOSE vs reference).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import openai
from evalhub.adapter import (
    DefaultCallbacks,
    ErrorInfo,
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
    configure_telemetry,
)
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials
from evalhub.models import MetricSchema, ResultType

logger = logging.getLogger(__name__)

_PROVIDER_ID = "tooleval"
_ADAPTER_VERSION = "0.3.0"
_SIDECAR_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PASS_SCORES = {"solved": 1.0, "unsure": 0.5, "unsolved": 0.0}
_DEBUG_IO_LIMIT = 2048
_debug_io_enabled = False

_BENCHMARKS: dict[str, dict[str, Any]] = {
    "single-tool": {
        "mode": "single_tool",
        "default_max_steps": 1,
        "default_instruction": (
            "Use the available tool to echo back the exact message: hello"
        ),
    },
    "multi-tool": {
        "mode": "multi_tool",
        "default_max_steps": 3,
        "default_instruction": (
            "First echo the message hello with the echo tool, then uppercase "
            "that same message with the uppercase tool. Finish when both are done."
        ),
    },
    "multi-step": {
        "mode": "multi_step",
        "default_max_steps": 5,
        "default_instruction": (
            "Solve this in steps: (1) echo the message hello, (2) uppercase the "
            "message hello, (3) finish with a short confirmation. Use tools one "
            "step at a time and observe each result before the next call."
        ),
    },
}

_DEFAULT_REFERENCE_CALLS = [
    {
        "category": "Tools",
        "tool_name": "echo",
        "api_name": "echo_message",
        "tool_input": {"message": "hello"},
    },
    {
        "category": "Tools",
        "tool_name": "uppercase",
        "api_name": "uppercase_message",
        "tool_input": {"message": "hello"},
    },
]


class ToolEvalAdapter(FrameworkAdapter):
    """FrameworkAdapter for ToolEval against a StableToolBench tool server."""

    def __init__(self, job_spec_path: Optional[str] = None) -> None:
        super().__init__(job_spec_path=job_spec_path)
        # (path, content_bytes, content_type) tuples for callbacks.mlflow.save
        self.mlflow_artifacts: list[tuple[str, bytes, str]] = []

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info(
            "Starting ToolEval job %s benchmark=%s model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing ToolEval adapter",
                        message_code="initializing",
                    ),
                )
            )

            bench = _BENCHMARKS.get(config.benchmark_id or "")
            if bench is None:
                raise ValueError(
                    f"Unsupported benchmark_id: {config.benchmark_id!r}; "
                    f"supported: {sorted(_BENCHMARKS)}"
                )
            mode = str(bench["mode"])

            params = config.parameters or {}
            _configure_debug_io(params)
            tool_server_url = str(params.get("tool_server_url") or "").strip().rstrip("/")
            if not tool_server_url:
                raise ValueError(
                    "tool_server_url is required "
                    "(deploy the ToolBench tool server and pass its ClusterIP URL)"
                )

            model_url = str(config.model.url or "").strip().rstrip("/")
            if not model_url:
                raise ValueError("config.model.url is required (EvalHub sidecar / MUT endpoint)")
            model_name = str(config.model.name or "").strip()
            if not model_name:
                raise ValueError("config.model.name is required")

            num_tasks = max(1, int(_param(params, "num_tasks", 5)))
            seed = int(_param(params, "seed", 42))
            tool_subset = str(params.get("tool_subset") or "default")
            timeout = float(_param(params, "tool_server_timeout_seconds", 30))
            mut_timeout = float(_param(params, "mut_timeout_seconds", 300))
            judge_timeout = float(_param(params, "judge_timeout_seconds", 300))
            toolbench_key = str(params.get("toolbench_key") or "evalhub-local")
            max_tokens = int(_param(params, "max_tokens", 512))
            temperature = float(_param(params, "temperature", 0.0))
            enable_judge = _as_bool(params.get("enable_judge", True))
            probe_models = _as_bool(params.get("probe_models", True))
            max_steps = max(
                1,
                int(_param(params, "max_steps", bench["default_max_steps"])),
            )
            if mode == "single_tool":
                max_steps = 1

            instruction = str(
                params.get("instruction") or bench["default_instruction"]
            ).strip()

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.05,
                    message=MessageInfo(
                        message=f"Checking tool server at {tool_server_url}",
                        message_code="running_evaluation",
                    ),
                )
            )

            mut_client = _build_openai_client(
                base_url=model_url,
                api_key=_resolve_mut_api_key(config),
                timeout=mut_timeout,
            )
            judge_client, judge_model, judge_url = _build_judge_client(
                config=config,
                params=params,
                model_url=model_url,
                model_name=model_name,
                timeout=judge_timeout,
            )

            if probe_models:
                _probe_chat_endpoint(mut_client, model_name, label="MUT")
                if enable_judge:
                    _probe_chat_endpoint(judge_client, judge_model, label="judge")

            mut_calls = 0
            judge_calls = 0
            pass_scores: list[float] = []
            win_scores: list[float] = []
            step_counts: list[int] = []
            succeeded = 0
            failed = 0
            episode_records: list[dict[str, Any]] = []
            self.mlflow_artifacts = []

            with httpx.Client(base_url=tool_server_url, timeout=timeout) as tool_client:
                _require_tool_server_healthy(tool_client)
                available_tools = _list_tools(tool_client)
                if not available_tools:
                    raise ValueError("tool server returned no tools")

                # Optional single-tool filter from params
                category = str(params.get("category") or "").strip()
                tool_name = str(params.get("tool_name") or "").strip()
                api_name = str(params.get("api_name") or "").strip()
                tool_input_template = params.get("tool_input")
                if not isinstance(tool_input_template, dict):
                    tool_input_template = {"message": "hello"}

                if mode == "single_tool":
                    if not tool_name or not api_name:
                        resolved_tool, resolved_api, resolved_cat = _resolve_tool(
                            available_tools,
                            category=category or "Tools",
                            tool_name=tool_name,
                            api_name=api_name,
                        )
                        tool_name, api_name, category = (
                            resolved_tool,
                            resolved_api,
                            resolved_cat,
                        )
                    default_refs = [
                        {
                            "category": category or "Tools",
                            "tool_name": tool_name,
                            "api_name": api_name,
                            "tool_input": dict(tool_input_template),
                        }
                    ]
                    # Job-level fallback; per-task catalogs are built in the task loop.
                    catalog = _single_tool_catalog(
                        available_tools, tool_name=tool_name, category=category or ""
                    )
                else:
                    catalog = available_tools
                    default_refs = list(_DEFAULT_REFERENCE_CALLS)
                    if isinstance(params.get("reference_calls"), list):
                        default_refs = [
                            c for c in params["reference_calls"] if isinstance(c, dict)
                        ] or default_refs

                tasks = _build_tasks(
                    num_tasks=num_tasks,
                    instruction=instruction,
                    default_reference_calls=default_refs,
                    params=params,
                    mode=mode,
                )

                for i, task in enumerate(tasks):
                    progress = 0.1 + (0.75 * (i + 1) / len(tasks))
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.RUNNING_EVALUATION,
                            progress=progress,
                            message=MessageInfo(
                                message=(
                                    f"Running {mode} task {i + 1}/{len(tasks)} "
                                    f"(max_steps={max_steps})"
                                ),
                                message_code="running_evaluation",
                            ),
                        )
                    )

                    try:
                        task_catalog = catalog
                        if mode == "single_tool" and task["reference_calls"]:
                            ref0 = task["reference_calls"][0]
                            task_catalog = _single_tool_catalog(
                                available_tools,
                                tool_name=str(ref0.get("tool_name") or tool_name),
                                category=str(ref0.get("category") or category or ""),
                            ) or catalog
                        episode, mut_delta = _run_agent_episode(
                            mut_client=mut_client,
                            model_name=model_name,
                            tool_client=tool_client,
                            catalog=task_catalog,
                            instruction=task["instruction"],
                            max_steps=max_steps,
                            mode=mode,
                            toolbench_key=toolbench_key,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        mut_calls += mut_delta
                        step_counts.append(len(episode["steps"]))
                        episode_records.append(
                            {
                                "task_index": i,
                                "instruction": task["instruction"],
                                "reference_calls": task["reference_calls"],
                                "episode": episode,
                            }
                        )

                        if enable_judge:
                            pass_label = _judge_pass_trajectory(
                                judge_client,
                                judge_model,
                                instruction=task["instruction"],
                                episode=episode,
                                max_tokens=max_tokens,
                            )
                            judge_calls += 1
                            win_label = _judge_win_trajectory(
                                judge_client,
                                judge_model,
                                instruction=task["instruction"],
                                episode=episode,
                                reference_calls=task["reference_calls"],
                                max_tokens=max_tokens,
                            )
                            judge_calls += 1
                        else:
                            pass_label, win_label = _structural_score(
                                episode, task["reference_calls"], mode=mode
                            )

                        pass_score = _PASS_SCORES.get(pass_label, 0.0)
                        if pass_label == "solved" and not episode.get("any_virtual_ok"):
                            pass_score = 0.0
                            pass_label = "unsolved"
                        win_score = 1.0 if win_label == "win" else 0.0
                        pass_scores.append(pass_score)
                        win_scores.append(win_score)
                        if pass_score >= 1.0:
                            succeeded += 1
                        else:
                            failed += 1
                        logger.info(
                            "task=%s mode=%s pass=%s win=%s steps=%s final=%s",
                            i + 1,
                            mode,
                            pass_label,
                            win_label,
                            len(episode["steps"]),
                            episode.get("final_answer"),
                        )
                    except Exception:
                        logger.exception("ToolEval task %s failed", i + 1)
                        pass_scores.append(0.0)
                        win_scores.append(0.0)
                        step_counts.append(0)
                        failed += 1

            n = len(pass_scores) or 1
            pass_rate = sum(pass_scores) / n
            win_rate = sum(win_scores) / n
            avg_steps = (sum(step_counts) / len(step_counts)) if step_counts else 0.0
            duration = time.time() - start_time

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.9,
                    message=MessageInfo(
                        message="Computing ToolEval metrics",
                        message_code="post_processing",
                    ),
                )
            )

            evaluation_metadata = {
                "framework": "tooleval",
                "tool_server_url": tool_server_url,
                "num_tasks": len(pass_scores),
                "seed": seed,
                "tool_subset": tool_subset,
                "mode": mode,
                "max_steps": max_steps,
                "avg_steps": avg_steps,
                "tasks_succeeded": succeeded,
                "tasks_failed": failed,
                "adapter_version": _ADAPTER_VERSION,
                "provider_id": config.provider_id or _PROVIDER_ID,
                "judge_model": judge_model,
                "judge_url": judge_url,
                "mut_calls": mut_calls,
                "judge_calls": judge_calls,
                "enable_judge": enable_judge,
                "tools_available": len(catalog),
            }
            evaluation_results = [
                EvaluationResult(
                    metric_name="pass_rate",
                    metric_value=round(pass_rate, 6),
                    metric_type="float",
                    num_samples=len(pass_scores),
                ),
                EvaluationResult(
                    metric_name="win_rate",
                    metric_value=round(win_rate, 6),
                    metric_type="float",
                    num_samples=len(win_scores),
                ),
            ]

            # Trajectory artifact for MLflow (ragas/promptfoo pattern)
            summary_payload = {
                "pass_rate": pass_rate,
                "win_rate": win_rate,
                "mode": mode,
                "num_tasks": len(pass_scores),
                "avg_steps": avg_steps,
                "benchmark_id": config.benchmark_id,
                "job_id": config.id,
            }
            trajectories_json = json.dumps(episode_records, indent=2).encode("utf-8")
            summary_json = json.dumps(summary_payload, indent=2).encode("utf-8")
            self.mlflow_artifacts = [
                ("trajectories.json", trajectories_json, "application/json"),
                ("summary.json", summary_json, "application/json"),
            ]

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.95,
                    message=MessageInfo(
                        message="Persisting ToolEval results",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            oci_artifact = None
            if config.exports and getattr(config.exports, "oci", None):
                results_dir = _results_output_dir(self)
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "trajectories.json").write_bytes(trajectories_json)
                (results_dir / "summary.json").write_bytes(summary_json)
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=results_dir,
                        coordinates=config.exports.oci.coordinates,
                    )
                )
                logger.info(
                    "OCI artifact created: %s",
                    getattr(oci_artifact, "reference", oci_artifact),
                )
            else:
                logger.info("No OCI exports configured; skipping artifact persistence")

            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=getattr(config, "benchmark_index", 0) or 0,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=pass_rate,
                num_examples_evaluated=len(pass_scores),
                duration_seconds=round(duration, 2),
                completed_at=datetime.now(tz=UTC),
                evaluation_metadata=evaluation_metadata,
                additional_info={
                    "adapter_version": _ADAPTER_VERSION,
                    "framework": "tooleval",
                    "mode": mode,
                    "max_steps": max_steps,
                    "pass_rate": round(pass_rate, 6),
                    "win_rate": round(win_rate, 6),
                    "judge_model": judge_model,
                    "tool_server_url": tool_server_url,
                },
                metrics_schema=[
                    MetricSchema(name="pass_rate", type=ResultType.NUMERIC),
                    MetricSchema(name="win_rate", type=ResultType.NUMERIC),
                ],
                oci_artifact=oci_artifact,
            )
            return job_results

        except Exception as exc:
            logger.exception("ToolEval evaluation failed")
            error_msg = str(exc)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(message=error_msg, message_code="failed"),
                    error=ErrorInfo(message=error_msg, message_code="evaluation_error"),
                    error_details={
                        "exception_type": type(exc).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )
            raise


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _param(params: dict[str, Any], key: str, default: Any) -> Any:
    """Return params[key], falling back to default when missing or explicitly null."""
    value = params.get(key, default)
    return default if value is None else value


def _configure_debug_io(params: dict[str, Any] | None = None) -> None:
    """Enable truncated MUT /virtual / judge I/O logs (param or TOOLEVAL_DEBUG_IO=1)."""
    global _debug_io_enabled
    env_on = os.getenv("TOOLEVAL_DEBUG_IO", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _debug_io_enabled = env_on or _as_bool((params or {}).get("debug_io"))
    if _debug_io_enabled:
        logger.info(
            "ToolEval debug_io enabled (payloads truncated to %s chars)",
            _DEBUG_IO_LIMIT,
        )


def _debug_preview(value: Any, limit: int = _DEBUG_IO_LIMIT) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, default=str)
    else:
        text = str(value if value is not None else "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _log_debug_io(kind: str, **fields: Any) -> None:
    if not _debug_io_enabled:
        return
    parts = " ".join(f"{key}={_debug_preview(val)}" for key, val in fields.items())
    logger.info("tooleval debug_io %s %s", kind, parts)


def _results_output_dir(adapter: FrameworkAdapter) -> Path:
    """Directory for OCI-exportable result files (ragas/lm-eval pattern)."""
    try:
        base = adapter.local_jobs_base_path
    except Exception:  # noqa: BLE001 — settings may be unavailable in unit tests
        base = None
    if base is not None:
        return Path(base) / "results"
    return Path(__file__).resolve().parent / "results"


def _openai_base_url(url: str) -> str:
    stripped = url.strip().rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def _resolve_mut_api_key(config: JobSpec) -> str:
    if config.model.auth and getattr(config.model.auth, "secret_ref", None):
        try:
            creds = resolve_model_credentials()
            if creds and creds.api_key:
                return creds.api_key
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve_model_credentials failed: %s", exc)
    ref = read_model_auth_key("api-key")
    if ref:
        return ref
    return os.getenv("OPENAI_API_KEY", "DUMMY")


def _build_openai_client(*, base_url: str, api_key: str, timeout: float) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=_openai_base_url(base_url),
        api_key=api_key or "DUMMY",
        timeout=timeout,
    )


def _build_judge_client(
    *,
    config: JobSpec,
    params: dict[str, Any],
    model_url: str,
    model_name: str,
    timeout: float,
) -> tuple[openai.OpenAI, str, str]:
    judge_model = str(params.get("judge_model") or model_name).strip() or model_name
    param_judge_url = str(params.get("judge_url") or "").strip().rstrip("/")
    param_judge_key = params.get("judge_api_key")
    secret_judge_key = read_model_auth_key("judge_api-key")
    secret_judge_url = read_model_auth_key("judge_url")

    if param_judge_url:
        judge_url = param_judge_url
        if param_judge_key:
            api_key = str(param_judge_key)
        elif urlparse(judge_url).hostname in _SIDECAR_HOSTS and secret_judge_key:
            api_key = secret_judge_key
        elif judge_url == model_url:
            api_key = _resolve_mut_api_key(config)
        else:
            api_key = (
                os.getenv("TOOLEVAL_JUDGE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or "DUMMY"
            )
    elif secret_judge_url and secret_judge_key:
        judge_url = secret_judge_url.strip().rstrip("/")
        api_key = secret_judge_key
    else:
        judge_url = model_url
        api_key = str(param_judge_key) if param_judge_key else _resolve_mut_api_key(config)

    return (
        _build_openai_client(base_url=judge_url, api_key=api_key, timeout=timeout),
        judge_model,
        judge_url,
    )


def _call_chat_model(
    client: openai.OpenAI,
    model_name: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    label: str = "chat",
) -> str:
    _log_debug_io(label, model=model_name, request=prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not response.choices:
        _log_debug_io(label, model=model_name, response="")
        return ""
    content = response.choices[0].message.content or ""
    _log_debug_io(label, model=model_name, response=content)
    return content


def _probe_chat_endpoint(client: openai.OpenAI, model_name: str, *, label: str) -> None:
    """Fail-fast connectivity/auth check for MUT or judge."""
    try:
        _call_chat_model(
            client,
            model_name,
            'Reply with exactly: ok',
            max_tokens=8,
            temperature=0.0,
            label=f"{label}_probe",
        )
    except Exception as exc:
        raise ValueError(
            f"{label} endpoint probe failed for model={model_name!r}: {exc}"
        ) from exc


def _run_agent_episode(
    *,
    mut_client: openai.OpenAI,
    model_name: str,
    tool_client: httpx.Client,
    catalog: list[dict[str, Any]],
    instruction: str,
    max_steps: int,
    mode: str,
    toolbench_key: str,
    max_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], int]:
    """Run select→call→observe loop. Returns (episode, mut_call_count)."""
    steps: list[dict[str, Any]] = []
    mut_calls = 0
    any_virtual_ok = False
    final_answer = ""

    for step_idx in range(max_steps):
        prompt = _agent_step_prompt(
            instruction=instruction,
            catalog=catalog,
            history=steps,
            mode=mode,
            step_idx=step_idx,
            max_steps=max_steps,
        )
        raw = _call_chat_model(
            mut_client,
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            label="mut",
        )
        mut_calls += 1
        action = _parse_agent_action(raw, catalog=catalog)

        if action.get("action") == "finish":
            final_answer = str(action.get("final_answer") or "")
            steps.append(
                {
                    "step": step_idx + 1,
                    "action": "finish",
                    "final_answer": final_answer,
                    "raw": raw,
                }
            )
            break

        call = {
            "category": str(action.get("category") or "Tools"),
            "tool_name": str(action.get("tool_name") or ""),
            "api_name": str(action.get("api_name") or ""),
            "tool_input": action.get("tool_input")
            if isinstance(action.get("tool_input"), dict)
            else {},
        }
        virtual_body = _run_virtual_call(
            tool_client,
            category=call["category"],
            tool_name=call["tool_name"],
            api_name=call["api_name"],
            tool_input=call["tool_input"],
            toolbench_key=toolbench_key,
        )
        virtual_ok = (virtual_body.get("error") or "") == ""
        any_virtual_ok = any_virtual_ok or virtual_ok
        steps.append(
            {
                "step": step_idx + 1,
                "action": "call",
                "call": call,
                "tool_response": virtual_body,
                "virtual_ok": virtual_ok,
                "raw": raw,
            }
        )

        # single_tool always stops after one call
        if mode == "single_tool":
            break

    return (
        {
            "steps": steps,
            "final_answer": final_answer,
            "any_virtual_ok": any_virtual_ok,
            "predicted_calls": [
                s["call"] for s in steps if s.get("action") == "call" and "call" in s
            ],
        },
        mut_calls,
    )


def _agent_step_prompt(
    *,
    instruction: str,
    catalog: list[dict[str, Any]],
    history: list[dict[str, Any]],
    mode: str,
    step_idx: int,
    max_steps: int,
) -> str:
    tools_text = json.dumps(catalog, indent=2)
    history_text = json.dumps(history, indent=2) if history else "[]"
    mode_hint = {
        "single_tool": "Make exactly one tool call, then you are done (no finish needed).",
        "multi_tool": (
            "You may call different tools across steps. Prefer calling each needed "
            "tool once. When the request is satisfied, respond with action=finish."
        ),
        "multi_step": (
            "Work step-by-step. After each observation, decide the next call or finish. "
            "Do not skip observing prior tool results."
        ),
    }.get(mode, "")

    return (
        "You are a tool-using assistant evaluating tool selection.\n"
        f"Mode: {mode}. Step {step_idx + 1} of {max_steps}. {mode_hint}\n\n"
        f"Available tools (from GET /tools):\n{tools_text}\n\n"
        f"User request:\n{instruction}\n\n"
        f"History so far (prior calls/observations):\n{history_text}\n\n"
        "Respond with ONLY a JSON object (no markdown) in one of these forms:\n"
        '1) {"action":"call","category":"...","tool_name":"...","api_name":"...","tool_input":{...}}\n'
        '2) {"action":"finish","final_answer":"..."}\n'
    )


def _parse_agent_action(text: str, *, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    data = _extract_json_object(text) or {}
    action = str(data.get("action") or "call").strip().lower()
    if action == "finish":
        return {"action": "finish", "final_answer": str(data.get("final_answer") or "")}

    default_tool = catalog[0] if catalog else {}
    tool_input = data.get("tool_input", data.get("arguments", {}))
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            tool_input = {"raw": tool_input}
    if not isinstance(tool_input, dict):
        tool_input = {}

    api_name = str(data.get("api_name") or "").strip()
    tool_name = str(data.get("tool_name") or default_tool.get("tool_name") or "").strip()
    if not api_name:
        chosen = next(
            (t for t in catalog if str(t.get("tool_name") or "") == tool_name),
            default_tool,
        )
        api_name = _first_api_name(chosen)

    return {
        "action": "call",
        "category": str(data.get("category") or default_tool.get("category") or "Tools"),
        "tool_name": tool_name,
        "api_name": api_name,
        "tool_input": tool_input,
    }


def _judge_pass_trajectory(
    client: openai.OpenAI,
    model: str,
    *,
    instruction: str,
    episode: dict[str, Any],
    max_tokens: int,
) -> str:
    prompt = (
        "You evaluate whether a tool-using agent solved the user request.\n"
        "Reply with exactly one word: Solved, Unsure, or Unsolved.\n\n"
        f"User request:\n{instruction}\n\n"
        f"Agent trajectory JSON:\n{json.dumps(episode, indent=2)}\n"
    )
    return _normalize_pass_label(
        _call_chat_model(
            client, model, prompt, max_tokens=max_tokens, temperature=0.0, label="judge_pass"
        )
    )


def _judge_win_trajectory(
    client: openai.OpenAI,
    model: str,
    *,
    instruction: str,
    episode: dict[str, Any],
    reference_calls: list[dict[str, Any]],
    max_tokens: int,
) -> str:
    prompt = (
        "Compare the agent's tool-call trajectory to the reference tool calls.\n"
        "Reply with exactly one word: WIN or LOSE.\n"
        "WIN if the agent used equivalent correct tools/APIs/arguments to solve "
        "the request (order may vary for multi-tool unless causality requires order).\n\n"
        f"User request:\n{instruction}\n\n"
        f"Reference tool calls JSON:\n{json.dumps(reference_calls, indent=2)}\n\n"
        f"Agent trajectory JSON:\n{json.dumps(episode, indent=2)}\n"
    )
    return _normalize_win_label(
        _call_chat_model(
            client, model, prompt, max_tokens=max_tokens, temperature=0.0, label="judge_win"
        )
    )


def _structural_score(
    episode: dict[str, Any],
    reference_calls: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[str, str]:
    predicted = episode.get("predicted_calls") or []
    if not predicted:
        return "unsolved", "lose"
    if not all(s.get("virtual_ok") for s in episode.get("steps", []) if s.get("action") == "call"):
        return "unsolved", "lose"

    if mode == "single_tool":
        ok = bool(reference_calls) and _calls_match(predicted[0], reference_calls[0])
        return ("solved", "win") if ok else ("unsolved", "lose")

    if mode == "multi_step":
        # Ordered: predicted calls must match reference_calls in the same order and count.
        ok = len(predicted) == len(reference_calls) and all(
            _calls_match(pred, ref)
            for pred, ref in zip(predicted, reference_calls)
        )
        return ("solved", "win") if ok else ("unsolved", "lose")

    # multi_tool: every reference call must appear (order-insensitive set match)
    unmatched = list(reference_calls)
    for pred in predicted:
        for idx, ref in enumerate(unmatched):
            if _calls_match(pred, ref):
                unmatched.pop(idx)
                break
    ok = len(unmatched) == 0
    return ("solved", "win") if ok else ("unsolved", "lose")


def _normalize_pass_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    for label in ("solved", "unsure", "unsolved"):
        if re.search(rf"\b{label}\b", text):
            return label
    return "unsolved"


def _normalize_win_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    if re.search(r"\bwin\b", text):
        return "win"
    return "lose"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _calls_match(predicted: dict[str, Any], reference: dict[str, Any]) -> bool:
    return (
        str(predicted.get("tool_name") or "").strip()
        == str(reference.get("tool_name") or "").strip()
        and str(predicted.get("api_name") or "").strip()
        == str(reference.get("api_name") or "").strip()
        and predicted.get("tool_input") == reference.get("tool_input")
    )


def _build_tasks(
    *,
    num_tasks: int,
    instruction: str,
    default_reference_calls: list[dict[str, Any]],
    params: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    raw_tasks = params.get("tasks")
    if isinstance(raw_tasks, list) and raw_tasks:
        tasks: list[dict[str, Any]] = []
        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            refs = item.get("reference_calls")
            if not isinstance(refs, list) or not refs:
                if mode == "single_tool" and isinstance(item.get("tool_input"), dict):
                    ref = dict(default_reference_calls[0]) if default_reference_calls else {}
                    ref["tool_input"] = dict(item["tool_input"])
                    for key in ("category", "tool_name", "api_name"):
                        if item.get(key):
                            ref[key] = item[key]
                    refs = [ref]
                else:
                    refs = list(default_reference_calls)
            tasks.append(
                {
                    "instruction": str(item.get("instruction") or instruction),
                    "reference_calls": [c for c in refs if isinstance(c, dict)],
                }
            )
        if tasks:
            return tasks[:num_tasks]
    return [
        {
            "instruction": instruction,
            "reference_calls": list(default_reference_calls),
        }
        for _ in range(num_tasks)
    ]


def _require_tool_server_healthy(client: httpx.Client) -> None:
    resp = client.get("/health")
    if resp.status_code != 200:
        raise ValueError(
            f"tool server health check failed: HTTP {resp.status_code} "
            f"from {client.base_url}/health"
        )
    body = resp.json()
    if body.get("status") != "ok":
        raise ValueError(f"tool server unhealthy: {body!r}")


def _list_tools(client: httpx.Client) -> list[dict[str, Any]]:
    resp = client.get("/tools")
    resp.raise_for_status()
    tools = (resp.json() or {}).get("tools") or []
    return [t for t in tools if isinstance(t, dict)]


def _resolve_tool(
    tools: list[dict[str, Any]],
    *,
    category: str,
    tool_name: str,
    api_name: str,
) -> tuple[str, str, str]:
    chosen = None
    for tool in tools:
        if category and tool.get("category") != category:
            continue
        if tool_name and tool.get("tool_name") != tool_name:
            continue
        chosen = tool
        break
    if chosen is None:
        chosen = tools[0]
    resolved_category = str(chosen.get("category") or category or "Tools")
    resolved_tool = str(chosen.get("tool_name") or "").strip()
    if not resolved_tool:
        raise ValueError(f"invalid tool entry: {chosen!r}")
    resolved_api = api_name.strip() if api_name else _first_api_name(chosen)
    return resolved_tool, resolved_api, resolved_category


def _run_virtual_call(
    client: httpx.Client,
    *,
    category: str,
    tool_name: str,
    api_name: str,
    tool_input: dict[str, Any],
    toolbench_key: str,
) -> dict[str, Any]:
    payload = {
        "category": category,
        "tool_name": tool_name,
        "api_name": api_name,
        "tool_input": tool_input,
        "strip": "",
        "toolbench_key": toolbench_key,
    }
    log_payload = {**payload, "toolbench_key": "<redacted>"}
    _log_debug_io("virtual_request", payload=log_payload)
    resp = client.post("/virtual", json=payload)
    if resp.status_code != 200:
        body = {"error": f"http_{resp.status_code}", "response": resp.text[:500]}
        _log_debug_io("virtual_response", status=resp.status_code, body=body)
        return body
    try:
        parsed = resp.json()
    except ValueError:
        body = {"error": "invalid_json", "response": resp.text[:500]}
        _log_debug_io("virtual_response", status=resp.status_code, body=body)
        return body
    body = parsed if isinstance(parsed, dict) else {"error": "invalid_body", "response": parsed}
    _log_debug_io("virtual_response", status=resp.status_code, body=body)
    return body



def _single_tool_catalog(
    tools: list[dict[str, Any]],
    *,
    tool_name: str,
    category: str,
) -> list[dict[str, Any]]:
    """Restrict GET /tools entries to one tool; fall back to the first catalog entry."""
    matched = [
        t
        for t in tools
        if t.get("tool_name") == tool_name
        and (not category or t.get("category") == category)
    ]
    return matched or tools[:1]


def _first_api_name(tool: dict[str, Any] | None) -> str:
    """Prefer the first api_list entry from GET /tools; fixture fallback is echo_message."""
    for entry in (tool or {}).get("api_list") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name:
                return name
    return "echo_message"


def _local_only_run() -> bool:
    return os.getenv("TOOLEVAL_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")


def _callbacks_for_adapter(adapter: ToolEvalAdapter) -> DefaultCallbacks:
    """Match ragas/lighteval: DefaultCallbacks + MLflow backend.

    additional_info is set inline on JobResults in run_benchmark_job (SDK only
    calls generate_additional_info_fn when additional_info is None).
    """
    if _local_only_run():
        return DefaultCallbacks(
            job_id=adapter.job_spec.id,
            provider_id=adapter.job_spec.provider_id,
            benchmark_id=adapter.job_spec.benchmark_id,
            benchmark_index=adapter.job_spec.benchmark_index,
            sidecar_url=None,
            insecure=adapter.settings.evalhub_insecure,
            oci_auth_config_path=adapter.settings.oci_auth_config_path,
            oci_insecure=adapter.settings.oci_insecure,
            mlflow_backend=adapter.settings.mlflow_backend,
            primary_score=adapter.job_spec.primary_score,
        )
    return DefaultCallbacks.from_adapter(adapter)


def main() -> None:
    """Load JobSpec, run ToolEval, save MLflow metrics, report results (ragas pattern)."""
    from evalhub.adapter.mlflow import MlflowArtifact

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_telemetry()

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = ToolEvalAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )
        callbacks = _callbacks_for_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        artifacts = [
            MlflowArtifact(path, content, content_type)
            for path, content, content_type in getattr(adapter, "mlflow_artifacts", [])
        ]
        run_id = callbacks.mlflow.save(results, adapter.job_spec, artifacts=artifacts or None)
        if run_id:
            results.mlflow_run_id = run_id
            logger.info("MLflow run created: %s", run_id)

        # Do NOT report_status(COMPLETED) here — report_results() owns completion.
        callbacks.report_results(results)

        logger.info(
            "Done %s pass_rate=%s win_rate=%s n=%s %.2fs",
            results.id,
            next((r.metric_value for r in results.results if r.metric_name == "pass_rate"), None),
            next((r.metric_value for r in results.results if r.metric_name == "win_rate"), None),
            results.num_examples_evaluated,
            results.duration_seconds,
        )
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("Job spec not found: %s (set EVALHUB_JOB_SPEC_PATH)", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
