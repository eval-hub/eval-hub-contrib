#!/usr/bin/env python3
"""EvalHub adapter for the FollowBench benchmark."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    configure_telemetry,
)
from evalhub.adapter.auth import resolve_model_credentials

from _evaluation import (
    FollowBenchExample,
    ScoredConstraint,
    compute_metrics,
    load_examples,
    parse_judge_result,
)
from rules import (
    evaluate_example_constraint,
    evaluate_rule_constraint,
    rule_evaluation_format,
)


logger = logging.getLogger(__name__)

_BENCHMARK_ID = "followbench"
_ADAPTER_VERSION = "0.1.0"
_DATA_DIR = Path(__file__).with_name("data")
_FOLLOWBENCH_REVISION = "6278f4c1377b4eafab737267b8b21acd52ea0e52"
_FOLLOWBENCH_REPOSITORY = "https://github.com/YJiangcm/FollowBench"


def _resolve_model_api_key(config: JobSpec) -> str:
    """Resolve the evaluated-model credential."""
    if config.model.auth and getattr(config.model.auth, "secret_ref", None):
        try:
            credentials = resolve_model_credentials()

            if credentials and credentials.api_key:
                return credentials.api_key
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not resolve model credentials: %s", exc)

    return os.getenv("OPENAI_API_KEY", "DUMMY")


def _resolve_judge_api_key(
    parameters: dict[str, Any],
) -> str:
    """Resolve the external-judge credential."""
    configured_key = parameters.get("judge_api_key")

    if configured_key:
        return str(configured_key)

    return os.getenv("FOLLOWBENCH_JUDGE_API_KEY") or os.getenv(
        "OPENAI_API_KEY",
        "DUMMY",
    )


def _call_chat_model(
    client: Any,
    model_name: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if not response.choices:
        return ""

    return response.choices[0].message.content or ""


def _group_examples(
    examples: list[FollowBenchExample],
    num_examples: int | None,
) -> list[list[FollowBenchExample]]:
    """Group records by category and example ID."""
    groups: dict[tuple[str, int], list[FollowBenchExample]] = {}

    for example in examples:
        key = (example.category, example.example_id)
        groups.setdefault(key, []).append(example)

    ordered_groups = list(groups.values())

    for group in ordered_groups:
        group.sort(key=lambda example: example.level)

    if num_examples is not None and num_examples > 0:
        ordered_groups = ordered_groups[:num_examples]

    return ordered_groups


def _build_judge_prompt(
    group: list[FollowBenchExample],
    response: str,
) -> str:
    """Build the FollowBench external-judge prompt."""
    evaluated_examples = [
        example for example in group if example.level > 0 and example.instruction
    ]

    if not evaluated_examples:
        raise ValueError("FollowBench group has no evaluable constraints")

    current = evaluated_examples[-1]
    level = current.level

    evolution = [
        example.instruction
        for example in group
        if example.level <= level and example.instruction
    ]

    sections = [
        "You are evaluating whether a model response follows FollowBench constraints.",
        "",
        "The instructions evolve by adding one constraint at each level.",
        "",
        "Instruction evolution:",
    ]

    for index, instruction in enumerate(evolution):
        sections.extend(
            [
                "",
                f"Level {index}:",
                instruction,
            ]
        )

    sections.extend(
        [
            "",
            f"Model response at level {level}:",
            response,
            "",
            "Evaluation requirements:",
            f"1. Identify the {level} added constraint(s).",
            f"2. Determine whether the response satisfies each added constraint.",
            (
                f"3. Return only a Python-style list with {level} values. "
                "Each value must be YES, NO, PARTIAL, MAYBE, UNKNOWN, or N/A."
            ),
        ]
    )

    return "\n".join(sections)


def _build_model_client(
    config: JobSpec,
    request_timeout: int,
) -> Any:
    model_url = str(config.model.url or "").strip().rstrip("/")
    model_name = str(config.model.name or "").strip()

    if not model_url:
        raise ValueError("config.model.url is required for FollowBench")

    if not model_name:
        raise ValueError("config.model.name is required for FollowBench")

    return openai.OpenAI(
        base_url=model_url,
        api_key=_resolve_model_api_key(config),
        timeout=request_timeout,
    )


def _build_judge_client(
    config: JobSpec,
    parameters: dict[str, Any],
    request_timeout: int,
) -> tuple[Any, str]:
    model_url = str(config.model.url or "").strip().rstrip("/")
    judge_url = str(parameters.get("judge_url") or model_url).strip().rstrip("/")
    judge_model = str(parameters.get("judge_model") or config.model.name).strip()

    if not judge_url:
        raise ValueError("judge_url or config.model.url is required")

    if not judge_model:
        raise ValueError("judge_model or config.model.name is required")

    client = openai.OpenAI(
        base_url=judge_url,
        api_key=_resolve_judge_api_key(parameters),
        timeout=request_timeout,
    )

    return client, judge_model


def _select_num_examples(parameters: dict[str, Any]) -> int | None:
    value = parameters.get("num_examples")

    if value is None:
        return None

    value = int(value)

    if value <= 0:
        raise ValueError("parameters.num_examples must be greater than zero")

    return value


def _score_example(
    example: FollowBenchExample,
    response: str,
    group: list[FollowBenchExample],
    judge_client: Any,
    judge_model: str,
    *,
    max_tokens: int,
    temperature: float,
) -> ScoredConstraint:
    """Score one FollowBench record."""
    if example.category == "example":
        satisfied = evaluate_example_constraint(example.target, response)
        return ScoredConstraint(
            example_id=example.example_id,
            level=example.level,
            hard_satisfied=satisfied,
            soft_satisfied=float(satisfied),
        )

    if example.category == "format" and example.example_id in {22, 30}:
        satisfied = rule_evaluation_format(
            response,
            example.example_id,
            example.level,
        )
        return ScoredConstraint(
            example_id=example.example_id,
            level=example.level,
            hard_satisfied=satisfied,
            soft_satisfied=float(satisfied),
        )

    rule_result = evaluate_rule_constraint(
        source=example.source,
        generation=response,
        target=example.target,
        level=example.level,
        example_id=example.example_id,
    )

    if rule_result is not None:
        return ScoredConstraint(
            example_id=example.example_id,
            level=example.level,
            hard_satisfied=rule_result,
            soft_satisfied=float(rule_result),
        )

    judge_prompt = _build_judge_prompt(group, response)

    try:
        judge_response = _call_chat_model(
            judge_client,
            judge_model,
            judge_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        hard_satisfied, soft_satisfied = parse_judge_result(
            judge_response,
            example.level,
        )
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Invalid FollowBench judge response for example=%s level=%s: %s",
            example.example_id,
            example.level,
            exc,
        )
        hard_satisfied = 0
        soft_satisfied = 0.0

    return ScoredConstraint(
        example_id=example.example_id,
        level=example.level,
        hard_satisfied=bool(hard_satisfied),
        soft_satisfied=soft_satisfied,
    )


class FollowBenchAdapter(FrameworkAdapter):
    """EvalHub FrameworkAdapter for FollowBench."""

    def __init__(self, job_spec_path: str | None = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def generate_additional_info(
        self,
        results: JobResults,
    ) -> dict[str, Any] | None:
        metrics = {
            result.metric_name: result.metric_value
            for result in results.results
        }

        return {
            "hsr": metrics.get("hsr"),
            "ssr": metrics.get("ssr"),
            "csl": metrics.get("csl"),
            "dataset_revision": _FOLLOWBENCH_REVISION,
            "repository": _FOLLOWBENCH_REPOSITORY,
        }

    def run_benchmark_job(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
    ) -> JobResults:
        start_time = time.time()
        parameters = dict(config.parameters or {})

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing FollowBench adapter",
                        message_code="initializing",
                    ),
                )
            )

            if config.benchmark_id != _BENCHMARK_ID:
                raise ValueError(
                    f"Unsupported benchmark_id: {config.benchmark_id}"
                )

            max_tokens = int(parameters.get("max_tokens", 2048))
            temperature = float(parameters.get("temperature", 0.0))
            request_timeout = int(parameters.get("request_timeout", 120))
            num_examples = _select_num_examples(parameters)

            model_name = str(config.model.name or "").strip()
            model_client = _build_model_client(config, request_timeout)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Loading FollowBench data",
                        message_code="loading_data",
                    ),
                )
            )

            examples = load_examples(_DATA_DIR)
            groups = _group_examples(examples, num_examples)

            if not groups:
                raise ValueError("No FollowBench examples were selected")

            judge_client, judge_model = _build_judge_client(
                config,
                parameters,
                request_timeout,
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.25,
                    message=MessageInfo(
                        message="Generating and scoring FollowBench responses",
                        message_code="running_evaluation",
                    ),
                )
            )

            scored_results: list[ScoredConstraint] = []
            evaluated_records = [
                (group, example)
                for group in groups
                for example in group
                if example.level > 0
            ]

            total = len(evaluated_records)

            for index, (group, example) in enumerate(evaluated_records, start=1):
                response = _call_chat_model(
                    model_client,
                    model_name,
                    example.instruction,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                scored_results.append(
                    _score_example(
                        example,
                        response,
                        group,
                        judge_client,
                        judge_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                )

                if index == total or index % 10 == 0:
                    progress = 0.25 + (0.55 * index / total)
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.RUNNING_EVALUATION,
                            progress=progress,
                            message=MessageInfo(
                                message=(
                                    f"Processed {index}/{total} "
                                    "FollowBench records"
                                ),
                                message_code="running_evaluation",
                            ),
                        )
                    )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.85,
                    message=MessageInfo(
                        message="Computing FollowBench metrics",
                        message_code="post_processing",
                    ),
                )
            )

            metrics = compute_metrics(scored_results)

            evaluation_results = [
                EvaluationResult(
                    metric_name="hsr",
                    metric_value=round(metrics["hsr"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="ssr",
                    metric_value=round(metrics["ssr"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="csl",
                    metric_value=round(metrics["csl"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="n_evaluated",
                    metric_value=int(metrics["n_evaluated"]),
                    metric_type="int",
                ),
            ]

            duration = time.time() - start_time

            results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=model_name,
                results=evaluation_results,
                overall_score=metrics["hsr"],
                num_examples_evaluated=int(metrics["n_evaluated"]),
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "followbench",
                    "adapter_version": _ADAPTER_VERSION,
                    "dataset_revision": _FOLLOWBENCH_REVISION,
                    "repository": _FOLLOWBENCH_REPOSITORY,
                    "judge_model": judge_model,
                    "judge_url": parameters.get("judge_url"),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.95,
                    message=MessageInfo(
                        message="Finalizing FollowBench results",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            return results

        except Exception as exc:
            logger.exception("FollowBench evaluation failed")
            error_message = str(exc)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(
                        message=error_message,
                        message_code="failed",
                    ),
                    error=ErrorInfo(
                        message=error_message,
                        message_code="evaluation_error",
                    ),
                    error_details={
                        "exception_type": type(exc).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )

            raise


def main() -> None:
    """Load the job, execute FollowBench, and report results."""
    configure_telemetry()

    job_spec_path = os.environ["EVALHUB_JOB_SPEC_PATH"]
    adapter = FollowBenchAdapter(job_spec_path=job_spec_path)
    callbacks = DefaultCallbacks.from_adapter(adapter)

    with callbacks.tracer.evaluation_run():
        results = adapter.run_benchmark_job(
            adapter.job_spec,
            callbacks,
        )
        callbacks.report_results(results)

    logger.info(
        "FollowBench evaluation completed: score=%s",
        results.overall_score,
    )


if __name__ == "__main__":
    main()