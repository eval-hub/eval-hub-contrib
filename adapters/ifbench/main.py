#!/usr/bin/env python3
"""IFBench instruction-following benchmark adapter for eval-hub.

Loads the IFBench OOD test set (allenai/IFBench), calls a model endpoint for each
prompt, and scores responses with programmatic constraint verifiers from the
ifbench package. The primary metric is prompt-level loose accuracy, matching the
IFBench paper reporting convention.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any, Optional

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
from evalhub.adapter.models.cards import CapabilityEvalEntry, EvalCardMetadata, EnvironmentCardMetadata

from _evaluation import (
    InputExample,
    compute_scores,
    evaluate_instruction_following_loose,
    evaluate_instruction_following_strict,
    read_prompt_list,
)

logger = logging.getLogger(__name__)

_BENCHMARK_ID = "ifbench"
_ADAPTER_VERSION = "0.1.0"
_PAPER_URL = "https://arxiv.org/abs/2507.02833"
_REPO_URL = "https://github.com/allenai/IFBench"


def _resolve_api_key(config: JobSpec) -> str:
    if config.model.auth and getattr(config.model.auth, "secret_ref", None):
        try:
            creds = resolve_model_credentials()
            if creds and creds.api_key:
                return creds.api_key
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve_model_credentials failed: %s", exc)

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    return "not-required"


def _call_model(
    client: Any,
    model_name: str,
    prompt_text: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _load_test_data(num_examples: int | None) -> list[InputExample]:
    from ifbench import data_path  # noqa: PLC0415

    examples = read_prompt_list(data_path())
    if num_examples is not None:
        examples = examples[: min(num_examples, len(examples))]
    return examples


def _build_eval_card(accuracy: float, n_evaluated: int, scoring_mode: str) -> EvalCardMetadata:
    return EvalCardMetadata(
        modalities_input=["text"],
        modalities_output=["text"],
        languages_count=1,
        languages=["en"],
        capability_evaluations=[
            CapabilityEvalEntry(
                ability="instruction following",
                benchmark=f"IFBench ({scoring_mode}), n={n_evaluated}",
                metric="accuracy",
                zero_shot=round(accuracy, 4),
                alt_prompting=None,
                alt_prompting_description=None,
            )
        ],
        safety_evaluations=[],
        developer_footnotes=(
            "IFBench (AllenAI, Apache 2.0) evaluates 58 out-of-domain verifiable "
            "constraints with programmatic checkers. Primary score is prompt-level "
            f"{'loose' if scoring_mode == 'loose' else 'strict'} accuracy. "
            f"Reference: {_PAPER_URL}"
        ),
    )


def _build_env_card(model_name: str) -> EnvironmentCardMetadata:
    env = EnvironmentCardMetadata.capture(
        framework_name="ifbench",
        framework_version=_ADAPTER_VERSION,
        extra_packages=["ifbench", "openai", "httpx"],
    )
    env.model_id = model_name
    env.model_provider = "openai-compatible"
    return env


class IFBenchAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter for the IFBench instruction-following benchmark."""

    def __init__(self, job_spec_path: Optional[str] = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        metric = {r.metric_name: r.metric_value for r in results.results}
        return {
            "zero_shot": metric.get("accuracy"),
            "prompting_strategy": "zero-shot user prompt (no system message)",
            "dataset": "allenai/IFBench_test",
            "benchmark_paper": _PAPER_URL,
            "scoring_mode": results.evaluation_metadata.get("scoring_mode", "loose"),
        }

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info(
            "Starting IFBench job %s benchmark=%s model=%s",
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
                    message=MessageInfo(message="Initializing IFBench adapter", message_code="initializing"),
                )
            )

            if config.benchmark_id != _BENCHMARK_ID:
                raise ValueError(f"Unsupported benchmark_id: {config.benchmark_id}")

            scoring_mode = str(config.parameters.get("scoring_mode", "loose")).lower()
            if scoring_mode not in {"loose", "strict"}:
                raise ValueError("parameters.scoring_mode must be 'loose' or 'strict'")

            _param_val = config.parameters.get("num_examples")
            num_examples_param = (
                _param_val if _param_val is not None else getattr(config, "num_examples", None)
            )
            num_examples = int(num_examples_param) if num_examples_param is not None else None
            max_concurrent = int(config.parameters.get("max_concurrent", 4))
            max_tokens = int(config.parameters.get("max_tokens", 2048))
            temperature = float(config.parameters.get("temperature", 0.0))
            request_timeout = int(config.parameters.get("request_timeout", 120))

            model_url = config.model.url
            model_name = config.model.name
            if not model_url:
                raise ValueError("config.model.url is required for IFBench adapter")
            if not model_name:
                raise ValueError("config.model.name is required for IFBench adapter")

            api_key = _resolve_api_key(config)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(message="Loading IFBench test prompts", message_code="loading_data"),
                )
            )

            examples = _load_test_data(num_examples)
            total = len(examples)
            if total == 0:
                raise ValueError("No IFBench examples to evaluate (empty dataset or num_examples=0)")
            logger.info("Loaded %d IFBench examples", total)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.25,
                    message=MessageInfo(
                        message=f"Generating responses for {total} prompts",
                        message_code="running_evaluation",
                    ),
                )
            )

            import openai  # noqa: PLC0415

            client = openai.OpenAI(base_url=model_url, api_key=api_key, timeout=request_timeout)
            prompt_to_response: dict[str, str] = {}
            completed = 0

            def _generate_response(example: InputExample) -> tuple[str, str]:
                try:
                    response = _call_model(
                        client,
                        model_name,
                        example.prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("API call failed for prompt key=%s: %s", example.key, exc)
                    response = ""
                return example.prompt, response

            with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
                futures = [pool.submit(_generate_response, example) for example in examples]
                for future in as_completed(futures):
                    prompt, response = future.result()
                    prompt_to_response[prompt] = response
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        progress = 0.25 + (0.55 * completed / total)
                        callbacks.report_status(
                            JobStatusUpdate(
                                status=JobStatus.RUNNING,
                                phase=JobPhase.RUNNING_EVALUATION,
                                progress=progress,
                                message=MessageInfo(
                                    message=f"Generated {completed}/{total} responses",
                                    message_code="running_evaluation",
                                ),
                            )
                        )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.85,
                    message=MessageInfo(message="Scoring IFBench responses", message_code="post_processing"),
                )
            )

            strict_outputs = [
                evaluate_instruction_following_strict(example, prompt_to_response)
                for example in examples
            ]
            loose_outputs = [
                evaluate_instruction_following_loose(example, prompt_to_response)
                for example in examples
            ]
            strict_scores = compute_scores(strict_outputs)
            loose_scores = compute_scores(loose_outputs)
            scores = loose_scores if scoring_mode == "loose" else strict_scores

            evaluation_results = [
                EvaluationResult(
                    metric_name="accuracy",
                    metric_value=round(scores["accuracy"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="prompt_level_strict",
                    metric_value=round(strict_scores["prompt_level_accuracy"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="prompt_level_loose",
                    metric_value=round(loose_scores["prompt_level_accuracy"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="inst_level_strict",
                    metric_value=round(strict_scores["instruction_level_accuracy"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="inst_level_loose",
                    metric_value=round(loose_scores["instruction_level_accuracy"], 6),
                    metric_type="float",
                ),
                EvaluationResult(
                    metric_name="n_evaluated",
                    metric_value=int(scores["n_evaluated"]),
                    metric_type="int",
                ),
            ]
            overall_score = scores["accuracy"]
            n_evaluated = int(scores["n_evaluated"])

            eval_card = _build_eval_card(overall_score, n_evaluated, scoring_mode)
            env_card = _build_env_card(model_name)
            duration = time.time() - start_time

            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=n_evaluated,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "ifbench",
                    "dataset": "allenai/IFBench_test",
                    "adapter_version": _ADAPTER_VERSION,
                    "scoring_mode": scoring_mode,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "repository": _REPO_URL,
                },
                eval_card=eval_card,
                env_card=env_card,
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.95,
                    message=MessageInfo(
                        message="Finalizing IFBench evaluation artifacts",
                        message_code="persisting_artifacts",
                    ),
                )
            )

            logger.info(
                "Done %s score=%.4f n=%d %.2fs",
                config.id,
                overall_score,
                n_evaluated,
                duration,
            )
            return job_results

        except Exception as exc:
            logger.exception("IFBench evaluation failed")
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


def _local_only_run() -> bool:
    """Skip sidecar only for standalone runs.

    EvalHub local runtime sets EVALHUB_MODE=local *and* a callback_url pointing
    at the sidecar. Treating EVALHUB_MODE=local as no-sidecar leaves jobs pending.
    """
    return os.getenv("IFBENCH_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")


def _callbacks_for_adapter(adapter: FrameworkAdapter) -> JobCallbacks:
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
        )
    return DefaultCallbacks.from_adapter(adapter)


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    configure_telemetry()

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = IFBenchAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = _callbacks_for_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)

        logger.info(
            "Done %s score=%s n=%s %.2fs",
            results.id,
            results.overall_score,
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
