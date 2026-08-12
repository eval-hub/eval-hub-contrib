"""Standalone lm-evaluation-harness adapter for eval-hub.

Runs LM Evaluation Harness benchmarks against any OpenAI-compatible inference
endpoint — vLLM, Ollama, OpenRouter, and others — without loading model weights
locally and without any dependency on TrustyAI LMEvalJob or Kubernetes CRDs.

Scoring modes
-------------
Loglikelihood (multiple-choice tasks: MMLU, HellaSwag, ARC, etc.)
  Requires a completions endpoint that supports echo=True and logprobs.
  vLLM /v1/completions and Ollama in OpenAI-compat mode both qualify.
  Chat-only endpoints (OpenRouter, Groq) cannot serve these requests.
  The adapter raises a clear ValueError at startup if this mismatch is detected.

Generate-until (generation tasks: GSM8K, IFEval, TriviaQA, etc.)
  Works with any endpoint, including chat-only.

Dataset sources
---------------
hub   (default) — download from HuggingFace Hub; HF cache dir is configurable.
local           — data_dir points to a mounted filesystem path.
s3              — data_files uses s3:// URIs; credentials via parameters or env.

Custom tasks
------------
Set benchmark_id to 'lm-eval/custom', provide 'task' (lm-eval task name(s)),
and optionally 'custom_tasks_path' (path to a directory of task YAML files
mounted into the container) for user-authored benchmarks.
"""

import logging
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from evalhub.adapter import (
    CapabilityEvalEntry,
    EnvironmentCardMetadata,
    EvalCardMetadata,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
    resolve_model_credentials,
)

from _execution import (
    build_cmd,
    build_env,
    build_model_args,
    get_lmeval_version,
    run_lmeval,
)
from _results import (
    compute_overall_score,
    extract_evaluation_results,
    find_results_file,
    parse_results_file,
)
from _tasks import (
    build_endpoint_url,
    detect_api_style,
    lmeval_model_type,
    preflight_check,
    resolve_tasks,
)

logger = logging.getLogger(__name__)

# Maps benchmark_id → EvalCard ability category (CapabilityEvalEntry.ability).
_ABILITY_MAP: dict[str, str] = {
    "lm-eval/mmlu": "knowledge",
    "lm-eval/mmlu_pro": "knowledge",
    "lm-eval/arc_easy": "reasoning",
    "lm-eval/arc_challenge": "reasoning",
    "lm-eval/hellaswag": "reasoning",
    "lm-eval/winogrande": "reasoning",
    "lm-eval/truthfulqa_mc2": "knowledge",
    "lm-eval/boolq": "reasoning",
    "lm-eval/piqa": "reasoning",
    "lm-eval/openbookqa": "knowledge",
    "lm-eval/bbh": "reasoning",
    "lm-eval/gpqa": "knowledge",
    "lm-eval/musr": "reasoning",
    "lm-eval/gsm8k": "math",
    "lm-eval/math_500": "math",
    "lm-eval/ifeval": "instruction_following",
    "lm-eval/triviaqa": "knowledge",
    "lm-eval/nq_open": "knowledge",
    "lm-eval/open-llm-leaderboard-v2": "composite",
    "lm-eval/generation-suite": "composite",
}


def _benchmark_ability(benchmark_id: str) -> str:
    return _ABILITY_MAP.get(benchmark_id, "custom")


def _primary_metric_name(results: list) -> str:
    """Derive the primary metric label from the first result for EvalCard."""
    for r in results:
        parts = r.metric_name.rsplit(".", 1)
        if len(parts) == 2:
            return parts[1]
    return "acc"


class LMEvalAdapter(FrameworkAdapter):
    """lm-evaluation-harness adapter for OpenAI-compatible inference endpoints."""

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        params = config.parameters or {}
        logger.info(
            "Starting lm-eval job %s | benchmark=%s | model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )

        work_dir: Path | None = None
        try:
            # ── Phase 1: Initialise ──────────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
            )

            tasks = resolve_tasks(config.benchmark_id, params.get("task"))
            # Resolve model_url before api_style detection so URL-based auto-detection
            # uses the final resolved value (config.model.url may be supplemented by
            # a params["base_url"] fallback when the job spec omits model.url).
            model_url = config.model.url or params.get("base_url", "")
            if not model_url:
                raise ValueError(
                    "'model.url' is required. Provide the base URL of the "
                    "OpenAI-compatible inference endpoint (e.g. http://vllm:8000)."
                )
            api_style = detect_api_style(params, model_url)
            preflight_check(config.benchmark_id, api_style)

            tokenizer = params.get("tokenizer")
            tokenizer_backend = str(params.get("tokenizer_backend", "None"))
            if not tokenizer and tokenizer_backend in ("None", "none"):
                logger.warning(
                    "No tokenizer configured — context length enforcement is approximate. "
                    "Set the 'tokenizer' parameter to a HuggingFace model ID (e.g. "
                    "'meta-llama/Meta-Llama-3-8B') for accurate truncation."
                )

            env_card = EnvironmentCardMetadata.capture(
                framework_name="lm-evaluation-harness",
                framework_version=get_lmeval_version(),
                extra_packages=["lm_eval"],
            )

            work_dir = Path(tempfile.mkdtemp(prefix="lmeval_"))
            output_dir = work_dir / "output"
            output_dir.mkdir()

            creds = resolve_model_credentials()
            api_key = params.get("api_key") or (creds.api_key if creds else "") or ""

            endpoint_url = build_endpoint_url(model_url, api_style)
            model_args = build_model_args(
                base_url=endpoint_url,
                model_name=config.model.name,
                num_concurrent=int(params.get("num_concurrent", 4)),
                max_retries=int(params.get("max_retries", 3)),
                timeout_http=int(params.get("timeout_http", 300)),
                max_length=int(params.get("max_context_length", 4096)),
                max_gen_toks=int(params.get("max_gen_tokens", 512)),
                tokenizer=tokenizer,
                tokenizer_backend=tokenizer_backend,
            )
            logger.info(
                "model_type=%s endpoint=%s model=%s",
                lmeval_model_type(api_style),
                endpoint_url,
                config.model.name,
            )

            env = build_env(
                api_key=api_key,
                hf_datasets_cache=params.get("hf_datasets_cache"),
                dataset_source=params.get("dataset_source", "hub"),
                s3_endpoint=params.get("s3_endpoint"),
                aws_access_key_id=params.get("aws_access_key_id"),
                aws_secret_access_key=params.get("aws_secret_access_key"),
            )

            cmd = build_cmd(
                model_type=lmeval_model_type(api_style),
                model_args=model_args,
                tasks=tasks,
                output_dir=output_dir,
                batch_size=int(params.get("batch_size", 1)),
                num_fewshot=params.get("num_fewshot"),
                limit=config.num_examples,
                gen_kwargs=params.get("gen_kwargs"),
                include_path=params.get("custom_tasks_path"),
                apply_chat_template=bool(params.get("apply_chat_template", False)),
                system_instruction=params.get("system_instruction"),
            )
            # Command is already logged at INFO level inside run_lmeval().

            # ── Phase 2: Load data (lm_eval handles internally) ──────────────
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
            )

            # ── Phase 3: Run evaluation ──────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION
                )
            )
            self._run_lmeval(cmd, env, int(params.get("subprocess_timeout", 7200)))

            # ── Phase 4: Post-process ────────────────────────────────────────
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING
                )
            )
            raw = self._parse_results(output_dir)
            evaluation_results = extract_evaluation_results(raw)
            overall_score = compute_overall_score(evaluation_results)
            num_samples = self._total_samples(evaluation_results)
            lmeval_version = get_lmeval_version()

            logger.info(
                "Post-processing complete | score=%.4f | metrics=%d | samples=%s",
                overall_score or 0.0,
                len(evaluation_results),
                num_samples,
            )

            # ── Phase 5: Persist OCI artifact (optional) ─────────────────────
            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS
                    )
                )
                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update(
                    {
                        "org.opencontainers.image.created": datetime.now(
                            UTC
                        ).isoformat(),
                        "io.github.eval-hub.benchmark": config.benchmark_id,
                        "io.github.eval-hub.model": config.model.name,
                        "io.github.eval-hub.job_id": config.id,
                        "io.github.eval-hub.lmeval.tasks": tasks,
                        "io.github.eval-hub.lmeval.api_style": api_style,
                        "io.github.eval-hub.lmeval.version": lmeval_version,
                    }
                )
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(files_path=output_dir, coordinates=coords)
                )
                logger.info("OCI artifact: %s", oci_artifact.reference)

            num_fewshot = params.get("num_fewshot")
            _ability = _benchmark_ability(config.benchmark_id)
            _primary_metric = _primary_metric_name(evaluation_results)
            if num_fewshot and int(num_fewshot) > 0:
                cap_entry = CapabilityEvalEntry(
                    ability=_ability,
                    benchmark=config.benchmark_id,
                    metric=_primary_metric,
                    alt_prompting=overall_score,
                    alt_prompting_description=f"{num_fewshot}-Shot",
                )
            else:
                cap_entry = CapabilityEvalEntry(
                    ability=_ability,
                    benchmark=config.benchmark_id,
                    metric=_primary_metric,
                    zero_shot=overall_score,
                )

            eval_card = EvalCardMetadata(
                modalities_input=["text"],
                modalities_output=["text"],
                languages_count=int(params.get("languages_count", 1)),
                languages=params.get("languages", ["en"]),
                capability_evaluations=[cap_entry],
                developer_footnotes=(
                    f"lm-evaluation-harness {lmeval_version} | "
                    f"api_style={api_style} | tasks={tasks} | "
                    f"model={config.model.name}"
                ),
            )

            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=num_samples or 0,
                duration_seconds=time.time() - start_time,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "lm-evaluation-harness",
                    "framework_version": lmeval_version,
                    "tasks": tasks,
                    "api_style": api_style,
                    "endpoint_url": endpoint_url,
                    "num_fewshot": params.get("num_fewshot"),
                    "gen_kwargs": params.get("gen_kwargs"),
                    "tokenizer": tokenizer,
                    "benchmark_config": params,
                },
                oci_artifact=oci_artifact,
                eval_card=eval_card,
                env_card=env_card,
            )

            # Do NOT call report_status(COMPLETED) here — report_results() sends
            # results + COMPLETED atomically. Calling COMPLETED early causes a 409
            # that drops all metrics (same sidecar behaviour as inspect adapter).
            experiment_name = (
                getattr(config, "experiment_name", None) or ""
            ).strip() or params.get("mlflow_experiment_name", "")
            if experiment_name:
                try:
                    mlflow_run_id = callbacks.mlflow.save(job_results, config)  # type: ignore[attr-defined]
                    if mlflow_run_id:
                        job_results.mlflow_run_id = mlflow_run_id
                        logger.info("MLflow run: %s", mlflow_run_id)
                except Exception:
                    logger.warning("MLflow save failed", exc_info=True)

            return job_results

        except Exception as exc:
            logger.exception("lm-eval job failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(exc),
                        message_code="evaluation_error",
                    ),
                )
            )
            raise

        finally:
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.debug("Cleaned up: %s", work_dir)
                except Exception as cleanup_err:  # noqa: BLE001
                    logger.warning("Cleanup failed for %s: %s", work_dir, cleanup_err)

    # ── Delegators — thin wrappers for test monkeypatching ───────────────────

    def _run_lmeval(self, cmd: list[str], env: dict[str, str], timeout: int) -> None:
        run_lmeval(cmd, env, timeout)

    def _parse_results(self, output_dir: Path) -> dict:
        return parse_results_file(find_results_file(output_dir))

    @staticmethod
    def _total_samples(results: list) -> int | None:
        # Deduplicate by task name: each task carries the same num_samples on
        # every metric row, so summing all rows would multiply the count by the
        # number of metrics per task (e.g. HellaSwag has acc + acc_norm = 2×).
        seen: dict[str, int] = {}
        for r in results:
            task = r.metric_name.rsplit(".", 1)[0]
            if task not in seen and r.num_samples:
                seen[task] = r.num_samples
        total = sum(seen.values())
        return total if total else None


def main() -> None:
    """Container entry point."""
    import sys

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        from evalhub.adapter import DefaultCallbacks

        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = LMEvalAdapter(job_spec_path=job_spec_path)
        logger.info("Job:       %s", adapter.job_spec.id)
        logger.info("Benchmark: %s", adapter.job_spec.benchmark_id)
        logger.info("Model:     %s", adapter.job_spec.model.name)

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        logger.info(
            "Completed: %s | score=%s | samples=%s",
            results.id,
            results.overall_score,
            results.num_examples_evaluated,
        )
        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as exc:
        logger.error("Job spec not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
