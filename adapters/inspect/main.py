"""Inspect AI framework adapter for eval-hub.

Integrates the UK AISI Inspect AI framework with eval-hub, exposing three
Meridian Labs alignment-auditing tools through the FrameworkAdapter pattern:

- **Petri** (inspect-petri) — auditor/target/judge pipeline; 170+ seeds across
  40 alignment-behavior tag categories; 38 judge dimensions scored 1–10.
- **Bloom** (petri-bloom) — generates evaluation scenarios from high-level behavior
  descriptions; multi-step: bloom init → bloom scenarios → inspect eval.
- **Dish** — research-preview feature for real agent deployment scaffold testing;
  exposed via task_args pass-through until the API stabilizes.

Standard inspect-evals benchmarks (75 total) are also supported.

Execution is mode-routed on benchmark_id prefix:
  inspect/petri-*  →  Petri mode  (inspect_petri/audit, --model-role ×3)
  inspect/bloom-*  →  Bloom mode  (petri_bloom/bloom_audit, multi-step)
  everything else  →  Standard mode  (inspect eval, --model)

Module structure:
  _benchmarks.py  — benchmark catalogs and dimension constants
  _routing.py     — client selection and model spec building
  _bloom.py       — Bloom multi-step scenario generation
  _execution.py   — command building, env setup, subprocess
  _results.py     — log parsing and result extraction
"""

import logging
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
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
)

from _benchmarks import (
    BLOOM_TEMPLATE_MAP,
    PETRI_SEED_MAP,
    STANDARD_TASK_MAP,
)
from _bloom import bloom_prepare
from _execution import build_command, build_env, get_inspect_version, redact_cmd, run_inspect
from _results import compute_overall_score, extract_results, parse_log
from _routing import (
    build_role_spec,
    role_model_spec,
    route_model,
    select_client,
    target_model_spec,
)

logger = logging.getLogger(__name__)


class InspectAdapter(FrameworkAdapter):
    """Inspect AI framework adapter for Petri, Bloom, Dish, and inspect-evals tasks."""

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        mode = self._detect_mode(config.benchmark_id)
        logger.info(f"Starting Inspect job {config.id} | benchmark={config.benchmark_id} | mode={mode}")

        work_dir: Path | None = None

        try:
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
            )

            extra_packages = ["inspect-ai"]
            if mode in ("petri", "bloom"):
                extra_packages.append("inspect-petri")
            if mode == "bloom":
                extra_packages.append("petri-bloom")

            env_card = EnvironmentCardMetadata.capture(
                framework_name="inspect-ai",
                framework_version=get_inspect_version(),
                extra_packages=extra_packages,
            )

            self._validate_config(config, mode)
            work_dir = Path(tempfile.mkdtemp(prefix=f"inspect_{mode}_"))
            log_dir = work_dir / "logs"
            log_dir.mkdir()

            env = build_env(config, mode)

            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
            )

            behavior_dir: Path | None = None
            if mode == "bloom":
                behavior_dir = bloom_prepare(config, work_dir, env, callbacks)

            task_spec = self._resolve_task(config, mode, behavior_dir)
            cmd = build_command(config, mode, task_spec, log_dir, behavior_dir, env)
            logger.info(f"Inspect command: {redact_cmd(cmd)}")

            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION)
            )

            log_file = run_inspect(cmd, env, log_dir)
            eval_log = parse_log(log_file)

            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
            )

            evaluation_results, capability_entries, num_samples = extract_results(eval_log, config.benchmark_id, mode)
            overall_score = compute_overall_score(evaluation_results, mode)
            logger.info(f"Post-processing complete | samples={num_samples} | overall_score={overall_score}")

            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None and log_file.exists():
                callbacks.report_status(
                    JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
                )
                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update({
                    "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                    "io.github.eval-hub.benchmark": config.benchmark_id,
                    "io.github.eval-hub.model": config.model.name,
                    "io.github.eval-hub.job_id": config.id,
                    "io.github.eval-hub.inspect.task": task_spec,
                    "io.github.eval-hub.inspect.mode": mode,
                })
                oci_artifact = callbacks.create_oci_artifact(OCIArtifactSpec(files_path=log_dir, coordinates=coords))
                logger.info(f"OCI artifact created: {oci_artifact.reference}")

            inspect_version = get_inspect_version()
            eval_card = EvalCardMetadata(
                modalities_input=["text"], modalities_output=["text"],
                languages_count=config.parameters.get("languages_count", 1),
                languages=config.parameters.get("languages", ["en"]),
                capability_evaluations=capability_entries,
                developer_footnotes=(
                    f"Inspect AI {inspect_version} | mode={mode} | task={task_spec} | "
                    f"model={config.model.name} | samples={num_samples}."
                ),
            )

            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=num_samples,
                duration_seconds=time.time() - start_time,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "inspect-ai",
                    "framework_version": inspect_version,
                    "mode": mode,
                    "task": task_spec,
                    "inspect_status": eval_log.get("status"),
                    "benchmark_config": config.parameters,
                },
                oci_artifact=oci_artifact,
                eval_card=eval_card,
                env_card=env_card,
            )

            mlflow_run_id = callbacks.mlflow.save(job_results, config)
            if mlflow_run_id:
                job_results.mlflow_run_id = mlflow_run_id
                logger.info(f"MLflow run recorded: {mlflow_run_id}")

            # Do NOT call report_status(COMPLETED) here — report_results() in
            # main() sends the results payload AND the COMPLETED status in one
            # atomic event. Calling COMPLETED early locks the sidecar and causes
            # the subsequent report_results() to get a 409, dropping all metrics.
            return job_results

        except Exception as e:
            logger.exception("Inspect AI evaluation failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(e),
                        message_code="evaluation_error",
                    ),
                )
            )
            raise

        finally:
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.info(f"Cleaned up: {work_dir}")
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup failed for {work_dir}: {cleanup_err}")

    # ------------------------------------------------------------------
    # Mode detection, validation, task resolution
    # ------------------------------------------------------------------

    def _detect_mode(self, benchmark_id: str) -> str:
        if benchmark_id.startswith("inspect/petri-"):
            return "petri"
        if benchmark_id.startswith("inspect/bloom-"):
            return "bloom"
        return "standard"

    def _validate_config(self, config: JobSpec, mode: str) -> None:
        if mode == "petri" and config.benchmark_id not in PETRI_SEED_MAP:
            raise ValueError(f"Unknown Petri benchmark '{config.benchmark_id}'. Known: {sorted(PETRI_SEED_MAP)}")

        if mode == "bloom":
            if config.benchmark_id not in BLOOM_TEMPLATE_MAP:
                raise ValueError(f"Unknown Bloom benchmark '{config.benchmark_id}'. Known: {sorted(BLOOM_TEMPLATE_MAP)}")
            if (BLOOM_TEMPLATE_MAP[config.benchmark_id] is None
                    and not config.parameters.get("behavior_dir")
                    and not config.parameters.get("bloom_template")):
                raise ValueError("inspect/bloom-custom requires either 'behavior_dir' or 'bloom_template' in parameters.")

        if mode == "standard" and config.benchmark_id == "inspect/custom":
            if not config.parameters.get("task"):
                raise ValueError("inspect/custom requires a 'task' parameter.")

        if mode in ("petri", "bloom") and not config.parameters.get("max_samples"):
            logger.warning("Running Petri/Bloom without max_samples — this may be expensive.")

        task_args: dict[str, Any] = config.parameters.get("task_args", {})
        if any("dish" in k.lower() for k in task_args):
            logger.warning("Dish is a research-preview feature. Behaviour may change without notice.")

    def _resolve_task(self, config: JobSpec, mode: str, behavior_dir: Path | None) -> str:
        explicit = config.parameters.get("task")
        if explicit:
            return explicit
        if mode == "petri":
            return "inspect_petri/audit"
        if mode == "bloom":
            return "petri_bloom/bloom_audit"
        task = STANDARD_TASK_MAP.get(config.benchmark_id)
        if task is None:
            raise ValueError(
                f"Benchmark '{config.benchmark_id}' has no default task mapping. "
                f"Set 'task' in parameters, or use one of: {[k for k, v in STANDARD_TASK_MAP.items() if v]}"
            )
        return task

    def _loading_message(self, config: JobSpec, mode: str) -> str:
        if mode == "petri":
            return f"Preparing Petri audit | seed={PETRI_SEED_MAP.get(config.benchmark_id, '(all seeds)')}"
        if mode == "bloom":
            return f"Preparing Bloom audit | template={BLOOM_TEMPLATE_MAP.get(config.benchmark_id, 'custom')}"
        return f"Preparing Inspect task for {config.benchmark_id}"

    # ------------------------------------------------------------------
    # Thin delegators kept for test and API compatibility
    # ------------------------------------------------------------------

    def _build_env(self, config: JobSpec, mode: str) -> dict[str, str]:
        return build_env(config, mode)

    def _build_command(self, config, mode, task_spec, log_dir, behavior_dir, env=None):
        return build_command(config, mode, task_spec, log_dir, behavior_dir, env or {})

    def _run_inspect(self, cmd, env, log_dir):
        return run_inspect(cmd, env, log_dir)

    def _parse_log(self, log_file):
        return parse_log(log_file)

    def _extract_results(self, eval_log, benchmark_id, mode):
        return extract_results(eval_log, benchmark_id, mode)

    def _compute_overall_score(self, results, mode):
        return compute_overall_score(results, mode)

    def _get_inspect_version(self):
        return get_inspect_version()

    def _select_client(self, env, endpoint_url=None):
        return select_client(env, endpoint_url)

    def _route_model(self, model_name, client):
        return route_model(model_name, client)

    def _build_role_spec(self, model_name, global_env, role_base_url=None, role_api_key=None,
                         role_anthropic_base_url=None, role_anthropic_api_key=None):
        return build_role_spec(model_name, global_env, role_base_url, role_api_key,
                               role_anthropic_base_url, role_anthropic_api_key)

    def _target_model_string(self, config, env):
        return target_model_spec(config.model.name, config.model.url, config.parameters, env)

    def _role_model_string(self, model_name, env, role, params):
        return role_model_spec(model_name, role, params, env)


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
        adapter = InspectAdapter(job_spec_path=job_spec_path)
        logger.info(f"Job: {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        logger.info(f"Completed: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Samples evaluated: {results.num_examples_evaluated}")

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
