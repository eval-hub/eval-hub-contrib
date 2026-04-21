"""AgentDojo framework adapter for eval-hub.

This adapter integrates AgentDojo (https://github.com/ethz-spylab/agentdojo)
with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

AgentDojo is a dynamic environment to evaluate prompt injection attacks and defenses
for LLM agents. It benchmarks LLM agents against simulated tool-use environments
(workspace, travel, banking, slack) and measures both utility (task completion) and
security (resistance to prompt injection attacks).

The adapter:
1. Reads JobSpec from a mounted ConfigMap
2. Executes AgentDojo benchmark evaluations via subprocess
3. Reports progress via callbacks to the sidecar
4. Parses per-task JSON results from the logdir
5. Persists results as OCI artifacts
6. Returns structured JobResults with utility and security metrics
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
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
)

logger = logging.getLogger(__name__)

# AgentDojo benchmark suites
SUPPORTED_SUITES = ["workspace", "travel", "banking", "slack"]

# Available attacks in AgentDojo
SUPPORTED_ATTACKS = [
    "direct",
    "ignore_previous",
    "system_prompt",
    "tool_knowledge",
    "important_instructions",
    "important_instructions_no_user_name",
    "injecagent",
    "dos",
]

# Available defenses in AgentDojo
SUPPORTED_DEFENSES = [
    "tool_filter",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
]


class AgentDojoAdapter(FrameworkAdapter):
    """AgentDojo framework adapter.

    This adapter executes AgentDojo prompt injection benchmarks and integrates
    with the eval-hub service using the callback-based architecture. It evaluates
    LLM agents on both utility (task completion) and security (injection resistance)
    across multiple simulated tool-use environments.
    """

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute an AgentDojo benchmark evaluation job.

        Args:
            config: Job specification from mounted ConfigMap
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If AgentDojo evaluation fails
        """
        start_time = time.time()
        logger.info(
            f"Starting AgentDojo job {config.id} for benchmark {config.benchmark_id}"
        )

        logdir: Path | None = None

        try:
            # Phase 1: Initialize
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message=f"Initializing AgentDojo for benchmark {config.benchmark_id}",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)
            suites = self._resolve_suites(config.benchmark_id, config.parameters)
            attack = config.parameters.get("attack")
            defense = config.parameters.get("defense")
            benchmark_version = config.parameters.get("benchmark_version", "v1.2.2")
            logdir = Path(tempfile.mkdtemp(prefix="agentdojo_"))

            logger.info(
                f"Configuration validated. Suites: {suites}, Attack: {attack}, "
                f"Defense: {defense}, Version: {benchmark_version}, Logdir: {logdir}"
            )

            # Phase 2: Loading data (TODO: this reports but doesnt do anything)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.2,
                    message=MessageInfo(
                        message=f"Preparing AgentDojo evaluation for {len(suites)} suite(s)",
                        message_code="loading_data",
                    ),
                    current_step="Preparing evaluation",
                    total_steps=4,
                    completed_steps=1,
                )
            )

            # Phase 3: Run evaluation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running AgentDojo benchmark on {len(suites)} suite(s)",
                        message_code="running_evaluation",
                    ),
                    current_step="Executing benchmark",
                    total_steps=4,
                    completed_steps=2,
                )
            )

            provider_type = config.parameters.get(
                "provider_type", "openai-compatible"
            )

            self._run_agentdojo(
                provider_type=provider_type,
                model_name=config.model.name,
                model_url=config.model.url,
                suites=suites,
                attack=attack,
                defense=defense,
                benchmark_version=benchmark_version,
                logdir=logdir,
                parameters=config.parameters,
            )

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing AgentDojo results",
                        message_code="post_processing",
                    ),
                    current_step="Extracting metrics",
                    total_steps=4,
                    completed_steps=3,
                )
            )

            task_results = self._parse_results(logdir)
            evaluation_results = self._extract_evaluation_results(
                task_results, suites, attack
            )
            overall_score = self._compute_overall_score(evaluation_results)
            num_evaluated = len(task_results)

            # Save detailed results
            output_files = self._save_detailed_results(
                job_id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                task_results=task_results,
                evaluation_results=evaluation_results,
                logdir=logdir,
            )

            logger.info(
                f"Post-processing complete. Overall score: {overall_score}, "
                f"Evaluated: {num_evaluated} tasks, Files: {len(output_files)}"
            )

            # Phase 5: Persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting AgentDojo artifacts to OCI registry",
                        message_code="persisting_artifacts",
                    ),
                    current_step="Creating OCI artifact",
                    total_steps=4,
                    completed_steps=4,
                )
            )

            oci_artifact = None
            exports = getattr(config, "exports", None)
            oci_exports = getattr(exports, "oci", None) if exports else None
            if output_files and oci_exports:
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=output_files[0].parent,
                        coordinates=oci_exports.coordinates,
                    )
                )
                logger.info(f"OCI artifact persisted: {oci_artifact.digest}")
            elif output_files:
                logger.info(
                    "Skipping OCI artifact persistence (no exports.oci configured). "
                    f"Results saved locally at {output_files[0].parent}"
                )

            duration = time.time() - start_time

            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=num_evaluated,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "agentdojo",
                    "framework_version": self._get_agentdojo_version(),
                    "benchmark_version": benchmark_version,
                    "suites": suites,
                    "attack": attack,
                    "defense": defense,
                    "benchmark_config": config.parameters,
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("AgentDojo evaluation failed")
            error_msg = str(e)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(
                        message=error_msg,
                        message_code="failed",
                    ),
                    error=ErrorInfo(
                        message=error_msg,
                        message_code="evaluation_error",
                    ),
                    error_details={
                        "exception_type": type(e).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )
            raise

        finally:
            if logdir and logdir.exists():
                try:
                    shutil.rmtree(logdir)
                    logger.info(f"Cleaned up temporary directory: {logdir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {logdir}: {e}")

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration for AgentDojo.

        Args:
            config: Job specification to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.name:
            raise ValueError("model.name is required (e.g., 'gpt-4o-2024-05-13')")

        attack = config.parameters.get("attack")
        if attack and attack not in SUPPORTED_ATTACKS:
            logger.warning(
                f"Unknown attack '{attack}'. Known attacks: {SUPPORTED_ATTACKS}. "
                f"It may be a custom attack registered via --module-to-load."
            )

        defense = config.parameters.get("defense")
        if defense and defense not in SUPPORTED_DEFENSES:
            raise ValueError(
                f"Unknown defense '{defense}'. Valid defenses: {SUPPORTED_DEFENSES}"
            )

        logger.debug("Configuration validated successfully")

    def _resolve_suites(
        self, benchmark_id: str, parameters: dict[str, Any]
    ) -> list[str]:
        """Determine which suites to run based on benchmark_id and parameters.

        Args:
            benchmark_id: Benchmark identifier — can be a suite name, "agentdojo",
                or a category
            parameters: Additional parameters with optional 'suites' key

        Returns:
            List of suite names to evaluate
        """
        # Explicit suites in parameters take priority
        if "suites" in parameters and parameters["suites"]:
            suites = parameters["suites"]
            if isinstance(suites, str):
                suites = [suites]
            for s in suites:
                if s not in SUPPORTED_SUITES:
                    raise ValueError(
                        f"Unknown suite '{s}'. Valid suites: {SUPPORTED_SUITES}"
                    )
            return suites

        # benchmark_id is a specific suite
        if benchmark_id in SUPPORTED_SUITES:
            return [benchmark_id]

        # "agentdojo" or any other value runs all suites
        return list(SUPPORTED_SUITES)

    def _run_agentdojo(
        self,
        provider_type: str,
        model_name: str,
        model_url: str,
        suites: list[str],
        attack: str | None,
        defense: str | None,
        benchmark_version: str,
        logdir: Path,
        parameters: dict[str, Any],
    ) -> None:
        """Execute AgentDojo benchmark CLI.

        Args:
            provider_type: AgentDojo provider type (e.g., 'openai-compatible')
            model_name: Model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
            model_url: Optional model endpoint URL (for local/vllm models)
            suites: List of suite names to run
            attack: Attack name or None
            defense: Defense name or None
            benchmark_version: Benchmark version string
            logdir: Directory for result output
            parameters: Additional benchmark parameters

        Raises:
            RuntimeError: If AgentDojo CLI fails
        """
        logger.info(
            f"Running AgentDojo: provider={provider_type}, model={model_name}, "
            f"suites={suites}, attack={attack}, defense={defense}, "
            f"version={benchmark_version}"
        )

        # Click 8.2+ matches enum choices by name, not value.
        # Convert provider value (e.g. "openai-compatible") to enum name ("OPENAI_COMPATIBLE").
        from agentdojo.models import ModelsEnum

        try:
            cli_provider = ModelsEnum(provider_type).name
        except ValueError:
            cli_provider = provider_type

        cmd = [
            sys.executable,
            "-m",
            "agentdojo.scripts.benchmark",
            "--model",
            cli_provider,
            "--model-id",
            model_name,
            "--benchmark-version",
            benchmark_version,
            "--logdir",
            str(logdir),
            "--force-rerun",
        ]

        for suite in suites:
            cmd.extend(["--suite", suite])

        if attack:
            cmd.extend(["--attack", attack])

        if defense:
            cmd.extend(["--defense", defense])

        # Pass through user tasks if specified
        user_tasks = parameters.get("user_tasks", [])
        if isinstance(user_tasks, str):
            user_tasks = [user_tasks]
        for ut in user_tasks:
            cmd.extend(["--user-task", ut])

        # Pass through injection tasks if specified
        injection_tasks = parameters.get("injection_tasks", [])
        if isinstance(injection_tasks, str):
            injection_tasks = [injection_tasks]
        for it in injection_tasks:
            cmd.extend(["--injection-task", it])

        # System message overrides
        system_message = parameters.get("system_message")
        if system_message:
            cmd.extend(["--system-message", system_message])
        else:
            system_message_name = parameters.get("system_message_name")
            if system_message_name:
                cmd.extend(["--system-message-name", system_message_name])

        # Custom modules to load
        modules_to_load = parameters.get("modules_to_load", [])
        if isinstance(modules_to_load, str):
            modules_to_load = [modules_to_load]
        for module in modules_to_load:
            cmd.extend(["--module-to-load", module])

        from evalhub.adapter.auth import read_model_auth_key

        env = os.environ.copy()

        if provider_type in ("local", "vllm_parsed") and model_url:
            port = str(model_url).rstrip("/").split(":")[-1].split("/")[0]
            env["LOCAL_LLM_PORT"] = port
        elif provider_type == "openai-compatible" and model_url:
            env["OPENAI_COMPATIBLE_BASE_URL"] = model_url
            api_key = (
                read_model_auth_key("OPENAI_COMPATIBLE_API_KEY")
                or read_model_auth_key("LITELLM_API_KEY")
            )
            if api_key:
                env["OPENAI_COMPATIBLE_API_KEY"] = api_key

        timeout = parameters.get("timeout_seconds", 7200)

        logger.info(f"Executing AgentDojo CLI: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                logger.info(f"[agentdojo] {line}")

            process.wait(timeout=timeout)

            if process.returncode != 0:
                raise RuntimeError(
                    f"AgentDojo CLI failed with exit code {process.returncode}\n"
                    f"Output:\n" + "\n".join(output_lines[-50:])
                )

        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(
                f"AgentDojo evaluation timed out after {timeout} seconds"
            )

    def _parse_results(self, logdir: Path) -> list[dict[str, Any]]:
        """Parse all per-task JSON result files from the AgentDojo logdir.

        AgentDojo stores results at:
        {logdir}/{pipeline_name}/{suite}/{user_task}/{attack_type}/{injection_task}.json

        Each JSON file contains:
        - suite_name, pipeline_name, user_task_id, injection_task_id
        - attack_type, injections
        - messages (full conversation)
        - utility (bool), security (bool)
        - duration (float)

        Args:
            logdir: The directory containing AgentDojo output

        Returns:
            List of parsed task result dictionaries
        """
        result_files = list(logdir.rglob("*.json"))

        if not result_files:
            raise RuntimeError(
                f"No result files found in {logdir}. "
                f"Available files: {list(logdir.rglob('*'))}"
            )

        logger.info(f"Found {len(result_files)} result files in {logdir}")

        task_results = []
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                # Only include files that look like AgentDojo task results
                if "utility" in data and "suite_name" in data:
                    task_results.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed result file {result_file}: {e}")

        logger.info(f"Parsed {len(task_results)} valid task results")
        return task_results

    def _extract_evaluation_results(
        self,
        task_results: list[dict[str, Any]],
        suites: list[str],
        attack: str | None,
    ) -> list[EvaluationResult]:
        """Extract structured evaluation results from AgentDojo task outputs.

        Produces metrics at three levels:
        1. Per-task metrics (utility and security for each task)
        2. Per-suite aggregates (average utility/security per suite)
        3. Overall aggregates (average across all suites)

        Args:
            task_results: List of parsed task result dictionaries
            suites: List of suite names evaluated
            attack: Attack name or None

        Returns:
            List of structured EvaluationResult objects
        """
        evaluation_results = []

        # Aggregate per-suite
        suite_utility: dict[str, list[bool]] = {s: [] for s in suites}
        suite_security: dict[str, list[bool]] = {s: [] for s in suites}

        for task in task_results:
            suite_name = task.get("suite_name", "unknown")
            user_task_id = task.get("user_task_id", "unknown")
            injection_task_id = task.get("injection_task_id")
            utility = task.get("utility", False)
            security = task.get("security", True)
            duration = task.get("duration", 0.0)

            task_label = user_task_id
            if injection_task_id and injection_task_id != "none":
                task_label += f".{injection_task_id}"

            evaluation_results.append(
                EvaluationResult(
                    metric_name=f"{task_label}.utility",
                    metric_value=1.0 if utility else 0.0,
                    metric_type="float",
                    metadata={
                        "suite": suite_name,
                        "user_task": user_task_id,
                        "injection_task": injection_task_id,
                        "duration_seconds": duration,
                    },
                )
            )

            # Per-task security (only when attack is used)
            if attack:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=f"{task_label}.security",
                        metric_value=1.0 if security else 0.0,
                        metric_type="float",
                        metadata={
                            "suite": suite_name,
                            "user_task": user_task_id,
                            "injection_task": injection_task_id,
                            "duration_seconds": duration,
                        },
                    )
                )

            if suite_name in suite_utility:
                suite_utility[suite_name].append(utility)
                if attack:
                    suite_security[suite_name].append(security)

        # Per-suite aggregates
        for suite_name in suites:
            utilities = suite_utility.get(suite_name, [])
            if utilities:
                avg_utility = sum(utilities) / len(utilities)
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="avg_utility",
                        metric_value=round(avg_utility, 4),
                        metric_type="float",
                        num_samples=len(utilities),
                        metadata={"suite": suite_name},
                    )
                )

            if attack:
                securities = suite_security.get(suite_name, [])
                if securities:
                    avg_security = sum(securities) / len(securities)
                    evaluation_results.append(
                        EvaluationResult(
                            metric_name="avg_security",
                            metric_value=round(avg_security, 4),
                            metric_type="float",
                            num_samples=len(securities),
                            metadata={"suite": suite_name},
                        )
                    )

        return evaluation_results

    def _compute_overall_score(self, results: list[EvaluationResult]) -> float | None:
        """Compute overall score from evaluation results.

        Uses the average of per-suite avg_utility values as the primary score.
        If security metrics are present, the overall score is the average of
        utility and security.

        Args:
            results: List of evaluation results

        Returns:
            Overall score (0.0-1.0), or None if no results
        """
        utility_scores = []
        security_scores = []

        for r in results:
            if r.metric_name == "avg_utility":
                if isinstance(r.metric_value, (int, float)):
                    utility_scores.append(float(r.metric_value))
            elif r.metric_name == "avg_security":
                if isinstance(r.metric_value, (int, float)):
                    security_scores.append(float(r.metric_value))

        if not utility_scores:
            return None

        avg_utility = sum(utility_scores) / len(utility_scores)

        if security_scores:
            avg_security = sum(security_scores) / len(security_scores)
            return round((avg_utility + avg_security) / 2, 4)

        return round(avg_utility, 4)

    def _save_detailed_results(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        task_results: list[dict[str, Any]],
        evaluation_results: list[EvaluationResult],
        logdir: Path,
    ) -> list[Path]:
        """Save detailed results to files for OCI artifact.

        Args:
            job_id: Job identifier
            benchmark_id: Benchmark identifier
            model_name: Model name
            task_results: Raw AgentDojo task results
            evaluation_results: Structured evaluation results
            logdir: AgentDojo logdir with raw result files

        Returns:
            List of paths to saved files
        """
        output_dir = Path("/tmp/agentdojo_results") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Save raw task results (without full message logs to reduce size)
        raw_results_file = output_dir / "task_results.json"
        summarized_results = []
        for task in task_results:
            summarized_results.append(
                {
                    "suite_name": task.get("suite_name"),
                    "user_task_id": task.get("user_task_id"),
                    "injection_task_id": task.get("injection_task_id"),
                    "attack_type": task.get("attack_type"),
                    "utility": task.get("utility"),
                    "security": task.get("security"),
                    "duration": task.get("duration"),
                    "error": task.get("error"),
                }
            )
        with open(raw_results_file, "w") as f:
            json.dump(summarized_results, f, indent=2)
        files.append(raw_results_file)

        # Save structured results
        structured_results_file = output_dir / "results.json"
        with open(structured_results_file, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "model_name": model_name,
                    "framework": "agentdojo",
                    "results": [
                        {
                            "metric_name": r.metric_name,
                            "metric_value": r.metric_value,
                            "metric_type": r.metric_type,
                            "confidence_interval": r.confidence_interval,
                            "num_samples": r.num_samples,
                            "metadata": r.metadata,
                        }
                        for r in evaluation_results
                    ],
                },
                f,
                indent=2,
            )
        files.append(structured_results_file)

        # Save summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("AgentDojo Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Benchmark: {benchmark_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Total tasks evaluated: {len(task_results)}\n")
            f.write("\nMetrics:\n")
            f.write("-" * 70 + "\n")
            for result in evaluation_results:
                # Only show aggregate metrics in summary
                if result.metric_name in ("avg_utility", "avg_security"):
                    value = result.metric_value
                    if isinstance(value, float):
                        f.write(f"{result.metric_name}: {value * 100:.2f}%\n")
                    else:
                        f.write(f"{result.metric_name}: {value}\n")
        files.append(summary_file)

        # Copy raw logdir for full trace artifacts
        raw_logs_archive = output_dir / "raw_logs.tar.gz"
        try:
            import tarfile

            with tarfile.open(raw_logs_archive, "w:gz") as tar:
                tar.add(logdir, arcname="logs")
            files.append(raw_logs_archive)
        except Exception as e:
            logger.warning(f"Failed to archive raw logs: {e}")

        logger.info(f"Saved {len(files)} result files to {output_dir}")
        return files

    def _get_agentdojo_version(self) -> str:
        """Get AgentDojo package version.

        Returns:
            AgentDojo version string, or 'unknown' if not available
        """
        try:
            from importlib.metadata import version

            return version("agentdojo")
        except Exception:
            return "unknown"


def main() -> None:
    """Main entry point for AgentDojo adapter.

    Environment variables:
    - EVALHUB_MODE: "k8s" or "local" (default: local)
    - EVALHUB_JOB_SPEC_PATH: Override job spec path
    - OPENAI_API_KEY: Required for OpenAI models
    - ANTHROPIC_API_KEY: Required for Anthropic models
    - TOGETHER_API_KEY: Required for Together models
    - GCP_PROJECT / GCP_LOCATION: Required for Google models
    - REGISTRY_URL: OCI registry URL
    - REGISTRY_USERNAME: Registry username
    - REGISTRY_PASSWORD: Registry password/token
    - REGISTRY_INSECURE: Allow insecure HTTP (default: false)
    """

    from evalhub.adapter import DefaultCallbacks

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = AgentDojoAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks.from_adapter(adapter)

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Evaluated {results.num_examples_evaluated} tasks")

        callbacks.report_results(results)

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        logger.error(
            "Set EVALHUB_JOB_SPEC_PATH or ensure job spec exists at default location"
        )
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
