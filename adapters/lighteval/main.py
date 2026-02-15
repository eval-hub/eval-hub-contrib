"""LightEval framework adapter for eval-hub.

This adapter integrates LightEval (https://github.com/huggingface/lighteval)
with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

The adapter:
1. Reads JobSpec from a mounted ConfigMap
2. Executes LightEval benchmark evaluations
3. Reports progress via callbacks to the sidecar
4. Persists results as OCI artifacts
5. Returns structured JobResults
"""

import json
import logging
import os
import shutil
import subprocess
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


class LightEvalAdapter(FrameworkAdapter):
    """LightEval framework adapter.

    This adapter executes LightEval benchmarks and integrates with the eval-hub
    service using the callback-based architecture. It supports all LightEval
    tasks and model providers (transformers, vllm, openai, anthropic, endpoint).
    """

    # Supported LightEval task categories and their associated tasks
    SUPPORTED_TASKS = {
        "commonsense_reasoning": ["hellaswag", "winogrande", "openbookqa", "arc:easy"],
        "scientific_reasoning": ["arc:easy", "arc:challenge"],
        "physical_commonsense": ["piqa"],
        "truthfulness": ["truthfulqa:mc", "truthfulqa:generation"],
        "math": ["gsm8k", "math:algebra", "math:counting_and_probability"],
        "knowledge": ["mmlu", "triviaqa"],
        "language_understanding": ["glue:cola", "glue:sst2", "glue:mrpc"],
    }

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a LightEval benchmark evaluation job.

        Args:
            config: Job specification from mounted ConfigMap
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults: Evaluation results and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If LightEval evaluation fails
        """
        start_time = time.time()
        logger.info(f"Starting LightEval job {config.id} for benchmark {config.benchmark_id}")

        output_dir: Path | None = None

        try:
            # Phase 1: Initialize
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message=f"Initialising LightEval for benchmark {config.benchmark_id}",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)
            tasks = self._parse_benchmark_tasks(config.benchmark_id, config.benchmark_config)
            output_dir = Path(tempfile.mkdtemp(prefix="lighteval_"))
            logger.info(f"Configuration validated. Tasks: {tasks}, Output dir: {output_dir}")

            # Phase 2: Loading data (LightEval handles this internally)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.2,
                    message=MessageInfo(
                        message=f"LightEval loading benchmark data for {len(tasks)} task(s)",
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
                        message=f"Running LightEval on {len(tasks)} task(s)",
                        message_code="running_evaluation",
                    ),
                    current_step="Executing benchmark",
                    total_steps=4,
                    completed_steps=2,
                )
            )

            lighteval_results = self._run_lighteval(
                model_config=config.model,
                tasks=tasks,
                output_dir=output_dir,
                num_fewshot=config.benchmark_config.get("num_few_shot", 0),
                limit=config.num_examples,
                batch_size=config.benchmark_config.get("batch_size", 1),
                benchmark_config=config.benchmark_config,
            )

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing LightEval results",
                        message_code="post_processing",
                    ),
                    current_step="Extracting metrics",
                    total_steps=4,
                    completed_steps=3,
                )
            )

            evaluation_results = self._extract_evaluation_results(
                lighteval_results, config.benchmark_id
            )
            overall_score = self._compute_overall_score(evaluation_results)
            num_evaluated = self._extract_num_evaluated(lighteval_results)

            # Save detailed results
            output_files = self._save_detailed_results(
                job_id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                lighteval_results=lighteval_results,
                evaluation_results=evaluation_results,
            )

            logger.info(
                f"Post-processing complete. Overall score: {overall_score}, "
                f"Evaluated: {num_evaluated} examples, Files: {len(output_files)}"
            )

            # Phase 5: Persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting LightEval artifacts to OCI registry",
                        message_code="persisting_artifacts",
                    ),
                    current_step="Creating OCI artifact",
                    total_steps=4,
                    completed_steps=4,
                )
            )

            oci_artifact = None
            if output_files:
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files=output_files,
                        base_path=Path("/tmp/lighteval_results"),
                        title=f"LightEval results for {config.benchmark_id}",
                        description=f"Results from LightEval job {config.id}",
                        annotations={
                            "job_id": config.id,
                            "benchmark_id": config.benchmark_id,
                            "model_name": config.model.name,
                            "framework": "lighteval",
                            "overall_score": str(overall_score) if overall_score else "N/A",
                            "num_evaluated": str(num_evaluated),
                        },
                        id=config.id,
                        benchmark_id=config.benchmark_id,
                        model_name=config.model.name,
                    )
                )
                logger.info(f"OCI artifact persisted: {oci_artifact.digest}")

            # Compute final duration
            duration = time.time() - start_time

            # Return results
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=num_evaluated,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "lighteval",
                    "framework_version": self._get_lighteval_version(),
                    "num_few_shot": config.benchmark_config.get("num_few_shot", 0),
                    "random_seed": config.benchmark_config.get("random_seed"),
                    "benchmark_config": config.benchmark_config,
                    "tasks": tasks,
                    "model_provider": config.benchmark_config.get("provider", "endpoint"),
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("LightEval evaluation failed")
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
            # Clean up temporary directory
            if output_dir and output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                    logger.info(f"Cleaned up temporary directory: {output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {output_dir}: {e}")

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration for LightEval.

        Args:
            config: Job specification to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.url and not config.model.name:
            raise ValueError("Either model.url or model.name is required")

        if not config.model.name:
            raise ValueError("model.name is required")

        # Validate model provider (from benchmark_config)
        provider = config.benchmark_config.get("provider", "endpoint")
        valid_providers = ["transformers", "vllm", "openai", "anthropic", "endpoint", "litellm"]
        if provider not in valid_providers:
            logger.warning(
                f"Unknown model provider '{provider}'. "
                f"Valid providers: {valid_providers}"
            )

        logger.debug("Configuration validated successfully")

    def _parse_benchmark_tasks(
        self, benchmark_id: str, benchmark_config: dict[str, Any]
    ) -> list[str]:
        """Parse benchmark ID and config to determine LightEval tasks.

        Args:
            benchmark_id: Benchmark identifier (can be a single task or category)
            benchmark_config: Additional benchmark configuration with optional 'tasks' key

        Returns:
            List of LightEval task names

        Raises:
            ValueError: If benchmark_id is invalid or tasks cannot be determined
        """
        # Check if tasks are explicitly provided in config
        if "tasks" in benchmark_config and benchmark_config["tasks"]:
            tasks = benchmark_config["tasks"]
            if isinstance(tasks, str):
                tasks = [tasks]
            return tasks

        # Check if benchmark_id is a known category
        if benchmark_id in self.SUPPORTED_TASKS:
            return self.SUPPORTED_TASKS[benchmark_id]

        # Otherwise, treat benchmark_id as a single task name
        # LightEval task names can include colons (e.g., "arc:easy", "truthfulqa:mc")
        return [benchmark_id]

    def _run_lighteval(
        self,
        model_config: Any,
        tasks: list[str],
        output_dir: Path,
        num_fewshot: int,
        limit: int | None,
        batch_size: int,
        benchmark_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute LightEval CLI and return parsed results.

        Args:
            model_config: Model configuration from JobSpec
            tasks: List of LightEval task names
            output_dir: Directory for LightEval output files
            num_fewshot: Number of few-shot examples
            limit: Maximum number of examples to evaluate (None = all)
            batch_size: Batch size for evaluation

        Returns:
            Parsed LightEval results dictionary

        Raises:
            RuntimeError: If LightEval CLI fails or results cannot be parsed
        """
        logger.info(
            f"Running LightEval: model={model_config.name}, tasks={tasks}, "
            f"fewshot={num_fewshot}, limit={limit}, batch_size={batch_size}"
        )

        # Format tasks for LightEval CLI: task1|fewshot,task2|fewshot
        task_strings = [f"{task}|{num_fewshot}" for task in tasks]
        tasks_arg = ",".join(task_strings)

        # Determine model provider from benchmark_config
        provider = benchmark_config.get("provider", "endpoint")

        if provider == "transformers":
            # For HuggingFace transformers models
            model_args = f"pretrained={model_config.name}"
            device = benchmark_config.get("device")
            if device:
                model_args += f",device={device}"
            cmd = ["lighteval", "accelerate", model_args, tasks_arg]

        elif provider in ["vllm"]:
            # For vLLM models
            model_args = f"pretrained={model_config.name}"
            cmd = ["lighteval", "vllm", model_args, tasks_arg]

        elif provider in ["openai", "anthropic", "endpoint", "litellm"]:
            # For API-based models (OpenAI, Anthropic, custom endpoints)
            model_name = model_config.name

            # Add openai/ prefix if not present and using custom endpoint
            if model_config.url and not model_name.startswith(("openai/", "anthropic/", "azure/")):
                model_name = f"openai/{model_name}"
                logger.info(f"Added openai/ prefix for custom endpoint: {model_name}")

            model_args = f"model_name={model_name}"

            if model_config.url:
                model_args += f",base_url={model_config.url}"
                # Add dummy API key for custom endpoints that don't require auth
                model_args += ",api_key=dummy"

            # Add additional parameters from benchmark_config
            parameters = benchmark_config.get("parameters", {})
            if parameters:
                for key, value in parameters.items():
                    model_args += f",{key}={value}"

            cmd = ["lighteval", "endpoint", "litellm", model_args, tasks_arg]

        else:
            raise ValueError(f"Unsupported model provider: {provider}")

        # Add common arguments
        cmd.extend([
            "--output-dir", str(output_dir),
            "--no-push-to-hub",
            "--save-details",
        ])

        # Add max-samples limit if specified
        if limit is not None:
            cmd.extend(["--max-samples", str(limit)])
            logger.info(f"Limiting evaluation to {limit} samples per task")

        logger.info(f"Executing LightEval CLI: {' '.join(cmd)}")

        try:
            # Run LightEval CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
            )

            # Log output
            if result.stdout:
                logger.info(f"LightEval stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"LightEval stderr:\n{result.stderr}")

            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(
                    f"LightEval CLI failed with exit code {result.returncode}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )

            # Parse results from output directory
            # LightEval writes results to output_dir/results/model_name/results_*.json
            results_files = list(output_dir.rglob("results_*.json"))

            if not results_files:
                # Try alternative location
                results_files = list(output_dir.rglob("results.json"))

            if not results_files:
                raise RuntimeError(
                    f"No results file found in {output_dir}. "
                    f"Available files: {list(output_dir.rglob('*'))}"
                )

            logger.info(f"Found results file: {results_files[0]}")

            # Load and return results
            with open(results_files[0]) as f:
                results_data = json.load(f)

            return results_data

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"LightEval evaluation timed out after {e.timeout} seconds"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LightEval results file: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during LightEval evaluation: {type(e).__name__}: {e}"
            ) from e

    def _extract_evaluation_results(
        self, lighteval_results: dict[str, Any], benchmark_id: str
    ) -> list[EvaluationResult]:
        """Extract structured evaluation results from LightEval output.

        LightEval results format:
        {
            "results": {
                "task_name": {
                    "metric_name": value,
                    "metric_name_stderr": value,
                    ...
                }
            }
        }

        Args:
            lighteval_results: Raw LightEval results dictionary
            benchmark_id: Benchmark identifier for context

        Returns:
            List of structured EvaluationResult objects
        """
        evaluation_results = []

        # Extract results dict
        results_dict = lighteval_results.get("results", lighteval_results)

        for task_name, task_metrics in results_dict.items():
            if not isinstance(task_metrics, dict):
                continue

            # Process each metric
            processed_metrics = set()  # Track processed metrics to avoid duplicates

            for metric_name, metric_value in task_metrics.items():
                # Skip stderr metrics (we'll handle them separately)
                if metric_name.endswith("_stderr"):
                    continue

                # Skip if already processed
                if metric_name in processed_metrics:
                    continue

                # Look for stderr (standard error) for confidence interval
                stderr_key = f"{metric_name}_stderr"
                stderr = task_metrics.get(stderr_key)

                confidence_interval = None
                if stderr is not None:
                    # 95% confidence interval: value ± 1.96 * stderr
                    margin = 1.96 * stderr
                    confidence_interval = (
                        float(metric_value) - margin,
                        float(metric_value) + margin,
                    )

                # Determine metric type
                metric_type = "float"
                if isinstance(metric_value, int):
                    metric_type = "int"
                elif isinstance(metric_value, str):
                    metric_type = "string"
                elif isinstance(metric_value, bool):
                    metric_type = "bool"

                # Create hierarchical metric name: task.metric
                full_metric_name = f"{task_name}.{metric_name}"

                evaluation_results.append(
                    EvaluationResult(
                        metric_name=full_metric_name,
                        metric_value=metric_value,
                        metric_type=metric_type,
                        confidence_interval=confidence_interval,
                        num_samples=None,  # LightEval doesn't always provide this
                        metadata={
                            "task": task_name,
                            "metric": metric_name,
                            "stderr": stderr,
                        },
                    )
                )

                processed_metrics.add(metric_name)

        logger.info(f"Extracted {len(evaluation_results)} metrics from LightEval results")
        return evaluation_results

    def _compute_overall_score(self, results: list[EvaluationResult]) -> float | None:
        """Compute overall score from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Overall score (average of primary metrics), or None if not applicable
        """
        # Primary metrics to consider for overall score
        primary_metric_names = ["accuracy", "acc", "exact_match", "f1", "bleu"]

        primary_values = []
        for result in results:
            # Extract base metric name (after the task prefix)
            metric_parts = result.metric_name.split(".")
            if len(metric_parts) >= 2:
                base_metric = metric_parts[-1]
            else:
                base_metric = result.metric_name

            # Check if it's a primary metric
            if base_metric in primary_metric_names:
                if isinstance(result.metric_value, (int, float)):
                    value = float(result.metric_value)
                    # Assume metrics are already normalized to 0-1 or 0-100
                    if value > 1.0:
                        value = value / 100.0
                    primary_values.append(value)

        if primary_values:
            return sum(primary_values) / len(primary_values)
        return None

    def _extract_num_evaluated(self, lighteval_results: dict[str, Any]) -> int:
        """Extract number of examples evaluated from LightEval results.

        Args:
            lighteval_results: Raw LightEval results

        Returns:
            Number of examples evaluated, or 0 if not available
        """
        # LightEval doesn't always provide this directly
        # Try to extract from config or metadata
        if "config" in lighteval_results and "num_examples" in lighteval_results["config"]:
            return lighteval_results["config"]["num_examples"]

        # Otherwise return 0 (unknown)
        return 0

    def _save_detailed_results(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        lighteval_results: dict[str, Any],
        evaluation_results: list[EvaluationResult],
    ) -> list[Path]:
        """Save detailed results to files for OCI artifact.

        Args:
            job_id: Job identifier
            benchmark_id: Benchmark identifier
            model_name: Model name
            lighteval_results: Raw LightEval results
            evaluation_results: Structured evaluation results

        Returns:
            List of paths to saved files
        """
        output_dir = Path("/tmp/lighteval_results") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Save raw LightEval results
        raw_results_file = output_dir / "lighteval_results.json"
        with open(raw_results_file, "w") as f:
            json.dump(lighteval_results, f, indent=2)
        files.append(raw_results_file)

        # Save structured results
        structured_results_file = output_dir / "results.json"
        with open(structured_results_file, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "model_name": model_name,
                    "framework": "lighteval",
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
            f.write(f"LightEval Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Benchmark: {benchmark_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write("\nMetrics:\n")
            f.write("-" * 70 + "\n")
            for result in evaluation_results:
                f.write(f"{result.metric_name}: {result.metric_value}\n")
                if result.confidence_interval:
                    f.write(f"  95% CI: {result.confidence_interval}\n")
        files.append(summary_file)

        logger.info(f"Saved {len(files)} result files to {output_dir}")
        return files

    def _get_lighteval_version(self) -> str:
        """Get LightEval version.

        Returns:
            LightEval version string, or 'unknown' if not available
        """
        try:
            import lighteval
            return getattr(lighteval, "__version__", "unknown")
        except (ImportError, AttributeError):
            return "unknown"


def main() -> None:
    """Main entry point for LightEval adapter.

    The adapter automatically loads settings and JobSpec:
    1. AdapterSettings loads from environment (or uses defaults for mode)
    2. JobSpec is loaded from configured path (default: /meta/job.json in k8s mode)
    3. DefaultCallbacks communicate with localhost sidecar (if available)
    4. Adapter runs the benchmark job
    5. Results are persisted to OCI registry via callbacks

    Environment variables:
    - EVALHUB_MODE: "k8s" or "local" (default: local)
    - EVALHUB_JOB_SPEC_PATH: Override job spec path
    - REGISTRY_URL: OCI registry URL (e.g., ghcr.io)
    - REGISTRY_USERNAME: Registry username
    - REGISTRY_PASSWORD: Registry password/token
    - REGISTRY_INSECURE: Allow insecure HTTP (default: false)

    Note: The service URL for callbacks comes from job_spec.callback_url (mounted via ConfigMap)
    """
    import sys
    from evalhub.adapter import DefaultCallbacks

    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create adapter with job spec path from environment or default
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = LightEvalAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        # Create callbacks using adapter settings
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            registry_url=adapter.settings.registry_url,
            registry_username=adapter.settings.registry_username,
            registry_password=adapter.settings.registry_password,
            insecure=adapter.settings.registry_insecure,
        )

        # Run benchmark job
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Evaluated {results.num_examples_evaluated} examples")

        # Report final results
        callbacks.report_results(results)

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        logger.error("Set EVALHUB_JOB_SPEC_PATH or ensure job spec exists at default location")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
