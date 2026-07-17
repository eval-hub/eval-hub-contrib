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
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
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
    read_model_auth_key,
    resolve_model_credentials,
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
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
            )

            self._validate_config(config)
            tasks = self._parse_benchmark_tasks(config.benchmark_id, config.parameters)
            output_dir = Path(tempfile.mkdtemp(prefix="lighteval_"))
            logger.info(f"Configuration validated. Tasks: {tasks}, Output dir: {output_dir}")

            # Phase 2: Loading data (LightEval handles this internally)
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
            )

            # Phase 3: Run evaluation
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION)
            )

            lighteval_results = self._run_lighteval(
                model_config=config.model,
                tasks=tasks,
                output_dir=output_dir,
                num_fewshot=config.parameters.get("num_few_shot", 0),
                limit=config.num_examples,
                batch_size=config.parameters.get("batch_size", 1),
                benchmark_config=config.parameters,
            )

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
            )

            config_tasks = lighteval_results.get("config_tasks", {})

            evaluation_results = self._extract_evaluation_results(
                lighteval_results, config.benchmark_id
            )
            overall_score = self._compute_overall_score(evaluation_results)
            num_evaluated = self._extract_num_evaluated(lighteval_results)
            additional_info = self._build_additional_info(
                overall_score, lighteval_results, config_tasks
            )

            # Save detailed results
            output_files = self._save_detailed_results(
                job_id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                lighteval_results=lighteval_results,
                evaluation_results=evaluation_results,
                additional_info=additional_info,
            )

            logger.info(
                f"Post-processing complete. Overall score: {overall_score}, "
                f"Evaluated: {num_evaluated} examples, Files: {len(output_files)}"
            )

            # Phase 5: Persist artifacts
            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None and output_files:
                callbacks.report_status(
                    JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
                )
                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update(
                    {
                        "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                        "io.github.eval-hub.benchmark": config.benchmark_id,
                        "io.github.eval-hub.model": config.model.name,
                        "io.github.eval-hub.job_id": config.id,
                    }
                )
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=output_files[0].parent,
                        coordinates=coords,
                    )
                )
                logger.info(f"OCI artifact created: {oci_artifact.reference}")
            else:
                logger.info("No OCI exports configured; skipping artifact persistence")

            # Compute final duration
            duration = time.time() - start_time

            # Return results
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
                    "framework": "lighteval",
                    "framework_version": self._get_lighteval_version(),
                    "num_few_shot": config.parameters.get("num_few_shot", 0),
                    "random_seed": config.parameters.get("random_seed"),
                    "benchmark_config": config.parameters,
                    "tasks": tasks,
                    "model_provider": config.parameters.get("provider", "endpoint"),
                },
                oci_artifact=oci_artifact,
                additional_info=additional_info,
            )

        except Exception as e:
            logger.exception("LightEval evaluation failed")
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
        provider = config.parameters.get("provider", "endpoint")
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

        elif provider == "vllm":
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
                # Ensure base_url ends with /v1 exactly once. model.url is normally
                # the sidecar address (http://localhost:8080) with no path, but local
                # test setups may pass a full URL that already includes /v1.
                _stripped = model_config.url.rstrip("/")
                sidecar_url = _stripped if _stripped.endswith("/v1") else _stripped + "/v1"
                model_args += f",base_url={sidecar_url}"
                creds = resolve_model_credentials()
                if creds.api_key:
                    # Inject ref token so the sidecar resolves it to the real key.
                    model_args += f",api_key={creds.api_key}"
                else:
                    # litellm rejects a falsy api_key client-side before making any
                    # HTTP call. "dummy" passes the check; the sidecar sees an empty
                    # Bearer (isBearerEmpty) and injects the SA token, or for truly
                    # open models forwards the request as-is.
                    model_args += ",api_key=dummy"

            # Add additional parameters from benchmark_config
            parameters = benchmark_config.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise ValueError("parameters must be an object")
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

        safe_cmd = [re.sub(r"(,api_key=)[^,]*", r"\1***", arg) for arg in cmd]
        logger.info(f"Executing LightEval CLI: {' '.join(safe_cmd)}")

        env = None
        if "HF_TOKEN" not in os.environ:
            hf_token = read_model_auth_key("hf-token")
            if hf_token:
                env = {**os.environ, "HF_TOKEN": hf_token}
                logger.info("Injected HF_TOKEN from model secret")

        try:
            # Run LightEval CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False,
                env=env,
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

            for metric_name, metric_value in task_metrics.items():
                if metric_name.endswith("_stderr"):
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

                clean_metric = self._normalise_metric_name(metric_name)
                clean_task = self._normalise_task_name(task_name)

                evaluation_results.append(
                    EvaluationResult(
                        metric_name=f"{clean_task}.{clean_metric}",
                        metric_value=metric_value,
                        metric_type=metric_type,
                        confidence_interval=confidence_interval,
                        num_samples=None,  # LightEval doesn't always provide this
                        metadata={
                            "task": clean_task,
                            "metric": clean_metric,
                            "stderr": stderr,
                        },
                    )
                )

        logger.info(f"Extracted {len(evaluation_results)} metrics from LightEval results")
        return evaluation_results

    @staticmethod
    def _normalise_task_name(task_name: str) -> str:
        # Strip |N fewshot suffix LightEval appends to task names in results JSON
        # (e.g. math_500|0 → math_500)
        if "|" in task_name:
            return task_name.rsplit("|", 1)[0]
        return task_name

    @staticmethod
    def _normalise_metric_name(metric_name: str) -> str:
        # Normalise LightEval metric names to match collection primary_score.metric.
        # pass@k:k=1&n=3 → pass@1  (query-string: extract k value)
        # codegen_pass@1:16 → codegen_pass@1  (plain numeric suffix)
        if ":" not in metric_name:
            return metric_name
        base, _, suffix = metric_name.rpartition(":")
        if "@k" in base:
            m = re.search(r"\bk=(\d+)\b", suffix)
            if m:
                return base.replace("@k", f"@{m.group(1)}")
        if suffix.isdigit():
            return base
        return metric_name

    def _compute_overall_score(self, results: list[EvaluationResult]) -> float | None:
        """Compute overall score from evaluation results.

        Returns the first numeric metric value. The EvalHub service selects
        the primary metric from the full results list using primary_score.metric
        from the collection YAML; the adapter's job is simply to report a
        representative score for logging.

        Args:
            results: List of evaluation results

        Returns:
            First numeric metric value, normalised to [0, 1], or None.
        """
        for result in results:
            if isinstance(result.metric_value, (int, float)):
                value = float(result.metric_value)
                if value > 1.0:
                    value = value / 100.0
                return value
        return None

    def _extract_num_evaluated(self, lighteval_results: dict[str, Any]) -> int:
        """Extract number of examples evaluated from LightEval results.

        Args:
            lighteval_results: Raw LightEval results

        Returns:
            Number of examples evaluated, or 0 if not available
        """
        config_general = lighteval_results.get("config_general", {})
        max_samples = config_general.get("max_samples") or config_general.get("num_samples")
        if max_samples is not None:
            return int(max_samples)
        return 0

    @staticmethod
    def _resolve_dataset_shas(config_tasks: dict[str, Any]) -> list[dict[str, Any]]:
        repos: dict[tuple[str, str | None], dict] = {}
        for task_cfg in config_tasks.values():
            if not isinstance(task_cfg, dict):
                continue
            hf_repo = task_cfg.get("hf_repo")
            if not hf_repo:
                continue
            hf_revision = task_cfg.get("hf_revision")
            key = (hf_repo, hf_revision)
            if key not in repos:
                repos[key] = {
                    "revision": hf_revision,
                    "subsets": [],
                }
            hf_subset = task_cfg.get("hf_subset", "default")
            if hf_subset not in repos[key]["subsets"]:
                repos[key]["subsets"].append(hf_subset)

        if not repos:
            return []

        from huggingface_hub import HfApi

        hf_token = os.environ.get("HF_TOKEN") or read_model_auth_key("hf-token")
        api = HfApi(token=hf_token or None)

        for (repo_id, _revision), repo in repos.items():
            try:
                info = api.dataset_info(repo_id, revision=repo["revision"] or "main")
                repo["sha"] = info.sha
            except Exception:
                logger.warning("Failed to resolve SHA for %s", repo_id, exc_info=True)

        dataset = []
        for (repo_id, _revision), repo in repos.items():
            for subset in repo["subsets"]:
                entry: dict[str, Any] = {"hf_repo": repo_id, "hf_subset": subset}
                if "sha" in repo:
                    entry["sha"] = repo["sha"]
                dataset.append(entry)

        return dataset

    @staticmethod
    def _fewshot_fields(num_fewshots: int, score) -> dict[str, Any]:
        if num_fewshots == 0:
            return {"zero_shot": score}
        return {
            "alt_prompting": score,
            "alt_prompting_description": f"{num_fewshots}-Shot",
        }

    @staticmethod
    def _extract_generation_parameters(
        lighteval_results: dict[str, Any],
    ) -> dict[str, Any]:
        model_config = lighteval_results.get("config_general", {}).get("model_config", {})
        gen_params = model_config.get("generation_parameters", {})
        if not isinstance(gen_params, dict):
            return {}
        return {k: v for k, v in gen_params.items() if v is not None}

    def _build_additional_info(
        self,
        overall_score: float | None,
        lighteval_results: dict[str, Any] | None = None,
        config_tasks: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            config_tasks = config_tasks or {}
            num_fewshots = 0
            for task_cfg in config_tasks.values():
                if isinstance(task_cfg, dict) and "num_fewshots" in task_cfg:
                    num_fewshots = task_cfg["num_fewshots"]
                    break

            dataset = self._resolve_dataset_shas(config_tasks)
            prompting_fields = self._fewshot_fields(num_fewshots, overall_score)

            info: dict[str, Any] = {"dataset": dataset, **prompting_fields}

            if lighteval_results is not None:
                gen_params = self._extract_generation_parameters(lighteval_results)
                if gen_params:
                    info["generation_parameters"] = gen_params

            return info
        except Exception:
            logger.warning("Failed to build additional_info", exc_info=True)
            return None

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        try:
            metadata = results.evaluation_metadata or {}
            num_fewshots = metadata.get("num_few_shot", 0)
            prompting_fields = self._fewshot_fields(num_fewshots, results.overall_score)
            return {"dataset": [], **prompting_fields}
        except Exception:
            logger.warning("Failed to generate additional_info", exc_info=True)
            return None

    def _save_detailed_results(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        lighteval_results: dict[str, Any],
        evaluation_results: list[EvaluationResult],
        additional_info: dict[str, Any] | None = None,
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
        if self.local_jobs_base_path is not None:
            output_dir = self.local_jobs_base_path / "results"
        else:
            output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Save raw LightEval results
        raw_results_file = output_dir / "lighteval_results.json"
        with open(raw_results_file, "w") as f:
            json.dump(lighteval_results, f, indent=2)
        files.append(raw_results_file)

        # Save structured results
        structured_results_file = output_dir / "results.json"
        structured = {
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
        }
        if additional_info is not None:
            structured["additional_info"] = additional_info
        with open(structured_results_file, "w") as f:
            json.dump(structured, f, indent=2)
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
            from importlib.metadata import version
            return version("lighteval")
        except Exception:
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
        callbacks = DefaultCallbacks.from_adapter(adapter)

        # Run benchmark job
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Evaluated {results.num_examples_evaluated} examples")

        # Save metrics/params to MLflow
        run_id = callbacks.mlflow.save(results, adapter.job_spec)
        if run_id:
            results.mlflow_run_id = run_id
            logger.info(f"MLflow run created: {run_id}")

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
