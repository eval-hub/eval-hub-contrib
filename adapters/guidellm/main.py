#!/usr/bin/env python3
"""
GuideLLM Adapter for eval-hub

This adapter integrates GuideLLM benchmarking with the eval-hub evaluation service.
GuideLLM is a performance benchmarking platform that evaluates LLM inference servers
under realistic production conditions.

Architecture:
    1. Settings loaded automatically from environment
    2. JobSpec auto-loaded from mounted ConfigMap
    3. Progress updates via callbacks to sidecar
    4. Results persisted as OCI artifacts
    5. Structured metrics in JobResults format
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GuideLLMAdapter(FrameworkAdapter):
    """
    Adapter for GuideLLM benchmarking framework.

    GuideLLM provides performance benchmarking capabilities including:
    - Multiple execution profiles (sweep, throughput, concurrent, constant, poisson)
    - Comprehensive metrics (TTFT, ITL, latency, throughput)
    - Flexible data sources (synthetic, HuggingFace, local files)
    - Multiple output formats (JSON, CSV, HTML, YAML)
    """

    def __init__(self, job_spec_path: Optional[str] = None):
        """Initialize the GuideLLM adapter.

        Args:
            job_spec_path: Optional path to job specification file.
                          If not provided, uses EVALHUB_JOB_SPEC_PATH env var or default.
        """
        super().__init__(job_spec_path=job_spec_path)
        self.results_dir: Optional[Path] = None
        logger.info("GuideLLM adapter initialized")

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """
        Execute a GuideLLM benchmark job.

        This method:
        1. Validates the job specification
        2. Prepares benchmark configuration
        3. Runs GuideLLM via subprocess
        4. Collects and processes results
        5. Reports artifacts to sidecar
        6. Returns structured metrics

        Args:
            config: The benchmark job specification
            callbacks: Callbacks for status updates and artifact persistence

        Returns:
            JobResults containing performance metrics and artifacts
        """
        start_time = time.time()
        logger.info(f"Starting GuideLLM benchmark job: {config.id}")

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.0,
                message=MessageInfo(
                    message="Initializing GuideLLM benchmark",
                    message_code="initializing",
                ),
            )
        )

        try:
            # Create temporary directory for results
            self.results_dir = Path(tempfile.mkdtemp(prefix="guidellm_results_"))
            logger.info(f"Results directory: {self.results_dir}")

            # Build GuideLLM command
            cmd = self._build_guidellm_command(config)
            logger.info(f"Running command: {' '.join(cmd)}")

            # Execute GuideLLM
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message="Executing performance benchmark",
                        message_code="running_evaluation",
                    ),
                )
            )
            self._run_guidellm(cmd)

            # Parse results
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing benchmark results",
                        message_code="post_processing",
                    ),
                )
            )
            results_data = self._parse_results(config)

            # Extract overall score (requests per second or throughput)
            overall_score = results_data.get("requests_per_second")

            # Create evaluation results with performance metrics
            evaluation_results = []

            # Add requests per second metric
            if "requests_per_second" in results_data:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="requests_per_second",
                        metric_value=results_data["requests_per_second"],
                        metric_type="throughput",
                    )
                )

            # Add token throughput metrics
            if "prompt_tokens_per_second" in results_data:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="prompt_tokens_per_second",
                        metric_value=results_data["prompt_tokens_per_second"],
                        metric_type="throughput",
                    )
                )

            if "output_tokens_per_second" in results_data:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="output_tokens_per_second",
                        metric_value=results_data["output_tokens_per_second"],
                        metric_type="throughput",
                    )
                )

            # Add latency metrics
            if "mean_ttft_ms" in results_data:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="mean_ttft_ms",
                        metric_value=results_data["mean_ttft_ms"],
                        metric_type="latency",
                    )
                )

            if "mean_itl_ms" in results_data:
                evaluation_results.append(
                    EvaluationResult(
                        metric_name="mean_itl_ms",
                        metric_value=results_data["mean_itl_ms"],
                        metric_type="latency",
                    )
                )

            # Compute duration
            duration = time.time() - start_time

            # Create job results
            results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=results_data.get("total_requests", 0),
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "guidellm",
                    "profile": config.benchmark_config.get("profile", "sweep"),
                    "request_type": config.benchmark_config.get("request_type", "chat_completions"),
                },
            )

            # Report artifacts after creating results
            oci_artifact = self._report_artifacts(config, callbacks)
            if oci_artifact:
                results.oci_artifact = oci_artifact

            logger.info(f"Benchmark completed successfully: {config.id}")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.COMPLETED,
                    phase=JobPhase.COMPLETED,
                    progress=1.0,
                    message=MessageInfo(
                        message="Benchmark finished successfully",
                        message_code="completed",
                    ),
                )
            )
            return results

        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(
                        message=error_msg,
                        message_code="failed",
                    ),
                    error=ErrorInfo(
                        message=error_msg,
                        message_code="benchmark_error",
                    ),
                    error_details={
                        "exception_type": type(e).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )
            raise

    def _build_guidellm_command(self, job_spec: JobSpec) -> List[str]:
        """
        Build the GuideLLM CLI command from job specification.

        Args:
            job_spec: The benchmark job specification

        Returns:
            List of command arguments for subprocess execution
        """
        config = job_spec.benchmark_config
        model = job_spec.model

        # Base command
        cmd = [
            "guidellm",
            "benchmark",
            "--target", model.url,
            "--output-path", str(self.results_dir),
        ]

        # Execution profile
        profile = config.get("profile", "sweep")
        cmd.extend(["--profile", profile])

        # Rate parameter (meaning varies by profile)
        if "rate" in config:
            cmd.extend(["--rate", str(config["rate"])])

        # Duration limits
        if "max_seconds" in config:
            cmd.extend(["--max-seconds", str(config["max_seconds"])])

        # Max requests: prefer benchmark_config, fallback to job_spec.num_examples
        max_requests = config.get("max_requests")
        if max_requests is None and job_spec.num_examples is not None:
            max_requests = job_spec.num_examples
            logger.info(f"Using num_examples={max_requests} as max_requests")

        if max_requests is not None:
            cmd.extend(["--max-requests", str(max_requests)])

        # Error handling
        if "max_errors" in config:
            cmd.extend(["--max-errors", str(config["max_errors"])])

        # Warmup/cooldown periods
        # Note: GuideLLM expects integer values, not percentages
        if "warmup" in config:
            warmup = config["warmup"]
            # Convert percentage strings to integers (e.g., "10%" -> 10)
            if isinstance(warmup, str) and warmup.endswith("%"):
                warmup = warmup.rstrip("%")
            cmd.extend(["--warmup", str(warmup)])

        if "cooldown" in config:
            cooldown = config["cooldown"]
            # Convert percentage strings to integers (e.g., "10%" -> 10)
            if isinstance(cooldown, str) and cooldown.endswith("%"):
                cooldown = cooldown.rstrip("%")
            cmd.extend(["--cooldown", str(cooldown)])

        # Saturation detection
        if config.get("detect_saturation", False):
            cmd.append("--detect-saturation")

        # Data source
        data = config.get("data", "prompt_tokens=256,output_tokens=128")
        cmd.extend(["--data", data])

        # Data configuration
        if "data_args" in config:
            cmd.extend(["--data-args", json.dumps(config["data_args"])])

        if "data_column_mapper" in config:
            cmd.extend(["--data-column-mapper", json.dumps(config["data_column_mapper"])])

        if "data_samples" in config:
            cmd.extend(["--data-samples", str(config["data_samples"])])

        # Request type
        request_type = config.get("request_type", "chat_completions")
        cmd.extend(["--request-type", request_type])

        # Model name (optional, for identification)
        if model.name:
            cmd.extend(["--model", model.name])

        # Processor for synthetic data
        # For synthetic data (prompt_tokens=X,output_tokens=Y), GuideLLM needs a HuggingFace tokenizer.
        # The processor can be specified in benchmark_config, otherwise use a safe default.
        processor = config.get("processor")
        if processor is None and "prompt_tokens=" in data and "output_tokens=" in data:
            # Default to gpt2 tokenizer for synthetic data generation (widely available, small)
            processor = "gpt2"
            logger.info(
                f"No processor specified for synthetic data. Using default '{processor}'. "
                f"Specify a HuggingFace model in benchmark_config.processor for custom tokenization "
                f"(e.g., 'google/flan-t5-small' or 'meta-llama/Llama-3.1-8B-Instruct')"
            )

        if processor:
            cmd.extend(["--processor", processor])

        # Output formats (always generate all for comprehensive reporting)
        # Note: GuideLLM uses --outputs, not --output-format
        cmd.extend(["--outputs", "json,csv,html,yaml"])

        logger.debug(f"Built GuideLLM command: {' '.join(cmd)}")
        return cmd

    def _run_guidellm(self, cmd: List[str]) -> None:
        """
        Execute GuideLLM via subprocess with proper output handling.

        Args:
            cmd: Command arguments list

        Raises:
            RuntimeError: If GuideLLM execution fails
        """
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            if process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(f"GuideLLM: {line}")

            # Wait for completion
            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"GuideLLM failed with exit code {return_code}")

        except Exception as e:
            logger.error(f"Failed to execute GuideLLM: {e}")
            raise

    def _parse_results(self, job_spec: JobSpec) -> Dict[str, Any]:
        """
        Parse GuideLLM results from output files.

        GuideLLM generates multiple output formats:
        - benchmarks.json: Complete authoritative record
        - benchmarks.csv: Tabular view
        - benchmarks.html: Visual summary
        - benchmarks.yaml: Human-readable alternative

        Args:
            job_spec: The benchmark job specification

        Returns:
            Dictionary of metrics in standardized format
        """
        if not self.results_dir:
            raise RuntimeError("Results directory not initialized")

        results_file = self.results_dir / "benchmarks.json"

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        logger.info(f"Parsing results from: {results_file}")

        with open(results_file, "r") as f:
            raw_results = json.load(f)

        # Extract key metrics from GuideLLM results
        # GuideLLM structure: {benchmarks: [{metrics: {...}}]}
        # Each metric has nested structure with successful/errored/total statistics

        # Initialize metrics with framework info
        metrics: Dict[str, Any] = {
            "framework": "guidellm",
            "benchmark_id": job_spec.benchmark_id,
        }

        # Extract summary statistics from new GuideLLM structure
        if "benchmarks" in raw_results and raw_results["benchmarks"]:
            benchmark_metrics = raw_results["benchmarks"][0].get("metrics", {})

            # Extract requests per second (use successful.mean)
            if "requests_per_second" in benchmark_metrics:
                rps = benchmark_metrics["requests_per_second"].get("successful", {})
                if "mean" in rps:
                    metrics["requests_per_second"] = rps["mean"]

            # Extract prompt tokens per second
            if "prompt_tokens_per_second" in benchmark_metrics:
                pts = benchmark_metrics["prompt_tokens_per_second"].get("successful", {})
                if "mean" in pts:
                    metrics["prompt_tokens_per_second"] = pts["mean"]

            # Extract output tokens per second
            if "output_tokens_per_second" in benchmark_metrics:
                ots = benchmark_metrics["output_tokens_per_second"].get("successful", {})
                if "mean" in ots:
                    metrics["output_tokens_per_second"] = ots["mean"]

            # Extract time to first token (mean)
            if "time_to_first_token_ms" in benchmark_metrics:
                ttft = benchmark_metrics["time_to_first_token_ms"].get("successful", {})
                if "mean" in ttft:
                    metrics["mean_ttft_ms"] = ttft["mean"]

            # Extract inter-token latency (mean)
            if "inter_token_latency_ms" in benchmark_metrics:
                itl = benchmark_metrics["inter_token_latency_ms"].get("successful", {})
                if "mean" in itl:
                    metrics["mean_itl_ms"] = itl["mean"]

            # Extract total requests
            if "request_totals" in benchmark_metrics:
                totals = benchmark_metrics["request_totals"]
                metrics["total_requests"] = totals.get("successful", 0)

            # Add benchmark count
            metrics["benchmark_count"] = len(raw_results.get("benchmarks", []))

        logger.info(f"Extracted metrics: {json.dumps(metrics, indent=2)}")

        # Store full results separately for artifact reporting only
        # Not sent to the service to reduce payload size
        self._full_results = raw_results

        return metrics

    def _report_artifacts(self, config: JobSpec, callbacks: JobCallbacks):
        """
        Report all generated artifacts to the sidecar for OCI persistence.

        GuideLLM generates multiple output formats:
        - benchmarks.json: Complete data
        - benchmarks.csv: Spreadsheet format
        - benchmarks.html: Visual report
        - benchmarks.yaml: Human-readable

        Args:
            config: The benchmark job specification
            callbacks: Callbacks for artifact persistence

        Returns:
            OCI artifact information if persisted, None otherwise
        """
        if not self.results_dir:
            logger.warning("No results directory available for artifact reporting")
            return None

        # Collect all output files
        output_files = list(self.results_dir.glob("benchmarks.*"))

        if not output_files:
            logger.warning("No output files found in results directory")
            return None

        # Create OCI artifact with all files
        logger.info(f"Creating OCI artifact with {len(output_files)} files")
        oci_artifact = callbacks.create_oci_artifact(
            OCIArtifactSpec(
                files=output_files,
                base_path=self.results_dir,
                title=f"GuideLLM results for {config.benchmark_id}",
                description=f"Performance benchmark results from GuideLLM job {config.id}",
                annotations={
                    "job_id": config.id,
                    "benchmark_id": config.benchmark_id,
                    "model_name": config.model.name,
                    "framework": "guidellm",
                    "profile": config.benchmark_config.get("profile", "sweep"),
                },
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
            )
        )

        if oci_artifact:
            logger.info(f"OCI artifact persisted: {oci_artifact.digest}")

        return oci_artifact


def main() -> None:
    """Main entry point for GuideLLM adapter.

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
        adapter = GuideLLMAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        # Create callbacks using adapter settings
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            sidecar_url=adapter.job_spec.callback_url,
            registry_url=adapter.settings.registry_url,
            registry_username=adapter.settings.registry_username,
            registry_password=adapter.settings.registry_password,
            insecure=adapter.settings.registry_insecure,
        )

        # Run benchmark job
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        # Report results to service
        callbacks.report_results(results)

        logger.info(f"Job completed successfully: {results.id}")
        if results.overall_score:
            logger.info(f"Overall score: {results.overall_score:.2f} requests/sec")
        logger.info(f"Evaluated {results.num_examples_evaluated} requests")
        logger.info(f"Duration: {results.duration_seconds:.2f} seconds")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error in GuideLLM adapter: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
