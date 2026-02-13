"""MTEB (Massive Text Embedding Benchmark) adapter for eval-hub.

This adapter integrates MTEB (https://github.com/embeddings-benchmark/mteb)
with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

MTEB is a comprehensive benchmark for evaluating text embedding models across
diverse tasks including semantic textual similarity, retrieval, classification,
clustering, reranking, and more.

Architecture:
    1. JobSpec loaded from mounted ConfigMap (k8s mode) or local file
    2. Configuration validated and tasks resolved from benchmark_id or explicit list
    3. MTEB CLI executed via subprocess with proper arguments
    4. Results parsed from MTEB output JSON files
    5. Metrics extracted and normalized to EvaluationResult format
    6. Artifacts persisted as OCI images via sidecar callbacks
    7. Structured JobResults returned to eval-hub service

Example usage:
    # In Kubernetes (automatic):
    python main.py  # Reads /meta/job.json

    # Local development:
    EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
)

# Configure module logger
logger = logging.getLogger(__name__)


class MTEBAdapter(FrameworkAdapter):
    """MTEB (Massive Text Embedding Benchmark) adapter.

    Evaluates embedding models using the MTEB CLI and integrates with
    eval-hub via the callback-based architecture. Supports all MTEB task
    types including STS, Retrieval, Classification, Clustering, and more.

    The adapter:
    - Validates job configuration and resolves tasks from benchmark presets
    - Builds and executes MTEB CLI commands via subprocess
    - Parses JSON results from MTEB output directory
    - Extracts metrics and computes overall scores
    - Persists detailed results as OCI artifacts

    Attributes:
        BENCHMARK_PRESETS: Mapping of benchmark_id shortcuts to task lists
        SUPPORTED_TASK_TYPES: List of valid MTEB task type identifiers

    Example:
        >>> adapter = MTEBAdapter(job_spec_path="meta/job.json")
        >>> callbacks = DefaultCallbacks(...)
        >>> results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    """

    # Benchmark ID to task list mappings for common evaluation scenarios
    BENCHMARK_PRESETS: dict[str, list[str]] = {
        "mteb_sts": [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STS17",
            "STSBenchmark",
            "SICK-R",
        ],
        "mteb_retrieval": [
            "NFCorpus",
            "SciFact",
            "ArguAna",
            "TRECCOVID",
            "Touche2020",
        ],
        "mteb_classification": [
            "AmazonReviewsClassification",
            "Banking77Classification",
            "EmotionClassification",
        ],
        "mteb_clustering": [
            "ArxivClusteringP2P",
            "ArxivClusteringS2S",
            "BiorxivClusteringP2P",
        ],
        "mteb_reranking": [
            "AskUbuntuDupQuestions",
            "MindSmallReranking",
            "SciDocsRR",
        ],
    }

    # Supported MTEB task types for validation
    SUPPORTED_TASK_TYPES: list[str] = [
        "BitextMining",
        "Classification",
        "Clustering",
        "InstructionRetrieval",
        "MultilabelClassification",
        "PairClassification",
        "Reranking",
        "Retrieval",
        "STS",
        "Summarization",
    ]

    def run_benchmark_job(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
    ) -> JobResults:
        """Execute an MTEB evaluation job.

        Orchestrates the complete evaluation lifecycle:
        1. Validates configuration and resolves tasks
        2. Builds MTEB CLI command
        3. Executes subprocess and monitors progress
        4. Parses results and extracts metrics
        5. Persists artifacts and returns structured results

        Args:
            config: Job specification containing model, benchmark, and config details.
                Required fields: id, benchmark_id, model.name
                Optional: benchmark_config with tasks, batch_size, device, etc.
            callbacks: Callback interface for status updates and artifact persistence.
                Used to report progress phases and create OCI artifacts.

        Returns:
            JobResults containing:
                - Extracted metrics as EvaluationResult list
                - Computed overall_score (average of main_scores)
                - Evaluation metadata (framework version, tasks, config)
                - OCI artifact reference if persistence succeeded

        Raises:
            ValueError: If configuration is invalid (missing model name, etc.)
            RuntimeError: If MTEB CLI fails, times out, or produces no results

        Example:
            >>> results = adapter.run_benchmark_job(job_spec, callbacks)
            >>> print(f"Overall score: {results.overall_score}")
            >>> for r in results.results:
            ...     print(f"{r.metric_name}: {r.metric_value}")
        """
        start_time = time.time()
        logger.info(f"Starting MTEB job {config.id} for benchmark {config.benchmark_id}")

        output_dir: Path | None = None

        try:
            # Phase 1: Initialize and validate
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message=f"Initializing MTEB for benchmark {config.benchmark_id}",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)
            tasks = self._resolve_tasks(config.benchmark_id, config.benchmark_config)
            output_dir = Path(tempfile.mkdtemp(prefix="mteb_output_"))

            logger.info(
                f"Configuration validated. Tasks: {tasks}, Output dir: {output_dir}"
            )

            # Phase 2: Loading data (MTEB handles internally, but we signal the phase)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=MessageInfo(
                        message=f"MTEB preparing {len(tasks)} task(s) for evaluation",
                        message_code="loading_data",
                    ),
                    current_step="Preparing evaluation",
                    total_steps=5,
                    completed_steps=1,
                )
            )

            # Phase 3: Run MTEB evaluation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.2,
                    message=MessageInfo(
                        message=f"Running MTEB evaluation on {len(tasks)} task(s)",
                        message_code="running_evaluation",
                    ),
                    current_step="Executing benchmark",
                    total_steps=5,
                    completed_steps=2,
                )
            )

            cmd = self._build_mteb_command(
                model_name=config.model.name,
                tasks=tasks,
                output_dir=output_dir,
                benchmark_config=config.benchmark_config,
            )

            timeout = getattr(config, "timeout_seconds", None) or 7200  # Default 2 hours
            self._run_mteb_subprocess(cmd, timeout)

            # Phase 4: Post-processing - parse and extract results
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing MTEB evaluation results",
                        message_code="post_processing",
                    ),
                    current_step="Extracting metrics",
                    total_steps=5,
                    completed_steps=3,
                )
            )

            mteb_results = self._parse_results(output_dir, config.model.name)
            evaluation_results = self._extract_evaluation_results(mteb_results)
            overall_score = self._compute_overall_score(evaluation_results)

            logger.info(
                f"Post-processing complete. Overall score: {overall_score}, "
                f"Metrics extracted: {len(evaluation_results)}"
            )

            # Phase 5: Persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting MTEB artifacts to OCI registry",
                        message_code="persisting_artifacts",
                    ),
                    current_step="Creating OCI artifact",
                    total_steps=5,
                    completed_steps=4,
                )
            )

            output_files = self._save_detailed_results(
                job_id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                mteb_results=mteb_results,
                evaluation_results=evaluation_results,
            )

            oci_artifact = None
            if output_files:
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files=output_files,
                        base_path=Path("/tmp/mteb_results"),
                        title=f"MTEB results for {config.benchmark_id}",
                        description=f"Results from MTEB job {config.id}",
                        annotations={
                            "job_id": config.id,
                            "benchmark_id": config.benchmark_id,
                            "model_name": config.model.name,
                            "framework": "mteb",
                            "overall_score": str(overall_score) if overall_score else "N/A",
                            "num_tasks": str(len(tasks)),
                        },
                        id=config.id,
                        benchmark_id=config.benchmark_id,
                        model_name=config.model.name,
                    )
                )
                logger.info(f"OCI artifact persisted: {oci_artifact.digest}")

            # Compute final duration
            duration = time.time() - start_time

            # Build and return results
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=len(tasks),  # MTEB doesn't expose sample counts
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "mteb",
                    "framework_version": self._get_mteb_version(),
                    "tasks": tasks,
                    "benchmark_config": config.benchmark_config,
                    "output_dir": str(output_dir),
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("MTEB evaluation failed")
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
                        "model_name": config.model.name,
                    },
                )
            )
            raise

        finally:
            # Clean up temporary MTEB output directory
            if output_dir and output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                    logger.info(f"Cleaned up temporary directory: {output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {output_dir}: {e}")

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration for MTEB evaluation.

        Checks that all required fields are present and have valid values.
        Logs warnings for unknown task types but does not reject them
        (MTEB may support tasks not in our static list).

        Args:
            config: Job specification to validate

        Raises:
            ValueError: If required fields are missing or invalid:
                - benchmark_id is required
                - model.name is required (MTEB uses HuggingFace model names)

        Example:
            >>> adapter._validate_config(job_spec)  # Raises if invalid
        """
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.name:
            raise ValueError(
                "model.name is required for MTEB evaluation. "
                "Provide a HuggingFace model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')"
            )

        # Validate task_types if provided
        benchmark_config = config.benchmark_config or {}
        task_types = benchmark_config.get("task_types")
        if task_types:
            for task_type in task_types:
                if task_type not in self.SUPPORTED_TASK_TYPES:
                    logger.warning(
                        f"Unknown task type '{task_type}'. "
                        f"Supported types: {self.SUPPORTED_TASK_TYPES}"
                    )

        logger.debug("Configuration validated successfully")

    def _resolve_tasks(
        self,
        benchmark_id: str,
        benchmark_config: dict[str, Any] | None,
    ) -> list[str]:
        """Resolve benchmark_id and config to a list of MTEB tasks.

        Tasks can be specified in multiple ways (in priority order):
        1. Explicit 'tasks' list in benchmark_config
        2. 'task_types' filter in benchmark_config (MTEB will expand)
        3. Preset benchmark_id (e.g., 'mteb_sts' -> list of STS tasks)
        4. Single task name as benchmark_id (e.g., 'STS12')

        Args:
            benchmark_id: Benchmark identifier - can be a preset name or task name
            benchmark_config: Optional configuration dict with 'tasks' or 'task_types'

        Returns:
            List of MTEB task names to evaluate

        Raises:
            ValueError: If no tasks can be resolved from the configuration

        Example:
            >>> tasks = adapter._resolve_tasks("mteb_sts", {})
            >>> # Returns: ["STS12", "STS13", "STS14", ...]
            >>> tasks = adapter._resolve_tasks("custom", {"tasks": ["STS12"]})
            >>> # Returns: ["STS12"]
        """
        benchmark_config = benchmark_config or {}

        # Priority 1: Explicit tasks list
        if "tasks" in benchmark_config and benchmark_config["tasks"]:
            tasks = benchmark_config["tasks"]
            if isinstance(tasks, str):
                tasks = [tasks]
            logger.info(f"Using explicit task list: {tasks}")
            return tasks

        # Priority 2: Task types filter (let MTEB CLI handle expansion)
        if "task_types" in benchmark_config and benchmark_config["task_types"]:
            # When task_types are specified, we don't specify individual tasks
            # MTEB CLI will filter tasks by type
            logger.info(
                f"Using task_types filter: {benchmark_config['task_types']}. "
                "MTEB will select matching tasks."
            )
            # Return empty list - _build_mteb_command will use --task-types instead
            return []

        # Priority 3: Preset benchmark ID
        if benchmark_id in self.BENCHMARK_PRESETS:
            tasks = self.BENCHMARK_PRESETS[benchmark_id]
            logger.info(f"Using preset '{benchmark_id}': {tasks}")
            return tasks

        # Priority 4: Treat benchmark_id as a single task name
        logger.info(f"Treating benchmark_id '{benchmark_id}' as single task")
        return [benchmark_id]

    def _build_mteb_command(
        self,
        model_name: str,
        tasks: list[str],
        output_dir: Path,
        benchmark_config: dict[str, Any] | None,
    ) -> list[str]:
        """Build MTEB CLI command from configuration.

        Constructs the 'mteb run' command with appropriate arguments based on
        the job configuration. Handles model specification, task selection,
        batch size, device override, and other MTEB options.

        Args:
            model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            tasks: List of MTEB task names to run (can be empty if task_types used)
            output_dir: Directory where MTEB will write result JSON files
            benchmark_config: Optional configuration with additional parameters:
                - batch_size (int): Encoding batch size, default 32
                - device (str): Device override ('cuda', 'cpu', 'mps', 'cuda:0')
                - verbosity (int): MTEB verbosity level 0-3, default 2
                - co2_tracker (bool): Enable CO2 tracking, default False
                - overwrite_results (bool): Overwrite existing results, default True
                - task_types (list): Filter by task type instead of explicit tasks
                - task_categories (list): Filter by category (e.g., 's2s', 'p2p')
                - languages (list): Language codes to include, default ['eng']

        Returns:
            List of command arguments ready for subprocess execution

        Example:
            >>> cmd = adapter._build_mteb_command(
            ...     "sentence-transformers/all-MiniLM-L6-v2",
            ...     ["STS12", "STS13"],
            ...     Path("/tmp/output"),
            ...     {"batch_size": 64, "device": "cuda"}
            ... )
            >>> # Returns: ["mteb", "run", "--model", "...", "--tasks", "STS12", "STS13", ...]
        """
        benchmark_config = benchmark_config or {}

        # Base command
        cmd = [
            "mteb",
            "run",
            "--model",
            model_name,
            "--output-folder",
            str(output_dir),
        ]

        # Task specification (mutually exclusive with task_types)
        if tasks:
            cmd.append("--tasks")
            cmd.extend(tasks)
        elif "task_types" in benchmark_config and benchmark_config["task_types"]:
            # Use task type filtering
            cmd.append("--task-types")
            task_types = benchmark_config["task_types"]
            if isinstance(task_types, str):
                task_types = [task_types]
            cmd.extend(task_types)

        # Task categories (optional additional filter)
        if "task_categories" in benchmark_config and benchmark_config["task_categories"]:
            cmd.append("--categories")
            categories = benchmark_config["task_categories"]
            if isinstance(categories, str):
                categories = [categories]
            cmd.extend(categories)

        # Language filter
        languages = benchmark_config.get("languages", ["eng"])
        if languages:
            cmd.append("--languages")
            if isinstance(languages, str):
                languages = [languages]
            cmd.extend(languages)

        # Batch size
        batch_size = benchmark_config.get("batch_size", 32)
        cmd.extend(["--batch-size", str(batch_size)])

        # Verbosity level
        verbosity = benchmark_config.get("verbosity", 2)
        cmd.extend(["--verbosity", str(verbosity)])

        # Device override (optional)
        device = benchmark_config.get("device")
        if device:
            cmd.extend(["--device", device])

        # CO2 tracker (optional)
        if benchmark_config.get("co2_tracker", False):
            cmd.append("--co2-tracker")

        # Overwrite results (default True for eval-hub to ensure fresh results)
        if benchmark_config.get("overwrite_results", True):
            cmd.append("--overwrite")

        logger.info(f"Built MTEB command: {' '.join(cmd)}")
        return cmd

    def _run_mteb_subprocess(
        self,
        cmd: list[str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        """Execute MTEB CLI subprocess with output streaming and timeout.

        Runs the MTEB command, captures output for logging, and handles
        errors and timeouts appropriately.

        Args:
            cmd: Command arguments list (from _build_mteb_command)
            timeout: Maximum execution time in seconds

        Returns:
            CompletedProcess object with return code and output

        Raises:
            RuntimeError: If MTEB CLI exits with non-zero code or times out.
                The error message includes stdout/stderr for debugging.

        Example:
            >>> result = adapter._run_mteb_subprocess(cmd, timeout=3600)
            >>> print(f"MTEB completed with code {result.returncode}")
        """
        logger.info(f"Executing MTEB CLI with timeout={timeout}s: {' '.join(cmd)}")

        try:
            # Run MTEB with combined stdout/stderr for logging
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        logger.info(f"MTEB: {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.warning(f"MTEB stderr: {line}")

            # Check for errors
            if result.returncode != 0:
                raise RuntimeError(
                    f"MTEB CLI failed with exit code {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )

            logger.info("MTEB CLI completed successfully")
            return result

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"MTEB evaluation timed out after {timeout} seconds. "
                f"Consider increasing timeout_seconds or reducing the number of tasks."
            ) from e

    def _parse_results(
        self,
        output_dir: Path,
        model_name: str,
    ) -> dict[str, Any]:
        """Parse MTEB result JSON files from output directory.

        MTEB writes results to:
            <output_folder>/<model_name_sanitized>/<revision>/<TaskName>.json

        This method locates and parses all task result files, combining them
        into a single dictionary keyed by task name.

        Args:
            output_dir: MTEB output directory (passed to --output_folder)
            model_name: Model name (used to locate results subdirectory)

        Returns:
            Dictionary mapping task names to their result data:
                {
                    "STS12": {"scores": {...}, "task_name": "STS12", ...},
                    "STS13": {...},
                    ...
                }

        Raises:
            RuntimeError: If no result files are found or JSON parsing fails

        Example:
            >>> results = adapter._parse_results(Path("/tmp/mteb_out"), "model-name")
            >>> for task, data in results.items():
            ...     print(f"{task}: {data['scores']}")
        """
        logger.info(f"Parsing MTEB results from {output_dir}")

        # MTEB sanitizes model names for directory names
        # Find the model directory (may have different naming conventions)
        model_dirs = list(output_dir.iterdir())

        if not model_dirs:
            raise RuntimeError(
                f"No output directories found in {output_dir}. "
                "MTEB may have failed to produce results."
            )

        # Find all JSON result files
        result_files: list[Path] = []
        for model_dir in model_dirs:
            if model_dir.is_dir():
                # Results may be directly in model_dir or in a revision subdirectory
                result_files.extend(model_dir.rglob("*.json"))

        if not result_files:
            raise RuntimeError(
                f"No JSON result files found in {output_dir}. "
                f"Searched directories: {model_dirs}"
            )

        logger.info(f"Found {len(result_files)} result file(s)")

        # Parse each result file
        mteb_results: dict[str, Any] = {}
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    task_data = json.load(f)

                # Use task_name from the data, or derive from filename
                task_name = task_data.get("task_name", result_file.stem)
                mteb_results[task_name] = task_data
                logger.debug(f"Parsed results for task: {task_name}")

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse {result_file}: {e}")
                continue

        if not mteb_results:
            raise RuntimeError(
                f"Failed to parse any result files from {output_dir}. "
                f"Files found: {result_files}"
            )

        logger.info(f"Successfully parsed results for {len(mteb_results)} task(s)")
        return mteb_results

    def _extract_evaluation_results(
        self,
        mteb_results: dict[str, Any],
    ) -> list[EvaluationResult]:
        """Convert MTEB task results to EvaluationResult objects.

        Extracts metrics from MTEB's nested result structure and creates
        standardized EvaluationResult objects with hierarchical metric names.

        MTEB result structure:
            {
                "task_name": "STS12",
                "scores": {
                    "test": [
                        {
                            "main_score": 0.847,
                            "cosine_spearman": 0.847,
                            "hf_subset": "default",
                            "languages": ["eng-Latn"]
                        }
                    ]
                }
            }

        Output metric names follow the pattern:
            <task_name>.<split>.<metric_name>
            e.g., "STS12.test.main_score", "STS12.test.cosine_spearman"

        Args:
            mteb_results: Dictionary of task results from _parse_results

        Returns:
            List of EvaluationResult objects with:
                - metric_name: Hierarchical name (task.split.metric)
                - metric_value: Numeric value
                - metric_type: "float" for all MTEB metrics
                - metadata: Task, split, subset, and language info

        Example:
            >>> results = adapter._extract_evaluation_results(mteb_results)
            >>> for r in results:
            ...     print(f"{r.metric_name}: {r.metric_value}")
        """
        evaluation_results: list[EvaluationResult] = []

        for task_name, task_data in mteb_results.items():
            scores = task_data.get("scores", {})

            for split_name, split_scores in scores.items():
                # split_scores is a list of score entries (one per subset/language)
                if not isinstance(split_scores, list):
                    continue

                for score_entry in split_scores:
                    if not isinstance(score_entry, dict):
                        continue

                    # Extract main_score first (primary metric)
                    main_score = score_entry.get("main_score")
                    if main_score is not None:
                        evaluation_results.append(
                            EvaluationResult(
                                metric_name=f"{task_name}.{split_name}.main_score",
                                metric_value=main_score,
                                metric_type="float",
                                metadata={
                                    "task": task_name,
                                    "split": split_name,
                                    "subset": score_entry.get("hf_subset"),
                                    "languages": score_entry.get("languages"),
                                },
                            )
                        )

                    # Extract additional numeric metrics
                    skip_keys = {"main_score", "hf_subset", "languages"}
                    for metric_key, metric_value in score_entry.items():
                        if metric_key in skip_keys:
                            continue
                        if isinstance(metric_value, (int, float)):
                            evaluation_results.append(
                                EvaluationResult(
                                    metric_name=f"{task_name}.{split_name}.{metric_key}",
                                    metric_value=float(metric_value),
                                    metric_type="float",
                                    metadata={
                                        "task": task_name,
                                        "split": split_name,
                                    },
                                )
                            )

        logger.info(f"Extracted {len(evaluation_results)} metrics from MTEB results")
        return evaluation_results

    def _compute_overall_score(
        self,
        results: list[EvaluationResult],
    ) -> float | None:
        """Compute aggregate score from evaluation results.

        Calculates the average of all main_score metrics across tasks.
        This provides a single summary metric for the evaluation.

        Args:
            results: List of EvaluationResult objects from _extract_evaluation_results

        Returns:
            Average of main_score values, or None if no main_scores found

        Example:
            >>> score = adapter._compute_overall_score(results)
            >>> print(f"Overall MTEB score: {score:.3f}")
        """
        main_scores = [
            r.metric_value
            for r in results
            if r.metric_name.endswith(".main_score")
            and isinstance(r.metric_value, (int, float))
        ]

        if main_scores:
            overall = sum(main_scores) / len(main_scores)
            logger.info(
                f"Computed overall score: {overall:.4f} "
                f"(average of {len(main_scores)} main_score values)"
            )
            return overall

        logger.warning("No main_score metrics found, cannot compute overall score")
        return None

    def _save_detailed_results(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        mteb_results: dict[str, Any],
        evaluation_results: list[EvaluationResult],
    ) -> list[Path]:
        """Save detailed results to files for OCI artifact persistence.

        Creates multiple output files for comprehensive result reporting:
        - mteb_results.json: Raw MTEB output for all tasks
        - results.json: Structured eval-hub format with metrics
        - summary.txt: Human-readable evaluation summary

        Args:
            job_id: Job identifier for file organization
            benchmark_id: Benchmark identifier for metadata
            model_name: Model name for metadata
            mteb_results: Raw MTEB results dictionary
            evaluation_results: Extracted EvaluationResult objects

        Returns:
            List of Paths to created files for OCI artifact spec

        Example:
            >>> files = adapter._save_detailed_results(
            ...     "job-123", "mteb_sts", "model-name", mteb_results, eval_results
            ... )
            >>> print(f"Created {len(files)} output files")
        """
        output_dir = Path("/tmp/mteb_results") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        files: list[Path] = []

        # Save raw MTEB results
        raw_results_file = output_dir / "mteb_results.json"
        with open(raw_results_file, "w") as f:
            json.dump(mteb_results, f, indent=2)
        files.append(raw_results_file)
        logger.debug(f"Saved raw results to {raw_results_file}")

        # Compute overall score for structured output
        overall_score = self._compute_overall_score(evaluation_results)

        # Save structured results in eval-hub format
        structured_results_file = output_dir / "results.json"
        with open(structured_results_file, "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "model_name": model_name,
                    "framework": "mteb",
                    "framework_version": self._get_mteb_version(),
                    "overall_score": overall_score,
                    "num_tasks": len(mteb_results),
                    "results": [
                        {
                            "metric_name": r.metric_name,
                            "metric_value": r.metric_value,
                            "metric_type": r.metric_type,
                            "metadata": r.metadata,
                        }
                        for r in evaluation_results
                    ],
                },
                f,
                indent=2,
            )
        files.append(structured_results_file)
        logger.debug(f"Saved structured results to {structured_results_file}")

        # Save human-readable summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("MTEB Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Job ID:      {job_id}\n")
            f.write(f"Benchmark:   {benchmark_id}\n")
            f.write(f"Model:       {model_name}\n")
            f.write(f"Framework:   mteb {self._get_mteb_version()}\n")
            f.write(f"Tasks:       {len(mteb_results)}\n")
            if overall_score is not None:
                f.write(f"Overall:     {overall_score:.4f}\n")
            f.write("\n")

            # Group metrics by task
            f.write("Results by Task:\n")
            f.write("-" * 70 + "\n")

            task_scores: dict[str, list[tuple[str, float]]] = {}
            for r in evaluation_results:
                parts = r.metric_name.split(".")
                if len(parts) >= 3:
                    task = parts[0]
                    metric = ".".join(parts[1:])
                    if task not in task_scores:
                        task_scores[task] = []
                    if isinstance(r.metric_value, (int, float)):
                        task_scores[task].append((metric, float(r.metric_value)))

            for task in sorted(task_scores.keys()):
                f.write(f"\n{task}:\n")
                for metric, value in sorted(task_scores[task]):
                    f.write(f"  {metric}: {value:.4f}\n")

        files.append(summary_file)
        logger.debug(f"Saved summary to {summary_file}")

        logger.info(f"Saved {len(files)} result files to {output_dir}")
        return files

    def _get_mteb_version(self) -> str:
        """Get installed MTEB package version.

        Returns:
            Version string (e.g., "1.14.0") or "unknown" if unavailable
        """
        try:
            import mteb

            return getattr(mteb, "__version__", "unknown")
        except ImportError:
            return "unknown"


def main() -> None:
    """Main entry point for MTEB adapter.

    Orchestrates the complete adapter lifecycle:
    1. Configures logging from LOG_LEVEL environment variable
    2. Loads JobSpec from configured path (default: /meta/job.json in k8s mode)
    3. Creates DefaultCallbacks for sidecar communication
    4. Executes benchmark job via adapter
    5. Reports results and exits with appropriate code

    Environment Variables:
        EVALHUB_MODE: Execution mode, "k8s" or "local" (default: local)
        EVALHUB_JOB_SPEC_PATH: Path to job specification JSON
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        REGISTRY_URL: OCI registry URL for artifact storage
        REGISTRY_USERNAME: Registry authentication username
        REGISTRY_PASSWORD: Registry authentication password
        REGISTRY_INSECURE: Allow insecure HTTP registry (default: false)

    Exit Codes:
        0: Job completed successfully
        1: Job failed (configuration error, MTEB error, etc.)

    Example:
        # Kubernetes deployment (automatic ConfigMap mount)
        python main.py

        # Local development
        EVALHUB_MODE=local EVALHUB_JOB_SPEC_PATH=meta/job.json python main.py
    """
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create adapter with job spec path from environment or default
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = MTEBAdapter(job_spec_path=job_spec_path)

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

        # Log summary
        logger.info(f"Job completed successfully: {results.id}")
        if results.overall_score is not None:
            logger.info(f"Overall score: {results.overall_score:.4f}")
        logger.info(f"Duration: {results.duration_seconds:.2f} seconds")

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
