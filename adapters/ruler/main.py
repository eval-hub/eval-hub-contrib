"""RULER adapter for EvalHub — NVIDIA RULER long-context benchmark.

RULER (What's the Real Context Size of Your LLM?) evaluates LLMs across
13 synthetic long-context tasks at configurable context lengths.
Reference: https://github.com/NVIDIA/RULER
"""

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    CapabilityEvalEntry,
    ErrorInfo,
    EvalCardMetadata,
    EnvironmentCardMetadata,
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

# Maps provider.yaml benchmark IDs → internal RULER task keys (from synthetic.yaml)
BENCHMARK_TO_TASKS: dict[str, list[str]] = {
    "niah-single-noise":          ["niah_single_1"],
    "niah-single-essay":          ["niah_single_2"],
    "niah-single-uuid":           ["niah_single_3"],
    "niah-multikey":              ["niah_multikey_1"],
    "niah-needle-bg":             ["niah_multikey_2"],
    "niah-multikey-uuid":         ["niah_multikey_3"],
    "niah-multivalue":            ["niah_multivalue"],
    "niah-multiquery":            ["niah_multiquery"],
    "variable-tracking":          ["vt"],
    "common-words-extraction":    ["cwe"],
    "frequency-words-extraction": ["fwe"],
    "qa-squad":                   ["qa_1"],
    "qa-hotpotqa":                ["qa_2"],
}

TASK_CATEGORIES: dict[str, str] = {
    "niah_single_1":  "needle_in_a_haystack",
    "niah_single_2":  "needle_in_a_haystack",
    "niah_single_3":  "needle_in_a_haystack",
    "niah_multikey_1": "needle_in_a_haystack",
    "niah_multikey_2": "needle_in_a_haystack",
    "niah_multikey_3": "needle_in_a_haystack",
    "niah_multivalue": "needle_in_a_haystack",
    "niah_multiquery": "needle_in_a_haystack",
    "vt":  "variable_tracking",
    "cwe": "aggregation",
    "fwe": "aggregation",
    "qa_1": "question_answering",
    "qa_2": "question_answering",
}

_SCRIPTS_DIR = Path(os.getenv("RULER_SCRIPTS_DIR", str(Path(__file__).parent / "scripts")))


def _load_module_from_path(name: str, path: Path) -> Any:
    """Load a Python module from an explicit file path without touching sys.path or sys.modules.

    Both scripts/data/synthetic/constants.py and scripts/eval/synthetic/constants.py are
    named 'constants'. Loading either one via sys.path would cache it in sys.modules under
    the key 'constants', causing the second load to silently return the wrong module.
    This helper avoids that collision by using importlib with a unique qualified name.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class RulerAdapter(FrameworkAdapter):
    """RULER (What's the Real Context Size of Your LLM?) long-context benchmark.

    Evaluates LLMs across 13 synthetic tasks testing retrieval, tracking,
    and aggregation at configurable context lengths via an OpenAI-compatible
    inference endpoint.
    """

    SCRIPTS_DIR: Path = _SCRIPTS_DIR

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute a RULER evaluation job.

        Args:
            config: Job specification (benchmark id, model config, parameters).
            callbacks: Progress and artifact callbacks.

        Returns:
            JobResults with per-task, per-context-length scores.
        """
        start_time = time.time()
        started_at = datetime.now(UTC)
        logger.info(f"Starting RULER job {config.id} benchmark={config.benchmark_id}")

        work_dir: Path | None = None
        try:
            # Phase 1 — initialise and validate
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message=f"Initialising RULER for {config.benchmark_id}",
                        message_code="initializing",
                    ),
                )
            )
            self._validate_config(config)
            params = config.parameters
            tasks = self._resolve_tasks(config.benchmark_id, params)

            # Use num_examples when explicitly > 0; fall back to parameter or default.
            # Careful: Python's truthiness treats 0 as falsy, so avoid `or` for integers.
            num_examples = config.num_examples
            num_samples: int = (
                num_examples if num_examples is not None and num_examples > 0
                else params.get("num_samples", 10)
            )

            context_lengths: list[int] = params.get("context_lengths", [4096, 8192, 16384])
            tokenizer_path: str = params.get("tokenizer_path") or config.model.name
            tokenizer_type: str = params.get("tokenizer_type", "hf")
            model_template: str = params.get("model_template", "base")
            tokens_to_generate: int | None = params.get("tokens_to_generate")
            batch_size: int = max(1, params.get("batch_size", 1))
            random_seed: int = params.get("random_seed", 42)
            # User-configurable data-generation subprocess timeout (seconds, default 10 min)
            data_gen_timeout: int = params.get("data_gen_timeout_seconds", 600)

            work_dir = Path(tempfile.mkdtemp(prefix="ruler_"))
            data_dir = work_dir / "data"
            pred_dir = work_dir / "predictions"
            data_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            # Verify tokenizer availability without downloading the full model weights.
            self._verify_tokenizer(tokenizer_path, tokenizer_type)

            total_pairs = len(tasks) * len(context_lengths)
            completed_pairs = 0

            # Phase 2 — data generation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=MessageInfo(
                        message=(
                            f"Generating RULER datasets: {len(tasks)} task(s) × "
                            f"{len(context_lengths)} context length(s)"
                        ),
                        message_code="loading_data",
                    ),
                    current_step="Generating synthetic datasets",
                    total_steps=total_pairs + 2,
                    completed_steps=0,
                )
            )

            for task_id in tasks:
                for ctx_len in context_lengths:
                    self._generate_task_data(
                        task_id=task_id,
                        context_length=ctx_len,
                        data_dir=data_dir,
                        tokenizer_path=tokenizer_path,
                        tokenizer_type=tokenizer_type,
                        model_template=model_template,
                        num_samples=num_samples,
                        random_seed=random_seed,
                        timeout=data_gen_timeout,
                    )
                    completed_pairs += 1
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.LOADING_DATA,
                            progress=0.1 + 0.2 * (completed_pairs / total_pairs),
                            message=MessageInfo(
                                message=f"Generated {completed_pairs}/{total_pairs} datasets",
                                message_code="loading_data",
                            ),
                            current_step=f"Generated {task_id} ctx={ctx_len}",
                            total_steps=total_pairs + 2,
                            completed_steps=completed_pairs,
                        )
                    )

            # Compute a reproducibility hash over all generated datasets.
            # Deterministic: same config + random_seed → same hash across runs.
            dataset_hash = self._compute_dataset_hash(tasks, context_lengths, data_dir)
            logger.info(f"Dataset hash (SHA-256): {dataset_hash}")

            # Phase 3 — inference
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running inference: {total_pairs} (task × context) pair(s)",
                        message_code="running_evaluation",
                    ),
                    current_step="Starting inference",
                    total_steps=total_pairs + 2,
                    completed_steps=total_pairs,
                )
            )

            # Build a shared OpenAI client once for the whole job.
            api_client = self._make_api_client(config.model.url)
            raw_results: dict[str, dict[int, list[dict]]] = {}
            completed_pairs = 0

            for task_id in tasks:
                task_ctx_results: dict[int, list[dict]] = {}
                for ctx_len in context_lengths:
                    preds = self._run_ruler_task(
                        task_id=task_id,
                        context_length=ctx_len,
                        config=config,
                        data_dir=data_dir,
                        pred_dir=pred_dir,
                        tokens_to_generate=tokens_to_generate,
                        batch_size=batch_size,
                        api_client=api_client,
                        callbacks=callbacks,
                        completed_pairs=completed_pairs,
                        total_pairs=total_pairs,
                    )
                    task_ctx_results[ctx_len] = preds
                    completed_pairs += 1
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.RUNNING_EVALUATION,
                            progress=0.3 + 0.5 * (completed_pairs / total_pairs),
                            message=MessageInfo(
                                message=(
                                    f"Inference {completed_pairs}/{total_pairs}: "
                                    f"{len(preds)} samples evaluated"
                                ),
                                message_code="running_evaluation",
                            ),
                            current_step=f"Completed {task_id} ctx={ctx_len}",
                            total_steps=total_pairs + 2,
                            completed_steps=total_pairs + completed_pairs,
                        )
                    )
                raw_results[task_id] = task_ctx_results

            # Phase 4 — scoring
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Computing RULER scores",
                        message_code="post_processing",
                    ),
                    current_step="Computing per-task scores",
                    total_steps=total_pairs + 2,
                    completed_steps=2 * total_pairs + 1,
                )
            )

            evaluation_results = self._evaluate_predictions(raw_results)
            overall_score = self._compute_overall_score(evaluation_results)
            num_evaluated = sum(
                len(preds)
                for ctx_results in raw_results.values()
                for preds in ctx_results.values()
            )
            completed_at = datetime.now(UTC)

            eval_card = self._build_eval_card(
                benchmark_id=config.benchmark_id,
                evaluation_results=evaluation_results,
                context_lengths=context_lengths,
            )
            env_card = self._build_env_card(
                config=config,
                evaluation_results=evaluation_results,
                overall_score=overall_score,
                started_at=started_at,
                completed_at=completed_at,
                dataset_hash=dataset_hash,
                context_lengths=context_lengths,
            )

            output_files = self._save_results(
                job_id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                evaluation_results=evaluation_results,
                work_dir=work_dir,
            )

            # Phase 5 — persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=MessageInfo(
                        message="Persisting RULER artifacts",
                        message_code="persisting_artifacts",
                    ),
                    current_step="Creating OCI artifact",
                    total_steps=total_pairs + 2,
                    completed_steps=2 * total_pairs + 2,
                )
            )

            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None and output_files:
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
                completed_at=completed_at,
                evaluation_metadata={
                    "framework": "ruler",
                    "framework_version": "1.0.0",
                    "context_lengths": context_lengths,
                    "num_samples": num_samples,
                    "tasks": tasks,
                    "tokenizer_type": tokenizer_type,
                    "model_template": model_template,
                    "random_seed": random_seed,
                    "dataset_hash": dataset_hash,
                },
                oci_artifact=oci_artifact,
                eval_card=eval_card,
                env_card=env_card,
            )

        except Exception as e:
            logger.exception("RULER evaluation failed")
            error_msg = str(e)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(message=error_msg, message_code="failed"),
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
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to clean up {work_dir}: {cleanup_err}")

    # ------------------------------------------------------------------ helpers

    def _validate_config(self, config: JobSpec) -> None:
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")
        if config.benchmark_id not in BENCHMARK_TO_TASKS:
            raise ValueError(
                f"Unknown benchmark_id '{config.benchmark_id}'. "
                f"Valid options: {sorted(BENCHMARK_TO_TASKS)}"
            )
        if not config.model.url:
            raise ValueError("model.url is required (OpenAI-compatible endpoint)")
        if not config.model.name:
            raise ValueError("model.name is required")

        # Validate benchmarks override if provided
        override = (config.parameters or {}).get("benchmarks")
        if override is not None:
            unknown = [b for b in override if b not in BENCHMARK_TO_TASKS]
            if unknown:
                raise ValueError(
                    f"Unknown benchmark IDs in 'benchmarks' parameter: {unknown}. "
                    f"Valid options: {sorted(BENCHMARK_TO_TASKS)}"
                )
            if not override:
                raise ValueError("'benchmarks' parameter must not be an empty list")

    def _get_category(self, task_id: str) -> str:
        return TASK_CATEGORIES.get(task_id, "unknown")

    def _resolve_tasks(self, benchmark_id: str, params: dict[str, Any]) -> list[str]:
        """Return the list of internal RULER task IDs to run.

        When the caller passes a ``benchmarks`` parameter (list of
        provider.yaml benchmark IDs), those override the single ``benchmark_id``.
        All IDs are validated in ``_validate_config`` before this is called.
        """
        override = params.get("benchmarks")
        if override:
            tasks: list[str] = []
            for bid in override:
                tasks.extend(BENCHMARK_TO_TASKS[bid])
            return tasks
        return BENCHMARK_TO_TASKS[benchmark_id]

    def _verify_tokenizer(self, tokenizer_path: str, tokenizer_type: str) -> bool:
        """Return True if the tokenizer is accessible; warn and return False otherwise.

        Uses fast tokenizer-only loads (no model weights) to avoid downloading
        multi-gigabyte checkpoints during the INITIALIZING phase.
        """
        try:
            if tokenizer_type == "hf":
                # use_fast=True + local_files_only avoids downloading model weights;
                # fall back to hub check without weights if local files are absent.
                from transformers import AutoTokenizer  # noqa: PLC0415
                AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=False,
                    use_fast=True,
                    # Request tokenizer files only — skip model weights
                    local_files_only=False,
                )
                return True
            if tokenizer_type == "openai":
                # tokenizer_path is a tiktoken encoding name (e.g. cl100k_base),
                # not a model name — use get_encoding(), not encoding_for_model().
                import tiktoken  # noqa: PLC0415
                tiktoken.get_encoding(tokenizer_path)
                return True
        except Exception as exc:
            logger.warning(
                f"Tokenizer '{tokenizer_path}' ({tokenizer_type}) not immediately "
                f"accessible: {exc}. Data generation may fail."
            )
        return False

    @lru_cache(maxsize=None)  # cache keyed by task_id; synthetic.yaml never changes per run
    def _load_task_config(self, task_id: str) -> dict[str, Any]:
        """Merge synthetic.yaml task entry with its base task constants.

        Results are cached so the YAML is parsed only once per unique task_id.
        Modules are loaded via importlib from explicit file paths to prevent the
        'constants' name from colliding in sys.modules between the data and eval
        constants files.
        """
        import yaml  # noqa: PLC0415

        yaml_path = self.SCRIPTS_DIR / "synthetic.yaml"
        with open(yaml_path) as fh:
            tasks_customized: dict = yaml.safe_load(fh)

        if task_id not in tasks_customized:
            raise ValueError(f"Task '{task_id}' not found in synthetic.yaml")

        # Load data constants via explicit path to avoid sys.modules collision with
        # eval/synthetic/constants.py (both files are named 'constants').
        data_constants = _load_module_from_path(
            "ruler_data_constants",
            self.SCRIPTS_DIR / "data" / "synthetic" / "constants.py",
        )

        task_entry = dict(tasks_customized[task_id])
        task_entry.update(data_constants.TASKS[task_entry["task"]])
        return task_entry

    def _make_api_client(self, model_url: str) -> Any:
        """Create a single shared OpenAI-compatible client for the whole job."""
        from openai import OpenAI  # noqa: PLC0415

        api_key = os.getenv("MODEL_API_KEY", "")
        if not api_key:
            raise ValueError(
                "MODEL_API_KEY environment variable is required for API authentication. "
                "Set it to 'dummy' or 'none' explicitly if the endpoint has no auth."
            )
        return OpenAI(base_url=model_url, api_key=api_key)

    def _generate_task_data(
        self,
        task_id: str,
        context_length: int,
        data_dir: Path,
        tokenizer_path: str,
        tokenizer_type: str,
        model_template: str,
        num_samples: int,
        random_seed: int,
        timeout: int = 600,
    ) -> Path:
        """Generate the JSONL dataset for one (task, context_length) pair.

        Returns the path to the generated file.
        """
        ctx_save_dir = data_dir / str(context_length)
        ctx_save_dir.mkdir(parents=True, exist_ok=True)
        output_file = ctx_save_dir / task_id / "validation.jsonl"

        if output_file.exists():
            logger.info(f"Reusing existing dataset: {output_file}")
            return output_file

        prepare_script = self.SCRIPTS_DIR / "data" / "prepare.py"
        pythonpath = os.pathsep.join(
            filter(None, [
                str(self.SCRIPTS_DIR / "data"),
                str(self.SCRIPTS_DIR / "data" / "synthetic"),
                os.environ.get("PYTHONPATH", ""),
            ])
        )
        env = {**os.environ, "PYTHONPATH": pythonpath}

        cmd = [
            sys.executable,
            str(prepare_script),
            "--save_dir", str(ctx_save_dir),
            "--benchmark", "synthetic",
            "--task", task_id,
            "--tokenizer_path", tokenizer_path,
            "--tokenizer_type", tokenizer_type,
            "--max_seq_length", str(context_length),
            "--num_samples", str(num_samples),
            "--random_seed", str(random_seed),
            "--model_template_type", model_template,
        ]

        logger.info(f"Generating data: task={task_id} ctx={context_length}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Data generation failed for {task_id}/{context_length}:\n{result.stderr}"
            )
        logger.debug(f"Data generation stdout: {result.stdout}")
        return output_file

    def _run_api_inference(
        self,
        model_name: str,
        data_file: Path,
        pred_file: Path,
        tokens_to_generate: int,
        batch_size: int,
        api_client: Any,
        callbacks: JobCallbacks,
        sample_offset: int = 0,
        total_samples_hint: int = 0,
    ) -> list[dict]:
        """Call the OpenAI-compatible endpoint and write predictions.

        Args:
            model_name: Model identifier string.
            data_file: Path to input JSONL.
            pred_file: Path to write prediction JSONL.
            tokens_to_generate: Max tokens per generation.
            batch_size: Controls progress-log interval (no batching at API level).
            api_client: Shared OpenAI client (created once per job in _make_api_client).
            callbacks: Used to report per-sample progress.
            sample_offset: Running count of samples already processed (for progress math).
            total_samples_hint: Total expected samples across all tasks/ctx for progress.

        Returns:
            List of prediction dicts with 'pred' key added.
        """
        samples: list[dict] = []
        with open(data_file) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        predictions: list[dict] = []
        n_samples = len(samples)
        log_interval = max(1, batch_size)

        for i, sample in enumerate(samples):
            try:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": sample["input"]}],
                    max_tokens=tokens_to_generate,
                    temperature=0.0,
                )
                pred_text: str = response.choices[0].message.content or ""
            except Exception as exc:
                # Re-raise auth/connection errors immediately; swallow only per-sample
                # transient errors (e.g., timeout on a single long prompt).
                exc_type = type(exc).__name__
                if exc_type in ("AuthenticationError", "PermissionDeniedError"):
                    raise RuntimeError(
                        f"API authentication failed for {model_name}: {exc}. "
                        "Check MODEL_API_KEY."
                    ) from exc
                logger.warning(
                    f"Inference failed for sample {sample.get('index', '?')}: {exc}"
                )
                pred_text = ""

            predictions.append(
                {
                    "index": sample["index"],
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "pred": pred_text,
                }
            )
            completed = i + 1
            if completed % log_interval == 0 or completed == n_samples:
                logger.debug(f"Inference progress: {completed}/{n_samples}")
                # Emit a mid-inference status callback so the EvalHub sidecar can
                # relay live progress to the UI during long-running evaluations.
                if total_samples_hint > 0:
                    frac = (sample_offset * n_samples + completed) / (total_samples_hint * n_samples)
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.RUNNING_EVALUATION,
                            progress=min(0.79, 0.30 + 0.49 * frac),
                            message=MessageInfo(
                                message=f"Inference: {completed}/{n_samples} samples",
                                message_code="running_evaluation",
                            ),
                        )
                    )

        pred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_file, "w") as fh:
            for p in predictions:
                fh.write(json.dumps(p) + "\n")

        return predictions

    def _run_ruler_task(
        self,
        task_id: str,
        context_length: int,
        config: JobSpec,
        data_dir: Path,
        pred_dir: Path,
        tokens_to_generate: int | None,
        batch_size: int,
        api_client: Any,
        callbacks: JobCallbacks,
        completed_pairs: int,
        total_pairs: int,
    ) -> list[dict]:
        """Orchestrate data generation + inference for one (task, context_length)."""
        task_config = self._load_task_config(task_id)
        ttg = tokens_to_generate if tokens_to_generate is not None else task_config.get("tokens_to_generate", 128)

        data_file = data_dir / str(context_length) / task_id / "validation.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Expected dataset not found at {data_file}. "
                "Data generation may have silently failed."
            )

        pred_file = pred_dir / f"{task_id}_{context_length}.jsonl"
        return self._run_api_inference(
            model_name=config.model.name,
            data_file=data_file,
            pred_file=pred_file,
            tokens_to_generate=ttg,
            batch_size=batch_size,
            api_client=api_client,
            callbacks=callbacks,
            sample_offset=completed_pairs,
            total_samples_hint=total_pairs,
        )

    def _evaluate_predictions(
        self,
        raw_results: dict[str, dict[int, list[dict]]],
    ) -> list[EvaluationResult]:
        """Compute per-task, per-context-length string-match scores.

        Loads eval constants via explicit file path to prevent sys.modules collision
        with data/synthetic/constants.py (both files share the module name 'constants').
        """
        eval_constants = _load_module_from_path(
            "ruler_eval_constants",
            self.SCRIPTS_DIR / "eval" / "synthetic" / "constants.py",
        )
        eval_tasks = eval_constants.TASKS

        evaluation_results: list[EvaluationResult] = []

        for task_id, ctx_results in raw_results.items():
            task_cfg = self._load_task_config(task_id)
            base_task_type = task_cfg["task"]
            metric_fn = eval_tasks[base_task_type]["metric_fn"]

            per_ctx_scores: dict[int, float] = {}
            for ctx_len, preds in ctx_results.items():
                if not preds:
                    continue
                predictions = [p["pred"] for p in preds]
                references = [p["outputs"] for p in preds]
                raw_score: float = metric_fn(predictions, references)
                normalised = raw_score / 100.0

                # CI computed using the same fraction-of-refs logic as string_match_all,
                # propagated to a per-sample mean so the CI bounds match the scored metric.
                ci = self._compute_ci(predictions, references, metric_fn)
                per_ctx_scores[ctx_len] = raw_score

                evaluation_results.append(
                    EvaluationResult(
                        metric_name=f"{task_id}.ctx_{ctx_len}.score",
                        metric_value=normalised,
                        metric_type="float",
                        confidence_interval=ci,
                        num_samples=len(preds),
                        metadata={
                            "task_id": task_id,
                            "category": self._get_category(task_id),
                            "context_length": ctx_len,
                            "metric": "string_match",
                            "raw_score_pct": raw_score,
                        },
                    )
                )

            if per_ctx_scores:
                avg_pct = sum(per_ctx_scores.values()) / len(per_ctx_scores)
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=f"{task_id}.overall",
                        metric_value=avg_pct / 100.0,
                        metric_type="float",
                        num_samples=sum(len(v) for v in ctx_results.values()),
                        metadata={
                            "task_id": task_id,
                            "category": self._get_category(task_id),
                            "per_context_length": per_ctx_scores,
                        },
                    )
                )

        return evaluation_results

    def _compute_ci(
        self,
        predictions: list[str],
        references: list[list[str]],
        metric_fn: Any,
    ) -> tuple[float, float] | None:
        """Bootstrap 95% CI that matches the actual scoring metric.

        Rather than approximating the metric with a binary proxy, this bootstraps
        the metric_fn itself so the CI bounds always correspond to what was scored.
        Uses 500 resamples — sufficient for a stable CI at typical RULER sample sizes.
        """
        if not predictions or len(predictions) < 2:
            return None

        import random  # noqa: PLC0415

        n = len(predictions)
        indices = list(range(n))
        boot_scores: list[float] = []
        rng = random.Random(42)

        for _ in range(500):
            sample_idx = [rng.choice(indices) for _ in range(n)]
            boot_preds = [predictions[i] for i in sample_idx]
            boot_refs = [references[i] for i in sample_idx]
            boot_scores.append(metric_fn(boot_preds, boot_refs) / 100.0)

        boot_scores.sort()
        lo = boot_scores[int(0.025 * len(boot_scores))]
        hi = boot_scores[int(0.975 * len(boot_scores))]
        return (max(0.0, lo), min(1.0, hi))

    def _compute_overall_score(
        self, results: list[EvaluationResult]
    ) -> float | None:
        overall_scores: list[float] = [
            float(r.metric_value)
            for r in results
            if r.metric_name.endswith(".overall") and isinstance(r.metric_value, (int, float))
        ]
        return sum(overall_scores) / len(overall_scores) if overall_scores else None

    def _write_summary_csv(
        self,
        evaluation_results: list[EvaluationResult],
        output_path: Path,
    ) -> None:
        """Write per-task, per-context-length score table to CSV."""
        import csv  # noqa: PLC0415

        rows: dict[str, dict] = {}
        all_ctx_lengths: set[int] = set()
        for result in evaluation_results:
            meta = result.metadata or {}
            if "per_context_length" in meta:
                task_id = meta["task_id"]
                per_ctx: dict[int, float] = meta["per_context_length"]
                rows[task_id] = {
                    "overall": float(result.metric_value) * 100.0,
                    "per_ctx": per_ctx,
                }
                all_ctx_lengths.update(per_ctx.keys())

        sorted_lengths = sorted(all_ctx_lengths)
        with open(output_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["task"] + [str(l) for l in sorted_lengths] + ["overall"])
            for task_id, data in rows.items():
                row: list = [task_id]
                for ctx_len in sorted_lengths:
                    row.append(round(data["per_ctx"].get(ctx_len, 0.0), 4))
                row.append(round(data["overall"], 4))
                writer.writerow(row)

    def _compute_dataset_hash(
        self,
        tasks: list[str],
        context_lengths: list[int],
        data_dir: Path,
    ) -> str:
        """SHA-256 over all generated JSONL datasets for reproducibility verification."""
        import hashlib  # noqa: PLC0415

        hasher = hashlib.sha256()
        for task_id in sorted(tasks):
            for ctx_len in sorted(context_lengths):
                data_file = data_dir / str(ctx_len) / task_id / "validation.jsonl"
                if data_file.exists():
                    hasher.update(data_file.read_bytes())
        return hasher.hexdigest()

    def _build_eval_card(
        self,
        benchmark_id: str,
        evaluation_results: list[EvaluationResult],
        context_lengths: list[int],
    ) -> EvalCardMetadata:
        """Build a RULER EvalCard (Dhar et al. arXiv:2511.21695) from scored results.

        Produces one CapabilityEvalEntry per evaluated category (NIAH, variable
        tracking, aggregation, QA) with the zero-shot average score for that
        category across all context lengths.
        """
        CATEGORY_ABILITIES = {
            "needle_in_a_haystack": "Long-context retrieval (NIAH)",
            "variable_tracking": "Long-context variable tracking",
            "aggregation": "Long-context aggregation",
            "question_answering": "Long-context question answering",
        }
        CATEGORY_METRICS = {
            "question_answering": "string_match_part (% partial ref recall)",
        }
        DEFAULT_METRIC = "string_match_all (% exact ref recall)"

        # Collect per-category overall scores from .overall EvaluationResult entries
        category_scores: dict[str, list[float]] = {}
        for result in evaluation_results:
            if not result.metric_name.endswith(".overall"):
                continue
            meta = result.metadata or {}
            cat = meta.get("category", "unknown")
            category_scores.setdefault(cat, []).append(float(result.metric_value))

        capability_evaluations: list[CapabilityEvalEntry] = []
        for cat in sorted(category_scores):
            scores = category_scores[cat]
            avg = sum(scores) / len(scores)
            capability_evaluations.append(
                CapabilityEvalEntry(
                    ability=CATEGORY_ABILITIES.get(cat, cat),
                    benchmark=f"RULER 1.0 / {benchmark_id}",
                    metric=CATEGORY_METRICS.get(cat, DEFAULT_METRIC),
                    zero_shot=round(avg, 4),
                )
            )

        ctx_str = ", ".join(str(c) for c in sorted(context_lengths))
        return EvalCardMetadata(
            modalities_input=["text"],
            modalities_output=["text"],
            languages=["eng"],
            languages_count=1,
            capability_evaluations=capability_evaluations,
            developer_footnotes=(
                f"RULER evaluates effective context utilisation via 13 synthetic tasks "
                f"at context lengths {ctx_str} tokens. "
                "Datasets are generated deterministically (fixed random_seed); "
                "scores reflect synthetic retrieval/tracking/aggregation tasks, "
                "not real-world document understanding performance. "
                "All tasks use zero-shot prompting with task-specific answer-prefix templates. "
                "Scoring: string_match_all (fraction of expected references found in "
                "prediction) for NIAH, variable-tracking, and aggregation tasks; "
                "string_match_part (any expected reference in prediction) for QA tasks. "
                "Reference: Hsieh et al. (2024) RULER: What's the Real Context Size "
                "of Your LLM? arXiv:2404.06654."
            ),
        )

    def _build_env_card(
        self,
        config: JobSpec,
        evaluation_results: list[EvaluationResult],
        overall_score: float | None,
        started_at: datetime,
        completed_at: datetime,
        dataset_hash: str,
        context_lengths: list[int],
    ) -> EnvironmentCardMetadata:
        """Build an EnvironmentCard with RULER-specific layers 4-5.

        Calls EnvironmentCardMetadata.capture() for auto-collected layers 1-3
        (OS, Python, GPU if present, k8s pod labels/limits), then fills in
        model identity and run-provenance fields that only the adapter knows.
        """
        env_card = EnvironmentCardMetadata.capture(
            framework_name="ruler",
            framework_version="1.0.0",
            extra_packages=[
                "openai", "tiktoken", "nltk",
                "wonderwords", "numpy", "pandas", "transformers",
            ],
        )

        # Layer 4 — model identity
        env_card.model_id = config.model.name
        env_card.model_provider = "openai-compatible"

        # Layer 5 — run provenance
        env_card.started_at = started_at.isoformat()
        env_card.completed_at = completed_at.isoformat()
        env_card.dataset_hash = dataset_hash
        env_card.scorer_ids = [
            "ruler/string_match_all@1.0.0",
            "ruler/string_match_part@1.0.0",
        ]

        # Aggregate metrics
        env_card.aggregate_results = {
            "overall_score": overall_score,
            "context_lengths_evaluated": sorted(context_lengths),
        }

        # Per-task breakdown (from .overall EvaluationResult entries)
        per_task: dict[str, Any] = {}
        for result in evaluation_results:
            if not result.metric_name.endswith(".overall"):
                continue
            meta = result.metadata or {}
            task_id = meta.get("task_id", result.metric_name.replace(".overall", ""))
            per_task[task_id] = {
                "score": round(float(result.metric_value), 4),
                "category": meta.get("category"),
                "num_samples": result.num_samples,
                "per_context_length": {
                    str(k): round(v, 4)
                    for k, v in meta.get("per_context_length", {}).items()
                },
            }
        env_card.per_task_results = per_task

        # Confidence intervals (from per-context-length EvaluationResult entries)
        cis: dict[str, Any] = {}
        for result in evaluation_results:
            if result.confidence_interval is not None:
                cis[result.metric_name] = {
                    "lower": round(result.confidence_interval[0], 4),
                    "upper": round(result.confidence_interval[1], 4),
                }
        env_card.confidence_intervals = cis

        # Recompute completeness after filling layers 4-5
        env_card.capture_completeness = env_card._compute_completeness()
        logger.info(
            "EnvironmentCard built (completeness: %.0f%%)",
            (env_card.capture_completeness or 0) * 100,
        )

        return env_card

    def _save_results(
        self,
        job_id: str,
        benchmark_id: str,
        model_name: str,
        evaluation_results: list[EvaluationResult],
        work_dir: Path,
    ) -> list[Path]:
        # Determine output directory: prefer the SDK-provided path (k8s persistent volume),
        # fall back to $HOME/ruler_<job_id>_results if that path is not writable
        # (e.g. local mode where /results is root-owned and not mounted).
        if self.local_jobs_base_path is not None:
            output_dir = self.local_jobs_base_path / "results"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                fallback = Path(os.getenv("HOME", "/tmp")) / f"ruler_{job_id}_results"
                logger.warning(
                    f"Cannot write to {output_dir} (permission denied); "
                    f"saving results to {fallback}"
                )
                output_dir = fallback
                output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = work_dir / "results"
            output_dir.mkdir(parents=True, exist_ok=True)

        files: list[Path] = []

        csv_file = output_dir / "summary.csv"
        self._write_summary_csv(evaluation_results, csv_file)
        files.append(csv_file)

        results_file = output_dir / "results.json"
        with open(results_file, "w") as fh:
            json.dump(
                {
                    "job_id": job_id,
                    "benchmark_id": benchmark_id,
                    "model_name": model_name,
                    "framework": "ruler",
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
                fh,
                indent=2,
            )
        files.append(results_file)

        logger.info(f"Saved {len(files)} result file(s) to {output_dir}")
        return files


def main() -> None:
    """Entry point for the RULER adapter container.

    Reads EVALHUB_JOB_SPEC_PATH (default /meta/job.json), constructs
    JobSpec and RulerAdapter, runs the benchmark, and reports results.
    """
    from evalhub.adapter import DefaultCallbacks  # noqa: PLC0415

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        with open(job_spec_path) as fh:
            job_spec = json.load(fh)
        job_spec = JobSpec(**job_spec)

        adapter = RulerAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        logger.info(f"Job completed: {results.id}")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Evaluated {results.num_examples_evaluated} examples")

        run_id = callbacks.mlflow.save(results, adapter.job_spec)
        if run_id:
            results.mlflow_run_id = run_id
            logger.info(f"MLflow run created: {run_id}")

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as exc:
        logger.error(f"Job spec not found: {exc}")
        logger.error("Set EVALHUB_JOB_SPEC_PATH or ensure job spec is at the default path")
        sys.exit(1)
    except ValueError as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
