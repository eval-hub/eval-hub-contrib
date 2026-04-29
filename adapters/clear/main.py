#!/usr/bin/env python3
"""IBM CLEAR agentic adapter for eval-hub.

Loads a JobSpec, resolves trace JSON input, runs CLEAR's step-by-step agentic
pipeline (prepare traces → judge / analyze → clear_results.json), then maps
CLEAR output to evalhub-sdk JobResults and optional MLflow or OCI artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
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
    OCIArtifactResult,
    OCIArtifactSpec,
)
from evalhub.adapter.auth import resolve_model_credentials
from evalhub.adapter.mlflow import MlflowArtifact

try:
    from clear_eval.agentic.pipeline.run_clear_agentic_eval import (
        create_output_structure,
        prepare_traces_data,
        run_step_by_step_pipeline,
    )
    from clear_eval.agentic.pipeline.utils import get_run_output_dir
except ImportError as exc:
    raise ImportError(
        "Install IBM CLEAR from source (see requirements.txt; PyPI wheels omit agentic)."
    ) from exc

logger = logging.getLogger(__name__)


def _local_only_run() -> bool:
    """True when running without Eval Hub callbacks or MLflow (local iteration on job.json)."""
    mode = os.getenv("EVALHUB_MODE", "").strip().lower()
    if mode == "local":
        return True
    return os.getenv("CLEAR_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")


def _callbacks_for_adapter(adapter: FrameworkAdapter) -> JobCallbacks:
    """DefaultCallbacks with no sidecar when local; otherwise mirror job_spec.callback_url."""
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


def _run_clear_unified_pipeline(agentic_config: dict[str, Any], output_dir: Path) -> None:
    """Drive IBM CLEAR's agentic layout: output dirs, trace prep, step-by-step run."""
    results_dir = agentic_config["results_dir"]
    run_name = agentic_config["run_name"]
    resolved_out, _ = get_run_output_dir(results_dir, run_name)
    if resolved_out.resolve() != output_dir.resolve():
        logger.warning(
            "CLEAR output path mismatch: expected %s, resolved %s", output_dir, resolved_out
        )
    base = resolved_out
    output_paths = create_output_structure(base)
    data_dir = Path(agentic_config["data_dir"])
    from_raw = bool(agentic_config.get("from_raw_traces", True))
    traces_data_dir = prepare_traces_data(data_dir, from_raw, output_paths, agentic_config)
    if not traces_data_dir:
        raise RuntimeError("CLEAR failed to prepare traces_data")
    ok = run_step_by_step_pipeline(
        traces_data_dir, output_paths["step_by_step"], agentic_config
    )
    if not ok:
        raise RuntimeError("CLEAR step-by-step pipeline reported failure")


def _find_clear_results_json(output_dir: Path, eval_model_name: str) -> Path | None:
    """Locate clear_results.json after a run (preferred step_by_step path, then fallbacks)."""
    unified = (
        output_dir / "step_by_step" / "clear_results" / eval_model_name / "clear_results.json"
    )
    if unified.is_file():
        return unified
    # Agentic pipeline writes here before cleanup (see save_comprehensive_json_results).
    step_flat = output_dir / "step_by_step" / "clear_results.json"
    if step_flat.is_file():
        return step_flat
    legacy = output_dir / "clear_results" / eval_model_name / "clear_results.json"
    if legacy.is_file():
        return legacy
    flat = output_dir / "clear_results.json"
    if flat.is_file():
        return flat
    matches = sorted(output_dir.rglob("clear_results.json"))
    return matches[0] if matches else None


def _preserve_html_reports_from_clear_output(output_dir: Path) -> list[Path]:
    """
    Preserve CLEAR HTML artifacts before cleanup.

    CLEAR may write a static dashboard (e.g., `step_by_step/clear_results.html`) and other
    HTML files under intermediate directories. We copy key files to the run root so they
    survive adapter cleanup and can be shipped via MLflow/OCI.
    """
    saved: list[Path] = []

    # Keep the static dashboard "next to JSON" at the run root.
    step_static = output_dir / "step_by_step"
    for name in ("clear_results.html", "clear_results.dashboard_data.json"):
        src = step_static / name
        if src.is_file():
            dest = output_dir / name
            try:
                shutil.copy2(src, dest)
                saved.append(dest)
                logger.info("Preserved CLEAR dashboard: %s -> %s", src, dest.name)
            except OSError as exc:
                logger.warning("Could not preserve %s: %s", src, exc)

    for sub in ("step_by_step", "full_trajectory", "traces_data", "clear_data", "clear_results"):
        base = output_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.html")):
            if not p.is_file():
                continue
            if p.name == "clear_results.html" and p.parent == step_static:
                # Already copied above with a stable name.
                continue
            rel = p.relative_to(output_dir)
            tag = "__".join(rel.parts).replace(" ", "_")
            if len(tag) > 200:
                tag = tag[:180] + "__etc"
            dest = output_dir / f"clear_html__{tag}"
            try:
                shutil.copy2(p, dest)
                saved.append(dest)
                logger.info("Preserved CLEAR HTML: %s -> %s", p, dest.name)
            except OSError as exc:
                logger.warning("Could not preserve HTML %s: %s", p, exc)
    return saved


class ClearAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter that runs IBM CLEAR on trace JSON and returns JobResults."""

    def __init__(self, job_spec_path: Optional[str] = None) -> None:
        super().__init__(job_spec_path=job_spec_path)

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Execute one CLEAR job: validate, run pipeline, extract metrics, callbacks, return results."""
        start_time = time.time()
        logger.info(f"Starting CLEAR job {config.id} for benchmark {config.benchmark_id}")

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing CLEAR for agentic evaluation",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)

            data_dir: str | None = None
            test_data_path = Path("/test_data")
            if test_data_path.exists() and any(test_data_path.iterdir()):
                traces_sub = test_data_path / "traces"
                if traces_sub.exists() and any(traces_sub.iterdir()):
                    data_dir = "/test_data/traces"
                    logger.info("Using traces from /test_data/traces")
                elif any(test_data_path.glob("*.json")):
                    data_dir = "/test_data"
                    logger.info("Using traces from /test_data")
                else:
                    json_files = list(test_data_path.rglob("*.json"))
                    if json_files:
                        data_dir = str(json_files[0].parent)
                        logger.info("Using traces from %s (nested JSON)", data_dir)
                    else:
                        logger.warning(
                            "/test_data is non-empty but no JSON traces found: %s",
                            [c.name for c in test_data_path.iterdir()],
                        )

            if not data_dir:
                data_path = Path("/data")
                if data_path.exists() and any(data_path.iterdir()):
                    if (data_path / "traces").exists() and any((data_path / "traces").iterdir()):
                        data_dir = "/data/traces"
                        logger.info("Using S3-mounted data from /data/traces directory")
                    elif any(data_path.glob("*.json")):
                        data_dir = "/data"
                        logger.info("Using traces from /data")
                    else:
                        logger.warning(
                            "/data exists but no trace files: %s",
                            [c.name for c in data_path.iterdir()],
                        )

            if not data_dir:
                data_dir = config.parameters.get("data_dir") or config.parameters.get(
                    "traces_input_dir"
                )
                if not data_dir:
                    raise ValueError(
                        "No input traces: mount data under /test_data or /data, "
                        "or set parameters.data_dir (preferred) or parameters.traces_input_dir"
                    )
                logger.info("Using data_dir from parameters: %s", data_dir)
                # S3 / init staging often appears after the adapter process starts; wait before failing.
                _param_path = Path(data_dir)
                _deadline = time.monotonic() + 120.0
                while time.monotonic() < _deadline:
                    if _param_path.is_dir() and any(_param_path.glob("*.json")):
                        break
                    logger.info(
                        "Waiting for parameters.data_dir to contain *.json: %s", data_dir
                    )
                    time.sleep(3)
                else:
                    raise ValueError(
                        f"parameters.data_dir {data_dir!r} missing or has no *.json after 120s "
                        f"(confirm S3 staging path with: find /test_data /data -name '*.json')"
                    )

            if not Path(data_dir).exists():
                raise ValueError(f"data_dir not found: {data_dir}")

            trace_files = list(Path(data_dir).glob("*.json"))
            if not trace_files:
                raise ValueError(f"No JSON trace files found in {data_dir}")

            logger.info("Found %d trace file(s) in %s", len(trace_files), data_dir)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.2,
                    message=MessageInfo(
                        message="Processing MLflow traces",
                        message_code="loading_data",
                    ),
                )
            )

            results_dir_param = config.parameters.get("results_dir")
            run_name_param = config.parameters.get("run_name")
            if results_dir_param and run_name_param:
                output_dir = Path(results_dir_param) / str(run_name_param)
                output_dir.mkdir(parents=True, exist_ok=True)
            elif os.getenv("EVALHUB_MODE") == "k8s":
                output_dir = Path(tempfile.mkdtemp(prefix="clear_output_"))
            else:
                output_dir = Path(__file__).parent / "output"
                output_dir.mkdir(exist_ok=True)
            logger.info("CLEAR output base: %s", output_dir)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.4,
                    message=MessageInfo(
                        message="Running CLEAR agentic pipeline",
                        message_code="running_evaluation",
                    ),
                )
            )

            if config.model.url and config.parameters.get("inference_backend") != "endpoint":
                os.environ["OPENAI_BASE_URL"] = config.model.url
                self._ensure_openai_api_key_for_litellm()

            agentic_config = self._build_agentic_config(config, data_dir, output_dir)
            _run_clear_unified_pipeline(agentic_config, output_dir)

            eval_model_name = config.parameters.get("eval_model_name", "default").split("/")[-1]
            json_results_path = _find_clear_results_json(output_dir, eval_model_name)
            if not json_results_path:
                raise FileNotFoundError(
                    f"clear_results.json not found under {output_dir} "
                    f"(expected step_by_step/clear_results/{eval_model_name}/ or legacy paths)"
                )
            logger.info("Results file: %s", json_results_path)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Processing CLEAR results",
                        message_code="post_processing",
                    ),
                )
            )

            evaluation_results = self._extract_agentic_results(str(json_results_path))
            overall_score = self._compute_overall_score(evaluation_results)
            num_evaluated = self._extract_num_evaluated(str(json_results_path))

            metrics_summary = self._metrics_summary_dict(
                config, evaluation_results, overall_score, num_evaluated
            )

            clear_html_artifacts = _preserve_html_reports_from_clear_output(output_dir)
            self._cleanup_intermediate_files(output_dir, str(json_results_path))

            final_results_path = output_dir / "clear_results.json"
            oci_artifact = self._report_artifacts(
                config=config,
                callbacks=callbacks,
                json_results_path=final_results_path,
                evaluation_results=evaluation_results,
                overall_score=overall_score,
                num_evaluated=num_evaluated,
                extra_files=clear_html_artifacts,
            )

            duration = time.time() - start_time
            job_results = JobResults(
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
                    "framework": "clear",
                    "data_dir": data_dir,
                    "output_dir": str(output_dir),
                },
                oci_artifact=oci_artifact,
            )
            rid = self._save_results_to_mlflow(
                callbacks, config, job_results, output_dir, metrics_summary, clear_html_artifacts
            )
            if rid:
                job_results = job_results.model_copy(update={"mlflow_run_id": rid})
            return job_results

        except Exception as exc:
            logger.exception("CLEAR evaluation failed")
            error_msg = str(exc)
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
                        "exception_type": type(exc).__name__,
                        "benchmark_id": config.benchmark_id,
                    },
                )
            )
            raise

    def _validate_config(self, config: JobSpec) -> None:
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        has_test_data = Path("/test_data").exists() and any(Path("/test_data").iterdir())
        has_data = Path("/data").exists() and any(Path("/data").iterdir())
        has_traces_param = (
            "data_dir" in config.parameters or "traces_input_dir" in config.parameters
        )

        if not has_test_data and not has_data and not has_traces_param:
            raise ValueError(
                "Provide parameters.data_dir (or traces_input_dir) or mount test data under "
                "/test_data or /data"
            )

        if "eval_model_name" not in config.parameters:
            raise ValueError("eval_model_name is required in parameters")

        if "provider" not in config.parameters:
            raise ValueError("provider is required in parameters")

        self._validate_benchmark_contract(config)

    def _validate_benchmark_contract(self, config: JobSpec) -> None:
        """Enforce required parameters for catalog benchmark ids (see provider.yaml)."""
        bid = config.benchmark_id or ""
        params = config.parameters or {}

        if bid == "agentic-evaluation-custom-criteria":
            ec = params.get("evaluation_criteria")
            if not ec or not isinstance(ec, dict) or len(ec) == 0:
                raise ValueError(
                    "benchmark_id agentic-evaluation-custom-criteria requires a non-empty "
                    "parameters.evaluation_criteria object (criterion name -> description)."
                )

        if bid == "agentic-evaluation-predefined-issues":
            pi = params.get("predefined_issues")
            if not pi or not isinstance(pi, list) or len(pi) == 0:
                raise ValueError(
                    "benchmark_id agentic-evaluation-predefined-issues requires a non-empty "
                    "parameters.predefined_issues list of strings."
                )

    @staticmethod
    def _ensure_openai_api_key_for_litellm() -> None:
        """Set OPENAI_API_KEY from env or from model auth secret mount (not job parameters)."""
        if os.getenv("OPENAI_API_KEY"):
            return
        creds = resolve_model_credentials()
        if creds.api_key:
            os.environ["OPENAI_API_KEY"] = creds.api_key
            return
        logger.info("OPENAI_API_KEY not set; some OpenAI-compatible gateways still work without it")

    def _build_agentic_config(
        self, config: JobSpec, data_dir: str, output_dir: Path
    ) -> dict[str, Any]:
        agentic_config = {
            "data_dir": data_dir,
            "results_dir": str(output_dir.parent),
            "run_name": output_dir.name,
            "from_raw_traces": config.parameters.get("from_raw_traces", True),
            "run_step_by_step": config.parameters.get("run_step_by_step", True),
            "run_full_trajectory": config.parameters.get("run_full_trajectory", False),
            "provider": config.parameters["provider"],
            "eval_model_name": config.parameters["eval_model_name"],
            "agent_framework": config.parameters.get("agent_framework", "langgraph"),
            "observability_framework": config.parameters.get("observability_framework", "mlflow"),
            "separate_tools": config.parameters.get("separate_tools", False),
            "overwrite": config.parameters.get("overwrite", True),
            "resume_enabled": False,
            "agent_mode": True,
            "task": config.parameters.get("task", "general"),
            "inference_backend": config.parameters.get("inference_backend", "litellm"),
            "perform_generation": config.parameters.get("perform_generation", False),
            "is_reference_based": config.parameters.get("is_reference_based", False),
            "reference_column": config.parameters.get("reference_column", "ground_truth"),
            "model_output_column": config.parameters.get("model_output_column", "response"),
            "model_input_column": config.parameters.get("model_input_column", "model_input"),
            "question_column": config.parameters.get("question_column", "question"),
            "qid_column": config.parameters.get("qid_column", "id"),
            "max_examples_to_analyze": config.parameters.get("max_examples_to_analyze", None),
            "use_general_prompt": config.parameters.get("use_general_prompt", True),
            "perform_clustering": config.parameters.get("perform_clustering", True),
            "use_full_text_for_analysis": config.parameters.get("use_full_text_for_analysis", False),
            "max_workers": config.parameters.get("max_workers", 20),
            "max_shortcomings": config.parameters.get("max_shortcomings", 15),
            "min_shortcomings": config.parameters.get("min_shortcomings", 3),
            "max_eval_text_for_synthesis": config.parameters.get("max_eval_text_for_synthesis", 1000),
        }

        if agentic_config.get("inference_backend") == "endpoint":
            if "endpoint_url" in config.parameters:
                ep = config.parameters["endpoint_url"]
            elif "inference_url" in config.parameters:
                ep = config.parameters["inference_url"]
            elif config.model.url:
                ep = config.model.url
            else:
                raise ValueError(
                    "model.url or parameters.endpoint_url (or inference_url) is required "
                    "when inference_backend is 'endpoint'"
                )
            # CLEAR's eval pipeline expects endpoint_url; keep inference_url for compatibility.
            agentic_config["endpoint_url"] = ep
            agentic_config["inference_url"] = ep

        if "eval_model_params" in config.parameters:
            agentic_config["eval_model_params"] = config.parameters["eval_model_params"]
        else:
            agentic_config["eval_model_params"] = {"temperature": 0.0, "max_tokens": 8096}

        if config.parameters and "evaluation_criteria" in config.parameters:
            agentic_config["evaluation_criteria"] = config.parameters["evaluation_criteria"]
        if config.parameters and "predefined_issues" in config.parameters:
            agentic_config["predefined_issues"] = config.parameters["predefined_issues"]

        return agentic_config

    def _extract_agentic_results(self, json_results_path: str) -> list[EvaluationResult]:
        evaluation_results = []

        if not json_results_path or not Path(json_results_path).exists():
            logger.warning(f"Results file not found: {json_results_path}")
            return evaluation_results

        with open(json_results_path) as f:
            results = json.load(f)

        stats = results.get("metadata", {}).get("statistics", {})
        agents = results.get("agents", {})

        total_interactions = int(stats.get("total_interactions_analyzed", 0) or 0)
        total_issues = int(stats.get("total_issues_discovered", 0) or 0)
        interactions_with_issues = int(stats.get("total_interactions_with_issues", 0) or 0)
        interactions_no_issues = int(stats.get("total_interactions_no_issues", 0) or 0)
        total_agents_stat = stats.get("total_agents")
        total_agents = int(total_agents_stat) if total_agents_stat is not None else len(agents)

        evaluation_results.append(
            EvaluationResult(
                metric_name="total_interactions",
                metric_value=total_interactions,
                metric_type="int",
            )
        )

        evaluation_results.append(
            EvaluationResult(
                metric_name="total_issues",
                metric_value=total_issues,
                metric_type="int",
            )
        )

        evaluation_results.append(
            EvaluationResult(
                metric_name="interactions_with_issues",
                metric_value=interactions_with_issues,
                metric_type="int",
            )
        )
        evaluation_results.append(
            EvaluationResult(
                metric_name="interactions_no_issues",
                metric_value=interactions_no_issues,
                metric_type="int",
            )
        )
        evaluation_results.append(
            EvaluationResult(
                metric_name="total_agents",
                metric_value=total_agents,
                metric_type="int",
            )
        )

        if total_interactions > 0:
            pct_with_issues = 100.0 * interactions_with_issues / total_interactions
            issues_per_interaction = total_issues / total_interactions
        else:
            pct_with_issues = 0.0
            issues_per_interaction = 0.0
        evaluation_results.append(
            EvaluationResult(
                metric_name="pct_interactions_with_issues",
                metric_value=round(pct_with_issues, 4),
                metric_type="float",
            )
        )
        evaluation_results.append(
            EvaluationResult(
                metric_name="issues_per_interaction",
                metric_value=round(issues_per_interaction, 6),
                metric_type="float",
            )
        )

        agent_scores = []

        for agent_name, agent_data in agents.items():
            summary = agent_data.get("agent_summary", {})
            avg_score = summary.get("avg_score", 0.0)
            agent_scores.append(avg_score)

            evaluation_results.append(
                EvaluationResult(
                    metric_name=f"agent.{agent_name}.avg_score",
                    metric_value=float(avg_score),
                    metric_type="float",
                )
            )

            issues_catalog = agent_data.get("issues_catalog", {})
            num_issues = len(issues_catalog)

            evaluation_results.append(
                EvaluationResult(
                    metric_name=f"agent.{agent_name}.num_issues",
                    metric_value=num_issues,
                    metric_type="int",
                    metadata={"issues_catalog": issues_catalog},
                )
            )

        if agent_scores:
            overall_avg = sum(agent_scores) / len(agent_scores)
            evaluation_results.append(
                EvaluationResult(
                    metric_name="average_score",
                    metric_value=float(overall_avg),
                    metric_type="float",
                )
            )

        logger.info("Extracted %d metrics from CLEAR results", len(evaluation_results))
        return evaluation_results

    def _metrics_summary_dict(
        self,
        config: JobSpec,
        evaluation_results: list[EvaluationResult],
        overall_score: Optional[float],
        num_evaluated: int,
    ) -> dict[str, Any]:
        return {
            "job_id": config.id,
            "benchmark_id": config.benchmark_id,
            "benchmark_index": config.benchmark_index,
            "provider_id": config.provider_id,
            "model_name": config.model.name,
            "overall_score": overall_score,
            "num_examples_evaluated": num_evaluated,
            "metrics": {r.metric_name: r.metric_value for r in evaluation_results},
        }

    def _report_artifacts(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        json_results_path: Path | str,
        evaluation_results: list[EvaluationResult],
        overall_score: Optional[float],
        num_evaluated: int,
        extra_files: Optional[list[Path]] = None,
    ) -> Optional[OCIArtifactResult]:
        if not config.exports or not config.exports.oci:
            return None

        results_path = Path(json_results_path)
        if not results_path.exists():
            logger.warning("OCI skipped: missing %s", results_path)
            return None

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.PERSISTING_ARTIFACTS,
                progress=0.9,
                message=MessageInfo(
                    message="Persisting CLEAR artifacts",
                    message_code="persisting_artifacts",
                ),
            )
        )

        results_dir = Path("/tmp/clear_results") / config.id
        results_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(results_path, results_dir / "clear_results.json")

        summary = self._metrics_summary_dict(
            config, evaluation_results, overall_score, num_evaluated
        )
        (results_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        for html_f in extra_files or []:
            if isinstance(html_f, Path) and html_f.is_file():
                try:
                    shutil.copy2(html_f, results_dir / html_f.name)
                except OSError as exc:
                    logger.warning("OCI: could not copy %s: %s", html_f, exc)

        oci_artifact = callbacks.create_oci_artifact(
            OCIArtifactSpec(
                files_path=results_dir,
                coordinates=config.exports.oci.coordinates,
            )
        )
        if oci_artifact:
            logger.info("OCI artifact: %s", oci_artifact.digest)
        return oci_artifact

    def _compute_overall_score(self, results: list[EvaluationResult]) -> Optional[float]:
        for result in results:
            if result.metric_name == "average_score":
                return result.metric_value
        return None

    def _extract_num_evaluated(self, json_results_path: str) -> int:
        if not json_results_path or not Path(json_results_path).exists():
            return 0

        with open(json_results_path) as f:
            results = json.load(f)

        stats = results.get("metadata", {}).get("statistics", {})
        return stats.get("total_interactions_analyzed", 0)

    def _save_results_to_mlflow(
        self,
        callbacks: JobCallbacks,
        config: JobSpec,
        results: JobResults,
        output_dir: Path,
        metrics_summary: dict[str, Any],
        clear_html_artifacts: Optional[list[Path]] = None,
    ) -> str | None:
        if _local_only_run():
            logger.info("MLflow skipped: local-only run (EVALHUB_MODE=local or CLEAR_LOCAL_ONLY=1)")
            return None

        name = (config.experiment_name or "").strip()
        if not name:
            raw = config.parameters.get("mlflow_experiment_name")
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
        if not name:
            logger.info(
                "MLflow skipped: set experiment_name or parameters.mlflow_experiment_name on the job"
            )
            return None

        spec = config.model_copy(update={"experiment_name": name})
        clear_path = output_dir / "clear_results.json"
        if not clear_path.is_file():
            logger.warning("MLflow: clear_results.json not found at %s", clear_path)
            return None

        summary_bytes = json.dumps(metrics_summary, indent=2, default=str).encode("utf-8")

        mlflow_artifacts: list[MlflowArtifact] = [
            MlflowArtifact(
                "clear_results.json",
                clear_path.read_bytes(),
                "application/json",
            ),
            MlflowArtifact(
                "metrics_summary.json",
                summary_bytes,
                "application/json",
            ),
        ]
        for h in clear_html_artifacts or []:
            if h.is_file():
                mlflow_artifacts.append(
                    MlflowArtifact(
                        h.name,
                        h.read_bytes(),
                        "text/html",
                    )
                )
                logger.info("MLflow artifact: %s (HTML from CLEAR)", h.name)

        try:
            return callbacks.mlflow.save(
                results,
                spec,
                artifacts=mlflow_artifacts,
            )
        except Exception as e:
            logger.warning("MLflow artifact save failed (job still completes): %s", e, exc_info=True)
            return None

    def _cleanup_intermediate_files(self, output_dir: Path, json_results_path: str) -> None:
        final_json = output_dir / "clear_results.json"
        src = Path(json_results_path)
        if src.is_file() and src.resolve() != final_json.resolve():
            shutil.copy2(src, final_json)

        for dir_name in (
            "step_by_step",
            "full_trajectory",
            "traces_data",
            "clear_data",
            "clear_results",
        ):
            dir_path = output_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)


def main() -> None:
    """Load JobSpec, run ClearAdapter, emit JobResults via DefaultCallbacks."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = ClearAdapter(job_spec_path=job_spec_path)
        logger.info("Job %s benchmark=%s model=%s", adapter.job_spec.id, adapter.job_spec.benchmark_id, adapter.job_spec.model.name)

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

    except FileNotFoundError as e:
        logger.error("Job spec not found: %s (set EVALHUB_JOB_SPEC_PATH)", e)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
