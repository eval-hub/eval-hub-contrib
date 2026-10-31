#!/usr/bin/env python3
"""NVIDIA SkillEvaluator adapter for eval-hub (Tier 1: static skill checks).

Evaluates an agent skill (a directory containing ``SKILL.md``, staged under
``/test_data``) with the pinned ``skillevaluator`` CLI and maps its JSON report
onto EvalHub metrics:

* ``skill-quality``  : ``quality-check`` composite score (0-100 -> 0..1) and dimensions
* ``skill-security`` : ``security-scan`` (SkillSpector static analysis, no LLM)
* ``skill-pii``      : ``pii-scan`` (personal data, credentials, network identifiers)

No model is called and no skill content is executed: every benchmark here is
deterministic static analysis. ``JobSpec.model`` is required by the job schema
but unused. LLM-backed checks (rubric, Tier 2 dedup) and live agent evaluation
(Tier 3) are separate follow-ups.

SkillEvaluator is Apache-2.0 and Experimental (https://github.com/NVIDIA/SkillEvaluator);
the adapter pins an exact commit and parses a documented, versioned JSON report.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
    configure_telemetry,
)
from evalhub.adapter.models.cards import (
    CapabilityEvalEntry,
    EnvironmentCardMetadata,
    EvalCardMetadata,
    SafetyEvalEntry,
)
from evalhub.models import MetricSchema, ResultType

import _results
import _runner
import _skill_source

logger = logging.getLogger(__name__)

_ADAPTER_VERSION = "0.1.0"
_SKILLSPECTOR_PATH_ENV = "SKILLEVALUATOR_SKILLSPECTOR_PATH"

#: One entry per benchmark. ``category == "safety"`` routes to SafetyEvalEntry.
BENCHMARKS: dict[str, dict[str, Any]] = {
    "skill-quality": {
        "command": ["quality-check"],
        "category": "quality",
        "feature": "agent skill quality (correctness, discoverability, reliability, efficiency)",
        "metric": "quality_score",
    },
    "skill-security": {
        "command": ["security-scan", "--no-llm"],
        "category": "safety",
        "feature": "agent skill static security analysis (SkillSpector)",
        "metric": "gate_score (1.0 = no blocking findings)",
    },
    "skill-pii": {
        "command": ["pii-scan"],
        "category": "safety",
        "feature": "agent skill PII, credential and network-identifier exposure",
        "metric": "gate_score (1.0 = no blocking findings)",
    },
}
_SAFETY_BENCHMARKS = frozenset(
    k for k, v in BENCHMARKS.items() if v["category"] == "safety"
)

_ON_INCOMPLETE = ("fail", "report")
_DEFAULT_MIN_SCORE = 70
_DEFAULT_TIMEOUT = 300


class ScanIncompleteError(RuntimeError):
    """A check could not finish and ``on_incomplete=fail`` was requested."""


def _int_param(
    params: dict[str, Any], name: str, default: int, low: int, high: int
) -> int:
    """Read an integer parameter in ``low..high`` (booleans rejected), or return ``default``."""
    value = params.get(name)
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"parameters.{name} must be an integer in {low}..{high}")
    return value


def _tool_version(tool: str) -> str | None:
    """Version of an external scanner located like SkillEvaluator locates it."""
    explicit = os.environ.get(_SKILLSPECTOR_PATH_ENV, "").strip()
    exe = explicit or shutil.which(tool)
    return _runner.get_version(exe) if exe else None


def _metrics_schema(results: list[Any]) -> list[MetricSchema]:
    """Declare how the server should render each metric: strings are labels, the rest numeric."""
    return [
        MetricSchema(
            name=r.metric_name,
            type=ResultType.CATEGORICAL
            if isinstance(r.metric_value, str)
            else ResultType.NUMERIC,
        )
        for r in results
    ]


def _build_env_card(
    framework_version: str | None, skillspector_version: str | None
) -> EnvironmentCardMetadata:
    """Describe the runtime for the Environment Card; no model is involved in static checks."""
    env = EnvironmentCardMetadata.capture(
        framework_name="skillevaluator",
        framework_version=framework_version or "unknown",
        extra_packages=["eval-hub-sdk"],
    )
    env.model_id = "n/a (static analysis; no model called)"
    env.model_provider = "none"
    if skillspector_version:
        logger.info("SkillSpector version %s", skillspector_version)
    return env


def _build_eval_card(
    spec: dict[str, Any],
    benchmark_id: str,
    mapped: _results.Mapped,
    framework_version: str | None,
) -> EvalCardMetadata:
    """Build the Eval Card entry for the benchmark; an incomplete scan records ``incomplete`` as its score."""
    label = f"SkillEvaluator {framework_version or 'unknown'} / {benchmark_id}"
    score: float | str = (
        mapped.overall_score if mapped.overall_score is not None else "incomplete"
    )
    capability: list[CapabilityEvalEntry] = []
    safety: list[SafetyEvalEntry] = []
    if benchmark_id in _SAFETY_BENCHMARKS:
        safety.append(
            SafetyEvalEntry(
                feature=spec["feature"],
                benchmark=label,
                metric=spec["metric"],
                zero_shot=score,
            )
        )
    else:
        capability.append(
            CapabilityEvalEntry(
                ability=spec["feature"],
                benchmark=label,
                metric=spec["metric"],
                zero_shot=score,
            )
        )
    return EvalCardMetadata(
        modalities_input=["text"],
        modalities_output=["text"],
        languages_count=1,
        languages=["en"],
        capability_evaluations=capability,
        safety_evaluations=safety,
        developer_footnotes=(
            "Static analysis of an agent skill by NVIDIA SkillEvaluator (Apache-2.0, Experimental). "
            "No model is called and no skill content is executed. Incomplete checks produce no score. "
            "SkillSpector static analysis is not a malicious-skill gate on its own: it scored 0% malicious "
            "recall on MaliciousSkillBench (arXiv:2608.19901)."
        ),
    )


class SkillEvaluatorAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter for SkillEvaluator Tier 1 static checks."""

    def __init__(self, job_spec_path: str | None = None) -> None:
        """Load the job spec and prepare the buffer for the OCI artifact payloads."""
        super().__init__(job_spec_path=job_spec_path)
        self.mlflow_artifacts: list[tuple[str, bytes, str]] = []

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        """Return the extra Eval Card fields: tool version, benchmark and evaluated-skill digest."""
        metadata = results.evaluation_metadata or {}
        return {
            "zero_shot": results.overall_score,
            "prompting_strategy": "none (static analysis; no model calls)",
            "framework": "skillevaluator",
            "framework_version": metadata.get("framework_version"),
            "benchmark": results.benchmark_id,
            "skill_digest": metadata.get("skill_digest"),
        }

    @staticmethod
    def _validate_config(config: JobSpec) -> dict[str, Any]:
        """Validate the benchmark id and parameters and return the normalised options."""
        if config.benchmark_id not in BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark_id {config.benchmark_id!r}; expected one of {sorted(BENCHMARKS)}"
            )
        params = config.parameters or {}
        on_incomplete = params.get("on_incomplete", "fail")
        if on_incomplete not in _ON_INCOMPLETE:
            raise ValueError(
                f"parameters.on_incomplete must be one of {list(_ON_INCOMPLETE)}"
            )
        submitter = params.get("submitter")
        if submitter is not None and not isinstance(submitter, str):
            raise ValueError("parameters.submitter must be a string")
        return {
            "min_score": _int_param(params, "min_score", _DEFAULT_MIN_SCORE, 0, 100),
            "timeout": _int_param(
                params, "timeout_seconds", _DEFAULT_TIMEOUT, 10, 3600
            ),
            "on_incomplete": on_incomplete,
            "submitter": submitter,
        }

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run one static check against the staged skill and report status, metrics and cards."""
        start = time.time()
        logger.info(
            "Starting SkillEvaluator job %s benchmark=%s",
            config.id,
            config.benchmark_id,
        )
        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing SkillEvaluator adapter",
                        message_code="initializing",
                    ),
                )
            )
            opts = self._validate_config(config)
            spec = BENCHMARKS[config.benchmark_id]
            cli = _runner.find_cli()
            env = _runner.build_env(
                {"SKILLEVALUATOR_SUBMITTER": opts["submitter"] or ""}
            )
            framework_version = _runner.get_version(cli, env=env)
            skillspector_version = _tool_version("skillspector")

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Resolving skill under /test_data",
                        message_code="loading_data",
                    ),
                )
            )
            skill_dir = _skill_source.resolve_skill_path(config.parameters or {})

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running {spec['command'][0]}",
                        message_code="running_evaluation",
                    ),
                )
            )
            with tempfile.TemporaryDirectory(prefix="skillevaluator-") as tmp:
                work = Path(tmp)
                run = _runner.run_cli(
                    [*spec["command"], str(skill_dir)],
                    output_dir=work / "report",
                    timeout=opts["timeout"],
                    env=env,
                    executable=cli,
                )
                report_text = run.report_path.read_bytes()
                report = _runner.load_report(run.report_path)

                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.POST_PROCESSING,
                        progress=0.8,
                        message=MessageInfo(
                            message="Mapping SkillEvaluator report",
                            message_code="post_processing",
                        ),
                    )
                )
                if spec["category"] == "quality":
                    mapped = _results.map_quality(report, min_score=opts["min_score"])
                else:
                    mapped = _results.map_findings(report)

                if (
                    mapped.outcome.status == _results.OUTCOME_INCOMPLETE
                    and opts["on_incomplete"] == "fail"
                ):
                    detail = "; ".join(mapped.outcome.reasons) or "no reason reported"
                    scans = ", ".join(mapped.outcome.incomplete_scans) or "unknown"
                    raise ScanIncompleteError(
                        f"{config.benchmark_id} incomplete (scans: {scans}): {detail}"
                    )

                eval_card = _build_eval_card(
                    spec, config.benchmark_id, mapped, framework_version
                )
                env_card = _build_env_card(framework_version, skillspector_version)

                oci_artifact = None
                if config.exports and config.exports.oci:
                    callbacks.report_status(
                        JobStatusUpdate(
                            status=JobStatus.RUNNING,
                            phase=JobPhase.PERSISTING_ARTIFACTS,
                            progress=0.95,
                            message=MessageInfo(
                                message="Persisting SkillEvaluator report",
                                message_code="persisting_artifacts",
                            ),
                        )
                    )
                    results_dir = work / "results"
                    results_dir.mkdir()
                    (results_dir / f"{config.benchmark_id}.json").write_bytes(
                        report_text
                    )
                    oci_artifact = callbacks.create_oci_artifact(
                        OCIArtifactSpec(
                            files_path=results_dir,
                            coordinates=config.exports.oci.coordinates,
                        )
                    )
                    if oci_artifact:
                        env_card.oci_artifact_ref = oci_artifact.reference
                self.mlflow_artifacts = [
                    (f"{config.benchmark_id}.json", report_text, "application/json")
                ]

            duration = time.time() - start
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=mapped.results,
                metrics_schema=_metrics_schema(mapped.results),
                overall_score=mapped.overall_score,
                num_examples_evaluated=1,
                duration_seconds=round(duration, 2),
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "skillevaluator",
                    "framework_version": framework_version,
                    "skillspector_version": skillspector_version,
                    "adapter_version": _ADAPTER_VERSION,
                    "benchmark_id": config.benchmark_id,
                    "outcome": mapped.outcome.status,
                    "incomplete_scans": mapped.outcome.incomplete_scans,
                    "incomplete_reasons": mapped.outcome.reasons,
                    "skill_digest": _results.skill_digest(report),
                    **mapped.metadata,
                },
                oci_artifact=oci_artifact,
                eval_card=eval_card,
                env_card=env_card,
            )

        except Exception as exc:
            logger.exception("SkillEvaluator evaluation failed")
            code = (
                "scan_incomplete"
                if isinstance(exc, ScanIncompleteError)
                else "evaluation_error"
            )
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(message=str(exc), message_code="failed"),
                    error_message=MessageInfo(message=str(exc), message_code=code),
                )
            )
            raise


def _local_only_run() -> bool:
    """Return True when ``EVALHUB_MODE=local``, i.e. there is no sidecar to report to."""
    return os.getenv("EVALHUB_MODE", "").strip().lower() == "local"


def _callbacks_for_adapter(adapter: FrameworkAdapter) -> JobCallbacks:
    """Build callbacks that skip the sidecar in local mode and use the adapter's settings otherwise."""
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
    """Load JobSpec, run the adapter, emit JobResults via DefaultCallbacks."""
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_telemetry()

    try:
        adapter = SkillEvaluatorAdapter(
            job_spec_path=os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        )
        logger.info(
            "Job %s benchmark=%s", adapter.job_spec.id, adapter.job_spec.benchmark_id
        )
        callbacks = _callbacks_for_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        experiment = (
            adapter.job_spec.experiment_name or ""
        ).strip() or adapter.job_spec.parameters.get("mlflow_experiment_name", "")
        if experiment:
            from evalhub.adapter.mlflow import MlflowArtifact

            artifacts = [
                MlflowArtifact(p, c, t) for p, c, t in adapter.mlflow_artifacts
            ]
            try:
                run_id = callbacks.mlflow.save(
                    results, adapter.job_spec, artifacts=artifacts
                )  # type: ignore[attr-defined]
                if run_id:
                    results.mlflow_run_id = run_id
            except Exception:
                logger.warning("MLflow save failed", exc_info=True)

        callbacks.report_results(results)
        logger.info(
            "Done %s score=%s %.2fs",
            results.id,
            results.overall_score,
            results.duration_seconds,
        )
        sys.exit(0)

    except FileNotFoundError as exc:
        logger.error("Job spec not found: %s (set EVALHUB_JOB_SPEC_PATH)", exc)
        sys.exit(1)
    except (
        ValueError,
        _skill_source.SkillSourceError,
        _runner.RunnerError,
        ScanIncompleteError,
    ) as exc:
        logger.error("Job failed: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
