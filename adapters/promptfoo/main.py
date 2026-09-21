#!/usr/bin/env python3
"""promptfoo framework adapter for eval-hub.

Wraps promptfoo (https://github.com/promptfoo/promptfoo, MIT license), an
open-source LLM testing tool, exposing two benchmarks:

- promptfoo-eval: assertion-based prompt/model regression testing
- promptfoo-redteam: promptfoo's red-team plugin catalog (OWASP LLM Top 10
  and beyond)

Every completed job persists promptfoo's own native eval.json (obtained via
``promptfoo export eval <id> -o eval.json``, verified byte-identical to the
``-o`` flag on ``eval``/``redteam run`` themselves) through three independent
paths, so results can always be reopened in promptfoo's own viewer via
``promptfoo import``, regardless of which EvalHub exports the job configured:

1. Always embedded in ``JobResults.additional_info["promptfoo_eval_json"]``,
   size-gated by ``PROMPTFOO_EVAL_JSON_MAX_BYTES`` (the /events payload has no
   documented size ceiling in the SDK, but a multi-MB body is not safe to
   assume will always be accepted by every ingress in front of eval-hub).
2. Attached as an MLflow artifact via ``callbacks.mlflow.save(...,
   artifacts=[...])`` when ``job_spec.experiment_name`` is set.
3. Attached as an OCI artifact via ``callbacks.create_oci_artifact()`` when
   ``config.exports.oci`` is set.

VERIFIED OPERATIONAL CONSTRAINT (promptfoo 0.123.1, checked 2026-09-21):
``redteam generate`` / ``redteam run`` refuse to proceed non-interactively
unless ``PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION=1`` is set in the
environment — without it, every plugin (even fully deterministic ones like
sql-injection) blocks on an interactive email-verification prompt against
promptfoo's cloud service, which a headless k8s Job cannot satisfy. With the
flag set, generation runs locally against ``generation_provider`` (or
promptfoo's own default model, requiring OPENAI_API_KEY) instead of
promptfoo's hosted service. This adapter always sets that flag; it is not
exposed as a configurable parameter because there is no non-interactive
alternative.
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

import yaml
from evalhub.adapter import (
    DefaultCallbacks,
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
    configure_telemetry,
)
from evalhub.adapter.auth import resolve_model_credentials
from evalhub.adapter.mlflow import MlflowArtifact
from evalhub.adapter.models.cards import (
    CapabilityEvalEntry,
    EnvironmentCardMetadata,
    EvalCardMetadata,
    SafetyEvalEntry,
)

logger = logging.getLogger(__name__)

PROMPTFOO_VERSION = "0.123.1"
_ADAPTER_VERSION = "0.1.0"

# Above this size, the full eval.json is NOT embedded in additional_info
# (it is still always available via the MLflow/OCI artifact paths, when
# those exports are configured). 5MB is a conservative guess at what an
# ingress in front of eval-hub's /events endpoint will reliably accept;
# revisit once real payload-size limits are confirmed against a live
# eval-hub deployment.
PROMPTFOO_EVAL_JSON_MAX_BYTES = 5_000_000

_EVAL_ID_RE = re.compile(r"\(ID:\s*(eval-\S+)\)")

_OWASP_DEFAULT_PLUGINS = [
    "excessive-agency",
    "hallucination",
    "harmful:privacy",
    "pii:direct",
    "pii:session",
    "pii:api-db",
    "prompt-extraction",
    "rag-poisoning",
    "shell-injection",
    "sql-injection",
    "ssrf",
    "system-prompt-override",
    "cross-session-leak",
    "indirect-prompt-injection",
]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _resolve_api_key(config: JobSpec) -> str:
    """Return an API key string for the target model endpoint, or a sentinel."""
    if config.model.auth and getattr(config.model.auth, "secret_ref", None):
        try:
            creds = resolve_model_credentials()
            if creds and creds.api_key:
                return creds.api_key
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve_model_credentials failed: %s", exc)

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    return "not-required"


# ---------------------------------------------------------------------------
# promptfoo config generation
# ---------------------------------------------------------------------------


def _build_target_provider(
    config: JobSpec, api_key: str, request_timeout: int
) -> dict[str, Any]:
    """Build a promptfoo provider/target block pointing at the EvalHub model endpoint."""
    model_url = (config.model.url or "").rstrip("/")
    if not model_url:
        raise ValueError("config.model.url is required for the promptfoo adapter")
    base_url = model_url if model_url.endswith("/v1") else f"{model_url}/v1"
    if not config.model.name:
        raise ValueError("config.model.name is required for the promptfoo adapter")

    return {
        "id": f"openai:chat:{config.model.name}",
        "label": config.model.name,
        "config": {
            "apiBaseUrl": base_url,
            "apiKey": api_key,
            # Best-effort: promptfoo ignores unrecognised provider config keys
            # rather than erroring, so this is safe even if the key name
            # drifts across promptfoo versions.
            "timeoutMs": request_timeout * 1000,
        },
    }


def _build_eval_config(config: JobSpec, provider: dict[str, Any]) -> dict[str, Any]:
    """Assemble a promptfoo config dict for the promptfoo-eval benchmark."""
    params = config.parameters or {}

    config_yaml = params.get("config_yaml")
    if config_yaml:
        parsed = yaml.safe_load(config_yaml)
        if not isinstance(parsed, dict):
            raise ValueError("parameters.config_yaml must parse to a YAML mapping")
        # Credentials always come from EvalHub's model config, never from a
        # pass-through config_yaml — see provider.yaml's documented contract.
        parsed["providers"] = [provider]
        return parsed

    prompts = params.get("prompts")
    tests = params.get("tests")
    if not prompts or not tests:
        raise ValueError(
            "promptfoo-eval requires either parameters.config_yaml, or both "
            "parameters.prompts and parameters.tests"
        )

    return {
        "description": f"eval-hub job {config.id}",
        "prompts": prompts,
        "providers": [provider],
        "tests": tests,
    }


def _build_redteam_config(config: JobSpec, provider: dict[str, Any]) -> dict[str, Any]:
    """Assemble a promptfoo config dict for the promptfoo-redteam benchmark."""
    params = config.parameters or {}

    purpose = params.get("purpose", "An AI assistant")
    plugin_ids = params.get("plugins") or list(_OWASP_DEFAULT_PLUGINS)
    strategies = params.get("strategies", [])
    num_tests = int(params.get("num_tests_per_plugin", 5))

    plugins = [{"id": pid, "numTests": num_tests} for pid in plugin_ids]

    return {
        "description": f"eval-hub redteam job {config.id}",
        "targets": [provider],
        "redteam": {
            "purpose": purpose,
            "plugins": plugins,
            "strategies": strategies,
            "numTests": num_tests,
        },
    }


# ---------------------------------------------------------------------------
# promptfoo CLI invocation
# ---------------------------------------------------------------------------


def _promptfoo_env() -> dict[str, str]:
    env = dict(os.environ)
    # See module docstring: required for non-interactive red-team generation.
    env["PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION"] = "1"
    env.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "1")
    return env


def _run_promptfoo_cli(
    args: list[str], cwd: Path, timeout: int = 3600
) -> subprocess.CompletedProcess:
    """Invoke the promptfoo CLI and return the completed process.

    promptfoo returns exit code 0 even when individual test cases error out
    or fail assertions — pass/fail outcome must be read from the eval.json
    stats, not the return code. A non-zero return code means the CLI itself
    could not run (bad config, crash), which IS fatal.
    """
    cmd = ["promptfoo", *args]
    logger.info("Executing promptfoo CLI: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_promptfoo_env(),
    )
    if result.stdout:
        logger.info("promptfoo stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("promptfoo stderr:\n%s", result.stderr)
    return result


def _extract_eval_id(stdout: str) -> str:
    match = _EVAL_ID_RE.search(stdout)
    if not match:
        raise RuntimeError(
            "Could not find an eval ID in promptfoo output; expected a line "
            "containing '(ID: eval-...)'"
        )
    return match.group(1)


def _export_eval_json(eval_id: str, cwd: Path) -> tuple[dict[str, Any], bytes]:
    """Run `promptfoo export eval <id> -o eval.json` and return (parsed, raw bytes)."""
    out_path = cwd / "eval.json"
    result = _run_promptfoo_cli(
        ["export", "eval", eval_id, "-o", str(out_path)], cwd=cwd, timeout=300
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"promptfoo export eval failed (exit {result.returncode})\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    raw = out_path.read_bytes()
    return json.loads(raw), raw


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _compute_metrics(
    eval_json: dict[str, Any],
) -> tuple[list[EvaluationResult], float | None, int]:
    """Extract pass-rate metrics from a parsed promptfoo eval.json."""
    results_block = eval_json.get("results", {})
    stats = results_block.get("stats", {})
    successes = int(stats.get("successes", 0))
    failures = int(stats.get("failures", 0))
    errors = int(stats.get("errors", 0))
    total = successes + failures + errors

    pass_rate = successes / total if total > 0 else None

    results = [
        EvaluationResult(
            metric_name="n_evaluated", metric_value=total, metric_type="int"
        ),
        EvaluationResult(
            metric_name="n_passed", metric_value=successes, metric_type="int"
        ),
        EvaluationResult(
            metric_name="n_failed", metric_value=failures, metric_type="int"
        ),
        EvaluationResult(
            metric_name="n_errors", metric_value=errors, metric_type="int"
        ),
    ]
    if pass_rate is not None:
        results.insert(
            0,
            EvaluationResult(
                metric_name="pass_rate",
                metric_value=round(pass_rate, 6),
                metric_type="float",
            ),
        )

    return results, pass_rate, total


def _compute_plugin_breakdown(eval_json: dict[str, Any]) -> dict[str, Any]:
    """Best-effort per-plugin pass rate + severity for promptfoo-redteam results.

    promptfoo's redteam result rows carry metadata.pluginId and
    metadata.severity (verified against a real `redteam run` on
    promptfoo 0.123.1). Wrapped defensively — this is a nice-to-have
    breakdown, not part of the core metrics contract, and must not fail
    the job if promptfoo changes this shape.
    """
    try:
        rows = eval_json.get("results", {}).get("results", [])
        by_plugin: dict[str, dict[str, int]] = {}
        severity_by_plugin: dict[str, str] = {}
        for row in rows:
            meta = row.get("metadata") or row.get("testCase", {}).get("metadata", {})
            plugin_id = meta.get("pluginId")
            if not plugin_id:
                continue
            bucket = by_plugin.setdefault(plugin_id, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if row.get("success"):
                bucket["passed"] += 1
            if meta.get("severity"):
                severity_by_plugin[plugin_id] = meta["severity"]

        if not by_plugin:
            return {}

        return {
            "pass_rate_by_plugin": {
                pid: round(b["passed"] / b["total"], 4) if b["total"] else None
                for pid, b in by_plugin.items()
            },
            "severity_by_plugin": severity_by_plugin,
        }
    except Exception:
        logger.warning("Failed to compute per-plugin breakdown", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# EvalCard / EnvironmentCard
# ---------------------------------------------------------------------------


def _build_eval_card(
    config: JobSpec, pass_rate: float | None, n_evaluated: int
) -> EvalCardMetadata:
    is_redteam = config.benchmark_id == "promptfoo-redteam"
    footnote = (
        "promptfoo (MIT license) red-team plugin catalog. pass_rate is the "
        "fraction of adversarial probes the model correctly resisted. "
        "Reference: https://github.com/promptfoo/promptfoo"
        if is_redteam
        else "promptfoo (MIT license) assertion-based regression testing. "
        "pass_rate is the fraction of test cases whose assertions all passed. "
        "Reference: https://github.com/promptfoo/promptfoo"
    )
    benchmark_label = (
        f"{config.benchmark_id} (promptfoo {PROMPTFOO_VERSION}), n={n_evaluated}"
    )
    zero_shot = round(pass_rate, 4) if pass_rate is not None else None

    if is_redteam:
        return EvalCardMetadata(
            modalities_input=["text"],
            modalities_output=["text"],
            languages_count=1,
            languages=["en"],
            capability_evaluations=[],
            safety_evaluations=[
                SafetyEvalEntry(
                    feature="red-team probe resistance",
                    benchmark=benchmark_label,
                    metric="pass_rate",
                    zero_shot=zero_shot,
                    alt_prompting=None,
                    alt_prompting_description=None,
                )
            ],
            developer_footnotes=footnote,
        )
    return EvalCardMetadata(
        modalities_input=["text"],
        modalities_output=["text"],
        languages_count=1,
        languages=["en"],
        capability_evaluations=[
            CapabilityEvalEntry(
                ability="prompt/model regression testing",
                benchmark=benchmark_label,
                metric="pass_rate",
                zero_shot=zero_shot,
                alt_prompting=None,
                alt_prompting_description=None,
            )
        ],
        safety_evaluations=[],
        developer_footnotes=footnote,
    )


def _build_env_card(config: JobSpec) -> EnvironmentCardMetadata:
    env = EnvironmentCardMetadata.capture(
        framework_name="promptfoo",
        framework_version=PROMPTFOO_VERSION,
        extra_packages=["promptfoo"],
    )
    env.model_id = config.model.name
    env.model_provider = "openai-compatible"
    return env


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PromptfooAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter wrapping promptfoo eval and redteam."""

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        metric = {r.metric_name: r.metric_value for r in results.results}
        return {
            "pass_rate": metric.get("pass_rate"),
            "promptfoo_version": PROMPTFOO_VERSION,
        }

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info(
            "Starting promptfoo job %s, benchmark=%s, model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )

        if config.benchmark_id not in ("promptfoo-eval", "promptfoo-redteam"):
            raise ValueError(
                f"Unsupported benchmark_id for promptfoo adapter: {config.benchmark_id}"
            )

        work_dir: Path | None = None
        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing promptfoo adapter",
                        message_code="initializing",
                    ),
                )
            )

            request_timeout = int((config.parameters or {}).get("request_timeout", 120))
            api_key = _resolve_api_key(config)
            provider = _build_target_provider(config, api_key, request_timeout)

            work_dir = Path(tempfile.mkdtemp(prefix="promptfoo_"))
            config_path = work_dir / "promptfooconfig.yaml"

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Generating promptfoo config",
                        message_code="loading_data",
                    ),
                )
            )

            is_redteam = config.benchmark_id == "promptfoo-redteam"
            pf_config = (
                _build_redteam_config(config, provider)
                if is_redteam
                else _build_eval_config(config, provider)
            )
            config_path.write_text(yaml.safe_dump(pf_config, sort_keys=False))

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running promptfoo {'redteam' if is_redteam else 'eval'} against {config.model.url}",
                        message_code="running_evaluation",
                    ),
                )
            )

            if is_redteam:
                generation_provider = (config.parameters or {}).get(
                    "generation_provider"
                )
                run_args = [
                    "redteam",
                    "run",
                    "-c",
                    str(config_path),
                    "--no-cache",
                    "--no-progress-bar",
                    "--force",
                ]
                if generation_provider:
                    # `redteam run` does generation+eval in one step and does not
                    # expose --provider directly; run generate (with -w to write
                    # the generated tests back into config_path itself) then eval
                    # against that same file, so the generation provider can be
                    # controlled explicitly.
                    gen_args = [
                        "redteam",
                        "generate",
                        "-c",
                        str(config_path),
                        "-w",
                        "--no-cache",
                        "--no-progress-bar",
                        "--force",
                        "--provider",
                        generation_provider,
                    ]
                    gen_result = _run_promptfoo_cli(gen_args, cwd=work_dir)
                    if gen_result.returncode != 0:
                        raise RuntimeError(
                            f"promptfoo redteam generate failed (exit {gen_result.returncode})\n"
                            f"stdout: {gen_result.stdout}\nstderr: {gen_result.stderr}"
                        )
                    run_args = [
                        "eval",
                        "-c",
                        str(config_path),
                        "--no-cache",
                        "--no-progress-bar",
                    ]
                result = _run_promptfoo_cli(run_args, cwd=work_dir)
            else:
                result = _run_promptfoo_cli(
                    ["eval", "-c", str(config_path), "--no-cache", "--no-progress-bar"],
                    cwd=work_dir,
                )

            if result.returncode != 0:
                raise RuntimeError(
                    f"promptfoo CLI failed (exit {result.returncode})\nstdout: {result.stdout}\nstderr: {result.stderr}"
                )

            eval_id = _extract_eval_id(result.stdout)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Parsing promptfoo results",
                        message_code="post_processing",
                    ),
                )
            )

            eval_json, eval_json_raw = _export_eval_json(eval_id, cwd=work_dir)
            evaluation_results, pass_rate, n_evaluated = _compute_metrics(eval_json)

            additional_info: dict[str, Any] = {"promptfoo_eval_id": eval_id}
            if is_redteam:
                breakdown = _compute_plugin_breakdown(eval_json)
                additional_info.update(breakdown)

            if len(eval_json_raw) <= PROMPTFOO_EVAL_JSON_MAX_BYTES:
                additional_info["promptfoo_eval_json"] = eval_json
            else:
                additional_info["promptfoo_eval_json_omitted"] = True
                additional_info["promptfoo_eval_json_size_bytes"] = len(eval_json_raw)
                logger.warning(
                    "eval.json is %d bytes (> %d limit); omitting from additional_info. "
                    "Still available via MLflow/OCI artifacts if those exports are configured.",
                    len(eval_json_raw),
                    PROMPTFOO_EVAL_JSON_MAX_BYTES,
                )

            eval_card = _build_eval_card(config, pass_rate, n_evaluated)
            env_card = _build_env_card(config)

            oci_artifact = None
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.PERSISTING_ARTIFACTS,
                        progress=0.95,
                        message=MessageInfo(
                            message="Persisting promptfoo artifacts",
                            message_code="persisting_artifacts",
                        ),
                    )
                )
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=work_dir, coordinates=oci_exports.coordinates
                    )
                )
                logger.info("OCI artifact created: %s", oci_artifact.reference)

            duration = time.time() - start_time
            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=pass_rate,
                num_examples_evaluated=n_evaluated,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "promptfoo",
                    "framework_version": PROMPTFOO_VERSION,
                    "adapter_version": _ADAPTER_VERSION,
                    "promptfoo_eval_id": eval_id,
                },
                eval_card=eval_card,
                env_card=env_card,
                oci_artifact=oci_artifact,
                additional_info=additional_info,
            )

            logger.info(
                "Done %s score=%s n=%d %.2fs",
                config.id,
                pass_rate if pass_rate is not None else "n/a",
                n_evaluated,
                duration,
            )
            return job_results

        except Exception as exc:
            logger.exception("promptfoo evaluation failed")
            error_msg = str(exc)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    message=MessageInfo(message=error_msg, message_code="failed"),
                    error_message=MessageInfo(
                        message=error_msg, message_code="evaluation_error"
                    ),
                )
            )
            raise
        finally:
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to clean up %s: %s", work_dir, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _local_only_run() -> bool:
    return os.getenv("EVALHUB_MODE", "").strip().lower() == "local"


def _callbacks_for_adapter(adapter: FrameworkAdapter) -> DefaultCallbacks:
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
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_telemetry()

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = PromptfooAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = _callbacks_for_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        # eval.json as a retained MLflow artifact (path 2 of 3 — see module
        # docstring). additional_info already carries path 1 (always-on
        # embed); OCI (path 3) is handled inside run_benchmark_job.
        eval_json = (results.additional_info or {}).get("promptfoo_eval_json")
        artifacts = None
        if eval_json is not None:
            artifacts = [
                MlflowArtifact(
                    "eval.json", json.dumps(eval_json).encode(), "application/json"
                ),
            ]
        run_id = callbacks.mlflow.save(results, adapter.job_spec, artifacts=artifacts)
        if run_id:
            results.mlflow_run_id = run_id
            logger.info("MLflow run created: %s", run_id)

        callbacks.report_results(results)

        logger.info(
            "Done %s score=%s n=%s %.2fs",
            results.id,
            results.overall_score,
            results.num_examples_evaluated,
            results.duration_seconds,
        )
        sys.exit(0)

    except FileNotFoundError as exc:
        logger.error("Job spec not found: %s (set EVALHUB_JOB_SPEC_PATH)", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
