#!/usr/bin/env python3
"""PyRIT framework adapter for eval-hub.

Wraps PyRIT (https://github.com/Azure/PyRIT, MIT license) via headless
``pyrit_scan`` (pinned at 1.1.0), exposing the ``pyrit-scan`` safety
benchmark against an EvalHub-supplied OpenAI-compatible chat endpoint.

Scaffold for RHAIRFE-3602 — helpers and ``run_benchmark_job`` are stubbed
to match the promptfoo CLI-wrapper shape and will be filled in by later
implementation steps.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobResults,
    JobSpec,
    configure_telemetry,
)

logger = logging.getLogger(__name__)

PYRIT_VERSION = "1.1.0"
_ADAPTER_VERSION = "0.1.0"

# Above this size, native scan/memory JSON is NOT embedded in additional_info
# (still available via MLflow/OCI artifact paths when configured).
PYRIT_RESULTS_JSON_MAX_BYTES = 5_000_000


# ---------------------------------------------------------------------------
# Credentials / env / conf (stubs — implement in adapter step)
# ---------------------------------------------------------------------------


def _resolve_api_key(config: JobSpec) -> str:
    """Return an API key string for the target model endpoint, or a sentinel."""
    raise NotImplementedError("scaffold: resolve target model API key")


def _build_pyrit_env(config: JobSpec, work_dir: Path, api_key: str) -> dict[str, str]:
    """Build job-local env with OPENAI_CHAT_ENDPOINT / KEY / MODEL for pyrit_scan."""
    raise NotImplementedError("scaffold: build PyRIT OpenAI chat env")


def _build_pyrit_conf(config: JobSpec) -> dict[str, Any]:
    """Build minimal pyrit conf (in_memory memory + target/scorer/technique)."""
    raise NotImplementedError("scaffold: build pyrit config mapping")


def _build_scan_argv(config: JobSpec, conf_path: Path) -> list[str]:
    """Build ``pyrit_scan run <scenario> --target openai_chat ...`` argv."""
    raise NotImplementedError("scaffold: build pyrit_scan CLI argv")


def _run_pyrit_scan(
    argv: list[str],
    env: dict[str, str],
    work_dir: Path,
) -> Any:
    """Run pyrit_scan via subprocess; return CompletedProcess[str]."""
    raise NotImplementedError("scaffold: run pyrit_scan subprocess")


def _compute_metrics(scan_export: dict[str, Any]) -> dict[str, Any]:
    """Map pyrit_scan / memory export JSON into pass_rate, n_evaluated, etc."""
    raise NotImplementedError("scaffold: map scan export to metrics")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PyritAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter wrapping headless pyrit_scan."""

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        """Return a compact summary dict for job additional_info."""
        metric = {r.metric_name: r.metric_value for r in results.results}
        return {
            "pass_rate": metric.get("pass_rate"),
            "pyrit_version": PYRIT_VERSION,
        }

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run a pyrit-scan benchmark and return JobResults.

        Planned phase sequence (mirrors promptfoo):
        INITIALIZING → LOADING_DATA → RUNNING_EVALUATION →
        POST_PROCESSING → PERSISTING_ARTIFACTS.
        """
        raise NotImplementedError(
            "scaffold: implement pyrit_scan CLI bridge and metrics mapping"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _local_only_run() -> bool:
    """Return True when EVALHUB_MODE=local, enabling offline callback routing."""
    return os.getenv("EVALHUB_MODE", "").strip().lower() == "local"


def _callbacks_for_adapter(adapter: FrameworkAdapter) -> DefaultCallbacks:
    """Return the appropriate DefaultCallbacks for the current run mode."""
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
    """Container entry point: load job spec, run benchmark, report results."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_telemetry()

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = PyritAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = _callbacks_for_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        run_id = callbacks.mlflow.save(results, adapter.job_spec, artifacts=None)
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
