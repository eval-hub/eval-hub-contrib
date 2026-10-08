"""EvalHub post-processing adapter: execute ordered operations on stored results."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from evalhub.adapter import (
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    configure_telemetry,
)

from post_processor.config import operations
from post_processor.operations import OPERATIONS, Context
from post_processor.sources import Sources
from post_processor.transport import PostProcessorCallbacks, Sidecar

logger = logging.getLogger(__name__)


class PostProcessorAdapter(FrameworkAdapter):
    def _load_job_spec(self) -> JobSpec:
        # The internal provider has no model endpoint. The SDK still requires
        # ModelConfig.url, so normalize that field only for this adapter.
        with self.settings.resolved_job_spec_path.open() as stream:
            raw = json.load(stream)
        model = raw.get("model") or {"name": "evaluation-post-processor"}
        raw["model"] = {"url": "", **model}
        return JobSpec.model_validate(raw)

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        started = time.monotonic()
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
        )
        sidecar = Sidecar(config.callback_url)
        try:
            requested = operations(config.parameters)
            for name, _ in requested:
                if name not in OPERATIONS:
                    raise ValueError(f"Unknown post-processing operation {name!r}")
            sources = Sources(
                sidecar, exports=config.exports.model_dump() if config.exports else {}
            )
            additional_info = {}
            with tempfile.TemporaryDirectory(prefix="evalhub-post-processor-") as temp:
                for index, (name, operation_config) in enumerate(requested):
                    directory = Path(temp) / f"operation-{index}"
                    directory.mkdir()
                    context = Context(config, callbacks, sidecar, sources, directory)
                    logger.info("Starting operation %s (%d/%d)", name, index + 1, len(requested))
                    additional_info[name] = OPERATIONS[name](context, operation_config)
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
            )
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=[],
                num_examples_evaluated=0,
                duration_seconds=time.monotonic() - started,
                additional_info=additional_info,
            )
        except Exception as exc:
            try:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.FAILED,
                        error_message=MessageInfo(
                            message=str(exc), message_code="post_processing_failed"
                        ),
                    )
                )
            except Exception:
                logger.exception("Failed to deliver post-processing failure event")
            raise
        finally:
            sidecar.close()


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
    )
    configure_telemetry()
    sidecar = None
    try:
        adapter = PostProcessorAdapter(
            job_spec_path=os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        )
        sidecar = Sidecar(adapter.job_spec.callback_url)
        callbacks = PostProcessorCallbacks(adapter.job_spec, sidecar)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)
        logger.info("Completed %d post-processing operation(s)", len(results.additional_info or {}))
        return 0
    except Exception:
        logger.exception("Post-processing failed")
        return 1
    finally:
        if sidecar is not None:
            sidecar.close()


if __name__ == "__main__":
    raise SystemExit(main())
