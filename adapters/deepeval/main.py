"""DeepEval adapter entrypoint for eval-hub.

Loads a JobSpec, reads test data (CSV/JSONL/JSON), builds DeepEval test cases,
runs the benchmark's metric via ``deepeval.evaluate()`` and reports JobResults
including the EvalCard and EnvironmentCard. The implementation lives in the
``deepeval_adapter`` package; the benchmark list, with the metric and dataset
columns each one uses, is ``deepeval_adapter.benchmarks.BENCHMARKS``.
"""

from __future__ import annotations

import logging
import os
import sys

from evalhub.adapter import DefaultCallbacks, configure_telemetry

from deepeval_adapter.adapter import DeepEvalAdapter

logger = logging.getLogger(__name__)

__all__ = ["DeepEvalAdapter", "main"]


def main() -> None:
    """Load JobSpec, run DeepEvalAdapter, emit JobResults via DefaultCallbacks."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    configure_telemetry()

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = DeepEvalAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s | benchmark=%s | model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = DefaultCallbacks.from_adapter(adapter)
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)

        logger.info(
            "Done %s | score=%s | n=%s | %.2fs",
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
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
