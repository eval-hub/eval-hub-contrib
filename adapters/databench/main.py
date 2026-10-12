#!/usr/bin/env python3
"""EvalHub adapter for DataBench and DataBench Lite direct-answer QA."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import openai
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
    OCIArtifactSpec,
    configure_telemetry,
)
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials
from evalhub.models import MetricSchema, ResultType

from _evaluation import (
    BENCHMARKS,
    DATASET,
    REVISION,
    build_prompt,
    load_questions,
    load_table,
    make_evaluator,
    score_answer,
)

logger = logging.getLogger(__name__)


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1 or str(result) != str(value):
        raise ValueError(f"{name} must be a positive integer")
    return result


class DataBenchAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start = time.monotonic()
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
        )
        if config.benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark_id: {config.benchmark_id}")
        params = config.parameters or {}
        lite = BENCHMARKS[config.benchmark_id]
        limit = params.get("num_examples", config.num_examples)
        limit = positive_int(limit, "num_examples") if limit is not None else None
        max_tokens = positive_int(params.get("max_tokens", 2048), "max_tokens")
        max_chars = positive_int(
            params.get("max_prompt_chars", 1000000), "max_prompt_chars"
        )
        timeout = positive_int(params.get("request_timeout", 120), "request_timeout")
        temperature = float(params.get("temperature", 0))
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not config.model.url or not config.model.name:
            raise ValueError("model.url and model.name are required")
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
        )
        token = read_model_auth_key("hf-token") or os.getenv("HF_TOKEN")
        qa = load_questions(limit, params.get("table_id"), token)
        evaluator = make_evaluator(qa)
        credentials = resolve_model_credentials()
        api_key = (
            (credentials.api_key if credentials else None)
            or os.getenv("MODEL_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "DUMMY"
        )
        directory = Path(tempfile.mkdtemp(prefix="databench-results-"))
        samples_path = directory / "samples.jsonl"
        correct = 0
        tables = {}
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.0,
            )
        )
        with (
            openai.OpenAI(
                base_url=config.model.url, api_key=api_key, timeout=timeout
            ) as client,
            samples_path.open("w") as output,
        ):
            for index, row in enumerate(qa):
                table_id = row["dataset"]
                if table_id not in tables:
                    # Only retain the current table: full DataBench includes large tables.
                    tables.clear()
                    tables[table_id] = load_table(table_id, lite, token)
                prompt = build_prompt(row, tables[table_id], max_chars)
                completion = client.chat.completions.create(
                    model=config.model.name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if (
                    not completion.choices
                    or not (completion.choices[0].message.content or "").strip()
                ):
                    raise RuntimeError("Model returned no answer")
                choice = completion.choices[0]
                answer = choice.message.content
                passed = score_answer(evaluator, answer, row, lite)
                correct += int(passed)
                output.write(
                    json.dumps(
                        {
                            "sample_index": index,
                            "table_id": table_id,
                            "question": row["question"],
                            "answer_type": row["type"],
                            "expected_answer": row[
                                "sample_answer" if lite else "answer"
                            ],
                            "model_answer": answer,
                            "correct": passed,
                            "finish_reason": choice.finish_reason,
                            "table_rows": len(tables[table_id]),
                            "table_columns": list(tables[table_id].columns),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.RUNNING_EVALUATION,
                        progress=(index + 1) / len(qa),
                    )
                )
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
        )
        accuracy = correct / len(qa)
        metadata = {
            "framework": "databench-eval",
            "framework_version": "4.0.1",
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_license": "MIT (pinned dataset card)",
            "dataset_attribution": "Jorge Osés Grijalba, Luis Alfonso Ureña-López, Eugenio Martínez Cámara, Jose Camacho-Collados (LREC-COLING 2024)",
            "table_id_filter": params.get("table_id"),
            "num_examples_requested": limit,
            "evaluation_scope": "subset"
            if limit is not None or params.get("table_id")
            else "full_split",
            "metric_scale": "0-1",
            "higher_is_better": True,
            "dataset_config": "qa",
            "split": "train",
            "table_variant": "sample" if lite else "all",
            "prompting": "direct_answer_csv",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_correct": correct,
            "results_directory": str(directory),
        }
        (directory / "results.json").write_text(
            json.dumps(
                {"accuracy": accuracy, "num_examples": len(qa), **metadata}, indent=2
            )
        )
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS
            )
        )
        artifact = None
        if config.exports and config.exports.oci:
            artifact = callbacks.create_oci_artifact(
                OCIArtifactSpec(
                    files_path=directory,
                    coordinates=config.exports.oci.coordinates.model_copy(deep=True),
                )
            )
        logger.info(
            "DataBench accuracy=%.6f (%d/%d); diagnostics=%s",
            accuracy,
            correct,
            len(qa),
            directory,
        )
        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=[EvaluationResult(metric_name="accuracy", metric_value=accuracy)],
            metrics_schema=[MetricSchema(name="accuracy", type=ResultType.NUMERIC)],
            overall_score=accuracy,
            num_examples_evaluated=len(qa),
            duration_seconds=time.monotonic() - start,
            completed_at=datetime.now(UTC),
            evaluation_metadata=metadata,
            oci_artifact=artifact,
        )


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    configure_telemetry()
    adapter = DataBenchAdapter(
        job_spec_path=os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    )
    callbacks = DefaultCallbacks.from_adapter(adapter)
    try:
        result = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(result)
    except Exception:
        callbacks.report_status(JobStatusUpdate(status=JobStatus.FAILED))
        logger.exception("DataBench evaluation failed")
        raise


if __name__ == "__main__":
    main()
