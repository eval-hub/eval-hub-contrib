"""RAGAS framework adapter for eval-hub.

This adapter integrates RAGAS (https://github.com/explodinggradients/ragas)
with the eval-hub evaluation service using the evalhub-sdk framework adapter pattern.

The adapter:
1. Reads JobSpec from a mounted ConfigMap
2. Loads evaluation data from /test_data (S3 init container) or /data
3. Runs RAGAS metrics against the model in the job spec
4. Reports progress via callbacks to the sidecar
5. Persists results as OCI artifacts
6. Returns structured JobResults

RAGAS evaluates RAG pipelines on metrics like faithfulness, answer relevancy,
context precision/recall, and more. Models are accessed via OpenAI-compatible
completions/embeddings endpoints.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from evalhub.adapter import (
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
from evalhub.adapter.auth import resolve_model_credentials
from ragas import EvaluationDataset
from ragas.run_config import RunConfig

try:
    from openai import AsyncOpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

logger = logging.getLogger(__name__)

# When test_data_ref.s3 is set, EvalHub's init container downloads objects here
TEST_DATA_DIR = Path("/test_data")
DEFAULT_DATA_DIR = Path("/data")
DEFAULT_DATASET_FILENAME = "dataset.jsonl"
_DATA_SUFFIXES = (".jsonl", ".json")

# ---------------------------------------------------------------------------
# RAGAS metrics — collections API (ragas >= 0.4.x)
# ---------------------------------------------------------------------------
from ragas.metrics.collections import (
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
    ResponseGroundedness,
    SemanticSimilarity,
)


@dataclass
class _MetricDef:
    """Canonical name and lazy factory for a RAGAS metric.

    Metrics require an LLM (and sometimes embeddings) at instantiation time,
    so they are created inside _run_ragas rather than at module import.
    """

    name: str
    factory: Callable


METRIC_MAPPING: dict[str, "_MetricDef"] = {
    d.name: d
    for d in [
        _MetricDef("answer_relevancy", lambda llm, emb: AnswerRelevancy(llm=llm, embeddings=emb)),
        _MetricDef("answer_correctness", lambda llm, emb: AnswerCorrectness(llm=llm)),
        _MetricDef("context_precision", lambda llm, emb: ContextPrecision(llm=llm)),
        _MetricDef("faithfulness", lambda llm, emb: Faithfulness(llm=llm)),
        _MetricDef("context_recall", lambda llm, emb: ContextRecall(llm=llm)),
        _MetricDef("context_entity_recall", lambda llm, emb: ContextEntityRecall(llm=llm)),
        _MetricDef("factual_correctness", lambda llm, emb: FactualCorrectness(llm=llm)),
        _MetricDef("noise_sensitivity", lambda llm, emb: NoiseSensitivity(llm=llm)),
        _MetricDef("context_relevance", lambda llm, emb: ContextRelevance(llm=llm)),
        _MetricDef("answer_accuracy", lambda llm, emb: AnswerAccuracy(llm=llm)),
        _MetricDef("response_groundedness", lambda llm, emb: ResponseGroundedness(llm=llm)),
        _MetricDef("semantic_similarity", lambda llm, emb: SemanticSimilarity(embeddings=emb)),
    ]
}

DEFAULT_METRICS = [
    "answer_relevancy",
    "context_precision",
    "faithfulness",
    "context_recall",
]


@dataclass
class _CollectionsEvaluationResult:
    """Minimal stand-in for legacy ragas.evaluate() result objects."""

    dataframe: Any

    def to_pandas(self):
        return self.dataframe


def _records_from_dataset(eval_dataset: EvaluationDataset) -> list[dict[str, Any]]:
    if hasattr(eval_dataset, "to_list"):
        return eval_dataset.to_list()
    return [dict(sample) for sample in eval_dataset]


def _metric_input_fields(metric: Any) -> list[str]:
    return [
        name
        for name, param in inspect.signature(metric.ascore).parameters.items()
        if name != "self" and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def _build_metric_inputs(
    records: list[dict[str, Any]], fields: list[str]
) -> list[dict[str, Any]]:
    return [{field: record[field] for field in fields if field in record} for record in records]


def _run_collections_evaluation(
    *,
    eval_dataset: EvaluationDataset,
    metric_defs: list["_MetricDef"],
    llm: Any,
    embeddings: Any,
) -> _CollectionsEvaluationResult:
    """Evaluate dataset rows with ragas.metrics.collections metrics."""
    import pandas as pd

    records = _records_from_dataset(eval_dataset)
    dataframe = pd.DataFrame(records)

    for metric_def in metric_defs:
        metric = metric_def.factory(llm, embeddings)
        fields = _metric_input_fields(metric)
        if records:
            missing = [field for field in fields if field not in records[0]]
            if missing:
                raise ValueError(
                    f"Metric {metric_def.name} requires fields {fields}, "
                    f"but dataset is missing {missing} (columns: {list(records[0].keys())})"
                )
        inputs = _build_metric_inputs(records, fields)

        batch_results = metric.batch_score(inputs)
        column_name = getattr(metric, "name", metric_def.name)
        dataframe[column_name] = [result.value for result in batch_results]

    return _CollectionsEvaluationResult(dataframe=dataframe)


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM wrapper
# ---------------------------------------------------------------------------
def _openai_credentials(base_url: str) -> tuple[str, str]:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    creds = resolve_model_credentials()
    api_key = creds.api_key
    if not api_key:
        auth_value = creds.auth_headers.get("Authorization", "")
        if auth_value.startswith("Bearer "):
            api_key = auth_value.removeprefix("Bearer ").strip()
    return url, api_key or "DUMMY"


def _async_openai_client(base_url: str) -> Any:
    if not _HAS_OPENAI:
        raise RuntimeError(
            "openai package is required — install with: pip install openai>=1.0.0"
        )
    url, api_key = _openai_credentials(base_url)
    return AsyncOpenAI(base_url=url, api_key=api_key)


def _create_ragas_llm(
    base_url: str,
    model_id: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    """Build an InstructorLLM for ragas.metrics.collections via llm_factory."""
    from ragas.llms import llm_factory

    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Pass raw AsyncOpenAI — llm_factory patches with instructor internally.
    # Do not pre-patch or pass mode= (conflicts with llm_factory's own instructor setup).
    client = _async_openai_client(base_url)
    return llm_factory(model_id, client=client, **kwargs)


def _create_ragas_embeddings(
    base_url: str,
    model_id: str,
    *,
    run_config: RunConfig | None = None,
) -> Any:
    """Build embeddings for ragas.metrics.collections via embedding_factory."""
    from ragas.embeddings.base import embedding_factory

    client = _async_openai_client(base_url)
    return embedding_factory(
        "openai",
        model=model_id,
        client=client,
        interface="modern",
        run_config=run_config,
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _first_dataset_in_dir(path: Path) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in _DATA_SUFFIXES and f.is_file():
            return f
    return None


def _resolve_data_path(config: JobSpec) -> Path:
    bc = config.parameters or {}
    explicit = bc.get("data_path")
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else DEFAULT_DATA_DIR / explicit

    test_data_file = TEST_DATA_DIR / DEFAULT_DATASET_FILENAME
    if test_data_file.exists():
        return test_data_file
    first_in_test = _first_dataset_in_dir(TEST_DATA_DIR)
    if first_in_test is not None:
        return first_in_test

    default_file = DEFAULT_DATA_DIR / DEFAULT_DATASET_FILENAME
    if default_file.exists():
        return default_file
    first_in_data = _first_dataset_in_dir(DEFAULT_DATA_DIR)
    if first_in_data is not None:
        return first_in_data
    return default_file


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix.lower() == ".json":
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            raise ValueError(
                f"JSON dataset must be a list or {{'data': list}}, got {type(data)}"
            )
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _apply_column_map(
    records: list[dict[str, Any]], column_map: dict[str, str] | None
) -> list[dict[str, Any]]:
    if not column_map:
        return records
    return [{column_map.get(k, k): v for k, v in row.items()} for row in records]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class RagasAdapter(FrameworkAdapter):
    """EvalHub framework adapter that runs RAGAS evaluation."""

    def _resolve_metrics(self, bc: dict[str, Any]) -> list["_MetricDef"]:
        metric_names = (
            bc.get("metrics") or bc.get("scoring_functions") or DEFAULT_METRICS
        )
        defs = [METRIC_MAPPING[n] for n in metric_names if n in METRIC_MAPPING]
        unknown = [n for n in metric_names if n not in METRIC_MAPPING]
        if unknown:
            logger.warning("Unknown metrics (skipped): %s", unknown)
        if not defs:
            logger.info("No valid metrics specified, using defaults")
            defs = [METRIC_MAPPING[m] for m in DEFAULT_METRICS]
        return defs

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info(
            "Starting RAGAS job %s benchmark=%s model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )

        try:
            # --- INITIALIZING ---
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
            )
            self._validate_config(config)
            bc = config.parameters or {}

            # --- LOADING_DATA ---
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
            )
            data_path = _resolve_data_path(config)
            records = _load_dataset(data_path)

            column_map = bc.get("column_map")
            if isinstance(column_map, dict):
                records = _apply_column_map(records, column_map)

            if config.num_examples and config.num_examples > 0:
                records = records[: config.num_examples]

            if not records:
                raise ValueError(
                    f"No records in dataset at {data_path} (or after limit)"
                )

            eval_dataset = EvaluationDataset.from_list(records)
            logger.info(
                "Dataset loaded: path=%s records=%d columns=%s",
                data_path,
                len(records),
                list(records[0].keys()) if records else [],
            )

            # --- RUNNING_EVALUATION ---
            metric_defs = self._resolve_metrics(bc)
            model_url = config.model.url.strip().rstrip("/")
            model_name = config.model.name
            embedding_model = bc.get("embedding_model") or model_name
            embedding_url = bc.get("embedding_url") or model_url

            # In k8s mode the service rewrites model.url to the local credential-
            # injection sidecar (http://localhost:…).  If a caller also supplies an
            # explicit embedding_url that points at an external host, the embedding
            # calls would bypass the sidecar and reach the model endpoint with the
            # unresolved "api-key:ref" ref token, causing a 401.  Detect this and
            # fall back to the sidecar URL so both LLM and embedding calls share the
            # same credential-injection path.
            _sidecar_hosts = {"localhost", "127.0.0.1", "::1"}
            _model_is_local = urlparse(model_url).hostname in _sidecar_hosts
            _embed_is_external = urlparse(embedding_url).hostname not in _sidecar_hosts
            if _model_is_local and _embed_is_external and embedding_url != model_url:
                logger.warning(
                    "embedding_url %r is external but model.url points to the local "
                    "sidecar (%r); falling back to the sidecar URL for embeddings so "
                    "that credential injection applies to both LLM and embedding calls.",
                    embedding_url,
                    model_url,
                )
                embedding_url = model_url

            max_workers = min(max(int(bc.get("max_workers") or 1), 1), 10)
            run_config = RunConfig(max_workers=max_workers)
            llm = _create_ragas_llm(
                model_url,
                model_name,
                max_tokens=bc.get("max_tokens"),
                temperature=bc.get("temperature"),
            )
            embeddings = _create_ragas_embeddings(
                embedding_url,
                embedding_model,
                run_config=run_config,
            )

            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION)
            )

            ragas_result = self._run_ragas(
                eval_dataset=eval_dataset,
                metric_defs=metric_defs,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            # --- POST_PROCESSING ---
            callbacks.report_status(
                JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
            )

            result_df = ragas_result.to_pandas()
            n_evaluated = len(result_df)
            # Keep per-row results for main() to attach to the MLflow run.
            self.mlflow_artifacts = [
                ("results.jsonl", result_df.to_json(orient="records", lines=True).encode(), "application/json"),
                ("results.csv", result_df.to_csv(index=False).encode(), "text/csv"),
            ]
            evaluation_results: list[EvaluationResult] = []
            scores_for_overall: list[float] = []

            for metric_name in [d.name for d in metric_defs]:
                # Some metrics (e.g. FactualCorrectness, NoiseSensitivity) report
                # their score column with a mode suffix: "factual_correctness(mode=f1)".
                column = metric_name
                if column not in result_df.columns:
                    column = next(
                        (
                            c
                            for c in result_df.columns
                            if c.startswith(f"{metric_name}(")
                        ),
                        None,
                    )
                if column is None:
                    logger.warning(
                        "Metric %s missing from RAGAS results (columns: %s)",
                        metric_name,
                        list(result_df.columns),
                    )
                    continue
                series = result_df[column].dropna()
                values = series.tolist()
                if not values:
                    continue
                avg = sum(values) / len(values)
                scores_for_overall.append(avg)
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        metric_value=round(avg, 6),
                        metric_type="float",
                        num_samples=len(values),
                        metadata={"min": min(values), "max": max(values)},
                    )
                )

            overall_score = (
                sum(scores_for_overall) / len(scores_for_overall)
                if scores_for_overall
                else None
            )

            oci_artifact = None
            if config.exports and config.exports.oci:
                callbacks.report_status(
                    JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.PERSISTING_ARTIFACTS)
                )
                if self.local_jobs_base_path is not None:
                    results_dir = self.local_jobs_base_path / "results"
                else:
                    results_dir = Path(__file__).parent / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                results_file = results_dir / "results.jsonl"
                result_df.to_json(results_file, orient="records", lines=True)
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=results_dir,
                        coordinates=config.exports.oci.coordinates,
                    )
                )

            duration = time.time() - start_time
            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=n_evaluated,
                duration_seconds=round(duration, 2),
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "ragas",
                    "ragas_version": importlib.metadata.version("ragas"),
                    "judge_llm": model_name,
                    "embedding_model": embedding_model,
                    "data_path": str(data_path),
                    "metrics": [d.name for d in metric_defs],
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("RAGAS EvalHub job %s failed", config.id)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error_message=MessageInfo(
                        message=str(e),
                        message_code="job_failed",
                    ),
                )
            )
            raise

    def _run_ragas(self, *, eval_dataset, metric_defs, llm, embeddings, run_config):
        del run_config  # collections metrics use batch_score; RunConfig applies at factory time
        return _run_collections_evaluation(
            eval_dataset=eval_dataset,
            metric_defs=metric_defs,
            llm=llm,
            embeddings=embeddings,
        )

    def _validate_config(self, config: JobSpec) -> None:
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")
        if not config.model or not config.model.url:
            raise ValueError("model.url is required")
        if not config.model.name:
            raise ValueError("model.name is required")


def main() -> None:
    """Load JobSpec, run RagasAdapter, emit JobResults via DefaultCallbacks."""
    from evalhub.adapter import DefaultCallbacks

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = RagasAdapter(job_spec_path=job_spec_path)
        logger.info(
            "Job %s benchmark=%s model=%s",
            adapter.job_spec.id,
            adapter.job_spec.benchmark_id,
            adapter.job_spec.model.name,
        )

        callbacks = DefaultCallbacks.from_adapter(adapter)

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        from evalhub.adapter.mlflow import MlflowArtifact

        artifacts = [
            MlflowArtifact(path, content, content_type)
            for path, content, content_type in getattr(adapter, "mlflow_artifacts", [])
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
