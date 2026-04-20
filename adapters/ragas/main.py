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

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalhub.adapter import (
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
from langchain_core.language_models.llms import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas import EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

try:
    from openai import OpenAI

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
# RAGAS metrics — import singletons and class-based metrics
# ---------------------------------------------------------------------------
from ragas.metrics import (
    AnswerAccuracy,
    ContextRelevance,
    FactualCorrectness,
    NoiseSensitivity,
    ResponseGroundedness,
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)

_SINGLETON_METRICS = [
    answer_relevancy,
    answer_similarity,
    context_precision,
    faithfulness,
    context_recall,
    context_entity_recall,
]

_CLASS_METRICS = [
    AnswerAccuracy(),
    ContextRelevance(),
    FactualCorrectness(),
    NoiseSensitivity(),
    ResponseGroundedness(),
]

METRIC_MAPPING = {m.name: m for m in _SINGLETON_METRICS + _CLASS_METRICS}

DEFAULT_METRICS = [
    "answer_relevancy",
    "context_precision",
    "faithfulness",
    "context_recall",
]


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM wrapper
# ---------------------------------------------------------------------------
_SECRET_DIR = Path("/var/run/secrets/model")


def _get_api_key() -> str:
    for name in ("OPENAICOMPATIBLE_API_KEY", "OPENAI_API_KEY"):
        if val := os.environ.get(name):
            return val
        secret_file = _SECRET_DIR / name
        if secret_file.exists():
            return secret_file.read_text().strip()
    return "DUMMY"


def _openai_client(base_url: str) -> Any:
    if not _HAS_OPENAI:
        raise RuntimeError(
            "openai package is required — install with: pip install openai>=1.0.0"
        )
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return OpenAI(base_url=url, api_key=_get_api_key())


class EvalHubOpenAILLM(BaseRagasLLM):
    """RAGAS LLM that calls an OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        run_config: RunConfig | None = None,
    ):
        if run_config is None:
            run_config = RunConfig()
        super().__init__(run_config, multiple_completion_supported=True)
        self._base_url = base_url
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        client = _openai_client(self._base_url)
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt.to_string(),
            "n": n,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        t = temperature if temperature is not None else self._temperature
        if t is not None:
            kwargs["temperature"] = t
        if stop:
            kwargs["stop"] = stop

        try:
            response = client.completions.create(**kwargs)
        except Exception as e:
            logger.error("Completion request failed: %s", e)
            raise

        generations = []
        for choice in getattr(response, "choices", []) or []:
            text = getattr(choice, "text", "") or ""
            generations.append(Generation(text=text))

        if not generations:
            generations = [Generation(text="")]

        return LLMResult(
            generations=[generations], llm_output={"provider": "evalhub_openai"}
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        return self.generate_text(
            prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks
        )

    def get_temperature(self, n: int) -> float:
        if self._temperature is not None:
            return self._temperature
        return 0.3 if n > 1 else 1e-8


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings wrapper
# ---------------------------------------------------------------------------
class EvalHubOpenAIEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings that call an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        run_config: RunConfig | None = None,
    ):
        super().__init__()
        self._base_url = base_url
        self._model_id = model_id
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> list[float]:
        client = _openai_client(self._base_url)
        try:
            r = client.embeddings.create(input=text, model=self._model_id)
            if not r.data:
                raise ValueError("Embeddings response had no data")
            emb = r.data[0].embedding
            if isinstance(emb, str):
                raise ValueError("Expected float embeddings, got base64 string")
            return emb
        except Exception as e:
            logger.error("Embed query failed: %s", e)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = _openai_client(self._base_url)
        try:
            r = client.embeddings.create(input=texts, model=self._model_id)
            result = []
            for d in r.data:
                if isinstance(d.embedding, str):
                    raise ValueError("Expected float embeddings, got base64 string")
                result.append(d.embedding)
            return result
        except Exception as e:
            logger.error("Embed documents failed: %s", e)
            raise

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


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

    def _resolve_metrics(self, bc: dict[str, Any]) -> list:
        metric_names = (
            bc.get("metrics") or bc.get("scoring_functions") or DEFAULT_METRICS
        )
        metrics = [METRIC_MAPPING[n] for n in metric_names if n in METRIC_MAPPING]
        unknown = [n for n in metric_names if n not in METRIC_MAPPING]
        if unknown:
            logger.warning("Unknown metrics (skipped): %s", unknown)
        if not metrics:
            logger.info("No valid metrics specified, using defaults")
            metrics = [METRIC_MAPPING[m] for m in DEFAULT_METRICS]
        return metrics

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
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Validating configuration and resolving data path",
                        message_code="initializing",
                    ),
                )
            )
            self._validate_config(config)
            bc = config.parameters or {}

            # --- LOADING_DATA ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Loading evaluation dataset",
                        message_code="loading_data",
                    ),
                )
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
            metrics = self._resolve_metrics(bc)
            model_url = config.model.url.strip().rstrip("/")
            model_name = config.model.name
            embedding_model = bc.get("embedding_model") or model_name
            embedding_url = bc.get("embedding_url") or model_url

            run_config = RunConfig(max_workers=1)
            llm = EvalHubOpenAILLM(
                base_url=model_url,
                model_id=model_name,
                max_tokens=bc.get("max_tokens"),
                temperature=bc.get("temperature"),
                run_config=run_config,
            )
            embeddings = EvalHubOpenAIEmbeddings(
                base_url=embedding_url,
                model_id=embedding_model,
                run_config=run_config,
            )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running RAGAS evaluation ({len(metrics)} metrics)",
                        message_code="running_evaluation",
                    ),
                )
            )

            ragas_result = self._run_ragas(
                eval_dataset=eval_dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            # --- POST_PROCESSING ---
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.85,
                    message=MessageInfo(
                        message="Processing RAGAS results",
                        message_code="post_processing",
                    ),
                )
            )

            result_df = ragas_result.to_pandas()
            n_evaluated = len(result_df)
            evaluation_results: list[EvaluationResult] = []
            scores_for_overall: list[float] = []

            for metric_name in [m.name for m in metrics]:
                if metric_name not in result_df.columns:
                    continue
                series = result_df[metric_name].dropna()
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

            # --- PERSISTING_ARTIFACTS ---
            oci_artifact = None
            if config.exports and config.exports.oci:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.PERSISTING_ARTIFACTS,
                        progress=0.9,
                        message=MessageInfo(
                            message="Persisting results to OCI",
                            message_code="persisting_artifacts",
                        ),
                    )
                )
                results_dir = Path("/tmp/ragas_evalhub_results") / config.id
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
                    "data_path": str(data_path),
                    "metrics": [m.name for m in metrics],
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception("RAGAS EvalHub job %s failed", config.id)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error=ErrorInfo(message=str(e), message_code="job_failed"),
                    error_details={"exception_type": type(e).__name__},
                )
            )
            raise

    def _run_ragas(self, *, eval_dataset, metrics, llm, embeddings, run_config):
        return ragas_evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
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
