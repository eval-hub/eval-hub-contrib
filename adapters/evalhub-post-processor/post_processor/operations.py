"""Operation registry and prediction-powered confidence interval computation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from evalhub.adapter import JobCallbacks, JobPhase, JobSpec, JobStatus, JobStatusUpdate

from .config import ci_config, object_value, text_value
from .data import numeric, ppi_arrays, read_table
from .sources import Sources
from .transport import Sidecar


@dataclass
class Context:
    spec: JobSpec
    callbacks: JobCallbacks
    sidecar: Sidecar
    sources: Sources
    directory: Path

    def phase(self, phase: JobPhase) -> None:
        self.callbacks.report_status(JobStatusUpdate(status=JobStatus.RUNNING, phase=phase))


Operation = Callable[[Context, dict], dict]
OPERATIONS: dict[str, Operation] = {}


def register(name: str):
    def decorator(operation: Operation) -> Operation:
        if name in OPERATIONS:
            raise ValueError(f"Operation {name!r} is already registered")
        OPERATIONS[name] = operation
        return operation

    return decorator


def identity(benchmark: dict) -> tuple:
    index = benchmark.get("benchmark_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Benchmark metadata must include a nonnegative benchmark_index")
    return (
        text_value(benchmark.get("provider_id"), "benchmark provider_id"),
        text_value(benchmark.get("id"), "benchmark id"),
        index,
    )


def source_job(context: Context, job_id: str) -> dict:
    if job_id == context.spec.id:
        raise ValueError("A post-processing job cannot reference itself")
    job = context.sidecar.get(f"/api/v1/evaluations/jobs/{quote(job_id, safe='')}")
    state = job.get("status", {})
    if state.get("state") != "completed":
        raise ValueError("Source evaluation job must be completed successfully")
    statuses = state.get("benchmarks", [])
    benchmarks = job.get("results", {}).get("benchmarks", [])
    if not statuses or not benchmarks or any(b.get("status") != "completed" for b in statuses):
        raise ValueError("Every source benchmark must be completed and have stored results")
    status_ids = [identity(b) for b in statuses]
    result_ids = [identity(b) for b in benchmarks]
    if (
        len(set(result_ids)) != len(result_ids)
        or len(set(status_ids)) != len(status_ids)
        or set(status_ids) != set(result_ids)
    ):
        raise ValueError(
            "Source benchmark statuses and results do not have matching unique identities"
        )
    if len({item[2] for item in result_ids}) != len(result_ids):
        raise ValueError("Source benchmark indexes must be unique")
    return job


def primary_metric(context: Context, job: dict, result: dict) -> str:
    # The stored test records the metric actually used when the source job ran.
    metric = result.get("test", {}).get("primary_score_metric")
    if not metric:
        for index, benchmark in enumerate(job.get("benchmarks", [])):
            if (
                benchmark.get("benchmark_index", index) == result["benchmark_index"]
                and benchmark.get("id") == result["id"]
                and benchmark.get("provider_id") == result["provider_id"]
            ):
                metric = benchmark.get("primary_score", {}).get("metric")
                break
    if not metric:
        provider_id = quote(result["provider_id"], safe="")
        provider = context.sidecar.get(f"/api/v1/evaluations/providers/{provider_id}")
        for benchmark in provider.get("benchmarks", []):
            if benchmark.get("id") == result["id"]:
                metric = benchmark.get("primary_score", {}).get("metric")
                break
    metric = text_value(metric, "source benchmark primary_score.metric")
    schema = {
        entry["name"]: entry.get("type", "numeric") for entry in result.get("metrics_schema", [])
    }
    if schema.get(metric, "numeric") != "numeric":
        raise ValueError(f"Primary metric {metric!r} is not numeric")
    value = result.get("metrics", {}).get(metric)
    if value is None and result.get("test", {}).get("primary_score_metric") == metric:
        value = result["test"].get("primary_score")
    numeric(value, "stored primary score")
    return metric


def artifact_sources(job: dict, benchmark: dict) -> list[dict]:
    """Prefer MLflow when both exports exist, with OCI as an explicit fallback."""
    artifacts = object_value(benchmark.get("artifacts") or {}, "benchmark artifacts")
    candidates = []
    if job.get("experiment") is not None and benchmark.get("mlflow_run_id"):
        mlflow = artifacts.get("mlflow", {})
        path = mlflow.get("artifact_path", "") if isinstance(mlflow, dict) else ""
        candidates.append(
            {
                "mlflow": {
                    "run_id": benchmark["mlflow_run_id"],
                    "artifact_path": path,
                    "path_hint": benchmark.get("logs_path"),
                }
            }
        )
    export = (job.get("exports") or {}).get("oci")
    if export is not None:
        coords = dict(export.get("coordinates") or {})
        digest = artifacts.get("oci_digest")
        reference = artifacts.get("oci_reference")
        if reference:
            from oras.container import Container

            parsed = Container(reference.removeprefix("oci://"))
            coords.update(
                oci_host=parsed.registry,
                oci_repository="/".join(filter(None, (parsed.namespace, parsed.repository))),
                oci_tag=parsed.tag,
            )
            digest = digest or parsed.digest
        if not digest and not reference and not candidates:
            # The SDK creates a distinct default tag for each benchmark.
            # Never guess that exports.oci's job-level tag identifies this result.
            raise ValueError("OCI source benchmark is missing artifacts.oci_reference/oci_digest")
        if digest or reference:
            oci = {"coordinates": coords, "artifact_path": artifacts.get("artifact_path", "")}
            if digest:
                oci["digest"] = digest
            if export.get("k8s"):
                oci["k8s"] = export["k8s"]
            candidates.append({"oci": oci})
    if not candidates:
        raise ValueError(
            "Source benchmark needs experiment + mlflow_run_id or exports.oci + artifact coordinates"
        )
    return candidates


def result_config(config: dict, target: dict) -> dict:
    descriptor = config.get("results_data_config", {})
    if isinstance(descriptor, dict):
        return descriptor
    if not isinstance(descriptor, list):
        raise ValueError(
            "results_data_config must be an object or a list of selected configurations"
        )
    matches = []
    for entry in descriptor:
        entry = object_value(entry, "results_data_config entry")
        selection = entry.get("selection", {})
        if selection and all(str(target.get(k)) == str(v) for k, v in selection.items()):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("results_data_config must select exactly one format for each benchmark")
    return matches[0]


def interval(results, calibration, target: dict, *, shared: bool, alpha: float) -> dict:
    from ppi_py import ppi_mean_ci

    labels, predicted, unlabeled = ppi_arrays(results, calibration, target, shared=shared)
    # Original PPI (lambda=1), rather than estimating a PPI++ tuning parameter.
    lower, upper = ppi_mean_ci(labels, predicted, unlabeled, alpha=alpha, lam=1)
    # PPI returns length-one ndarrays for scalar mean estimation.
    lower = lower.item() if hasattr(lower, "item") else lower
    upper = upper.item() if hasattr(upper, "item") else upper
    lower = numeric(lower, "PPI lower bound")
    upper = numeric(upper, "PPI upper bound")
    if lower > upper:
        raise ValueError("PPI returned inverted confidence bounds")
    return {"lower": lower, "upper": upper}


@register("confidence_interval")
def confidence_interval(context: Context, raw: dict) -> dict:
    config = ci_config(raw)
    reference = config["results_data_ref"]
    context.phase(JobPhase.LOADING_DATA)
    job = source_job(context, reference["eval_job"]["id"]) if "eval_job" in reference else None
    calibration = []
    for index, ref in enumerate(config["calibration_data_ref"]):
        source_ref = {key: value for key, value in ref.items() if key != "data_config"}
        directory = context.sources.download(source_ref, context.directory / f"calibration-{index}")
        calibration.append(read_table(directory, ref.get("data_config", {})))
    alpha = config["significance_level"]
    if job is None:
        directory = context.sources.download(reference, context.directory / "results")
        data = read_table(directory, result_config(config, {}))
        metric = text_value(data.primary_score, "external result manifest primary_score.metric")
        context.phase(JobPhase.RUNNING_EVALUATION)
        return {
            "confidence_interval": interval(
                data, calibration, {"metric": metric}, shared=False, alpha=alpha
            )
        }

    # Both collection and explicit benchmark submissions use this authoritative list.
    benchmarks = job["results"]["benchmarks"]
    targets = []
    for benchmark in benchmarks:
        target = {
            "benchmark_id": benchmark["id"],
            "provider_id": benchmark["provider_id"],
            "benchmark_index": benchmark["benchmark_index"],
            "metric": primary_metric(context, job, benchmark),
        }
        targets.append((benchmark, target, artifact_sources(job, benchmark)))
    context.phase(JobPhase.RUNNING_EVALUATION)

    def compute(entry: tuple) -> dict:
        benchmark, target, candidates = entry
        errors = []
        for index, candidate in enumerate(candidates):
            directory = (
                context.directory / f"benchmark-{target['benchmark_index']}" / f"source-{index}"
            )
            try:
                data = read_table(
                    context.sources.download(candidate, directory), result_config(config, target)
                )
                bounds = interval(
                    data, calibration, target, shared=len(benchmarks) > 1, alpha=alpha
                )
                return {
                    "id": benchmark["id"],
                    "provider_id": benchmark["provider_id"],
                    "benchmark_index": benchmark["benchmark_index"],
                    "confidence_interval": bounds,
                }
            except Exception as exc:
                errors.append(f"{next(iter(candidate))}: {type(exc).__name__}: {exc}")
        raise ValueError(
            f"Cannot compute CI for benchmark index {target['benchmark_index']}: "
            + "; ".join(errors)
        )

    # executor.map preserves source result order even when downloads finish out of order.
    with ThreadPoolExecutor(max_workers=config["num_parallel_threads"]) as executor:
        results = list(executor.map(compute, targets))
    return {"benchmarks": results}
