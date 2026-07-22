"""NeMo Guardrails adapter for EvalHub.

Evaluates NeMo Guardrails configurations against classification datasets.
Starts a local NeMo Guardrails server, sends prompts to the /v1/guardrail/checks
endpoint, and computes accuracy, precision, recall, F1, and latency metrics.
"""

import asyncio
import atexit
import csv
import enum
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import signal
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import jq
import requests
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from evalhub.adapter import (
    EvaluationResult,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
)
from evalhub.adapter.callbacks import DefaultCallbacks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Version & hashing utilities
# ---------------------------------------------------------------------------

def _get_nemo_version() -> str:
    try:
        return importlib.metadata.version("nemoguardrails")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _get_nemo_commit() -> str | None:
    try:
        import nemoguardrails
        pkg_dir = os.path.dirname(os.path.abspath(nemoguardrails.__file__))
        candidate = pkg_dir
        for _ in range(5):
            git_dir = os.path.join(candidate, ".git")
            if os.path.exists(git_dir):
                head_file = os.path.join(git_dir, "HEAD")
                if os.path.isfile(head_file):
                    with open(head_file) as f:
                        head = f.read().strip()
                    if head.startswith("ref: "):
                        ref_path = os.path.join(git_dir, head[5:])
                        if os.path.isfile(ref_path):
                            with open(ref_path) as f:
                                return f.read().strip()[:12]
                    else:
                        return head[:12]
            candidate = os.path.dirname(candidate)
    except Exception:
        pass
    return None


def _hash_config_dir(config_path: str) -> str:
    hasher = hashlib.sha256()
    abs_path = os.path.abspath(config_path)
    file_entries = []
    for root, _dirs, files in os.walk(abs_path):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, abs_path)
            file_entries.append((rel, full))
    for rel_path, full_path in sorted(file_entries):
        hasher.update(rel_path.encode("utf-8"))
        with open(full_path, "rb") as f:
            hasher.update(f.read())
    return f"sha256:{hasher.hexdigest()}"


# ---------------------------------------------------------------------------
# Config path validation
# ---------------------------------------------------------------------------

def _validate_nemo_config_path(path: str) -> str:
    """Resolve and validate a NeMo config path (must be absolute)."""
    resolved = os.path.realpath(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"NeMo config directory not found: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _compile_transform(expr: str | None):
    if expr is None:
        return None
    return jq.compile(expr)


def _map_labels(raw_label, block_labels, pass_labels, transform=None) -> bool | None:
    label = transform.input_value(raw_label).first() if transform else raw_label
    for bl in block_labels:
        if label == bl or str(label) == str(bl):
            return True
    for pl in pass_labels:
        if label == pl or str(label) == str(pl):
            return False
    return None


def _balance_and_limit(samples: list[dict], eval_limit: int | None) -> list[dict]:
    blocked = [s for s in samples if s["expected_blocked"]]
    allowed = [s for s in samples if not s["expected_blocked"]]
    logger.info("  Class distribution: %d blocked, %d allowed", len(blocked), len(allowed))

    if eval_limit is None:
        return samples

    per_class = eval_limit // 2
    random.shuffle(blocked)
    random.shuffle(allowed)
    blocked = blocked[:per_class]
    allowed = allowed[:per_class]
    result = blocked + allowed
    random.shuffle(result)
    logger.info(
        "  After balancing (eval_limit=%d): %d blocked, %d allowed, %d total",
        eval_limit, len(blocked), len(allowed), len(result),
    )
    return result


def _load_huggingface(config: dict) -> list[dict]:
    from datasets import load_dataset

    split = config.get("split", "test")
    download_limit = config.get("download_limit")
    if download_limit and ":" not in split:
        split = f"{split}[:{download_limit}]"

    ds = load_dataset(config["hf_name"], name=config.get("subset"), split=split)
    prompt_col = config["prompt_column"]
    label_col = config["label_column"]
    block_labels = config["block_labels"]
    pass_labels = config["pass_labels"]
    transform = _compile_transform(config.get("label_transform"))

    samples = []
    for row in ds:
        expected = _map_labels(row[label_col], block_labels, pass_labels, transform)
        if expected is None:
            continue
        samples.append({
            "prompt": str(row[prompt_col]),
            "expected_blocked": expected,
        })
    return samples


def _load_csv(config: dict) -> list[dict]:
    path = config["csv_path"]
    if not os.path.isabs(path):
        datasets_dir = os.environ.get("NEMO_DATASETS_DIR", "datasets")
        path = os.path.join(datasets_dir, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    prompt_col = config["prompt_column"]
    label_col = config["label_column"]
    block_labels = config["block_labels"]
    pass_labels = config["pass_labels"]
    transform = _compile_transform(config.get("label_transform"))
    download_limit = config.get("download_limit")

    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if download_limit and i >= download_limit:
                break
            expected = _map_labels(row[label_col], block_labels, pass_labels, transform)
            if expected is None:
                continue
            samples.append({
                "prompt": row[prompt_col],
                "expected_blocked": expected,
            })
    return samples


def load_samples(dataset_config: dict) -> list[dict]:
    source = dataset_config["source"]
    if source == "huggingface":
        samples = _load_huggingface(dataset_config)
    elif source == "csv":
        samples = _load_csv(dataset_config)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}. Use 'huggingface' or 'csv'.")
    return _balance_and_limit(samples, dataset_config.get("eval_limit"))


# ---------------------------------------------------------------------------
# NeMo server management
# ---------------------------------------------------------------------------

def _find_config_id(config_path: str) -> str:
    if os.path.isfile(os.path.join(config_path, "config.yml")):
        return os.path.basename(os.path.abspath(config_path))

    subdirs = [
        d for d in os.listdir(config_path)
        if os.path.isdir(os.path.join(config_path, d))
        and os.path.isfile(os.path.join(config_path, d, "config.yml"))
    ]
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No config.yml found in {config_path} or its subdirectories")
    raise ValueError(
        f"Multiple config directories found in {config_path}: {subdirs}. "
        "Point nemo_config at a specific config directory."
    )


def _resolve_server_config_path(config_path: str) -> tuple[str, str]:
    abs_path = os.path.abspath(config_path)
    if os.path.isfile(os.path.join(abs_path, "config.yml")):
        return os.path.dirname(abs_path), os.path.basename(abs_path)
    config_id = _find_config_id(abs_path)
    return abs_path, config_id


def _wait_for_server(host: str, port: int, timeout: int = 60) -> None:
    url = f"http://{host}:{port}/v1/rails/configs"
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.ConnectionError as e:
            last_error = e
        time.sleep(1)
    raise TimeoutError(
        f"NeMo Guardrails server did not become healthy within {timeout}s. "
        f"Last error: {last_error}"
    )


def _ensure_nemo_examples_dir():
    """Create the examples/bots directory that NeMo's server import expects."""
    import nemoguardrails.utils as _u
    bots_dir = os.path.normpath(
        os.path.join(os.path.dirname(_u.__file__), "..", "examples", "bots")
    )
    os.makedirs(bots_dir, exist_ok=True)


def _start_server(
    config_path: str, port: int = 9999, host: str = "localhost",
    log_path: str | None = None,
) -> tuple[subprocess.Popen, str]:
    _ensure_nemo_examples_dir()
    server_root, config_id = _resolve_server_config_path(config_path)
    cmd = [
        "nemoguardrails", "server",
        "--config", server_root,
        "--default-config-id", config_id,
        "--port", str(port),
        "--verbose",
    ]
    if log_path is None:
        log_path = os.path.join(server_root, "server.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    return proc, os.path.abspath(log_path)


def warmup_server(host: str, port: int, attempts: int = 30, timeout: int = 15) -> None:
    url = f"http://{host}:{port}/v1/guardrail/checks"
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hello"}],
    }
    last_error = None
    for _ in range(attempts):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return
        except (requests.RequestException, requests.HTTPError) as e:
            last_error = e
            time.sleep(1)
    raise TimeoutError(f"Server warm-up failed after {attempts} attempts. Last error: {last_error}")


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextmanager
def managed_server(config_path: str, port: int = 9999, host: str = "localhost",
                   startup_timeout: int = 60, log_path: str | None = None):
    proc, resolved_log_path = _start_server(config_path, port, host, log_path)

    def _cleanup(*_args):
        if proc.poll() is None:
            _stop_server(proc)

    atexit.register(_cleanup)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        _cleanup()
        if callable(prev_sigterm):
            prev_sigterm(signum, frame)
        else:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        _wait_for_server(host, port, startup_timeout)
        yield f"http://{host}:{port}", resolved_log_path
    finally:
        _cleanup()
        atexit.unregister(_cleanup)
        signal.signal(signal.SIGTERM, prev_sigterm)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class NemoResponses(enum.Enum):
    ALLOW = "success"
    BLOCKED = "blocked"
    MODIFIED = "modified"
    ERROR = "error"


def _evaluate_prompt(server_url: str, prompt: str) -> dict:
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{server_url}/v1/guardrail/checks", json=payload, timeout=120)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:500]
            return {
                "predicted_blocked": NemoResponses.ERROR,
                "response_time_ms": round(elapsed_ms, 1),
                "response_time_ms_per_character": elapsed_ms / len(prompt),
                "error": f"HTTP {r.status_code}: {detail}",
            }

        data = r.json()
        return {
            "predicted_blocked": NemoResponses(data.get("status")),
            "response_time_ms": round(elapsed_ms, 1),
            "response_time_ms_per_character": elapsed_ms / len(prompt),
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "predicted_blocked": NemoResponses.ERROR,
            "response_time_ms": round(elapsed_ms, 1),
            "response_time_ms_per_character": elapsed_ms / len(prompt),
            "error": str(e),
        }


async def _evaluate_prompt_async(server_url: str, prompt: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _evaluate_prompt, server_url, prompt)


def run_evaluation(
    server_url: str,
    samples: list[dict],
    *,
    workers: int = 1,
    verbose: bool = False,
    no_color: bool = True,
) -> list[dict]:
    if workers <= 1:
        results = []
        errors = 0
        pbar = None if verbose else tqdm(samples, desc="Evaluating", unit="sample", dynamic_ncols=True)
        for i, sample in enumerate(samples):
            result = _evaluate_prompt(server_url, sample["prompt"])
            result["prompt"] = sample["prompt"]
            result["expected_blocked"] = sample["expected_blocked"]
            results.append(result)
            if result["error"]:
                errors += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix(errors=errors, refresh=False)
            if verbose:
                _print_result(i, len(samples), sample, result, no_color)
        if pbar:
            pbar.close()
        return results

    async def _run():
        sem = asyncio.Semaphore(workers)
        total = len(samples)
        results = [None] * total
        errors = 0
        pbar = None if verbose else tqdm(total=total, desc="Evaluating", unit="sample", dynamic_ncols=True)

        async def _process(i, sample):
            nonlocal errors
            result = await _evaluate_prompt_async(server_url, sample["prompt"], sem)
            result["prompt"] = sample["prompt"]
            result["expected_blocked"] = sample["expected_blocked"]
            results[i] = result
            if result["error"]:
                errors += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix(errors=errors, refresh=False)
            if verbose:
                _print_result(i, total, sample, result, no_color)

        await asyncio.gather(*[_process(i, s) for i, s in enumerate(samples)])
        if pbar:
            pbar.close()
        return results

    return asyncio.run(_run())


def _color(code: str, text: str, no_color: bool) -> str:
    if no_color:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_result(i: int, total: int, sample: dict, result: dict, no_color: bool = False) -> None:
    expected_blocked = sample["expected_blocked"]
    predicted = result["predicted_blocked"]
    expected_str = NemoResponses.BLOCKED.value if expected_blocked else NemoResponses.ALLOW.value
    predicted_str = predicted.value

    is_error = predicted == NemoResponses.ERROR
    is_correct = (
        (expected_blocked and predicted != NemoResponses.ALLOW)
        or (not expected_blocked and predicted == NemoResponses.ALLOW)
    )

    if is_error:
        marker = _color("33", "x ERR        ", no_color)
    elif is_correct:
        marker = "v            "
    elif expected_blocked:
        marker = _color("31", "x False Neg  ", no_color)
    else:
        marker = _color("35", "x False Pos  ", no_color)

    idx = f"[{i + 1}/{total}]"
    per_char = result.get("response_time_ms_per_character", 0)
    line = f"  {idx:>10s} {marker} expected={expected_str:5s} got={predicted_str:8s} | {result['response_time_ms']:>5.0f}ms ({per_char:.2f}ms/char)"
    line += f" | {repr(sample['prompt'][:100])}"
    if result["error"]:
        line += f"\n{'':>20s}ERROR: {result['error'][:200]}"
    print(line)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(results: list[dict]) -> tuple[dict, list]:
    y_true = []
    y_pred = []
    others = []

    for r in results:
        expected = r["expected_blocked"]
        actual = r["predicted_blocked"]
        if actual == NemoResponses.ERROR:
            others.append(r)
            continue
        y_true.append("blocked" if expected else "allowed")
        y_pred.append("allowed" if actual == NemoResponses.ALLOW else "blocked")

    if not y_true:
        return {
            "total": 0, "errors": len(others), "accuracy": 0.0,
            "classification_report": {}, "confusion_matrix": {},
        }, others

    labels = ["blocked", "allowed"]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "total": len(y_true),
        "errors": len(others),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "classification_report": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"],
            }
            for label in labels
        },
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }, others


def _percentile(sorted_vals: list[float], pct: float) -> float:
    idx = int(len(sorted_vals) * pct)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _compute_timing_stats(results: list[dict]) -> dict:
    times = [r["response_time_ms"] for r in results if r.get("response_time_ms") is not None]
    per_char = [r["response_time_ms_per_character"] for r in results if r.get("response_time_ms_per_character") is not None]

    if not times:
        return {
            "mean_ms": 0, "median_ms": 0, "p95_ms": 0, "total_ms": 0,
            "per_character": {"mean_ms": 0, "median_ms": 0, "p95_ms": 0},
        }

    times_sorted = sorted(times)
    per_char_sorted = sorted(per_char)
    return {
        "mean_ms": round(statistics.mean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(_percentile(times_sorted, 0.95), 1),
        "total_ms": round(sum(times), 1),
        "per_character": {
            "mean_ms": round(statistics.mean(per_char), 4),
            "median_ms": round(statistics.median(per_char), 4),
            "p95_ms": round(_percentile(per_char_sorted, 0.95), 4),
        },
    }


# ---------------------------------------------------------------------------
# Benchmark dataset loading from provider.yaml
# ---------------------------------------------------------------------------

def _load_benchmark_datasets(benchmark_id: str) -> list[dict]:
    provider_path = os.path.join(ADAPTER_DIR, "provider.yaml")
    with open(provider_path) as f:
        provider = yaml.safe_load(f)

    for bench in provider.get("benchmarks", []):
        if bench["id"] == benchmark_id:
            datasets = bench.get("datasets")
            if not datasets:
                raise ValueError(f"Benchmark '{benchmark_id}' has no 'datasets' entries")
            return datasets

    available = [b["id"] for b in provider.get("benchmarks", [])]
    raise ValueError(f"Benchmark '{benchmark_id}' not found. Available: {available}")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class NemoGuardrailsAdapter(FrameworkAdapter):
    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.monotonic()
        params = config.parameters or {}

        nemo_config_name = params.get("nemo_config", config.benchmark_id)
        server_port = int(params.get("server_port", 9999))
        server_host = params.get("server_host", "localhost")
        startup_timeout = int(params.get("startup_timeout", 120))
        workers = int(params.get("workers", 1))
        verbose = str(params.get("verbose", "false")).lower() in ("true", "1", "yes")

        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
        )

        config_path = _validate_nemo_config_path(nemo_config_name)
        nemo_version = _get_nemo_version()
        nemo_commit = _get_nemo_commit()
        config_hash = _hash_config_dir(config_path)

        logger.info("NeMo Guardrails version: %s (commit: %s)", nemo_version, nemo_commit)
        logger.info("Config: %s (hash: %s)", config_path, config_hash)

        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.LOADING_DATA)
        )

        dataset_configs = _load_benchmark_datasets(config.benchmark_id)
        samples = []
        for dc in dataset_configs:
            logger.info("Loading dataset: %s (%s)", dc.get("name", "unnamed"), dc["source"])
            ds_samples = load_samples(dc)
            logger.info("  Loaded %d samples", len(ds_samples))
            samples.extend(ds_samples)

        logger.info("Total samples: %d", len(samples))

        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.RUNNING_EVALUATION)
        )

        logger.info("Starting NeMo server on port %d", server_port)

        with managed_server(config_path, server_port, server_host, startup_timeout) as (server_url, _):
            logger.info("NeMo server ready at %s", server_url)
            logger.info("Warming up server...")
            warmup_server(server_host, server_port)
            logger.info("Server warm-up complete")
            logger.info("Evaluating %d samples with %d worker(s)", len(samples), workers)

            results = run_evaluation(
                server_url, samples, workers=workers, verbose=verbose, no_color=True
            )

        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.POST_PROCESSING)
        )

        duration = time.monotonic() - start_time
        logger.info("Evaluation finished in %.1fs, processing %d results", duration, len(results))

        metrics, _ = _compute_metrics(results)
        timing = _compute_timing_stats(results)

        logger.info(
            "Metrics: total=%d, errors=%d, accuracy=%.4f",
            metrics["total"], metrics["errors"], metrics["accuracy"],
        )

        eval_results = [
            EvaluationResult(metric_name="accuracy", metric_value=metrics["accuracy"]),
        ]

        cr = metrics.get("classification_report", {})
        for label in ["blocked", "allowed"]:
            if label in cr:
                for metric_key in ["precision", "recall", "f1"]:
                    eval_results.append(
                        EvaluationResult(
                            metric_name=f"{label}_{metric_key}",
                            metric_value=cr[label][metric_key],
                        )
                    )

        eval_results.extend([
            EvaluationResult(metric_name="mean_latency_ms", metric_value=timing["mean_ms"]),
            EvaluationResult(metric_name="median_latency_ms", metric_value=timing["median_ms"]),
            EvaluationResult(metric_name="p95_latency_ms", metric_value=timing["p95_ms"]),
            EvaluationResult(metric_name="errors", metric_value=float(metrics["errors"])),
        ])

        return JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name if config.model else "nemo-guardrails",
            results=eval_results,
            overall_score=metrics["accuracy"],
            num_examples_evaluated=metrics["total"],
            duration_seconds=duration,
            completed_at=datetime.now(UTC),
            evaluation_metadata={
                "framework": "nemo-guardrails",
                "framework_version": "0.1.0",
                "nemo_version": nemo_version,
                "nemo_commit": nemo_commit,
                "config_hash": config_hash,
                "nemo_config": nemo_config_name,
                "datasets": [dc.get("name", dc.get("hf_name", "unknown")) for dc in dataset_configs],
                "workers": workers,
                "errors": metrics["errors"],
                "confusion_matrix": metrics.get("confusion_matrix", {}),
                "timing": timing,
                "parameters": params,
            },
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting NeMo Guardrails adapter")
    try:
        job_spec_path = os.environ.get("EVALHUB_JOB_SPEC_PATH", os.path.join(ADAPTER_DIR, "meta", "job.json"))
        adapter = NemoGuardrailsAdapter(job_spec_path=job_spec_path)
        callbacks = DefaultCallbacks.from_adapter(adapter)
        callbacks.report_status(
            JobStatusUpdate(status=JobStatus.RUNNING, phase=JobPhase.INITIALIZING)
        )
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        callbacks.report_results(results)
        logger.info("EVALUATION COMPLETE")
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
