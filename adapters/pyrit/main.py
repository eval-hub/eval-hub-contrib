#!/usr/bin/env python3
"""PyRIT framework adapter for eval-hub.

Wraps PyRIT (https://github.com/Azure/PyRIT, MIT license) via headless
``pyrit_scan`` (pinned at 1.1.0), exposing the ``pyrit-scan`` safety
benchmark against an EvalHub-supplied OpenAI-compatible chat endpoint.

Flow:
1. Build job-local env (``OPENAI_CHAT_*`` + optional attacker/scorer) and a
   minimal ``.pyrit_conf`` (in-memory memory, target/scorer/dataset
   initializers, ``auth_mode: none``).
2. Run ``pyrit_scan run <scenario> --target openai_chat --start-server ...``.
3. Parse the scenario result id from CLI stdout, fetch
   ``/api/scenarios/runs/{id}/results`` as JSON while the detached backend
   is still up, then best-effort ``pyrit_scan stop-server``.
4. Map attack outcomes to ``pass_rate`` (= resisted / n_evaluated ==
   ``1 - attack_success_rate``).

Attacker/scorer upstreams and credentials are resolved from
``model.auth.secret_ref`` role keys (``attacker_url`` / ``attacker_api-key``,
``scorer_url`` / ``scorer_api-key``), not from plaintext job parameters.
The target API key written into the job-local ``.env`` must never appear in
``additional_info``, MLflow params, or OCI artifacts — only the scan
results JSON is exported.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
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
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials
from evalhub.adapter.mlflow import MlflowArtifact
from evalhub.adapter.models.cards import (
    EnvironmentCardMetadata,
    EvalCardMetadata,
    SafetyEvalEntry,
)

logger = logging.getLogger(__name__)

PYRIT_VERSION = "1.1.0"
_ADAPTER_VERSION = "0.1.0"

# Above this size, native scan JSON is NOT embedded in additional_info
# (still available via MLflow/OCI artifact paths when configured).
PYRIT_RESULTS_JSON_MAX_BYTES = 5_000_000

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_SCAN_TIMEOUT_SECONDS = 7_200
SUPPORTED_BENCHMARKS = frozenset({"pyrit-scan"})

# Legacy plaintext credential parameter names. Rejected if present so jobs
# cannot bypass model.auth.secret_ref role routing. Keep denylisted so any
# residual values never appear in additional_info / MLflow / OCI.
SECRET_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "attacker_provider_api_key",
        "scorer_provider_api_key",
    }
)

_RESULT_ID_RE = re.compile(
    r"Result ID:\s*([0-9a-fA-F-]{36}|\S+)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Credentials / URL helpers
# ---------------------------------------------------------------------------


def _strip_secret_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of job parameters with secret keys removed."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if k not in SECRET_PARAM_NAMES}


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


def _normalize_openai_base_url(url: str) -> str:
    """Normalize an OpenAI-compatible base URL for PyRIT OpenAIChatTarget.

    PyRIT expects a base such as ``http://host/v1`` and appends
    ``/chat/completions`` itself. Strip a trailing chat-completions path if
    the caller included one, and append ``/v1`` when missing.
    """
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError("config.model.url is required for the pyrit adapter")

    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("/")
            break

    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _reject_plaintext_aux_secrets(params: dict[str, Any] | None) -> None:
    """Fail fast if a job still passes plaintext attacker/scorer API keys."""
    if not params:
        return
    present = sorted(name for name in SECRET_PARAM_NAMES if params.get(name))
    if present:
        raise ValueError(
            "plaintext credential parameters are not supported: "
            + ", ".join(present)
            + ". Configure attacker_api-key / scorer_api-key (and optional "
            "attacker_url / scorer_url) in model.auth.secret_ref instead."
        )


def _role_connection(config: JobSpec, role: str, *, target_api_key: str) -> tuple[str, str]:
    """Resolve an auxiliary role URL/key from model.auth.secret_ref.

    Keys read from the projected secret directory:
    - ``{role}_url``
    - ``{role}_api-key``

    When the role URL is omitted, falls back to the target model URL. When the
    role API key is omitted, falls back to the target model's key (or
    ``not-required``).
    """
    role_key = read_model_auth_key(f"{role}_api-key")
    role_url = read_model_auth_key(f"{role}_url")
    if role_url and not role_key:
        raise ValueError(
            f"{role}_url in model.auth.secret_ref requires {role}_api-key"
        )
    base = (role_url or config.model.url or "").strip()
    endpoint = _normalize_openai_base_url(base)
    api_key = role_key or target_api_key or "not-required"
    return endpoint, api_key


def _resolve_scan_timeout(params: dict[str, Any] | None) -> int:
    """Return scan_timeout seconds, treating missing/null as the default."""
    if not params or params.get("scan_timeout") is None:
        return DEFAULT_SCAN_TIMEOUT_SECONDS
    return int(params["scan_timeout"])


def _role_metadata(config: JobSpec, env: dict[str, str]) -> dict[str, Any]:
    """Describe attacker/scorer routing (no secrets) for additional_info / cards."""
    target_model = config.model.name or ""
    target_endpoint = env.get("OPENAI_CHAT_ENDPOINT", "")
    attacker_model = env.get("ADVERSARIAL_CHAT_MODEL", "")
    scorer_model = env.get("OBJECTIVE_SCORER_CHAT_MODEL", "")
    attacker_endpoint = env.get("ADVERSARIAL_CHAT_ENDPOINT", "")
    scorer_endpoint = env.get("OBJECTIVE_SCORER_CHAT_ENDPOINT", "")
    attacker_uses_target = (
        attacker_model == target_model and attacker_endpoint == target_endpoint
    )
    scorer_uses_target = (
        scorer_model == target_model and scorer_endpoint == target_endpoint
    )
    return {
        "target_model": target_model,
        "attacker_model": attacker_model,
        "scorer_model": scorer_model,
        "attacker_uses_target": attacker_uses_target,
        "scorer_uses_target": scorer_uses_target,
    }


# ---------------------------------------------------------------------------
# Env / conf / argv
# ---------------------------------------------------------------------------


def _build_pyrit_env(config: JobSpec, work_dir: Path, api_key: str) -> dict[str, str]:
    """Build job-local env with OPENAI_CHAT_* (+ attacker/scorer roles) for pyrit_scan."""
    params = config.parameters or {}
    _reject_plaintext_aux_secrets(params)
    endpoint = _normalize_openai_base_url(config.model.url or "")
    if not config.model.name:
        raise ValueError("config.model.name is required for the pyrit adapter")

    env = os.environ.copy()
    # Keep PyRIT's ~/.pyrit + appdirs data under the job work dir.
    env["HOME"] = str(work_dir)
    env["OPENAI_CHAT_ENDPOINT"] = endpoint
    env["OPENAI_CHAT_KEY"] = api_key
    env["OPENAI_CHAT_MODEL"] = config.model.name

    attacker_url, attacker_key = _role_connection(
        config, "attacker", target_api_key=api_key
    )
    env["ADVERSARIAL_CHAT_ENDPOINT"] = attacker_url
    env["ADVERSARIAL_CHAT_KEY"] = attacker_key
    env["ADVERSARIAL_CHAT_MODEL"] = str(
        params.get("attacker_provider") or config.model.name
    )

    scorer_url, scorer_key = _role_connection(
        config, "scorer", target_api_key=api_key
    )
    env["OBJECTIVE_SCORER_CHAT_ENDPOINT"] = scorer_url
    env["OBJECTIVE_SCORER_CHAT_KEY"] = scorer_key
    env["OBJECTIVE_SCORER_CHAT_MODEL"] = str(
        params.get("scorer_provider") or config.model.name
    )

    role_info = _role_metadata(config, env)
    if role_info["attacker_uses_target"] or role_info["scorer_uses_target"]:
        logger.warning(
            "Auxiliary role(s) fall back to the target model "
            "(attacker_uses_target=%s, scorer_uses_target=%s). "
            "Self-grading may understate risk; set attacker_*/scorer_* "
            "secret_ref keys and optional parameters.attacker_provider / "
            "scorer_provider for independent models.",
            role_info["attacker_uses_target"],
            role_info["scorer_uses_target"],
        )

    return env


def _dotenv_payload(env: dict[str, str]) -> str:
    """Serialize PyRIT-relevant env vars into a dotenv document."""
    keys = [
        "OPENAI_CHAT_ENDPOINT",
        "OPENAI_CHAT_KEY",
        "OPENAI_CHAT_MODEL",
        "ADVERSARIAL_CHAT_ENDPOINT",
        "ADVERSARIAL_CHAT_KEY",
        "ADVERSARIAL_CHAT_MODEL",
        "OBJECTIVE_SCORER_CHAT_ENDPOINT",
        "OBJECTIVE_SCORER_CHAT_KEY",
        "OBJECTIVE_SCORER_CHAT_MODEL",
    ]
    lines = [f"{key}={env[key]}" for key in keys if key in env and env[key] is not None]
    return "\n".join(lines) + "\n"


def _build_pyrit_conf(*, env_file: Path, server_url: str = DEFAULT_SERVER_URL) -> dict[str, Any]:
    """Build minimal pyrit conf (in_memory memory + target/scorer/datasets)."""
    return {
        "memory_db_type": "in_memory",
        "silent": True,
        "env_files": [str(env_file)],
        "initializers": [
            # Order matters: techniques must be registered before scenario
            # catalog metadata is built (otherwise GET .../scenarios/catalog/*
            # returns 500: AttackTechniqueRegistry is empty).
            "technique",
            {"name": "target", "args": {"tags": ["default"]}},
            "scorer",
            "load_default_datasets",
        ],
        "server": {
            "url": server_url,
            "startup_timeout": 120,
            "auth_mode": "none",
        },
    }


def _resolve_scenario(config: JobSpec) -> str:
    """Return the required pyrit_scan scenario id from job parameters."""
    params = config.parameters or {}
    scenario = params.get("scenario")
    if scenario is None or not str(scenario).strip():
        raise ValueError(
            "parameters.scenario is required (pyrit_scan scenario id, e.g. "
            "'foundry.red_team_agent' or 'airt.leakage')"
        )
    return str(scenario).strip()


def _build_scan_argv(config: JobSpec, conf_path: Path) -> list[str]:
    """Build ``pyrit_scan run <scenario> --target openai_chat ...`` argv."""
    if config.benchmark_id not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark_id for pyrit adapter: {config.benchmark_id}"
        )

    scenario = _resolve_scenario(config)
    params = config.parameters or {}
    argv = [
        "pyrit_scan",
        "run",
        scenario,
        "--target",
        "openai_chat",
        "--start-server",
        "--auth-mode",
        "none",
        "--config-file",
        str(conf_path),
        "--server-url",
        DEFAULT_SERVER_URL,
    ]

    max_objectives = params.get("max_objectives")
    if max_objectives is not None:
        argv.extend(["--max-dataset-size", str(int(max_objectives))])

    techniques = params.get("techniques")
    if techniques:
        if isinstance(techniques, str):
            tech_list = [techniques]
        elif isinstance(techniques, list):
            tech_list = [str(t) for t in techniques]
        else:
            raise ValueError("parameters.techniques must be a string or list of strings")
        argv.extend(["--techniques", *tech_list])

    return argv


def _run_pyrit_scan(
    argv: list[str],
    env: dict[str, str],
    work_dir: Path,
    *,
    timeout: int = DEFAULT_SCAN_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run pyrit_scan via subprocess; return CompletedProcess[str]."""
    logger.info("Running: %s (cwd=%s)", " ".join(argv), work_dir)
    return subprocess.run(
        argv,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _stop_pyrit_server(conf_path: Path, env: dict[str, str], work_dir: Path) -> None:
    """Best-effort stop of the detached pyrit_backend started by --start-server."""
    argv = [
        "pyrit_scan",
        "stop-server",
        "--config-file",
        str(conf_path),
        "--server-url",
        DEFAULT_SERVER_URL,
        "--auth-mode",
        "none",
    ]
    try:
        result = subprocess.run(
            argv,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "pyrit_scan stop-server exited %s: %s",
                result.returncode,
                (result.stderr or result.stdout or "").strip(),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to stop pyrit server: %s", exc)


# ---------------------------------------------------------------------------
# Results fetch / metrics
# ---------------------------------------------------------------------------


def _parse_scenario_result_id(output: str) -> str | None:
    """Extract the scenario result UUID from pyrit_scan stdout/stderr."""
    match = _RESULT_ID_RE.search(output or "")
    return match.group(1) if match else None


def _fetch_scan_results(
    *,
    server_url: str,
    scenario_result_id: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """GET /api/scenarios/runs/{id}/results as a JSON dict."""
    base = server_url.rstrip("/")
    url = f"{base}/api/scenarios/runs/{scenario_result_id}/results"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _iter_attack_rows(scan_export: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ScenarioResult.attack_results into a list of outcome rows."""
    attack_results = scan_export.get("attack_results") or {}
    rows: list[dict[str, Any]] = []
    if isinstance(attack_results, dict):
        for atomic_name, results in attack_results.items():
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.setdefault("atomic_attack_name", atomic_name)
                rows.append(row)
    elif isinstance(attack_results, list):
        rows = [r for r in attack_results if isinstance(r, dict)]
    return rows


def _compute_metrics(
    scan_export: dict[str, Any],
) -> tuple[list[EvaluationResult], float | None, int]:
    """Map pyrit_scan / memory export JSON into pass_rate, n_evaluated, etc.

    Only ``success`` and ``failure`` outcomes are scored. ``error`` /
    ``undetermined`` probes are excluded from ``n_evaluated`` so a broken
    target/scorer endpoint cannot report ``pass_rate = 1.0``.

    ``pass_rate`` = failures / n_evaluated (== ``1 - attack_success_rate``).
    When nothing was scored (``n_evaluated == 0``), ``pass_rate`` is ``None``
    (unknown) — never ``1.0``.
    """
    rows = _iter_attack_rows(scan_export)
    pass_rate: float | None
    attack_success_rate: float | None

    if not rows:
        # Per-attack rows are required. CLI summary rates cannot distinguish
        # error/undetermined from true resistance, so they are never scored.
        n_evaluated = 0
        pass_rate = None
        attack_success_rate = None
    else:
        successes = 0
        failures = 0
        for row in rows:
            outcome = str(row.get("outcome", "")).lower()
            if outcome == "success":
                successes += 1
            elif outcome == "failure":
                failures += 1
        n_evaluated = successes + failures
        if n_evaluated == 0:
            # All rows were error/undetermined — not a resistance score.
            pass_rate = None
            attack_success_rate = None
        else:
            attack_success_rate = successes / n_evaluated
            pass_rate = failures / n_evaluated

    evaluation_results: list[EvaluationResult] = [
        EvaluationResult(
            metric_name="n_evaluated",
            metric_value=n_evaluated,
            metric_type="int",
        ),
    ]
    if pass_rate is not None:
        evaluation_results.insert(
            0,
            EvaluationResult(
                metric_name="pass_rate",
                metric_value=round(pass_rate, 6),
                metric_type="float",
            ),
        )
    if attack_success_rate is not None:
        evaluation_results.append(
            EvaluationResult(
                metric_name="attack_success_rate",
                metric_value=round(attack_success_rate, 6),
                metric_type="float",
            )
        )
    return evaluation_results, pass_rate, n_evaluated


def _build_eval_card(
    config: JobSpec,
    pass_rate: float | None,
    n_evaluated: int,
    *,
    role_info: dict[str, Any] | None = None,
) -> EvalCardMetadata:
    """Build EvalCardMetadata describing the PyRIT safety scan."""
    scenario = (config.parameters or {}).get("scenario", "unknown")
    zero_shot = round(pass_rate, 4) if pass_rate is not None else None
    roles = role_info or {}
    attacker_model = roles.get("attacker_model") or config.model.name
    scorer_model = roles.get("scorer_model") or config.model.name
    benchmark = (
        f"{config.benchmark_id}/{scenario} (pyrit {PYRIT_VERSION}), "
        f"n={n_evaluated}; attacker={attacker_model}; scorer={scorer_model}"
    )
    footnotes = [
        "PyRIT (MIT license) pyrit_scan scenario. pass_rate is the "
        "fraction of attack objectives the model resisted "
        "(== 1 - attack_success_rate).",
        f"Attacker model: {attacker_model}"
        + (" (same as target)" if roles.get("attacker_uses_target") else "")
        + f"; scorer model: {scorer_model}"
        + (" (same as target)" if roles.get("scorer_uses_target") else "")
        + ".",
        "Reference: https://github.com/Azure/PyRIT",
    ]
    return EvalCardMetadata(
        modalities_input=["text"],
        modalities_output=["text"],
        languages_count=1,
        languages=["en"],
        capability_evaluations=[],
        safety_evaluations=[
            SafetyEvalEntry(
                feature="adversarial scenario resistance",
                benchmark=benchmark,
                metric="pass_rate",
                zero_shot=zero_shot,
                alt_prompting=None,
                alt_prompting_description=None,
            )
        ],
        developer_footnotes=" ".join(footnotes),
    )


def _build_env_card(config: JobSpec) -> EnvironmentCardMetadata:
    """Build EnvironmentCardMetadata capturing the PyRIT runtime and model info."""
    env = EnvironmentCardMetadata.capture(
        framework_name="pyrit",
        framework_version=PYRIT_VERSION,
        extra_packages=["pyrit"],
    )
    env.model_id = config.model.name
    env.model_provider = "openai-compatible"
    return env


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PyritAdapter(FrameworkAdapter):
    """eval-hub FrameworkAdapter wrapping headless pyrit_scan."""

    def generate_additional_info(self, results: JobResults) -> dict[str, Any] | None:
        """Return a compact summary dict for job additional_info (no secrets)."""
        metric = {r.metric_name: r.metric_value for r in results.results}
        info: dict[str, Any] = {
            "pass_rate": metric.get("pass_rate"),
            "pyrit_version": PYRIT_VERSION,
        }
        # Defense in depth: never echo secret job parameters if a future
        # implementation copies parameters into additional_info.
        for key in SECRET_PARAM_NAMES:
            info.pop(key, None)
        return info

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run a pyrit-scan benchmark and return JobResults."""
        start_time = time.time()
        logger.info(
            "Starting pyrit job %s, benchmark=%s, model=%s",
            config.id,
            config.benchmark_id,
            config.model.name,
        )

        if config.benchmark_id not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark_id for pyrit adapter: {config.benchmark_id}"
            )

        work_dir: Path | None = None
        env: dict[str, str] | None = None
        conf_path: Path | None = None
        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Initializing pyrit adapter",
                        message_code="initializing",
                    ),
                )
            )

            api_key = _resolve_api_key(config)
            work_dir = Path(tempfile.mkdtemp(prefix="pyrit_"))
            # Nested home so PyRIT's ~/.pyrit and appdirs paths stay job-local.
            home_dir = work_dir / "home"
            home_dir.mkdir()
            env = _build_pyrit_env(config, home_dir, api_key)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.15,
                    message=MessageInfo(
                        message="Generating pyrit env/conf",
                        message_code="loading_data",
                    ),
                )
            )

            env_file = work_dir / ".env"
            env_file.write_text(_dotenv_payload(env), encoding="utf-8")
            conf = _build_pyrit_conf(env_file=env_file, server_url=DEFAULT_SERVER_URL)
            conf_path = work_dir / ".pyrit_conf"
            conf_path.write_text(yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")

            argv = _build_scan_argv(config, conf_path)
            scenario = _resolve_scenario(config)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"Running pyrit_scan scenario '{scenario}' against {config.model.url}",
                        message_code="running_evaluation",
                    ),
                )
            )

            scan_timeout = _resolve_scan_timeout(config.parameters)
            completed = _run_pyrit_scan(
                argv, env=env, work_dir=work_dir, timeout=scan_timeout
            )
            combined_output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"pyrit_scan failed (exit {completed.returncode})\n"
                    f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
                )

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Parsing pyrit_scan results",
                        message_code="post_processing",
                    ),
                )
            )

            scenario_result_id = _parse_scenario_result_id(combined_output)
            if not scenario_result_id:
                raise RuntimeError(
                    "Could not parse scenario result id from pyrit_scan output; "
                    "refusing to score from CLI summary "
                    "(summary rates cannot distinguish error/undetermined)."
                )
            try:
                scan_export = _fetch_scan_results(
                    server_url=DEFAULT_SERVER_URL,
                    scenario_result_id=scenario_result_id,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Failed to fetch pyrit scenario results for "
                    f"{scenario_result_id}: {exc}. "
                    "Refusing to score from CLI summary "
                    "(summary rates cannot distinguish error/undetermined)."
                ) from exc

            evaluation_results, pass_rate, n_evaluated = _compute_metrics(scan_export)
            if n_evaluated == 0:
                raise RuntimeError(
                    "pyrit_scan completed but no probes were scored "
                    "(all outcomes were error/undetermined, or attack_results "
                    "were empty). Check the target and scorer endpoints; "
                    "a broken endpoint must not report a completed score."
                )

            results_path = work_dir / "scan_results.json"
            results_json_raw = json.dumps(scan_export, default=str).encode("utf-8")
            results_path.write_bytes(results_json_raw)

            role_info = _role_metadata(config, env)
            additional_info: dict[str, Any] = {
                "scenario": scenario,
                "scenario_result_id": scenario_result_id
                or scan_export.get("scenario_result_id")
                or scan_export.get("id"),
                "pyrit_version": PYRIT_VERSION,
                "metrics_source": "results_api",
                **role_info,
            }
            # Never copy raw job parameters (may contain API keys).
            safe_params = _strip_secret_params(config.parameters)
            if "max_objectives" in safe_params:
                additional_info["max_objectives"] = safe_params["max_objectives"]
            if "techniques" in safe_params:
                additional_info["techniques"] = safe_params["techniques"]

            if len(results_json_raw) <= PYRIT_RESULTS_JSON_MAX_BYTES:
                additional_info["pyrit_scan_json"] = scan_export
            else:
                additional_info["pyrit_scan_json_omitted"] = True
                additional_info["pyrit_scan_json_size_bytes"] = len(results_json_raw)
                logger.warning(
                    "scan_results.json is %d bytes (> %d limit); omitting from "
                    "additional_info. Still available via MLflow/OCI artifacts "
                    "if those exports are configured.",
                    len(results_json_raw),
                    PYRIT_RESULTS_JSON_MAX_BYTES,
                )

            eval_card = _build_eval_card(
                config, pass_rate, n_evaluated, role_info=role_info
            )
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
                            message="Persisting pyrit artifacts",
                            message_code="persisting_artifacts",
                        ),
                    )
                )
                # Export ONLY scan_results.json — never work_dir (.env holds
                # plaintext API keys for target/attacker/scorer).
                artifact_dir = work_dir / "artifact"
                artifact_dir.mkdir()
                shutil.copy(results_path, artifact_dir / "scan_results.json")
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=artifact_dir, coordinates=oci_exports.coordinates
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
                    "framework": "pyrit",
                    "framework_version": PYRIT_VERSION,
                    "adapter_version": _ADAPTER_VERSION,
                    "scenario": scenario,
                    "scenario_result_id": additional_info.get("scenario_result_id"),
                    "pyrit_scan_json_b64": base64.b64encode(results_json_raw).decode(
                        "ascii"
                    ),
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
            logger.exception("pyrit evaluation failed")
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
            if conf_path is not None and env is not None and work_dir is not None:
                _stop_pyrit_server(conf_path, env, work_dir)
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to clean up %s: %s", work_dir, exc)


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

        scan_json_b64 = (results.evaluation_metadata or {}).get("pyrit_scan_json_b64")
        artifacts = None
        if scan_json_b64 is not None:
            artifacts = [
                MlflowArtifact(
                    "scan_results.json",
                    base64.b64decode(scan_json_b64),
                    "application/json",
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
