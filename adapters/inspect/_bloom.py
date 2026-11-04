"""Bloom multi-step scenario generation for the Inspect AI adapter."""

import logging
import subprocess
from pathlib import Path

from evalhub.adapter import JobCallbacks, JobPhase, JobSpec, JobStatus, JobStatusUpdate, MessageInfo

from _benchmarks import BLOOM_TEMPLATE_MAP
from _routing import _is_ollama_endpoint, route_model, select_client

logger = logging.getLogger(__name__)


def bloom_prepare(
    config: JobSpec,
    work_dir: Path,
    env: dict[str, str],
    callbacks: JobCallbacks,
) -> Path:
    """Run bloom init + bloom scenarios, or return a pre-built behavior directory."""
    behavior_dir_param = config.parameters.get("behavior_dir")
    if behavior_dir_param:
        behavior_dir = Path(behavior_dir_param)
        if not behavior_dir.exists():
            raise FileNotFoundError(f"behavior_dir '{behavior_dir}' does not exist.")
        logger.info(f"Using pre-built behavior dir: {behavior_dir}")
        return behavior_dir

    behavior_dir = work_dir / "behavior"

    template = (
        config.parameters.get("bloom_template")
        or BLOOM_TEMPLATE_MAP.get(config.benchmark_id)
    )
    if not template:
        raise ValueError(
            "Bloom requires a bloom_template parameter or a built-in template mapping."
        )

    logger.info(f"Running bloom init --from {template} {behavior_dir}")
    init_result = subprocess.run(
        ["bloom", "init", "--from", template, str(behavior_dir)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    if init_result.returncode != 0:
        raise RuntimeError(
            f"bloom init failed (exit {init_result.returncode}): {init_result.stderr[-500:]}"
        )

    p = config.parameters
    scenarios_name = p.get("scenarios_model") or p.get("auditor_model") or "claude-sonnet-4-6"
    scenarios_base_url = p.get("scenarios_base_url") or None
    scenarios_api_key = p.get("scenarios_api_key") or None

    # bloom CLI doesn't support the JSON dict model spec (model_args etc.) —
    # use a bare client/model string, but resolve the client from the
    # scenarios-specific endpoint so requests go to the right provider.
    client = select_client(env, endpoint_url=scenarios_base_url)
    scenarios_model = route_model(scenarios_name, client)
    logger.info(f"Running bloom scenarios with model={scenarios_model}")

    # Overlay scenarios routing overrides onto a copy of the global env so the
    # bloom subprocess targets the correct provider without affecting other roles.
    scenarios_env = dict(env)
    if scenarios_base_url:
        if _is_ollama_endpoint(scenarios_base_url):
            scenarios_env["OLLAMA_BASE_URL"] = scenarios_base_url
        else:
            scenarios_env["OPENAI_BASE_URL"] = scenarios_base_url
    if scenarios_api_key:
        scenarios_env["OPENAI_API_KEY"] = scenarios_api_key

    callbacks.report_status(
        JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.LOADING_DATA,
            progress=0.4,
            message=MessageInfo(
                message=f"Generating Bloom scenarios from template '{template}'",
                message_code="generating_scenarios",
            ),
            current_step="bloom scenarios",
            total_steps=4,
            completed_steps=1,
        )
    )

    scenarios_result = subprocess.run(
        ["bloom", "scenarios", str(behavior_dir), "--model-role", f"scenarios={scenarios_model}"],
        env=scenarios_env, capture_output=True, text=True, timeout=600,
    )
    if scenarios_result.returncode != 0:
        raise RuntimeError(
            f"bloom scenarios failed (exit {scenarios_result.returncode}): "
            f"{scenarios_result.stderr[-500:]}"
        )

    logger.info(f"Bloom scenarios generated in {behavior_dir}")
    return behavior_dir
