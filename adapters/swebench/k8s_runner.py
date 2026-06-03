"""Kubernetes-based evaluation backend for SWE-bench.

Runs each evaluation instance as a K8s Job, replacing the Docker-based
``run_instance`` with a cloud-native equivalent.  Grading reuses the
existing ``get_eval_report`` infrastructure.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_WORKDIR,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import (
    TestSpec,
    get_test_specs_from_dataset,
)

logger = logging.getLogger(__name__)

# Patch application strategies, same as run_evaluation.py
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

# Max length for K8s resource names
_K8S_NAME_MAX = 63


def _job_name(instance_id: str, run_id: str) -> str:
    """Generate a DNS-safe K8s Job name from an instance ID."""
    safe = instance_id.lower().replace("__", "-").replace("_", "-")
    suffix = run_id[:8] if run_id else ""
    base = f"sweb-{safe}"
    max_base = _K8S_NAME_MAX - len(suffix) - 1
    base = base[:max_base].rstrip("-")
    return f"{base}-{suffix}" if suffix else base


def _image_ref(test_spec: TestSpec, registry: str) -> str:
    """Build the full image reference from a TestSpec and registry prefix.

    Handles the ``namespace/sweb.eval.x86_64.<id>:tag`` format that
    ``instance_image_key`` produces and prepends the user-supplied registry.

    Image names use the ``_1776_`` convention to replace ``__``, which is
    the standard SWE-bench naming for container images across all registries
    (Docker Hub, Quay, OpenShift internal registry, etc.).
    """
    key = test_spec.instance_image_key
    # Strip any existing namespace prefix
    if "/" in key:
        _, _, image_name = key.partition("/")
    else:
        image_name = key

    image_name = image_name.replace("__", "_1776_").lower()

    return f"{registry}/{image_name}"


def _create_eval_script(test_spec: TestSpec, patch: str) -> str:
    """Build a self-contained bash script that applies the patch and runs tests."""
    apply_lines = []
    for i, cmd in enumerate(GIT_APPLY_CMDS):
        keyword = "if" if i == 0 else "elif"
        apply_lines.append(f'{keyword} {cmd} /tmp/patch.diff; then')
        apply_lines.append(f'  echo "{APPLY_PATCH_PASS}"')
    apply_lines.append("else")
    apply_lines.append(f'  echo "{APPLY_PATCH_FAIL}"')
    apply_lines.append("  exit 1")
    apply_lines.append("fi")

    return "\n".join([
        "#!/bin/bash",
        "set -uxo pipefail",
        "",
        "export HOME=/tmp",
        'export PATH="$HOME/.local/bin:$PATH"',
        f"cd {DOCKER_WORKDIR}",
        f'git config --global --add safe.directory {DOCKER_WORKDIR}',
        # Ignore file mode changes from the init container chown
        "git config core.fileMode false",
        "",
        "# Write model patch",
        "cat > /tmp/patch.diff << 'SWEBENCH_PATCH_EOF'",
        patch,
        "SWEBENCH_PATCH_EOF",
        "",
        "# Apply patch",
        "\n".join(apply_lines),
        "",
        "# Git diff before eval",
        f"cd {DOCKER_WORKDIR} && git -c core.fileMode=false diff > /tmp/diff_before.txt",
        "",
        "# Run eval script",
        test_spec.eval_script,
        "",
        "# Git diff after eval",
        f"cd {DOCKER_WORKDIR} && git -c core.fileMode=false diff > /tmp/diff_after.txt",
    ])


def _create_job(
    test_spec: TestSpec,
    prediction: dict,
    run_id: str,
    registry: str,
    timeout: int,
    service_account: str | None = None,
) -> k8s_client.V1Job:
    """Build a V1Job manifest for evaluating a single instance."""
    instance_id = test_spec.instance_id
    image = _image_ref(test_spec, registry)
    patch = prediction.get(KEY_PREDICTION, "") or ""
    script = _create_eval_script(test_spec, patch)
    name = _job_name(instance_id, run_id)

    return k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(
            name=name,
            labels={
                "app": "swebench-eval",
                "swebench/run-id": run_id[:_K8S_NAME_MAX],
                "swebench/instance-id": instance_id.replace("__", "-")[:_K8S_NAME_MAX],
            },
        ),
        spec=k8s_client.V1JobSpec(
            backoff_limit=0,
            active_deadline_seconds=timeout,
            ttl_seconds_after_finished=3600,
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(
                    labels={
                        "app": "swebench-eval",
                        "swebench/run-id": run_id[:_K8S_NAME_MAX],
                    },
                ),
                spec=k8s_client.V1PodSpec(
                    restart_policy="Never",
                    service_account_name=service_account,
                    # Init container runs as root to fix permissions
                    # on /testbed so the eval container can run as a
                    # non-root UID on OpenShift.
                    init_containers=[
                        k8s_client.V1Container(
                            name="fix-permissions",
                            image=image,
                            image_pull_policy="IfNotPresent",
                            command=["/bin/sh", "-c",
                                     f"chown -R 1001:0 {DOCKER_WORKDIR} 2>/dev/null; "
                                     "chown -R 1001:0 /opt/miniconda3 2>/dev/null; "
                                     "chown -R 1001:0 /tmp 2>/dev/null; "
                                     "true"],
                            security_context=k8s_client.V1SecurityContext(
                                allow_privilege_escalation=False,
                                run_as_user=0,
                                run_as_non_root=False,
                                capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
                                privileged=False,
                            ),
                        ),
                    ],
                    containers=[
                        k8s_client.V1Container(
                            name="eval",
                            image=image,
                            image_pull_policy="IfNotPresent",
                            command=["/bin/bash", "-c"],
                            args=[script],
                            working_dir=DOCKER_WORKDIR,
                            security_context=k8s_client.V1SecurityContext(
                                allow_privilege_escalation=False,
                                run_as_non_root=True,
                                run_as_user=1001,
                                capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
                            ),
                        ),
                    ],
                ),
            ),
        ),
    )


def _get_pod_logs(core_v1: k8s_client.CoreV1Api, job_name: str, namespace: str) -> str:
    """Retrieve logs from the pod created by a Job."""
    try:
        pods = core_v1.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}",
        )
        if pods.items:
            pod_name = pods.items[0].metadata.name
            try:
                logs = core_v1.read_namespaced_pod_log(pod_name, namespace)
            except ApiException:
                # Retry with size limit — some API servers reject
                # very large log responses (e.g. matplotlib tests).
                logger.warning("Retrying log fetch for %s with 10MB limit", job_name)
                logs = core_v1.read_namespaced_pod_log(
                    pod_name, namespace, limit_bytes=10 * 1024 * 1024,
                )
            # Some versions of the kubernetes Python client return pod
            # logs with literal escape sequences (``\\n``, ``\\x1b``,
            # etc.) instead of real characters.  Decode them so
            # downstream log parsers work correctly.
            if isinstance(logs, str) and ("\\n" in logs or "\\x1b" in logs):
                import re
                def _decode_escape(m):
                    return bytes([int(m.group(1), 16)]).decode("latin-1")
                # Decode \\xNN hex escapes (e.g. \\x1b -> ESC)
                logs = re.sub(r"\\x([0-9a-fA-F]{2})", _decode_escape, logs)
                # Decode common character escapes
                logs = logs.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'")
            return logs
    except ApiException as e:
        logger.error("Error getting logs for job %s: %s", job_name, e.reason)
    return ""


def _wait_for_job(
    batch_v1: k8s_client.BatchV1Api,
    job_name: str,
    namespace: str,
    poll_interval: int = 10,
) -> bool:
    """Poll a Job until completion. Returns True if succeeded."""
    while True:
        try:
            job = batch_v1.read_namespaced_job(job_name, namespace)
        except ApiException as e:
            logger.error("Error reading job %s: %s", job_name, e.reason)
            return False

        if job.status.succeeded and job.status.succeeded > 0:
            return True
        if job.status.failed and job.status.failed > 0:
            return False

        time.sleep(poll_interval)


def _delete_job(
    batch_v1: k8s_client.BatchV1Api, job_name: str, namespace: str,
) -> None:
    """Delete a single Job and its pods."""
    try:
        batch_v1.delete_namespaced_job(
            job_name, namespace, propagation_policy="Background",
        )
        logger.debug("Deleted job %s", job_name)
    except ApiException as e:
        if e.status != 404:
            logger.warning("Failed to delete job %s: %s", job_name, e.reason)


def run_instance_k8s(
    test_spec: TestSpec,
    prediction: dict,
    run_id: str,
    registry: str,
    namespace: str,
    timeout: int,
    batch_v1: k8s_client.BatchV1Api,
    core_v1: k8s_client.CoreV1Api,
    image_tag: str = "latest",
    service_account: str | None = None,
) -> dict:
    """Evaluate a single instance via a K8s Job.

    Returns dict with ``completed`` and ``resolved`` keys.
    """
    instance_id = test_spec.instance_id
    model_name = prediction.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name / instance_id

    # Skip if already evaluated
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report.get(instance_id, {}).get("resolved", False),
        }

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create and submit the Job
    job = _create_job(test_spec, prediction, run_id, registry, timeout, service_account=service_account)
    job_name = job.metadata.name

    try:
        batch_v1.create_namespaced_job(namespace, job)
        logger.info("Created job %s for %s", job_name, instance_id)
    except ApiException as e:
        if e.status == 409:
            logger.info("Job %s already exists, waiting for it", job_name)
        else:
            logger.error("Error creating job %s: %s", job_name, e.reason)
            return {"completed": False, "resolved": False}

    # Wait, collect logs, then clean up the Job immediately
    try:
        success = _wait_for_job(batch_v1, job_name, namespace)
        logs = _get_pod_logs(core_v1, job_name, namespace)
    finally:
        _delete_job(batch_v1, job_name, namespace)

    # Save test output
    test_output_path = log_dir / LOG_TEST_OUTPUT
    test_output_path.write_text(logs)

    if not success:
        logger.info(
            "Job %s exited non-zero for %s (%s)",
            job_name, instance_id,
            "patch apply failed" if APPLY_PATCH_FAIL in logs else "non-zero exit",
        )
        if APPLY_PATCH_FAIL in logs:
            return {"completed": False, "resolved": False}

    if not logs.strip():
        logger.error("No logs captured for %s", instance_id)
        return {"completed": False, "resolved": False}

    # Grade
    try:
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)
        logger.info("Result for %s: resolved=%s", instance_id, resolved)

        if not resolved:
            instance_report = report.get(instance_id, {})
            tests_status = instance_report.get("tests_status", {})
            if tests_status:
                f2p = tests_status.get("FAIL_TO_PASS", {})
                p2p = tests_status.get("PASS_TO_PASS", {})
                logger.info(
                    "Unresolved %s: f2p=%d/%d p2p=%d/%d",
                    instance_id,
                    len(f2p.get("success", [])),
                    len(f2p.get("success", [])) + len(f2p.get("failure", [])),
                    len(p2p.get("success", [])),
                    len(p2p.get("success", [])) + len(p2p.get("failure", [])),
                )

        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))

        return {"completed": True, "resolved": resolved}
    except Exception as e:
        logger.error("Grading error for %s: %s", instance_id, e)
        return {"completed": False, "resolved": False}


def cleanup_jobs(
    batch_v1: k8s_client.BatchV1Api, run_id: str, namespace: str,
) -> None:
    """Delete all Jobs for a given run."""
    try:
        jobs = batch_v1.list_namespaced_job(
            namespace, label_selector=f"swebench/run-id={run_id}",
        )
        for job in jobs.items:
            batch_v1.delete_namespaced_job(
                job.metadata.name, namespace, propagation_policy="Background",
            )
            logger.debug("Deleted job %s", job.metadata.name)
    except ApiException as e:
        logger.error("Error cleaning up jobs: %s", e.reason)


def run_instances_k8s(
    predictions: dict,
    dataset: list,
    run_id: str,
    registry: str,
    namespace: str,
    max_workers: int,
    timeout: int,
    cleanup: bool = True,
    instance_image_tag: str = "latest",
    service_account: str | None = None,
) -> dict:
    """Run all instances as K8s Jobs in parallel.

    This is the main entry point called by the eval-hub adapter.
    Images must already exist in the registry specified by ``registry``.

    Returns dict mapping instance_id to result.
    """
    # Load K8s config
    try:
        k8s_config.load_incluster_config()
        logger.info("Using in-cluster K8s config")
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
        logger.info("Using kubeconfig")

    batch_v1 = k8s_client.BatchV1Api()
    core_v1 = k8s_client.CoreV1Api()

    test_specs = get_test_specs_from_dataset(dataset, instance_image_tag=instance_image_tag)
    logger.info("Running evaluation for %d instances (max_workers=%d)", len(test_specs), max_workers)

    stats = {"completed": 0, "resolved": 0, "error": 0}
    results = {}
    lock = threading.Lock()

    def process(ts: TestSpec):
        pred = predictions[ts.instance_id]
        result = run_instance_k8s(
            test_spec=ts,
            prediction=pred,
            run_id=run_id,
            registry=registry,
            namespace=namespace,
            timeout=timeout,
            batch_v1=batch_v1,
            core_v1=core_v1,
            image_tag=instance_image_tag,
            service_account=service_account,
        )
        with lock:
            results[ts.instance_id] = result
            if result["completed"]:
                stats["completed"] += 1
                if result["resolved"]:
                    stats["resolved"] += 1
            else:
                stats["error"] += 1
            total = stats["completed"] + stats["error"]
            logger.info(
                "Progress: %d/%d (resolved: %d, errors: %d)",
                total, len(test_specs), stats["resolved"], stats["error"],
            )

    # Run with bounded parallelism
    semaphore = threading.Semaphore(max_workers)
    threads = []

    for ts in test_specs:
        def run(ts=ts):
            with semaphore:
                process(ts)
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Summary
    total = stats["completed"] + stats["error"]
    resolve_rate = stats["resolved"] / max(stats["completed"], 1)
    logger.info(
        "Evaluation complete: %d/%d completed, %d resolved (%.1f%%), %d errors",
        stats["completed"], total, stats["resolved"], resolve_rate * 100, stats["error"],
    )

    # Write summary
    summary_dir = RUN_EVALUATION_LOG_DIR / run_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(json.dumps({"stats": stats, "results": results}, indent=2))
    logger.info("Summary written to %s", summary_path)

    if cleanup:
        logger.info("Cleaning up jobs...")
        cleanup_jobs(batch_v1, run_id, namespace)

    return results
