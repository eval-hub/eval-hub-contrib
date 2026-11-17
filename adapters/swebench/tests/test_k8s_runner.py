"""Security contract for child Jobs that execute untrusted model patches."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kubernetes import client


@pytest.mark.parametrize("account", [None, "trusted-runner"])
def test_child_job_never_automounts_service_account_credentials(monkeypatch, account):
    """Serialize a real Kubernetes pod model and check both container credential paths."""
    constants = {
        name: name
        for name in (
            "APPLY_PATCH_FAIL",
            "APPLY_PATCH_PASS",
            "DOCKER_WORKDIR",
            "KEY_MODEL",
            "KEY_PREDICTION",
            "LOG_REPORT",
            "LOG_TEST_OUTPUT",
            "RUN_EVALUATION_LOG_DIR",
        )
    }
    # Job construction needs only harness interfaces, not dataset/grading dependencies.
    monkeypatch.setitem(
        sys.modules, "swebench.harness.constants", SimpleNamespace(**constants)
    )
    monkeypatch.setitem(
        sys.modules, "swebench.harness.grading", SimpleNamespace(get_eval_report=None)
    )
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.test_spec.test_spec",
        SimpleNamespace(TestSpec=object, get_test_specs_from_dataset=None),
    )
    spec = importlib.util.spec_from_file_location(
        "swebench_security_runner",
        Path(__file__).resolve().parents[1] / "k8s_runner.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_create_eval_script", lambda *args: "exit 0")
    job = module._create_job(
        SimpleNamespace(
            instance_id="repo__project-1", instance_image_key="instance:latest"
        ),
        {},
        "test-run",
        "registry.example",
        60,
        service_account=account,
    )
    pod = client.ApiClient().sanitize_for_serialization(job)["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod.get("serviceAccountName") == account
    assert not pod.get("volumes")
    for container in pod["containers"] + pod["initContainers"]:
        assert not container.get("volumeMounts")
