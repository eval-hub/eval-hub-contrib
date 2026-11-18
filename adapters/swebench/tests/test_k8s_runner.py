"""Security contract for child Jobs that execute untrusted model patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import k8s_runner
import pytest
from kubernetes import client
from kubernetes.client.rest import ApiException


@pytest.fixture
def test_spec():
    """Build a small harness-compatible spec without datasets or a cluster."""
    return SimpleNamespace(
        instance_id="repo__project-1",
        instance_image_key="instance:latest",
        eval_script="exit 0",
    )


@pytest.mark.parametrize("account", [None, "trusted-runner"])
def test_child_job_never_automounts_service_account_credentials(test_spec, account):
    """Serialize a real Kubernetes pod model and check both container credential paths."""
    job = k8s_runner._create_job(
        test_spec,
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


@pytest.mark.parametrize("automount", [None, True, False])
def test_existing_job_requires_explicit_token_protection(
    test_spec, monkeypatch, tmp_path, automount
):
    """A conflicting legacy Job must never reach log collection or grading."""
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    existing = k8s_runner._create_job(test_spec, {}, "retry", "registry.example", 60)
    existing.spec.template.spec.automount_service_account_token = automount
    batch = MagicMock()
    batch.create_namespaced_job.side_effect = ApiException(status=409)
    batch.read_namespaced_job.return_value = existing
    wait = MagicMock(return_value=True)
    logs = MagicMock(return_value="test output")
    grade = MagicMock(return_value={test_spec.instance_id: {"resolved": True}})
    monkeypatch.setattr(k8s_runner, "_wait_for_job", wait)
    monkeypatch.setattr(k8s_runner, "_get_pod_logs", logs)
    monkeypatch.setattr(k8s_runner, "get_eval_report", grade)

    result = k8s_runner.run_instance_k8s(
        test_spec, {}, "retry", "registry.example", "tenant", 60, batch, MagicMock()
    )

    batch.read_namespaced_job.assert_called_once_with(existing.metadata.name, "tenant")
    batch.delete_namespaced_job.assert_called_once()
    if automount is False:
        assert result == {"completed": True, "resolved": True}
        wait.assert_called_once()
        logs.assert_called_once()
        grade.assert_called_once()
    else:
        assert result == {"completed": False, "resolved": False}
        wait.assert_not_called()
        logs.assert_not_called()
        grade.assert_not_called()


def test_unreadable_existing_job_is_not_reused(test_spec, monkeypatch, tmp_path):
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    batch = MagicMock()
    batch.create_namespaced_job.side_effect = ApiException(status=409)
    batch.read_namespaced_job.side_effect = ApiException(status=403)
    wait = MagicMock()
    logs = MagicMock()
    monkeypatch.setattr(k8s_runner, "_wait_for_job", wait)
    monkeypatch.setattr(k8s_runner, "_get_pod_logs", logs)
    result = k8s_runner.run_instance_k8s(
        test_spec, {}, "retry", "registry.example", "tenant", 60, batch, MagicMock()
    )
    assert result == {"completed": False, "resolved": False}
    wait.assert_not_called()
    logs.assert_not_called()
