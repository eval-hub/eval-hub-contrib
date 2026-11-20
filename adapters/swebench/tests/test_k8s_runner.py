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


@pytest.mark.parametrize("logs", ["partial test output", ""])
def test_failed_child_job_is_not_graded(test_spec, monkeypatch, tmp_path, logs):
    """A failed Job must remain diagnostic even when partial output is gradeable."""
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    monkeypatch.setattr(k8s_runner, "_wait_for_job", MagicMock(return_value=False))
    monkeypatch.setattr(k8s_runner, "_get_pod_logs", MagicMock(return_value=logs))
    grade = MagicMock(return_value={test_spec.instance_id: {"resolved": False}})
    monkeypatch.setattr(k8s_runner, "get_eval_report", grade)
    batch = MagicMock()
    result = k8s_runner.run_instance_k8s(
        test_spec, {}, "failed-run", "registry.example", "tenant", 60, batch, MagicMock()
    )
    assert result == {"completed": False, "resolved": False}
    grade.assert_not_called()
    batch.delete_namespaced_job.assert_called_once()
    assert not list(tmp_path.rglob(k8s_runner.LOG_REPORT))
    assert next(tmp_path.rglob(k8s_runner.LOG_TEST_OUTPUT)).read_text() == logs


def test_successful_child_with_failing_tests_is_a_scored_observation(test_spec, monkeypatch, tmp_path):
    """An orderly completed test run can legitimately be unresolved."""
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    monkeypatch.setattr(k8s_runner, "_wait_for_job", MagicMock(return_value=True))
    monkeypatch.setattr(k8s_runner, "_get_pod_logs", MagicMock(return_value="FAILED test_example"))
    monkeypatch.setattr(k8s_runner, "get_eval_report", MagicMock(return_value={test_spec.instance_id: {"resolved": False}}))
    result = k8s_runner.run_instance_k8s(
        test_spec, {}, "unresolved-run", "registry.example", "tenant", 60, MagicMock(), MagicMock()
    )
    assert result == {"completed": True, "resolved": False}


@pytest.mark.parametrize("reason", ["DeadlineExceeded", "BackoffLimitExceeded"])
def test_terminal_job_failure_without_failed_pod_count(reason):
    """Terminal Job conditions stop polling even when pod counts lag behind."""
    batch = MagicMock()
    batch.read_namespaced_job.return_value = client.V1Job(status=client.V1JobStatus(
        conditions=[client.V1JobCondition(type="Failed", status="True", reason=reason)]
    ))
    assert k8s_runner._wait_for_job(batch, "job", "tenant", poll_interval=0) is False
    batch.read_namespaced_job.assert_called_once()


def test_rejected_patch_is_a_scored_zero(test_spec, monkeypatch, tmp_path):
    """Patch rejection belongs in the resolve-rate denominator without test grading."""
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    monkeypatch.setattr(k8s_runner, "_wait_for_job", MagicMock(return_value=False))
    monkeypatch.setattr(k8s_runner, "_get_pod_logs", MagicMock(return_value=k8s_runner.APPLY_PATCH_FAIL + "\n"))
    grade = MagicMock()
    monkeypatch.setattr(k8s_runner, "get_eval_report", grade)
    batch = MagicMock()
    result = k8s_runner.run_instance_k8s(
        test_spec, {}, "rejected", "registry.example", "tenant", 60, batch, MagicMock()
    )
    assert result == {"completed": True, "resolved": False}
    grade.assert_not_called()
    batch.delete_namespaced_job.assert_called_once()


@pytest.mark.parametrize("error", [OSError("disk full"), AttributeError("job status absent")])
def test_worker_exception_is_retained_as_diagnostic(test_spec, monkeypatch, tmp_path, error):
    """One failed thread cannot silently shrink the selected instance population."""
    monkeypatch.setattr(k8s_runner, "RUN_EVALUATION_LOG_DIR", tmp_path)
    monkeypatch.setattr(k8s_runner.k8s_config, "load_incluster_config", MagicMock())
    monkeypatch.setattr(k8s_runner.k8s_client, "BatchV1Api", MagicMock())
    monkeypatch.setattr(k8s_runner.k8s_client, "CoreV1Api", MagicMock())
    other = SimpleNamespace(instance_id="other-instance")
    monkeypatch.setattr(k8s_runner, "get_test_specs_from_dataset", lambda *a, **k: [test_spec, other])

    def evaluate(**kwargs):
        if kwargs["test_spec"] is test_spec:
            raise error
        return {"completed": True, "resolved": False}

    monkeypatch.setattr(k8s_runner, "run_instance_k8s", evaluate)
    results = k8s_runner.run_instances_k8s(
        predictions={test_spec.instance_id: {}, other.instance_id: {}}, dataset=[],
        run_id="threads", registry="registry.example", namespace="tenant",
        max_workers=2, timeout=60, cleanup=False,
    )
    assert len(results) == 2
    assert results[test_spec.instance_id] == {"completed": False, "resolved": False, "error": str(error)}
    assert results[other.instance_id] == {"completed": True, "resolved": False}
    import json
    summary = json.loads((tmp_path / "threads" / "summary.json").read_text())
    assert summary["stats"] == {"completed": 1, "resolved": 0, "error": 1}
