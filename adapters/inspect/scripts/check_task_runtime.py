"""Smoke-check real task imports and bundled GDM data without model/tool execution."""

import importlib
import socket
from unittest.mock import patch

from prepare_task_data import ARCHIVE_SHA256, verify_archive


def check_runtime() -> None:
    # Fail immediately if imports or data setup try to fetch missing assets.
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("Unexpected network access"),
    ):
        for module in (
            "inspect_cyber",
            "swebench",
            "inspect_evals.cybench",
            "inspect_evals.cybergym",
            "inspect_evals.theagentcompany",
            "inspect_evals.swe_bench",
        ):
            importlib.import_module(module)
            print(f"Imported {module}")

        # These are imported lazily by Inspect's SWE-bench scorers.
        for module, symbol in (
            ("swebench.harness.constants", "SWEbenchInstance"),
            ("swebench.harness.test_spec.test_spec", "make_test_spec"),
            ("swebench.harness.grading", "get_eval_report"),
        ):
            getattr(importlib.import_module(module), symbol)
        print("Loaded SWE-bench scorer APIs")

        from inspect_evals.gdm_self_proliferation import data_setup
        from inspect_evals.gdm_self_proliferation.dataset import get_task_sample

        verify_archive(data_setup.DATA_DIR / "sp08" / "secrets.zip", ARCHIVE_SHA256)
        if data_setup._needs_setup(data_setup.DATA_DIR):
            raise AssertionError("GDM task data still requires runtime preparation")
        for task_id in ("sp01", "sp08"):
            samples = get_task_sample(task_id)
            if not samples:
                raise AssertionError(f"No prepared samples for {task_id}")
            print(f"Loaded prepared GDM {task_id} samples")


if __name__ == "__main__":
    check_runtime()
