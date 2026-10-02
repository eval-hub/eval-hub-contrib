from __future__ import annotations

import json
from pathlib import Path

import pytest
from evalhub.adapter import JobSpec


@pytest.fixture
def job_spec() -> JobSpec:
    base = Path(__file__).resolve().parent.parent
    payload = json.loads((base / "meta" / "job.json").read_text())
    trajectory_path = Path(payload["parameters"]["trajectory_path"])
    if not trajectory_path.is_absolute():
        trajectory_path = base / trajectory_path
    payload["parameters"]["trajectory_path"] = str(trajectory_path.resolve())
    return JobSpec.model_validate(payload)
