"""No-network installed-image smoke test; run as the image's non-root user."""

import json
from pathlib import Path

from evalhub.adapter import JobCallbacks, JobSpec

from main import ATIFAdapter


class Callbacks(JobCallbacks):
    def __init__(self):
        self.mlflow = self

    def report_status(self, update):
        pass

    def report_results(self, results):
        pass

    def create_oci_artifact(self, spec):
        raise AssertionError("Unexpected OCI request")

    def save(self, result, spec, artifacts):
        assert artifacts and result.evaluation_metadata["atif_trajectories"]
        return "image-smoke-run"


class OfflineAdapter(ATIFAdapter):
    async def _judge_call(self, payload):
        if "step" not in payload:
            return json.dumps(
                {"criteria": [{"name": "quality", "description": "Task quality"}]}
            )
        return json.dumps({"score": 0.8})


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    spec = JobSpec.from_file(base / "meta/job.json")
    spec.parameters["trajectory_path"] = str(base / "tests/fixtures/trajectory.json")
    result = OfflineAdapter(job_spec_path=base / "meta/job.json").run_benchmark_job(
        spec, Callbacks()
    )
    assert result.overall_score == 0.8 and result.mlflow_run_id == "image-smoke-run"
    print("ATIF image smoke passed (no network; mocked judge and artifact backend)")
