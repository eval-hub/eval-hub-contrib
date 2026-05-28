# Copyright [yyyy] eval-hub contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RULER adapter configuration and data loading."""

import json
import sys
from pathlib import Path

import pytest

import os
ADAPTER_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(os.environ.get("RULER_SCRIPTS_DIR", str(ADAPTER_DIR / "scripts")))

# Ensure scripts are on path for downstream imports
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "data"))
sys.path.insert(0, str(SCRIPTS_DIR / "data" / "synthetic"))


class TestMetaJobSpec:
    """Validate meta/job.json structure and defaults."""

    @pytest.fixture
    def job_spec(self) -> dict:
        spec_path = ADAPTER_DIR / "meta" / "job.json"
        with open(spec_path) as f:
            return json.load(f)

    def test_spec_has_required_fields(self, job_spec: dict):
        for field in ("id", "provider_id", "benchmark_id", "model", "parameters"):
            assert field in job_spec, f"Missing required field: {field}"

    def test_spec_has_model_url(self, job_spec: dict):
        assert "url" in job_spec["model"]
        assert job_spec["model"]["url"].startswith("http")

    def test_spec_defaults(self, job_spec: dict):
        params = job_spec["parameters"]
        assert params["num_samples"] == 10
        assert params["random_seed"] == 42
        assert params["batch_size"] == 1
        assert params["tokenizer_type"] == "hf"
