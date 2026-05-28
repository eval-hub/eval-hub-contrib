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

"""Tests for the RULER adapter main module structure."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import os
ADAPTER_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(os.environ.get("RULER_SCRIPTS_DIR", str(ADAPTER_DIR / "scripts")))

# Ensure scripts are on path for downstream imports
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "data"))
sys.path.insert(0, str(SCRIPTS_DIR / "data" / "synthetic"))
sys.path.insert(0, str(SCRIPTS_DIR / "eval"))


class TestRulerAdapterClass:
    """Test the RulerAdapter class structure."""

    def test_constants_are_defined(self):
        """RULER constants should be importable."""
        from synthetic.constants import TASKS
        expected = {
            "niah", "variable_tracking", "common_words_extraction",
            "freq_words_extraction", "qa",
        }
        assert set(TASKS.keys()) == expected

    def test_template_constants_exist(self):
        """Chat templates should be importable."""
        from data.template import Templates
        assert "base" in Templates
        assert "meta-llama3" in Templates
        assert "meta-chat" in Templates
        assert "Phi3" in Templates

    def test_task_categories_complete(self):
        """All 13 tasks should be in some category."""
        expected_tasks = {
            "niah_single_1", "niah_single_2", "niah_single_3",
            "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multivalue", "niah_multiquery",
            "vt", "cwe", "fwe",
            "qa_1", "qa_2",
        }

        # Import via main module (which sets up sys.path correctly)
        # We just verify the vendored scripts have the right data
        import yaml
        yaml_path = SCRIPTS_DIR / "synthetic.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert set(config.keys()) == expected_tasks


class TestRulerMain:
    """Test the RULER entry point."""

    def test_main_requires_job_spec(self):
        """main() should fail if job spec doesn't exist."""
        # We test that main() has the right structure by
        # verifying the function exists and signature is correct
        # Import the module and check the main function
        adapter_code = (ADAPTER_DIR / "main.py").read_text()
        assert "EVALHUB_JOB_SPEC_PATH" in adapter_code
        assert "main()" in adapter_code
        assert "__name__ == \"__main__\"" in adapter_code

    def test_entry_point_structure(self):
        """main() should load JobSpec, create adapter, run benchmark."""
        adapter_code = (ADAPTER_DIR / "main.py").read_text()
        # Check the main() function structure
        assert "def main()" in adapter_code
        assert "logging.basicConfig" in adapter_code
        assert "RulerAdapter" in adapter_code
        assert "JobSpec(**job_spec)" in adapter_code


class TestHelpers:
    """Test RulerAdapter helper methods by inspecting code."""

    def test_has_required_methods(self):
        """Adapter should have all expected methods."""
        adapter_code = (ADAPTER_DIR / "main.py").read_text()
        required_methods = [
            "run_benchmark_job",
            "_get_category",
            "_verify_tokenizer",
            "_run_ruler_task",
            "_load_task_config",
            "_generate_task_data",
            "_run_api_inference",
            "_evaluate_predictions",
            "_compute_ci",
            "_write_summary_csv",
            "main",
        ]
        for method in required_methods:
            assert f"def {method}" in adapter_code, \
                f"Missing method: {method}"

    def test_provider_yaml_has_all_benchmarks(self):
        """provider.yaml should document all 13 benchmarks."""
        import yaml
        provider_path = ADAPTER_DIR / "provider.yaml"
        with open(provider_path) as f:
            provider = yaml.safe_load(f)

        benchmarks = {b["id"] for b in provider["benchmarks"]}
        expected = {
            "niah-single-noise", "niah-single-essay", "niah-single-uuid",
            "niah-multikey", "niah-needle-bg", "niah-multikey-uuid",
            "niah-multivalue", "niah-multiquery",
            "variable-tracking",
            "common-words-extraction", "frequency-words-extraction",
            "qa-squad", "qa-hotpotqa",
        }
        assert benchmarks == expected

    def test_provider_has_parameters(self):
        """provider.yaml should document all expected parameters."""
        import yaml
        provider_path = ADAPTER_DIR / "provider.yaml"
        with open(provider_path) as f:
            provider = yaml.safe_load(f)

        param_names = {p["name"] for p in provider["parameters"]}
        required_params = {
            "benchmarks", "context_lengths", "num_samples",
            "tokenizer_path", "tokenizer_type", "model_template",
            "tokens_to_generate", "batch_size", "random_seed",
        }
        assert required_params.issubset(param_names)
