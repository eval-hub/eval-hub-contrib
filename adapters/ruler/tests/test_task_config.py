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

"""Unit tests for RULER adapter task configuration and YAML parsing."""

import pytest
import yaml
from pathlib import Path
from main import TASK_CATEGORIES

import os
SCRIPTS_DIR = Path(os.environ.get("RULER_SCRIPTS_DIR", str(Path(__file__).resolve().parent.parent / "scripts")))


def _load_synthetic_yaml():
    """Helper to load synthetic.yaml once."""
    yaml_path = SCRIPTS_DIR / "synthetic.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


class TestSyntheticYaml:
    """Test that synthetic.yaml is well-formed."""

    def test_yaml_is_valid(self):
        config = _load_synthetic_yaml()
        assert isinstance(config, dict)
        assert len(config) >= 13

    def test_yaml_has_all_ruler_tasks(self):
        config = _load_synthetic_yaml()
        expected = {
            "niah_single_1", "niah_single_2", "niah_single_3",
            "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multivalue", "niah_multiquery",
            "vt", "cwe", "fwe",
            "qa_1", "qa_2",
        }
        assert set(config.keys()) == expected

    def test_all_tasks_have_task_and_args(self):
        config = _load_synthetic_yaml()
        for task_id, task_def in config.items():
            assert "task" in task_def, f"{task_id} missing 'task'"
            assert "args" in task_def, f"{task_id} missing 'args'"
            assert isinstance(task_def["args"], dict)

    def test_task_names_are_valid(self):
        """All task names should map to known RULER task generators."""
        config = _load_synthetic_yaml()
        valid_task_names = {
            "niah", "variable_tracking", "common_words_extraction",
            "freq_words_extraction", "qa",
        }
        for task_id, task_def in config.items():
            assert task_def["task"] in valid_task_names, \
                f"{task_id}: unknown task type '{task_def['task']}'"

    def test_niah_has_8_variants(self):
        config = _load_synthetic_yaml()
        niah_tasks = [k for k in config if k.startswith("niah_")]
        assert len(niah_tasks) == 8

    def test_qa_has_2_variants(self):
        config = _load_synthetic_yaml()
        qa_tasks = [k for k in config if k.startswith("qa_")]
        assert len(qa_tasks) == 2


class TestConstants:
    """Test RULER constants.py (vendored from NVIDIA/RULER)."""

    def test_task_constants_have_templates(self):
        """TASKS dict should define templates for all 5 task types."""
        # We test by importing the vendored constants module
        # (it's on sys.path via conftest or test setup)
        try:
            from scripts.data.synthetic.constants import TASKS
        except ImportError:
            from synthetic.constants import TASKS

        expected = {
            "niah", "variable_tracking", "common_words_extraction",
            "freq_words_extraction", "qa",
        }
        assert set(TASKS.keys()) == expected

    def test_task_constants_have_tokens_to_generate(self):
        try:
            from scripts.data.synthetic.constants import TASKS
        except ImportError:
            from synthetic.constants import TASKS

        for task_name, cfg in TASKS.items():
            assert "tokens_to_generate" in cfg, \
                f"Missing tokens_to_generate in {task_name}"
            assert isinstance(cfg["tokens_to_generate"], int)
            assert cfg["tokens_to_generate"] > 0

    def test_task_templates_contain_placeholders(self):
        try:
            from scripts.data.synthetic.constants import TASKS
        except ImportError:
            from synthetic.constants import TASKS

        # Aggregation tasks (cwe, fwe) embed the question directly in the
        # template — no {query} slot — which is correct RULER upstream behaviour.
        tasks_without_query = {"common_words_extraction", "freq_words_extraction"}

        for task_name, cfg in TASKS.items():
            template = cfg["template"]
            assert "{context}" in template, \
                f"{task_name}: missing {{context}} in template"
            if task_name not in tasks_without_query:
                assert "{query}" in template, \
                    f"{task_name}: missing {{query}} in template"
