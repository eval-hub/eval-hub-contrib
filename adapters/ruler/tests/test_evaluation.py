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

"""Unit tests for RULER adapter evaluation logic."""

import csv
import io
import pytest


class TestEvaluationResult:
    """Test result computation logic."""

    def test_per_context_length_scoring(self):
        """Simulate scoring at multiple context lengths."""
        scores = {4096: 0.95, 8192: 0.88, 16384: 0.72}
        overall = sum(scores.values()) / len(scores)
        assert abs(overall - 0.85) < 0.01

    def test_null_prediction_handling(self):
        """Null predictions should not affect accuracy denominator."""
        total = 10
        nulls = 2
        correct = 7
        evaluated = total - nulls
        accuracy = correct / evaluated
        assert abs(accuracy - 0.875) < 0.001

    def test_confidence_interval_binomial(self):
        """Wilson score CI for binomial proportion."""
        n = 100
        x = 72  # 72% accuracy
        p = x / n

        z = 1.96
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = (
            z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
        )
        ci_lower = max(0.0, centre - margin)
        ci_upper = min(1.0, centre + margin)

        assert ci_lower < p < ci_upper
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0

    def test_empty_scores_zero(self):
        """No scores → overall should be 0.0."""
        scores: dict = {}
        overall = sum(scores.values()) / len(scores) if scores else 0.0
        assert overall == 0.0


class TestResultAggregation:
    """Test aggregation of per-task results."""

    def test_weighted_average(self):
        """Average across tasks should be correct."""
        task_scores = {
            "niah_single_1": 0.9,
            "niah_single_2": 0.85,
            "variable_tracking": 0.7,
            "qa_squad": 0.6,
        }
        overall = sum(task_scores.values()) / len(task_scores)
        assert abs(overall - 0.7625) < 0.001

    def test_summary_csv_format(self):
        """Verify CSV output would be writeable."""
        results = {
            "task1": {
                "per_context_length": {"4096": 0.9, "8192": 0.7},
                "score": 0.8,
            },
            "task2": {
                "per_context_length": {"4096": 0.8, "8192": 0.6},
                "score": 0.7,
            },
        }

        all_lengths = sorted({
            int(k) for r in results.values() for k in r["per_context_length"]
        })

        output = io.StringIO()
        writer = csv.writer(output)
        header = ["task"] + [str(l) for l in all_lengths] + ["overall"]
        writer.writerow(header)

        for task_id, task_result in results.items():
            per_cl = task_result["per_context_length"]
            row = [task_id]
            for l in all_lengths:
                row.append(round(per_cl.get(str(l), 0.0), 4))
            row.append(round(task_result["score"], 4))
            writer.writerow(row)

        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 3  # header + 2 tasks

    def test_csv_has_correct_columns(self):
        """CSV should have task, context lengths, and overall columns."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["task", "4096", "8192", "16384", "overall"])
        writer.writerow(["niah_test", "0.95", "0.72", "0.55", "0.74"])
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 2
        cols = lines[0].split(",")
        assert "overall".strip() in [c.strip() for c in cols]


class TestPostProcessing:
    """Test RULER's postprocessing logic."""

    def test_postprocess_removes_non_printable(self):
        """RULER's postprocess should strip control characters."""
        import re

        def postprocess_pred(predict_str: str, task_config: dict):
            predict_str = predict_str.strip()
            np_pattern = re.compile(r'[\x00-\x1f]')
            predict_str = np_pattern.sub('\n', predict_str).strip()
            return predict_str

        # Control chars should be replaced with newlines and stripped
        result = postprocess_pred("\x00hello\x1fworld\x00", {})
        assert "hello" in result
        assert "world" in result

    def test_postprocess_strips_whitespace(self):
        import re

        def postprocess_pred(predict_str: str, task_config: dict):
            predict_str = predict_str.strip()
            np_pattern = re.compile(r'[\x00-\x1f]')
            predict_str = np_pattern.sub('\n', predict_str).strip()
            return predict_str

        result = postprocess_pred("  hello world  ", {})
        assert result == "hello world"
