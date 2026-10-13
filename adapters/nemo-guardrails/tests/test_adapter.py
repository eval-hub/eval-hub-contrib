import json
import os
import random
from contextlib import contextmanager
from unittest.mock import create_autospec

import pytest

from main import (
    NemoGuardrailsAdapter,
    NemoResponses,
)

from evalhub.adapter import JobCallbacks, JobResults


def _make_canned_results(n_blocked=5, n_allowed=5):
    results = []
    for i in range(n_blocked):
        results.append({
            "prompt": f"blocked prompt {i}",
            "expected_blocked": True,
            "predicted_blocked": NemoResponses.BLOCKED,
            "dataset_type": "classification",
            "response_time_ms": 10.0 + i,
            "response_time_ms_per_character": 0.5,
            "content": "",
            "error": None,
        })
    for i in range(n_allowed):
        results.append({
            "prompt": f"allowed prompt {i}",
            "expected_blocked": False,
            "predicted_blocked": NemoResponses.ALLOW,
            "dataset_type": "classification",
            "response_time_ms": 5.0 + i,
            "response_time_ms_per_character": 0.3,
            "content": "safe response",
            "error": None,
        })
    return results


def _make_canned_samples(n_blocked=5, n_allowed=5):
    samples = []
    for i in range(n_blocked):
        samples.append({
            "prompt": f"blocked prompt {i}",
            "expected_blocked": True,
            "dataset_type": "classification",
        })
    for i in range(n_allowed):
        samples.append({
            "prompt": f"allowed prompt {i}",
            "expected_blocked": False,
            "dataset_type": "classification",
        })
    return samples


def _make_masking_result(values, content, mode="forbidden"):
    from main import _matched_chars_score
    if mode == "forbidden":
        value_results = [{"value": v, "score": 0.0 if v in content else 1.0} for v in values]
    else:
        value_results = [{"value": v, "score": _matched_chars_score(v, content)} for v in values]
    masking_accuracy = sum(r["score"] for r in value_results) / len(value_results) if value_results else 1.0
    key = "mask_forbidden" if mode == "forbidden" else "mask_required"
    return {
        "prompt": "test prompt",
        "dataset_type": "masking",
        key: values,
        "value_results": value_results,
        "masking_accuracy": masking_accuracy,
        "predicted_blocked": NemoResponses.ALLOW,
        "content": content,
        "response_time_ms": 10.0,
        "response_time_ms_per_character": 0.5,
        "error": None,
    }


def _load_job_spec(tmp_path, benchmark_id="prompt_injection", nemo_config="/tmp/test_config"):
    job_path = os.path.join(os.path.dirname(__file__), "..", "meta", "job.json")
    with open(job_path) as f:
        spec = json.load(f)
    spec["benchmark_id"] = benchmark_id
    spec["parameters"]["nemo_config"] = nemo_config
    out_path = tmp_path / "job.json"
    out_path.write_text(json.dumps(spec))
    return str(out_path)


@contextmanager
def _fake_managed_server(*args, **kwargs):
    yield "http://localhost:9999", "/tmp/server.log"


@pytest.mark.integration
class TestNemoGuardrailsAdapter:
    def test_prompt_injection_benchmark(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "test_config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("rails: {}")

        job_spec_path = _load_job_spec(tmp_path, nemo_config=str(config_dir))
        adapter = NemoGuardrailsAdapter(job_spec_path=job_spec_path)
        callbacks = create_autospec(JobCallbacks)

        canned_samples = _make_canned_samples()
        canned_results = _make_canned_results()

        monkeypatch.setattr("main.managed_server", _fake_managed_server)
        monkeypatch.setattr("main.warmup_server", lambda *a, **k: None)
        monkeypatch.setattr("main.load_samples", lambda dc, **kwargs: canned_samples)
        monkeypatch.setattr("main.run_evaluation", lambda *a, **k: canned_results)

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        assert isinstance(results, JobResults)
        assert results.benchmark_id == "prompt_injection"
        assert results.overall_score == 1.0
        assert results.num_examples_evaluated == 10

        metric_names = {r.metric_name for r in results.results}
        assert "accuracy" in metric_names
        assert "blocked_precision" in metric_names
        assert "blocked_recall" in metric_names
        assert "blocked_f1" in metric_names
        assert "allowed_precision" in metric_names
        assert "allowed_recall" in metric_names
        assert "allowed_f1" in metric_names
        assert "mean_latency_ms" in metric_names
        assert "p95_latency_ms" in metric_names

    @pytest.mark.parametrize("benchmark_id", [
        "prompt_injection",
        "toxicity_profanity_safety",
        "pii",
        "pii_masking",
        "tool_response_injection",
    ])
    def test_all_benchmarks_have_datasets(self, benchmark_id):
        from main import _load_benchmark_datasets
        datasets = _load_benchmark_datasets(benchmark_id)
        assert len(datasets) > 0
        for ds in datasets:
            assert "source" in ds
            assert "prompt_column" in ds
            is_masking = "mask_column" in ds
            if is_masking:
                has_forbidden = "mask_transform_forbidden_values" in ds
                has_must_contain = "mask_transform_must_contain_values" in ds
                assert has_forbidden ^ has_must_contain, (
                    f"Masking dataset must have exactly one of mask_transform_forbidden_values "
                    f"or mask_transform_must_contain_values, got neither or both in {ds}"
                )
            else:
                assert "label_column" in ds

    def test_unknown_benchmark_raises(self):
        from main import _load_benchmark_datasets
        with pytest.raises(ValueError, match="not found"):
            _load_benchmark_datasets("nonexistent_benchmark")

    def test_classification_metrics_all_correct(self):
        from main import _compute_classification_metrics
        results = _make_canned_results(n_blocked=5, n_allowed=5)
        metrics, errors = _compute_classification_metrics(results)
        assert metrics["accuracy"] == 1.0
        assert metrics["errors"] == 0
        assert metrics["total"] == 10

    def test_classification_metrics_with_errors(self):
        from main import _compute_classification_metrics
        results = _make_canned_results(n_blocked=3, n_allowed=3)
        results.append({
            "prompt": "error prompt",
            "expected_blocked": True,
            "predicted_blocked": NemoResponses.ERROR,
            "dataset_type": "classification",
            "response_time_ms": 100.0,
            "response_time_ms_per_character": 1.0,
            "content": "",
            "error": "timeout",
        })
        metrics, errors = _compute_classification_metrics(results)
        assert metrics["errors"] == 1
        assert metrics["total"] == 6

    def test_classification_metrics_ignores_masking_results(self):
        from main import _compute_classification_metrics
        cls_results = _make_canned_results(n_blocked=5, n_allowed=5)
        mask_result = _make_masking_result(["secret"], "the content has no secret here")
        metrics, _ = _compute_classification_metrics(cls_results + [mask_result])
        assert metrics["total"] == 10

    def test_masking_metrics_forbidden_perfect(self):
        from main import _compute_masking_metrics
        result = _make_masking_result(["alice", "bob"], "no names here", mode="forbidden")
        metrics = _compute_masking_metrics([result])
        assert metrics["masking_accuracy"] == 1.0
        assert metrics["total"] == 2
        assert metrics["total_prompts"] == 1

    def test_masking_metrics_forbidden_full_leak(self):
        from main import _compute_masking_metrics
        result = _make_masking_result(["alice", "bob"], "alice and bob are here", mode="forbidden")
        metrics = _compute_masking_metrics([result])
        assert metrics["masking_accuracy"] == 0.0

    def test_masking_metrics_forbidden_partial_leak(self):
        from main import _compute_masking_metrics
        # "alice" leaks (score 0.0), "bob" is masked (score 1.0) → mean 0.5
        result = _make_masking_result(["alice", "bob"], "alice is here", mode="forbidden")
        metrics = _compute_masking_metrics([result])
        assert metrics["masking_accuracy"] == 0.5

    def test_masking_metrics_must_contain_perfect(self):
        from main import _compute_masking_metrics
        result = _make_masking_result(["hello world"], "hello world", mode="must_contain")
        metrics = _compute_masking_metrics([result])
        assert metrics["masking_accuracy"] == 1.0

    def test_masking_metrics_must_contain_with_gap(self):
        from main import _compute_masking_metrics
        # "Hello, my name is Adam. I like apples" with "Adam" redacted —
        # most characters match; score should be well above 0.8 but below 1.0.
        value = "Hello, my name is Adam. I like apples"
        content = "Hello, my name is <redacted>. I like apples"
        result = _make_masking_result([value], content, mode="must_contain")
        metrics = _compute_masking_metrics([result])
        assert 0.8 < metrics["masking_accuracy"] < 1.0

    def test_masking_metrics_empty_returns_empty(self):
        from main import _compute_masking_metrics
        cls_results = _make_canned_results()
        metrics = _compute_masking_metrics(cls_results)
        assert metrics == {}

    def test_matched_chars_score_full_match(self):
        from main import _matched_chars_score
        assert _matched_chars_score("hello", "say hello there") == 1.0

    def test_matched_chars_score_no_match(self):
        from main import _matched_chars_score
        assert _matched_chars_score("xyz", "abc def") == 0.0

    def test_matched_chars_score_with_gap(self):
        from main import _matched_chars_score
        # "AB__CD" where "__" is replaced: matched = "AB" + "CD" = 4 / 6
        score = _matched_chars_score("ABCD", "AB--CD")
        assert score == 1.0  # all 4 chars of "ABCD" appear (A,B matched; C,D matched)

    def test_matched_chars_score_partial(self):
        from main import _matched_chars_score
        score = _matched_chars_score("0123456789", "0123456")
        assert abs(score - 0.7) < 0.01

    def test_resolve_mask_transform_both_raises(self):
        from main import _resolve_mask_transform
        with pytest.raises(ValueError, match="exactly one"):
            _resolve_mask_transform({
                "mask_transform_forbidden_values": ".",
                "mask_transform_must_contain_values": ".",
            })

    def test_resolve_mask_transform_neither_raises(self):
        from main import _resolve_mask_transform
        with pytest.raises(ValueError, match="must specify either"):
            _resolve_mask_transform({})

    def test_resolve_mask_transform_forbidden(self):
        from main import _resolve_mask_transform
        mode, program = _resolve_mask_transform({"mask_transform_forbidden_values": "[.[].value]"})
        assert mode == "forbidden"
        assert program is not None

    def test_resolve_mask_transform_must_contain(self):
        from main import _resolve_mask_transform
        mode, program = _resolve_mask_transform({"mask_transform_must_contain_values": "[.]"})
        assert mode == "must_contain"
        assert program is not None

    def test_timing_stats(self):
        from main import _compute_timing_stats
        results = _make_canned_results(n_blocked=5, n_allowed=5)
        timing = _compute_timing_stats(results)
        assert timing["mean_ms"] > 0
        assert timing["p95_ms"] > 0
        assert timing["total_ms"] > 0

    def test_chunk_prompt_short_prompt_unchanged(self):
        from main import _chunk_prompt
        assert _chunk_prompt("hello", 2000) == ["hello"]

    def test_chunk_prompt_disabled(self):
        from main import _chunk_prompt
        long = "x" * 10000
        assert _chunk_prompt(long, 0) == [long]

    def test_chunk_prompt_covers_whole_prompt_with_overlap(self):
        from main import _chunk_prompt
        prompt = "".join(str(i % 10) for i in range(5000))
        chunks = _chunk_prompt(prompt, max_chars=2000, overlap=0.05)
        assert len(chunks) > 1
        assert all(len(c) <= 2000 for c in chunks)
        reconstructed = chunks[0]
        for c in chunks[1:]:
            reconstructed += c[100:]  # drop the overlap
        assert reconstructed == prompt

    def test_chunk_prompt_overlap_fraction_sets_step(self):
        from main import _chunk_prompt
        prompt = "".join(random.choices("0123456789abcdef", k=5000))
        chunks = _chunk_prompt(prompt, max_chars=1000, overlap=0.10)
        assert chunks[0] == prompt[0:1000]
        assert chunks[1] == prompt[900:1900]
        assert chunks[0][-100:] == chunks[1][:100]

    def test_chunk_prompt_zero_overlap(self):
        from main import _chunk_prompt
        prompt = "".join(random.choices("0123456789abcdef", k=5000))
        chunks = _chunk_prompt(prompt, max_chars=1000, overlap=0.0)
        assert chunks[0] == prompt[0:1000]
        assert chunks[1] == prompt[1000:2000]

    def test_evaluate_prompt_blocks_if_any_chunk_blocks(self, monkeypatch):
        import main
        long_prompt = "safe text " * 500
        calls = {"n": 0}

        def fake_chunk(server_url, text):
            calls["n"] += 1
            status = NemoResponses.BLOCKED if calls["n"] == 2 else NemoResponses.ALLOW
            return {
                "predicted_blocked": status,
                "content": "",
                "response_time_ms": 5.0,
                "response_time_ms_per_character": 0.1,
                "error": None,
            }

        monkeypatch.setattr(main, "_evaluate_chunk", fake_chunk)
        result = main._evaluate_prompt("http://x", long_prompt, chunk_strategy="chunk", chunk_size=2000)
        assert result["predicted_blocked"] == NemoResponses.BLOCKED
        assert calls["n"] == 2

    def test_evaluate_prompt_allows_when_all_chunks_allow(self, monkeypatch):
        import main
        long_prompt = "safe text " * 500

        def fake_chunk(server_url, text):
            return {
                "predicted_blocked": NemoResponses.ALLOW,
                "content": "safe",
                "response_time_ms": 5.0,
                "response_time_ms_per_character": 0.1,
                "error": None,
            }

        monkeypatch.setattr(main, "_evaluate_chunk", fake_chunk)
        result = main._evaluate_prompt("http://x", long_prompt, chunk_strategy="chunk", chunk_size=2000)
        assert result["predicted_blocked"] == NemoResponses.ALLOW
        assert result["error"] is None

    def test_evaluate_prompt_limit_strategy_truncates_to_first_window(self, monkeypatch):
        import main
        long_prompt = "A" * 5000
        seen = []

        def fake_chunk(server_url, text):
            seen.append(text)
            return {
                "predicted_blocked": NemoResponses.ALLOW,
                "content": "",
                "response_time_ms": 5.0,
                "response_time_ms_per_character": 0.1,
                "error": None,
            }

        monkeypatch.setattr(main, "_evaluate_chunk", fake_chunk)
        main._evaluate_prompt("http://x", long_prompt, chunk_strategy="limit", chunk_size=2000)
        assert len(seen) == 1
        assert seen[0] == "A" * 2000

    def test_evaluate_prompt_none_strategy_sends_prompt_whole(self, monkeypatch):
        import main
        long_prompt = "A" * 5000
        seen = []

        def fake_chunk(server_url, text):
            seen.append(text)
            return {
                "predicted_blocked": NemoResponses.ALLOW,
                "content": "",
                "response_time_ms": 5.0,
                "response_time_ms_per_character": 0.1,
                "error": None,
            }

        monkeypatch.setattr(main, "_evaluate_chunk", fake_chunk)
        main._evaluate_prompt("http://x", long_prompt, chunk_strategy="none", chunk_size=2000)
        assert len(seen) == 1
        assert seen[0] == long_prompt
