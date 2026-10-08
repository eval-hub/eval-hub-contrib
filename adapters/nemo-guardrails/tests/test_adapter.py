import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from unittest.mock import create_autospec

import pytest
from evalhub.adapter import JobCallbacks, JobResults

from main import (
    NemoGuardrailsAdapter,
    NemoResponses,
)


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
        metrics, _errors = _compute_classification_metrics(results)
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
        metrics, _errors = _compute_classification_metrics(results)
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


# ---------------------------------------------------------------------------
# Staged /test_data data (airgap) — _load_huggingface prefers staged rows
# ---------------------------------------------------------------------------


def _write_job_spec_with_test_data_ref(tmp_path, ref: dict | None) -> str:
    spec = {"id": "j-staged", "parameters": {}}
    if ref is not None:
        spec["test_data_ref"] = ref
    out = tmp_path / "meta"
    out.mkdir(exist_ok=True)
    path = out / "job.json"
    path.write_text(json.dumps(spec))
    return str(path)


def _classification_config() -> dict:
    return {
        "hf_name": "acme/safety",
        "prompt_column": "prompt",
        "label_column": "label",
        "block_labels": ["blocked"],
        "pass_labels": ["allowed"],
    }


def test_staged_rows_used_when_test_data_ref(tmp_path, monkeypatch):
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    (test_data / "test.jsonl").write_text(
        json.dumps(
            [
                {"prompt": "p1", "label": "blocked"},
                {"prompt": "p2", "label": "allowed"},
            ]
        )
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))

    samples = main._load_huggingface(_classification_config())
    assert len(samples) == 2
    assert samples[0]["expected_blocked"] is True
    assert samples[1]["expected_blocked"] is False


def test_no_staged_data_falls_back_to_hub_loader(tmp_path, monkeypatch):
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH", _write_job_spec_with_test_data_ref(tmp_path, None)
    )

    fake_rows = [{"prompt": "p", "label": "blocked"}]

    fake_datasets = type(sys)("datasets")

    def _fail(*a, **kw):
        raise AssertionError("load_dataset must not be called when staged rows resolve")

    def _ok(*a, **kw):
        return fake_rows

    fake_datasets.load_dataset = _ok
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(main, "_load_staged_rows", lambda config, split: None)

    samples = main._load_huggingface(_classification_config())
    assert len(samples) == 1
    assert samples[0]["expected_blocked"] is True


def test_staged_jsonl_bare_array_line(tmp_path, monkeypatch):
    """A .jsonl staged file containing one JSON array on a single line loads fully."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    (test_data / "test.jsonl").write_text(
        json.dumps(
            [{"prompt": "p1", "label": "blocked"}, {"prompt": "p2", "label": "allowed"}]
        )
    )

    samples = main._load_huggingface(_classification_config())
    assert len(samples) == 2
    assert samples[0]["expected_blocked"] is True


def test_staged_parquet_rows(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow.parquet")
    import pyarrow

    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(
            tmp_path, {"s3": {"bucket": "b", "key": "k", "secret_ref": "s"}}
        ),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    table = pyarrow.table(
        {
            "prompt": ["p1", "p2"],
            "label": ["blocked", "allowed"],
        }
    )
    pyarrow.parquet.write_table(table, str(test_data / "data.parquet"))

    samples = main._load_huggingface(_classification_config())
    assert len(samples) == 2
    assert samples[0]["expected_blocked"] is True


def test_staged_unparseable_jsonl_lines_skipped_and_logged(tmp_path, monkeypatch, caplog):
    """Unparseable JSONL lines are skipped with a warning, not silently."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    good = "\n".join(
        json.dumps(r)
        for r in [{"prompt": "p1", "label": "blocked"}, {"prompt": "p2", "label": "allowed"}]
    )
    (test_data / "test.jsonl").write_text(f"{good}\n{{not json}}\nalso bad\n")

    with caplog.at_level(logging.WARNING, logger="main"):
        samples = main._load_huggingface(_classification_config())
    assert len(samples) == 2
    assert "2 unparseable" in caplog.text


def test_staged_empty_file_raises_instead_of_zero_samples(tmp_path, monkeypatch):
    """An empty or all-malformed staged file raises — no silent zero-sample eval."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    (test_data / "test.jsonl").write_text("{not json}\nalso bad\n")

    with pytest.raises(RuntimeError, match="contains no rows"):
        main._load_huggingface(_classification_config())


def test_staged_download_limit_caps_rows(tmp_path, monkeypatch):
    """download_limit caps the staged rows the same way the Hub path caps them."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data" / "safety"
    test_data.mkdir(parents=True)
    rows = json.dumps([{"prompt": f"p{i}", "label": "blocked"} for i in range(5)])
    (test_data / "test.jsonl").write_text(rows)

    config = _classification_config()
    config["download_limit"] = 2
    samples = main._load_huggingface(config)
    assert len(samples) == 2


def test_staged_test_data_ref_with_no_material_raises(tmp_path, monkeypatch):
    """test_data_ref job, no matching staged material → clear error, no Hub call."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    (tmp_path / "test_data").mkdir(parents=True)  # empty staged root

    fake_datasets = type(sys)("datasets")

    def _must_not_load(*a, **kw):
        raise AssertionError("Hub must not be contacted for a staged-data job")

    fake_datasets.load_dataset = _must_not_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))

    with pytest.raises(RuntimeError, match="no usable dataset material"):
        main._load_huggingface(_classification_config())


def test_staged_ambiguity_raises_without_dataset_path(tmp_path, monkeypatch):
    """Two same-rank staged files raise instead of silently picking one."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True)
    rows = json.dumps([{"prompt": "p1", "label": "blocked"}])
    (test_data / "part-a.jsonl").write_text(rows)
    (test_data / "part-b.jsonl").write_text(rows)

    with pytest.raises(RuntimeError, match="Ambiguous staged dataset"):
        main._load_huggingface(_classification_config())


def test_staged_dataset_path_resolves_ambiguity(tmp_path, monkeypatch):
    """dataset_path in the dataset config wins over an ambiguous staged layout."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True)
    rows = json.dumps([{"prompt": "p1", "label": "blocked"}, {"prompt": "p2", "label": "allowed"}])
    (test_data / "part-a.jsonl").write_text(rows)
    (test_data / "part-b.jsonl").write_text(rows)

    config = _classification_config()
    config["dataset_path"] = "part-a.jsonl"
    samples = main._load_huggingface(config)
    assert len(samples) == 2


def test_staged_discovery_skips_hidden_and_vcs_dirs(tmp_path, monkeypatch):
    """Dataset files inside .git / dot-dirs are never candidates."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data"
    git_dir = test_data / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "objects.jsonl").write_text(json.dumps([{"prompt": "p", "label": "blocked"}]))
    (test_data / ".ipynb_checkpoints").mkdir()
    (test_data / ".ipynb_checkpoints" / "data.jsonl").write_text(
        json.dumps([{"prompt": "p", "label": "blocked"}])
    )

    fake_datasets = type(sys)("datasets")

    def _must_not_load(*a, **kw):
        raise AssertionError("Hub must not be contacted for a staged-data job")

    fake_datasets.load_dataset = _must_not_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # Nothing usable staged (only hidden dirs) → clear error, never the Hub.
    with pytest.raises(RuntimeError, match="no usable dataset material"):
        main._load_huggingface(_classification_config())


def test_staged_split_aware_discovery_prefers_matching_split(tmp_path, monkeypatch):
    """With split=test, test files rank above train files without ambiguity."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True)
    (test_data / "train.jsonl").write_text(
        json.dumps([{"prompt": "train-p", "label": "blocked"}])
    )
    (test_data / "test.jsonl").write_text(
        json.dumps([{"prompt": "test-p", "label": "blocked"}])
    )

    config = _classification_config()
    config["split"] = "test"
    samples = main._load_huggingface(config)
    assert len(samples) == 1
    assert samples[0]["prompt"] == "test-p"


# ---------------------------------------------------------------------------
# Job-parameter forwarding and multi-dataset staged isolation
# ---------------------------------------------------------------------------


def test_prepare_dataset_configs_forwards_hf_revision(monkeypatch):
    """Job-level hf_revision reaches every dataset entry; per-dataset pins win."""
    import main

    monkeypatch.delenv("EVALHUB_TEST_DATA_DIR", raising=False)
    original = [
        {"name": "a", "source": "huggingface", "hf_name": "o/a"},
        {"name": "b", "source": "huggingface", "hf_name": "o/b", "hf_revision": "v9"},
    ]
    prepared = main._prepare_dataset_configs(original, {"hf_revision": "abc123"})

    assert prepared[0]["hf_revision"] == "abc123"
    assert prepared[1]["hf_revision"] == "v9"
    assert "hf_revision" not in original[0]  # provider.yaml entries are not mutated


def test_prepare_dataset_configs_exports_test_data_dir(monkeypatch):
    """evalhub_test_data_dir overrides the staged root via EVALHUB_TEST_DATA_DIR."""
    import main

    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", "/inherited")
    main._prepare_dataset_configs([{"source": "csv"}], {"evalhub_test_data_dir": "/custom/root/"})
    assert main._test_data_root() == "/custom/root"
    monkeypatch.delenv("EVALHUB_TEST_DATA_DIR")  # undo the module-level env write


def test_prepare_dataset_configs_flags_multi_dataset_only(monkeypatch):
    import main

    monkeypatch.delenv("EVALHUB_TEST_DATA_DIR", raising=False)
    single = main._prepare_dataset_configs([{"source": "csv"}], {})
    multi = main._prepare_dataset_configs([{"source": "csv"}, {"source": "csv"}], {})
    assert "_multi_dataset" not in single[0]
    assert all(dc["_multi_dataset"] for dc in multi)


def _multi_dataset_config(hf_name: str) -> dict:
    config = _classification_config()
    config["hf_name"] = hf_name
    config["_multi_dataset"] = True
    return config


def test_multi_dataset_unstaged_dataset_never_gets_another_datasets_rows(tmp_path, monkeypatch):
    """Only one of several datasets is staged: the others raise, not reuse its rows."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    staged = tmp_path / "test_data" / "prompt-injections"
    staged.mkdir(parents=True)
    (staged / "test.parquet.jsonl").write_text(
        json.dumps([{"prompt": "p1", "label": "blocked"}])
    )

    # The dataset with its own directory resolves normally...
    assert len(main._load_huggingface(_multi_dataset_config("deepset/prompt-injections"))) == 1
    # ...the others must fail loudly and point at the directory to stage.
    with pytest.raises(RuntimeError, match=r"Stage it under .*jackhhao_x/"):
        main._load_huggingface(_multi_dataset_config("jackhhao/jackhhao_x"))


def test_no_mount_and_no_spec_ref_means_hub_path(tmp_path, monkeypatch):
    """No /test_data mount and no test_data_ref: staged loading steps aside (Hub)."""
    import main

    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", _write_job_spec_with_test_data_ref(tmp_path, None))
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(main, "_test_data_mount_usable", lambda: False)

    assert main._load_staged_rows(_multi_dataset_config("o/other"), "test") is None


def test_mount_without_spec_key_is_treated_as_staged_job(tmp_path, monkeypatch):
    """eval-hub mounts /test_data but does not put test_data_ref in /meta/job.json.

    Verified on-cluster: the spec has no test_data_ref key, so the usable mount is
    the signal. A dataset that matches nothing must raise, not fall to the Hub.
    """
    import main

    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", _write_job_spec_with_test_data_ref(tmp_path, None))
    root = tmp_path / "test_data"
    (root / "prompt-injections").mkdir(parents=True)
    (root / "prompt-injections" / "test.jsonl").write_text(
        json.dumps([{"prompt": "p1", "label": "blocked"}])
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(root))

    assert len(main._load_staged_rows(_multi_dataset_config("deepset/prompt-injections"), "test")) == 1
    with pytest.raises(RuntimeError, match=r"Stage it under .*other-dataset/"):
        main._load_staged_rows(_multi_dataset_config("acme/other-dataset"), "test")


def test_single_dataset_root_wide_fallback_is_split_aware(tmp_path, monkeypatch):
    """A single-dataset benchmark still resolves test.jsonl beside train.jsonl."""
    import main

    monkeypatch.setenv(
        "EVALHUB_JOB_SPEC_PATH",
        _write_job_spec_with_test_data_ref(tmp_path, {"pvc": {"claim_name": "d"}}),
    )
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "test_data"))
    root = tmp_path / "test_data"
    root.mkdir(parents=True)
    (root / "train.jsonl").write_text(json.dumps([{"prompt": "train-p", "label": "blocked"}]))
    (root / "test.jsonl").write_text(json.dumps([{"prompt": "test-p", "label": "blocked"}]))

    rows = main._load_staged_rows(_classification_config(), "test")
    assert rows == [{"prompt": "test-p", "label": "blocked"}]


def test_custom_root_empty_but_default_mount_populated_still_fails_fast(tmp_path, monkeypatch):
    """A mistyped evalhub_test_data_dir must not send a staged job to the Hub.

    The default /test_data mount is populated (a staged job), the configured root
    is not: the dataset raises the "Stage it under" error rather than falling
    through to the HuggingFace Hub on an air-gapped cluster.
    """
    import main

    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", _write_job_spec_with_test_data_ref(tmp_path, None))
    default_mount = tmp_path / "default_mount"
    (default_mount / "somewhere").mkdir(parents=True)
    monkeypatch.setattr(main, "_DEFAULT_TEST_DATA_DIR", str(default_mount))
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "typo-root"))  # does not exist

    with pytest.raises(RuntimeError, match=r"Stage it under .*typo-root/"):
        main._load_staged_rows(_classification_config(), "test")


def test_mount_check_handles_missing_and_unreadable_roots(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "_DEFAULT_TEST_DATA_DIR", str(tmp_path / "no-default"))
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(tmp_path / "no-custom"))
    assert main._test_data_mount_usable() is False

    populated = tmp_path / "custom"
    populated.mkdir()
    (populated / "x").write_text("1")
    monkeypatch.setenv("EVALHUB_TEST_DATA_DIR", str(populated))
    assert main._test_data_mount_usable() is True
