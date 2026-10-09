"""Asset materialization must reject pointers and preserve data on failed downloads."""
import hashlib
import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from prepare_assets import ESSAY_FILE, materialize_asset, materialize_essays, prepare_assets, required_assets


PAYLOAD = b'{"0": "reference-word"}'
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize("payload", [PAYLOAD, b'[{"question": "Q", "answer": "A", "context": []}]'])
def test_materializes_lfs_pointer_with_verified_json(tmp_path, payload):
    path = tmp_path / "words.json"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with patch("prepare_assets.urlopen", return_value=io.BytesIO(payload)):
        materialize_asset(path, "https://example.org/data", hashlib.sha256(payload).hexdigest())
    assert path.read_bytes() == payload


def test_verified_asset_requires_no_network(tmp_path):
    path = tmp_path / "words.json"
    path.write_bytes(PAYLOAD)
    with patch("prepare_assets.urlopen") as download:
        materialize_asset(path, "https://example.org/data", DIGEST)
    download.assert_not_called()


def test_bad_download_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "words.json"
    path.write_bytes(b"original")
    with patch("prepare_assets.urlopen", return_value=io.BytesIO(b"wrong")):
        with pytest.raises(ValueError, match="Checksum mismatch"):
            materialize_asset(path, "https://example.org/data", DIGEST)
    assert path.read_bytes() == b"original"


def test_matching_checksum_still_requires_json(tmp_path):
    payload = b"not-json"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "words.json"
    with patch("prepare_assets.urlopen", return_value=io.BytesIO(payload)):
        with pytest.raises(ValueError):
            materialize_asset(path, "https://example.org/data", digest)
    assert not path.exists()


def test_generation_failure_exposes_nested_stdout(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import main

    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    monkeypatch.setattr(adapter, "_hf_token", lambda: None)
    monkeypatch.setattr(main.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1,
        stdout="Error output: JSONDecodeError: unresolved LFS pointer",
        stderr="CalledProcessError: generator failed",
    ))
    with pytest.raises(RuntimeError, match="JSONDecodeError: unresolved LFS pointer"):
        adapter._generate_task_data(
            "cwe", 4096, tmp_path, "cl100k_base", "openai", "base", 3, 42,
        )


def test_essay_download_and_generation_use_writable_job_cache(monkeypatch, tmp_path):
    import main

    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    scripts = tmp_path / "read-only-scripts"
    monkeypatch.setattr(adapter, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(adapter, "_load_task_config", lambda _: {"args": {"type_haystack": "essay"}})
    monkeypatch.setattr(adapter, "_hf_token", lambda: None)
    data_dir = tmp_path / "data"
    essay_file = data_dir / "assets" / "PaulGrahamEssays.json"
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if Path(cmd[1]).name == "download_paulgraham_essay.py":
            download_dir = Path(kwargs["env"]["RULER_DATA_DIR"])
            assert download_dir.parent == essay_file.parent
            (download_dir / ESSAY_FILE).write_text(json.dumps({"text": "An essay for a synthetic haystack."}))
        else:
            assert kwargs["env"]["RULER_DATA_DIR"] == str(essay_file.parent)
            assert json.loads(essay_file.read_text())["text"]
            output = data_dir / "4096" / "niah_single_2" / "validation.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text('{"input": "prompt", "outputs": ["42"]}\n')
        return SimpleNamespace(returncode=0, stdout="Downloaded essays", stderr="")

    monkeypatch.setattr(main.subprocess, "run", run)
    adapter._prepare_task_assets(["niah_single_2"], data_dir, timeout=60)
    output = adapter._generate_task_data(
        "niah_single_2", 4096, data_dir, "cl100k_base", "openai", "base", 1, 42,
    )
    assert output.is_file()
    assert len(calls) == 2
    assert not scripts.exists()


def test_valid_essay_cache_requires_no_download(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    essay_file = cache / "PaulGrahamEssays.json"
    essay_file.write_text('{"text": "Cached essay text"}')
    with patch("prepare_assets.subprocess.run") as download:
        materialize_essays(essay_file, tmp_path / "scripts", timeout=60)
    download.assert_not_called()


@pytest.mark.parametrize("payload", ['not-json', '{}', '{"text": " "}', '{"text": []}', '[]'])
def test_invalid_essay_cache_is_replaced_only_after_validation(monkeypatch, tmp_path, payload):
    essay_file = tmp_path / "PaulGrahamEssays.json"
    essay_file.write_text(payload)
    def download(*args, **kwargs):
        path = Path(kwargs["env"]["RULER_DATA_DIR"]) / ESSAY_FILE
        path.write_text('{"text": "Fresh essay text"}')
        return SimpleNamespace(returncode=0, stdout="Downloaded essays", stderr="")

    monkeypatch.setattr("prepare_assets.subprocess.run", download)
    materialize_essays(essay_file, tmp_path / "scripts", timeout=60)
    assert json.loads(essay_file.read_text())["text"] == "Fresh essay text"
    assert not list(tmp_path.glob("essays_*"))


@pytest.mark.parametrize("payload", ['not-json', '{}', '{"text": " "}', '{"text": []}', '[]'])
def test_invalid_essay_download_does_not_replace_cache(monkeypatch, tmp_path, payload):
    essay_file = tmp_path / ESSAY_FILE
    essay_file.write_text("old invalid cache")

    def download(*args, **kwargs):
        path = Path(kwargs["env"]["RULER_DATA_DIR"]) / ESSAY_FILE
        path.write_text(payload)
        return SimpleNamespace(returncode=0, stdout="Downloaded essays", stderr="")

    monkeypatch.setattr("prepare_assets.subprocess.run", download)
    with pytest.raises(RuntimeError, match="non-empty JSON text"):
        materialize_essays(essay_file, tmp_path / "scripts", timeout=60)
    assert essay_file.read_text() == "old invalid cache"
    assert not list(tmp_path.glob("essays_*"))


def test_failed_essay_download_reports_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr("prepare_assets.subprocess.run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="Fail download essay: HTTP 503", stderr="No essays downloaded",
    ))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        materialize_essays(tmp_path / ESSAY_FILE, tmp_path / "scripts", timeout=60)
    assert not (tmp_path / "PaulGrahamEssays.json").exists()


def test_essay_downloader_reads_original_sources_and_writes_cache(monkeypatch, tmp_path):
    import main

    source = tmp_path / "source"
    source.mkdir()
    downloader = source / "download_paulgraham_essay.py"
    original = main.RulerAdapter.SCRIPTS_DIR / "data" / "synthetic" / "json" / downloader.name
    downloader.write_text(original.read_text())
    html_url = "http://www.paulgraham.com/test.html"
    github_url = "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/essay.txt"
    raw_url = "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/essay.txt"
    (source / "PaulGrahamEssays_URLs.txt").write_text(f"{html_url}\n{github_url}\n")
    essay_file = tmp_path / "cache" / "PaulGrahamEssays.json"
    monkeypatch.setenv("RULER_DATA_DIR", str(essay_file.parent))
    responses = {
        html_url: b"<html><font><p>HTML essay text.</p></font></html>",
        raw_url: b"Repository essay text.\n",
    }
    requested = []

    def download(url, timeout):
        assert timeout == 30
        requested.append(url)
        return io.BytesIO(responses[url])

    monkeypatch.setattr("urllib.request.urlopen", download)
    runpy.run_path(str(downloader), run_name="__main__")
    text = json.loads(essay_file.read_text())["text"]
    assert "HTML essay text." in text
    assert "Repository essay text." in text
    assert requested == [html_url, raw_url]
    assert not (source / "PaulGrahamEssays.json").exists()
    assert not essay_file.with_suffix(".json.download").exists()
    assert not (essay_file.parent / "essay_repo").exists()
    assert not (essay_file.parent / "essay_html").exists()


@pytest.mark.parametrize("with_essay", [True, False])
def test_job_prepares_required_assets_before_first_dataset(monkeypatch, tmp_path, with_essay):
    import main

    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    benchmarks = ["common-words-extraction"]
    if with_essay:
        benchmarks.append("niah-single-essay")
    config = SimpleNamespace(
        id="asset-preparation-test", benchmark_id=benchmarks[0], num_examples=1,
        model=SimpleNamespace(name="test-model", url="https://model.example/v1"),
        parameters={"benchmarks": benchmarks, "context_lengths": [4096]},
    )
    callbacks = SimpleNamespace(report_status=lambda _: None)
    monkeypatch.setattr(main.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "work"))
    monkeypatch.setattr(adapter, "_verify_tokenizer", lambda *args: True)
    operations = []
    def prepare(configs, cache_dir, scripts_dir, timeout):
        assert cache_dir == tmp_path / "work" / "data" / "assets"
        expected = {"english_words.json"}
        if with_essay:
            expected.add(ESSAY_FILE)
        assert required_assets(configs) == expected
        operations.append("assets")

    monkeypatch.setattr(main, "prepare_assets", prepare)

    def stop_after_first_dataset(**kwargs):
        assert kwargs["task_id"] == "cwe"
        operations.append("dataset")
        raise RuntimeError("End of data-preparation check")

    monkeypatch.setattr(adapter, "_generate_task_data", stop_after_first_dataset)
    with pytest.raises(RuntimeError, match="End of data-preparation check"):
        adapter.run_benchmark_job(config, callbacks)
    assert operations == ["assets", "dataset"]


@pytest.mark.parametrize("task,expected", [
    ("niah_single_1", set()),
    ("niah_single_2", {ESSAY_FILE}),
    ("niah_single_3", {ESSAY_FILE}),
    ("niah_multikey_1", {ESSAY_FILE}),
    ("niah_multikey_2", set()),
    ("niah_multikey_3", set()),
    ("niah_multivalue", {ESSAY_FILE}),
    ("niah_multiquery", {ESSAY_FILE}),
    ("vt", set()),
    ("cwe", {"english_words.json"}),
    ("fwe", set()),
    ("qa_1", {"squad.json"}),
    ("qa_2", {"hotpotqa.json"}),
])
def test_asset_dependencies_match_all_actual_task_configs(task, expected):
    import main

    adapter = main.RulerAdapter.__new__(main.RulerAdapter)
    assert required_assets([adapter._load_task_config(task)]) == expected


@pytest.mark.parametrize("tasks,expected", [
    ([{"task": "qa", "args": {"dataset": "squad"}}] * 2, {"squad.json"}),
    ([{"task": "common_words_extraction"}, {"args": {"type_haystack": "essay"}}],
     {"english_words.json", ESSAY_FILE}),
    ([{"task": "freq_words_extraction"}, {"task": "variable_tracking", "args": {"type_haystack": "noise"}}], set()),
])
def test_preparation_requests_only_needed_assets_once(monkeypatch, tmp_path, tasks, expected):
    requested = []
    monkeypatch.setattr("prepare_assets.materialize_asset", lambda path, *args, **kwargs: requested.append(path.name))
    monkeypatch.setattr("prepare_assets.materialize_essays", lambda path, *args, **kwargs: requested.append(path.name))
    prepare_assets(tasks, tmp_path / "assets", tmp_path / "scripts")
    assert set(requested) == expected
    assert len(requested) == len(expected)
