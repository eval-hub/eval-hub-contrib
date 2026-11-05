import json

import httpx
import pytest

from post_processor.data import read_table
from post_processor.sources import Sources
from post_processor.transport import Sidecar


@pytest.mark.parametrize("kind", ["s3", "git", "hf", "pvc"])
def test_kubernetes_uses_staged_data_without_download_or_copy(tmp_path, monkeypatch, kind):
    data = tmp_path / "staged"
    data.mkdir()
    (data / "samples.json").write_text('{"samples":[]}')
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_DATA_PATHS", json.dumps({"results": str(data)}))

    def unexpected(*args, **kwargs):
        pytest.fail("Kubernetes adapter must not download or copy staged source data")

    monkeypatch.setattr(Sources, "_" + kind, unexpected)
    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        source = Sources(Sidecar("http://sidecar", client=client))
        actual = source.download({kind: {}}, tmp_path / "scratch", staged_name="results")
    assert actual == data
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("kind", ["s3", "git", "hf", "pvc"])
def test_kubernetes_never_falls_back_to_direct_download(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("EVALHUB_MODE", "k8s")
    monkeypatch.delenv("EVALHUB_POST_PROCESSOR_DATA_PATHS", raising=False)
    with httpx.Client() as client:
        source = Sources(Sidecar("http://sidecar", client=client))
        with pytest.raises(ValueError, match="runtime-staged"):
            source.download({kind: {}}, tmp_path / "scratch", staged_name="results")


@pytest.mark.parametrize(
    "mapping", ["invalid", "[]", "{}", '{"results":null}', '{"results":"relative"}']
)
def test_staging_mapping_is_validated(tmp_path, monkeypatch, mapping):
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_DATA_PATHS", mapping)
    with httpx.Client() as client:
        source = Sources(Sidecar("http://sidecar", client=client))
        with pytest.raises(ValueError):
            source.download({"s3": {}}, tmp_path / "scratch", staged_name="results")


def test_staged_pvc_file_already_has_subpath_applied(tmp_path, monkeypatch):
    data = tmp_path / "labels.jsonl"
    data.write_text('{"sample_id":1,"label":1}\n')
    monkeypatch.setenv(
        "EVALHUB_POST_PROCESSOR_DATA_PATHS", json.dumps({"calibration-0": str(data)})
    )
    with httpx.Client() as client:
        source = Sources(Sidecar("http://sidecar", client=client))
        path = source.download(
            {"pvc": {"claim_name": "data", "sub_path": "nested/labels.jsonl"}},
            tmp_path / "scratch",
            staged_name="calibration-0",
        )
    assert path == data


def test_staged_symlink_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "data"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_DATA_PATHS", json.dumps({"results": str(link)}))
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="missing or invalid"):
            Sources(Sidecar("http://sidecar", client=client)).download(
                {"git": {}}, tmp_path / "scratch", staged_name="results"
            )


def test_staged_pvc_file_keeps_results_path_mapping(tmp_path, monkeypatch):
    data = tmp_path / "results.json"
    rows = [{"sample_id": "sample-1", "prediction": 0.6}]
    data.write_text(json.dumps({"primary_score": {"metric": "accuracy"}, "samples": rows}))
    monkeypatch.setenv("EVALHUB_POST_PROCESSOR_DATA_PATHS", json.dumps({"results": str(data)}))
    with httpx.Client() as client:
        path = Sources(Sidecar("http://sidecar", client=client)).download(
            {"pvc": {"claim_name": "results", "sub_path": "nested/results.json"}},
            tmp_path / "scratch",
            staged_name="results",
        )
    assert read_table(path, {"path": "results.json"}).rows == rows
