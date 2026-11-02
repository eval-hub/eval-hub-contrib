import json

import numpy as np
import pytest
from scipy.stats import norm

from post_processor.data import Table, ppi_arrays, read_table, within
from post_processor.operations import interval


@pytest.mark.parametrize("fmt", ["json", "auto"])
def test_nested_results_join_flat_calibration(tmp_path, fmt):
    records = [{"request_id": str(i), "latency": float(i)} for i in range(8)]
    (tmp_path / "results.json").write_text(
        json.dumps({"benchmarks": [{"requests": {"successful": records}}]})
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"primary_score": {"metric": "latency"}, "estimand": "mean"})
    )
    table = read_table(
        tmp_path,
        {
            "format": fmt,
            "records_path": "$.benchmarks[0].requests.successful",
            "columns": {"sample_id": "request_id", "prediction": "latency"},
        },
    )
    assert table.rows == records
    assert table.primary_score == "latency"
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"sample_id":"0","label":1}\n{"sample_id":"4","label":3}\n')
    cal = read_table(calibration, calibration=True)
    y, yhat, unlabeled = ppi_arrays(table, [cal], {"metric": "latency"}, shared=False)
    np.testing.assert_array_equal(y, [1, 3])
    np.testing.assert_array_equal(yhat, [0, 4])
    np.testing.assert_array_equal(unlabeled, [1, 2, 3, 5, 6, 7])


@pytest.mark.parametrize(
    "jsonpath,document",
    [
        ("$", [{"sample_id": "a", "prediction": 1}]),
        ("$['a/b']['~']['']", {"a/b": {"~": {"": [{"sample_id": "a", "prediction": 1}]}}}),
    ],
)
def test_records_jsonpath_root_and_escaped_keys(tmp_path, jsonpath, document):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(document))
    assert read_table(path, {"records_path": jsonpath}).rows == [
        {"sample_id": "a", "prediction": 1}
    ]


@pytest.mark.parametrize(
    "jsonpath,document,error",
    [
        ("records", {}, "JSONPath"),
        ("/records", {}, "JSONPath"),
        ("$.records[", {}, "JSONPath"),
        ("", {}, "JSONPath"),
        (None, {}, "JSONPath"),
        ("$.missing", {}, "exactly one"),
        ("$.records", {"records": 3}, "record array"),
        ("$.records", {"records": {}}, "record array"),
        ("$.records", {"records": [1]}, "record objects"),
        ("$[9]", [[]], "exactly one"),
        ("$[*]", [[], []], "exactly one"),
        ("$.records[*]", {"records": [{"sample_id": "a"}]}, "record array"),
    ],
)
def test_invalid_records_jsonpath(tmp_path, jsonpath, document, error):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=error):
        read_table(path, {"records_path": jsonpath})


@pytest.mark.parametrize("fmt", ["jsonl", "csv", "parquet"])
def test_records_jsonpath_requires_json(tmp_path, fmt):
    with pytest.raises(ValueError, match="requires JSON"):
        read_table(tmp_path, {"format": fmt, "records_path": "$"})


def test_records_jsonpath_rejects_non_json_file_and_calibration(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"sample_id":"a"}\n')
    with pytest.raises(ValueError, match="requires JSON"):
        read_table(path, {"records_path": "$"})
    with pytest.raises(ValueError, match="only for results"):
        read_table(path, {"records_path": "$"}, calibration=True)


def test_real_ppi_matches_analytical_original_mean_interval():
    result = Table(
        [{"sample_id": str(i), "prediction": p} for i, p in enumerate([0.2, 0.4, 0.8, 0.9])]
    )
    cal = Table(
        [
            {"sample_id": "a", "label": 1, "prediction": 0.7},
            {"sample_id": "b", "label": 0, "prediction": 0.2},
            {"sample_id": "c", "label": 1, "prediction": 0.8},
        ]
    )
    bounds = interval(result, [cal], {"metric": "accuracy"}, shared=False, alpha=0.05)
    residual = np.array([0.3, -0.2, 0.2])
    predicted = np.array([0.2, 0.4, 0.8, 0.9])
    center = predicted.mean() + residual.mean()
    width = norm.ppf(0.975) * np.sqrt(predicted.var() / 4 + residual.var() / 3)
    assert bounds["lower"] == pytest.approx(center - width)
    assert bounds["upper"] == pytest.approx(center + width)
    assert bounds["point_estimate"] == pytest.approx(center)
    assert {
        key: bounds[key] for key in ("n_labeled", "n_unlabeled", "alpha", "method", "lambda")
    } == {
        "n_labeled": 3,
        "n_unlabeled": 4,
        "alpha": 0.05,
        "method": "original_ppi",
        "lambda": 1,
    }


def test_join_labels_by_id_and_remove_calibration_overlap():
    result = Table([{"sample_id": str(i), "metrics": {"acc": float(i)}} for i in range(5)])
    cal = Table([{"sample_id": "3", "label": 2}, {"sample_id": "0", "label": 1}])
    y, yhat, unlabeled = ppi_arrays(result, [cal], {"metric": "acc"}, shared=False)
    np.testing.assert_array_equal(y, [2, 1])
    np.testing.assert_array_equal(yhat, [3, 0])
    np.testing.assert_array_equal(unlabeled, [1, 2, 4])


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv", "parquet"])
def test_format_and_column_mapping(tmp_path, fmt):
    records = [{"id": "a", "human": 0.9, "judge": 0.8}, {"id": "b", "human": 0.5, "judge": 0.4}]
    path = tmp_path / f"samples.{fmt}"
    if fmt == "json":
        path.write_text(json.dumps({"samples": records, "primary_score": {"metric": "accuracy"}}))
    elif fmt == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in records))
    elif fmt == "csv":
        path.write_text("id,human,judge\na,0.9,0.8\nb,0.5,0.4\n")
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(records), path)
    table = read_table(
        tmp_path,
        {"format": fmt, "columns": {"sample_id": "id", "label": "human", "prediction": "judge"}},
    )
    assert len(table.rows) == 2
    assert float(table.get(table.rows[0], "label")) == 0.9


def test_aggregate_artifacts_are_not_samples(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "metrics": {"accuracy": 0.8},
                "results": [{"metric_name": "accuracy", "metric_value": 0.8}],
            }
        )
    )
    with pytest.raises(ValueError, match="aggregate metrics"):
        read_table(tmp_path)


def test_reject_non_mean_estimand(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"estimand": "f1", "samples": [{"sample_id": "a", "prediction": 0.8}]})
    )
    with pytest.raises(ValueError, match="another estimand"):
        read_table(tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "wrong"])
def test_invalid_numeric_scores_fail(bad):
    result = Table([{"sample_id": "a", "prediction": bad}])
    with pytest.raises(ValueError, match="finite numeric"):
        ppi_arrays(result, [], {"metric": "accuracy"}, shared=False)


def test_duplicate_ids_and_unaligned_calibration_fail():
    results = Table([{"sample_id": "a", "prediction": 0.5}] * 2)
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        ppi_arrays(results, [], {"metric": "acc"}, shared=False)
    with pytest.raises(ValueError, match="matching evaluation"):
        ppi_arrays(
            Table([]), [Table([{"sample_id": "a", "label": 1}])], {"metric": "acc"}, shared=False
        )


def test_shared_calibration_requires_benchmark_identity():
    with pytest.raises(ValueError, match="both benchmark_id and provider_id"):
        ppi_arrays(
            Table([]),
            [Table([{"sample_id": "a", "label": 1}])],
            {"metric": "acc", "benchmark_id": "b", "provider_id": "p"},
            shared=True,
        )


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../../outside", "a\\b"])
def test_paths_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError):
        within(tmp_path, path)


def test_nested_categories_form_correct_ppi_arrays(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"id": index, "scores": {"judge": {"value": grade}}}
                    for index, grade in enumerate(["fail", "pass", "partial", "pass"])
                ]
            }
        )
    )
    config = {
        "columns": {"sample_id": "id", "prediction": "scores.judge.value"},
        "value_mappings": {"prediction": {"pass": 1, "fail": 0, "partial": 0.5}},
    }
    results = read_table(tmp_path, config)
    calibration = Table(
        [
            {"sample_id": "c1", "label": 1, "prediction": 0.8},
            {"sample_id": "c2", "label": 0, "prediction": 0.2},
        ]
    )
    y, yhat, unlabeled = ppi_arrays(results, [calibration], {"metric": "accuracy"}, shared=False)
    np.testing.assert_array_equal(y, [1, 0])
    np.testing.assert_array_equal(yhat, [0.8, 0.2])
    np.testing.assert_array_equal(unlabeled, [0, 1, 0.5, 1])


def test_value_mapping_applies_to_metric_fallback_and_labels(tmp_path):
    (tmp_path / "scores.json").write_text(
        json.dumps(
            [
                {"sample_id": "a", "metrics": {"acc": "bad"}},
                {"sample_id": "b", "metrics": {"acc": "good"}},
            ]
        )
    )
    results = read_table(tmp_path, {"value_mappings": {"prediction": {"good": 1, "bad": 0}}})
    cal = Table(
        [
            {"sample_id": "c1", "label": "yes", "prediction": 0.8},
            {"sample_id": "c2", "label": "no", "prediction": 0.2},
        ],
        config={"value_mappings": {"label": {"yes": 1, "no": 0}}},
    )
    labels, _, scores = ppi_arrays(results, [cal], {"metric": "acc"}, shared=False)
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(scores, [0, 1])


@pytest.mark.parametrize("value", ["unknown", 1, True, {"value": "pass"}])
def test_unmapped_categories_are_rejected(value):
    table = Table(
        [{"sample_id": "a", "prediction": value}],
        config={"value_mappings": {"prediction": {"pass": 1}}},
    )
    with pytest.raises(ValueError, match="Unmapped prediction"):
        ppi_arrays(table, [], {"metric": "accuracy"}, shared=False)


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"sample_id": {"x": 1}},
        {"prediction": {}},
        {"prediction": []},
        {"prediction": {1: 0}},
        {"prediction": {"pass": True}},
        {"prediction": {"pass": "1"}},
        {"prediction": {"pass": float("nan")}},
        {"prediction": {"pass": float("inf")}},
    ],
)
def test_invalid_value_mapping_fails_before_loading_data(tmp_path, mapping):
    with pytest.raises(ValueError, match="value_mappings"):
        read_table(tmp_path, {"value_mappings": mapping})
