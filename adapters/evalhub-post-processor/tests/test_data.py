import json

import numpy as np
import pytest
from scipy.stats import norm

from post_processor.data import Table, ppi_arrays, read_table, within
from post_processor.operations import interval


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
    assert bounds == pytest.approx({"lower": center - width, "upper": center + width})


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
