"""MTEB Python API runner for evaluations with a per-split sample limit."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import mteb
from datasets import Dataset, DatasetDict

logger = logging.getLogger("mteb_sample_runner")


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _limit_retrieval_split(
    split_data: MutableMapping[str, Any],
    limit: int,
    task_name: str,
    split_name: str,
) -> int:
    """Limit retrieval queries and keep their qrels aligned; retain the corpus."""
    queries = split_data.get("queries")
    relevant_docs = split_data.get("relevant_docs")
    if not isinstance(relevant_docs, Mapping) or not hasattr(queries, "select"):
        raise ValueError(
            f"MTEB task '{task_name}' cannot apply num_examples to retrieval split "
            f"'{split_name}': expected selectable queries and a relevant_docs mapping"
        )

    columns = getattr(queries, "column_names", [])
    id_column = next(
        (column for column in ("id", "_id", "query_id") if column in columns),
        None,
    )
    if id_column is None:
        raise ValueError(
            f"MTEB task '{task_name}' cannot apply num_examples to retrieval split "
            f"'{split_name}': query dataset has no supported ID column"
        )

    qrels_ids = {str(query_id) for query_id in relevant_docs}
    selected_indices: list[int] = []
    selected_ids: set[str] = set()
    for index, query_id in enumerate(queries[id_column]):
        normalized_id = str(query_id)
        if normalized_id in qrels_ids:
            selected_indices.append(index)
            selected_ids.add(normalized_id)
            if len(selected_indices) >= limit:
                break

    if not selected_indices:
        raise ValueError(
            f"MTEB task '{task_name}' cannot apply num_examples to retrieval split "
            f"'{split_name}': no queries have matching relevance judgments"
        )

    split_data["queries"] = queries.select(selected_indices)
    split_data["relevant_docs"] = {
        query_id: judgments
        for query_id, judgments in relevant_docs.items()
        if str(query_id) in selected_ids
    }

    top_ranked = split_data.get("top_ranked")
    if isinstance(top_ranked, Mapping):
        split_data["top_ranked"] = {
            query_id: ranking
            for query_id, ranking in top_ranked.items()
            if str(query_id) in selected_ids
        }

    # Deliberately leave split_data["corpus"] unchanged: retrieval queries are
    # the evaluated examples, while the corpus remains the search collection.
    return len(selected_indices)


def _limit_eval_data(
    data: Any,
    eval_splits: list[str],
    limit: int,
    task_name: str,
    *,
    split_value: bool = False,
) -> tuple[int, bool]:
    """Limit supported MTEB dataset structures and report (count, found)."""
    if isinstance(data, DatasetDict):
        count = 0
        found = False
        for split_name in eval_splits:
            if split_name not in data:
                continue
            dataset = data[split_name]
            data[split_name] = dataset.select(range(min(limit, len(dataset))))
            count += len(data[split_name])
            found = True
        return count, found

    if isinstance(data, Dataset):
        if not split_value:
            return 0, False
        selected = data.select(range(min(limit, len(data))))
        return len(selected), True

    if isinstance(data, MutableMapping):
        # MTEB retrieval data is nested by subset and split, with query/qrel
        # structures at the split level.
        if "queries" in data and "relevant_docs" in data:
            return 0, False

        matching_splits = [split for split in eval_splits if split in data]
        if matching_splits:
            count = 0
            for split_name in matching_splits:
                split_data = data[split_name]
                if (
                    isinstance(split_data, MutableMapping)
                    and "queries" in split_data
                    and "relevant_docs" in split_data
                ):
                    count += _limit_retrieval_split(
                        split_data, limit, task_name, split_name
                    )
                elif isinstance(split_data, (Dataset, DatasetDict)):
                    split_count, found = _limit_eval_data(
                        split_data,
                        eval_splits,
                        limit,
                        task_name,
                        split_value=True,
                    )
                    if not found:
                        raise ValueError(
                            f"MTEB task '{task_name}' cannot apply num_examples: "
                            f"evaluation split '{split_name}' has no supported data"
                        )
                    count += split_count
                else:
                    raise ValueError(
                        f"MTEB task '{task_name}' cannot apply num_examples to "
                        f"evaluation split '{split_name}': unsupported data structure "
                        f"{type(split_data).__name__}"
                    )
            return count, True

        # Multilingual tasks may nest DatasetDicts under an HF subset name.
        count = 0
        found = False
        for value in data.values():
            if isinstance(value, (Dataset, DatasetDict, Mapping)):
                nested_count, nested_found = _limit_eval_data(
                    value, eval_splits, limit, task_name
                )
                count += nested_count
                found = found or nested_found
        return count, found

    return 0, False


def _limit_task(task: Any, limit: int) -> int:
    task_name = task.metadata.name
    eval_splits = list(getattr(task.metadata, "eval_splits", None) or [])
    if not eval_splits:
        raise ValueError(
            f"MTEB task '{task_name}' cannot apply num_examples: it declares no "
            "evaluation splits"
        )

    task_type = getattr(task.metadata, "type", None)
    task_type = getattr(task_type, "value", task_type)
    if str(task_type).upper() == "STS" and limit < 2:
        raise ValueError(
            f"MTEB STS task '{task_name}' requires num_examples >= 2 because "
            "its correlation metrics need at least two evaluation examples"
        )

    task.load_data()
    count, found = _limit_eval_data(
        task.dataset, eval_splits, limit, task_name
    )
    if not found:
        dataset = getattr(task, "dataset", None)
        raise ValueError(
            f"MTEB task '{task_name}' cannot apply num_examples: unsupported "
            f"dataset structure {type(dataset).__name__}"
        )
    if count == 0:
        raise ValueError(
            f"MTEB task '{task_name}' has no examples in its evaluation splits"
        )
    return count


def run(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    verbosity = int(config.get("verbosity", 2))
    logging.basicConfig(
        level=logging.WARNING if verbosity <= 0 else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    task_names = _as_list(config.get("tasks"))
    languages = _as_list(config.get("languages", ["eng"]))
    if task_names:
        # The MTEB CLI treats explicit task names as the primary selector.
        tasks = mteb.get_tasks(tasks=task_names)
    else:
        tasks = mteb.get_tasks(
            languages=languages,
            task_types=_as_list(config.get("task_types")),
            categories=_as_list(config.get("task_categories")),
        )

    if not tasks:
        raise RuntimeError("MTEB task selection returned no tasks")

    num_examples = int(config["num_examples"])
    examples_by_task: dict[str, int] = {}
    for task in tasks:
        examples_by_task[task.metadata.name] = _limit_task(task, num_examples)

    # Load the model only after all selected tasks prove they support sampling.
    model = mteb.get_model(
        config["model_name"],
        device=config.get("device"),
    )
    output_dir = Path(config["output_dir"])
    cache = mteb.ResultCache(cache_path=output_dir)
    mteb.evaluate(
        model,
        tasks,
        cache=cache,
        encode_kwargs={"batch_size": int(config["batch_size"])},
        overwrite_strategy=(
            "always" if config.get("overwrite_results", True) else "only-missing"
        ),
        co2_tracker=config.get("co2_tracker", False),
        show_progress_bar=verbosity > 0,
    )

    sampling_info = {
        "num_examples_by_task": examples_by_task,
        "num_examples_evaluated": sum(examples_by_task.values()),
    }
    (output_dir / "evalhub_sampling_info.json").write_text(
        json.dumps(sampling_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: mteb_runner.py <config.json>")
    try:
        run(Path(sys.argv[1]))
    except Exception:
        logger.exception("Sample-limited MTEB evaluation failed")
        raise
