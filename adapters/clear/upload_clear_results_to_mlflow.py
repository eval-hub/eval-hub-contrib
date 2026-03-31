#!/usr/bin/env python3
"""Upload an existing ``clear_results.json`` to MLflow without running the full adapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from evalhub.adapter import (
    DefaultCallbacks,
    EvaluationResult,
    JobResults,
    JobSpec,
    ModelConfig,
)
from evalhub.adapter.mlflow import MlflowArtifact

logger = logging.getLogger(__name__)


def _extract_agentic_results(json_results_path: Path) -> list[EvaluationResult]:
    """Mirror ``ClearAdapter._extract_agentic_results`` for metrics logged to MLflow."""
    evaluation_results: list[EvaluationResult] = []

    if not json_results_path.is_file():
        logger.warning("Results file not found: %s", json_results_path)
        return evaluation_results

    with open(json_results_path, encoding="utf-8") as f:
        results = json.load(f)

    stats = results.get("metadata", {}).get("statistics", {})
    agents = results.get("agents", {})

    total_interactions = int(stats.get("total_interactions_analyzed", 0) or 0)
    total_issues = int(stats.get("total_issues_discovered", 0) or 0)
    interactions_with_issues = int(stats.get("total_interactions_with_issues", 0) or 0)
    interactions_no_issues = int(stats.get("total_interactions_no_issues", 0) or 0)
    total_agents_stat = stats.get("total_agents")
    total_agents = int(total_agents_stat) if total_agents_stat is not None else len(agents)

    evaluation_results.append(
        EvaluationResult(
            metric_name="total_interactions",
            metric_value=total_interactions,
            metric_type="int",
        )
    )
    evaluation_results.append(
        EvaluationResult(
            metric_name="total_issues",
            metric_value=total_issues,
            metric_type="int",
        )
    )
    evaluation_results.append(
        EvaluationResult(
            metric_name="interactions_with_issues",
            metric_value=interactions_with_issues,
            metric_type="int",
        )
    )
    evaluation_results.append(
        EvaluationResult(
            metric_name="interactions_no_issues",
            metric_value=interactions_no_issues,
            metric_type="int",
        )
    )
    evaluation_results.append(
        EvaluationResult(
            metric_name="total_agents",
            metric_value=total_agents,
            metric_type="int",
        )
    )

    if total_interactions > 0:
        pct_with_issues = 100.0 * interactions_with_issues / total_interactions
        issues_per_interaction = total_issues / total_interactions
    else:
        pct_with_issues = 0.0
        issues_per_interaction = 0.0
    evaluation_results.append(
        EvaluationResult(
            metric_name="pct_interactions_with_issues",
            metric_value=round(pct_with_issues, 4),
            metric_type="float",
        )
    )
    evaluation_results.append(
        EvaluationResult(
            metric_name="issues_per_interaction",
            metric_value=round(issues_per_interaction, 6),
            metric_type="float",
        )
    )

    agent_scores: list[float] = []
    for agent_name, agent_data in agents.items():
        summary = agent_data.get("agent_summary", {})
        avg_score = summary.get("avg_score", 0.0)
        agent_scores.append(float(avg_score))
        evaluation_results.append(
            EvaluationResult(
                metric_name=f"agent.{agent_name}.avg_score",
                metric_value=float(avg_score),
                metric_type="float",
            )
        )
        issues_catalog = agent_data.get("issues_catalog", {})
        num_issues = len(issues_catalog)
        evaluation_results.append(
            EvaluationResult(
                metric_name=f"agent.{agent_name}.num_issues",
                metric_value=num_issues,
                metric_type="int",
                metadata={"issues_catalog": issues_catalog},
            )
        )

    if agent_scores:
        overall_avg = sum(agent_scores) / len(agent_scores)
        evaluation_results.append(
            EvaluationResult(
                metric_name="average_score",
                metric_value=float(overall_avg),
                metric_type="float",
            )
        )

    logger.info("Extracted %d metrics from CLEAR results", len(evaluation_results))
    return evaluation_results


def _compute_overall_score(results: list[EvaluationResult]) -> Optional[float]:
    for result in results:
        if result.metric_name == "average_score":
            return float(result.metric_value)  # type: ignore[arg-type]
    return None


def _num_evaluated_from_json(json_results_path: Path) -> int:
    if not json_results_path.is_file():
        return 0
    with open(json_results_path, encoding="utf-8") as f:
        results = json.load(f)
    stats = results.get("metadata", {}).get("statistics", {})
    return int(stats.get("total_interactions_analyzed", 0) or 0)


def _metrics_summary_dict(
    config: JobSpec,
    evaluation_results: list[EvaluationResult],
    overall_score: Optional[float],
    num_evaluated: int,
) -> dict[str, Any]:
    return {
        "job_id": config.id,
        "benchmark_id": config.benchmark_id,
        "benchmark_index": config.benchmark_index,
        "provider_id": config.provider_id,
        "model_name": config.model.name,
        "overall_score": overall_score,
        "num_examples_evaluated": num_evaluated,
        "metrics": {r.metric_name: r.metric_value for r in evaluation_results},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload clear_results.json to MLflow (same SDK path as main.py)."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("output/clear_results.json"),
        help="Path to clear_results.json (default: output/clear_results.json)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name (creates experiment if missing).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Run/job id shown in MLflow (default: random UUID).",
    )
    parser.add_argument(
        "--benchmark-id",
        default="clear",
        help="JobSpec.benchmark_id (default: clear).",
    )
    parser.add_argument(
        "--provider-id",
        default="clear_adapter",
        help="JobSpec.provider_id (default: clear_adapter).",
    )
    parser.add_argument(
        "--model-name",
        default="manual-upload",
        help="Model name in JobResults / params (default: manual-upload).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    clear_path = args.json_path.resolve()
    if not clear_path.is_file():
        logger.error("File not found: %s", clear_path)
        return 1

    job_id = args.job_id or str(uuid.uuid4())
    experiment_name = args.experiment_name.strip()
    if not experiment_name:
        logger.error("--experiment-name must be non-empty")
        return 1

    evaluation_results = _extract_agentic_results(clear_path)
    overall_score = _compute_overall_score(evaluation_results)
    num_evaluated = _num_evaluated_from_json(clear_path)

    config = JobSpec(
        id=job_id,
        provider_id=args.provider_id,
        benchmark_id=args.benchmark_id,
        benchmark_index=0,
        model=ModelConfig(url="http://localhost", name=args.model_name),
        parameters={"source": "upload_clear_results_to_mlflow.py"},
        callback_url="http://127.0.0.1/no-sidecar",
        experiment_name=experiment_name,
        tags=[{"key": "source", "value": "manual_clear_upload"}],
    )

    metrics_summary = _metrics_summary_dict(
        config, evaluation_results, overall_score, num_evaluated
    )
    summary_bytes = json.dumps(metrics_summary, indent=2, default=str).encode("utf-8")
    clear_bytes = clear_path.read_bytes()

    results = JobResults(
        id=job_id,
        benchmark_id=config.benchmark_id,
        benchmark_index=config.benchmark_index,
        model_name=config.model.name,
        results=evaluation_results,
        overall_score=overall_score,
        num_examples_evaluated=num_evaluated,
        duration_seconds=0.0,
        evaluation_metadata={"upload_script": True},
    )

    spec = config.model_copy(update={"experiment_name": experiment_name})

    callbacks = DefaultCallbacks(
        job_id=job_id,
        benchmark_id=config.benchmark_id,
        provider_id=config.provider_id,
        benchmark_index=0,
        sidecar_url=None,
    )

    try:
        callbacks.mlflow.save(
            results,
            spec,
            artifacts=[
                MlflowArtifact(
                    "clear_results.json",
                    clear_bytes,
                    "application/json",
                ),
                MlflowArtifact(
                    "metrics_summary.json",
                    summary_bytes,
                    "application/json",
                ),
            ],
        )
    except Exception as e:
        logger.error("MLflow upload failed: %s", e, exc_info=True)
        return 1

    logger.info(
        "Uploaded to experiment %r run %r — %s",
        experiment_name,
        job_id,
        clear_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
