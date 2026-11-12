#!/usr/bin/env python3
"""Generate ``_benchmarks.py``, the adapter's single benchmark classification table.

Every fact the adapter needs about a benchmark (which endpoint it needs, whether
it executes model-generated code, whether it is a perplexity task, and which
metrics lm-eval reports for it) is read from the pinned lm-eval's own task
configs, so it cannot drift from the framework by hand-editing.

Usage (needs the adapter's ``requirements.txt`` installed, no network):

    python scripts/generate_benchmarks.py           # rewrite _benchmarks.py
    python scripts/generate_benchmarks.py --check   # exit 1 if it is stale

The benchmark IDs come from ``provider.yaml`` (curated ``lm-eval/*`` IDs map to a
task through ``CURATED``; every other ID is the lm-eval task name itself) plus
the members of the composite suites in ``_tasks.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lm_eval.api.metrics  # noqa: F401  (registers the built-in metrics)
import yaml
from lm_eval.api.registry import DEFAULT_METRIC_REGISTRY, is_higher_better
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

ADAPTER_DIR = Path(__file__).resolve().parent.parent
OUTPUT = ADAPTER_DIR / "_benchmarks.py"
MANIFEST = ADAPTER_DIR / "provider.yaml"

#: Curated benchmark ID -> lm-eval task. Everything else uses its own name.
CURATED: dict[str, str] = {
    "lm-eval/gsm8k": "gsm8k",
    "lm-eval/ifeval": "ifeval",
    "lm-eval/triviaqa": "triviaqa",
    "lm-eval/nq_open": "nq_open",
    "lm-eval/mmlu_pro": "mmlu_pro",
    "lm-eval/bbh": "bbh",
    # Registered as "hendrycks_math500"; "math_500" does not exist in lm-eval.
    "lm-eval/math_500": "hendrycks_math500",
    "lm-eval/mmlu": "mmlu",
    "lm-eval/arc_easy": "arc_easy",
    "lm-eval/arc_challenge": "arc_challenge",
    "lm-eval/hellaswag": "hellaswag",
    "lm-eval/winogrande": "winogrande",
    "lm-eval/truthfulqa_mc2": "truthfulqa_mc2",
    "lm-eval/boolq": "boolq",
    "lm-eval/piqa": "piqa",
    "lm-eval/openbookqa": "openbookqa",
    # The zero-shot diamond variant is the registered GPQA task; "gpqa" is a tag.
    "lm-eval/gpqa": "gpqa_diamond_zeroshot",
    # MuSR lives only under the leaderboard group; there is no bare "musr".
    "lm-eval/musr": "leaderboard_musr",
}

#: IDs that are not single tasks (resolved in ``_tasks.py``).
NON_TASK_IDS = {
    "lm-eval/custom",
    "lm-eval/open-llm-leaderboard-v2",
    "lm-eval/generation-suite",
}

#: Not registered in lm-eval 0.4.13 (the midstream ships them); kept for parity
#: with the LMEvalJob provider. lm-eval reports an unknown task if it lacks them.
UNREGISTERED = {"longbenchv2", "mrcr", "mrcr_2needle", "mrcr_4needle", "mrcr_8needle"}

LOGLIKELIHOOD_OUTPUTS = {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}


def _entry_config(index, name: str) -> dict:
    """Fully resolved config of a task entry (``include:`` merged), without importing code."""
    entry = index[name]
    cfg = dict(entry.cfg or {})
    if entry.yaml_path:
        base = load_yaml(entry.yaml_path, resolve_func=False)
        if "task_list" in base or "include" in cfg:
            includes = cfg.pop("include", None) or []
            for inc in includes if isinstance(includes, list) else [includes]:
                inc_cfg = load_yaml(
                    Path(entry.yaml_path).parent / inc, resolve_func=False
                )
                cfg = {**inc_cfg, **cfg}
        else:
            cfg = {**base, **cfg}
    return cfg


def _leaves(index, name: str, seen: set[str] | None = None) -> list[str]:
    """Concrete tasks under a task, group or tag."""
    seen = seen if seen is not None else set()
    if name in seen or name not in index:
        return []
    seen.add(name)
    entry = index[name]
    if "TASK" in str(entry.kind):
        return [name]
    subs = (entry.cfg or {}).get("task") or []
    subs = [subs] if isinstance(subs, str) else subs
    out: list[str] = []
    for sub in subs:
        if isinstance(sub, dict):
            sub = sub.get("task") or sub.get("group")
        if sub:
            out += _leaves(index, sub, seen)
    if not subs:  # a tag: every task that carries it
        out += sorted(
            n
            for n, e in index.items()
            if name in (e.tags or ()) and "TASK" in str(e.kind)
        )
    return out


def _function_returns_scalar(path: str) -> bool:
    """True if the ``!function`` metric at ``<dir>/<module>.<func>`` is annotated to return a scalar.

    A scalar metric is reported under the function's name. Anything else may be a
    dict that names its own result keys (humaneval's ``pass_at_k`` reports
    ``pass@1``), which cannot be known without running it. The source is parsed,
    never imported.
    """
    import ast

    module_path, _, func = path.rpartition(".")
    source = Path(module_path + ".py")
    if not source.is_file():
        return False
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return getattr(node.returns, "id", None) in ("float", "bool", "int")
    return False


def _task_metrics(cfg: dict) -> tuple[list[str], set[str], bool]:
    """``(emitted metric names, lower-is-better names, opaque)`` for a task config.

    A ``!function`` metric is reported under the function's name, unless the task
    has ``process_results``: then the keys come from that Python code, the metric
    is ``opaque`` and nothing can be said about them statically.
    """
    output_type = cfg.get("output_type", "generate_until")
    metric_list = cfg.get("metric_list")
    if not metric_list:
        defaults = list(DEFAULT_METRIC_REGISTRY.get(output_type, []))
        return defaults, {m for m in defaults if is_higher_better(m) is False}, False
    names: list[str] = []
    lower: set[str] = set()
    opaque = False
    for m in metric_list:
        name = m["metric"] if isinstance(m, dict) else m
        if isinstance(name, str) and "/" in name:  # an unresolved !function path
            if cfg.get("process_results") or not _function_returns_scalar(name):
                opaque = True
                continue
            name = name.rsplit(".", 1)[-1]
        if not isinstance(name, str):
            opaque = True
            continue
        names.append(name)
        declared = m.get("higher_is_better") if isinstance(m, dict) else None
        if declared is None:  # the config is silent: lm-eval's registry decides
            declared = is_higher_better(name)
        if declared is False:
            lower.add(name)
    return names, lower, opaque


def _spec(index, task: str) -> dict:
    """Classify one lm-eval task, group or tag."""
    if task not in index:
        # Registered nowhere in the pinned lm-eval: exposed for parity only.
        return {
            "scoring": "generation",
            "rolling": False,
            "unsafe_code": False,
            "registered": False,
            "primary_row": task,
            "emits": (),
            "lower": (),
            "opaque": True,
        }
    entry = index[task]
    leaf_names = _leaves(index, task)
    leaf_cfgs = [_entry_config(index, n) for n in leaf_names]
    outputs = {c.get("output_type", "generate_until") for c in leaf_cfgs}

    kind = str(entry.kind)
    if "TASK" in kind:
        emits, lower, opaque = _task_metrics(leaf_cfgs[0])
        primary_row = task
    elif "GROUP" in kind:
        aggregates = (entry.cfg or {}).get("aggregate_metric_list") or []
        aggregates = [aggregates] if isinstance(aggregates, dict) else aggregates
        emits = [a["metric"] for a in aggregates]
        # The group row only exists if its leaves report the aggregated metric;
        # when some leaf does not (african_flores aggregates "acc" over
        # translation tasks), which keys appear cannot be read statically.
        leaf_emits = [set(_task_metrics(c)[0]) for c in leaf_cfgs]
        lower = set()
        opaque = not emits or any(not set(emits) <= e for e in leaf_emits)
        primary_row = task
    else:  # tag: no results row of its own; the "<tag>_all" leaf is the overall one
        emits, lower, opaque = [], set(), True
        all_leaf = f"{task}_all"
        primary_row = all_leaf if all_leaf in leaf_names else None
        if primary_row:
            emits, lower, opaque = _task_metrics(_entry_config(index, primary_row))

    return {
        "scoring": "loglikelihood" if outputs & LOGLIKELIHOOD_OUTPUTS else "generation",
        "rolling": "loglikelihood_rolling" in outputs,
        "unsafe_code": any(c.get("unsafe_code") for c in leaf_cfgs),
        "registered": True,
        "primary_row": primary_row,
        "emits": tuple(dict.fromkeys(emits)),
        "lower": tuple(sorted(lower)),
        "opaque": opaque,
    }


def _composite_members() -> list[str]:
    """Task names of the composite suites in ``_tasks.py`` (parsed, not imported)."""
    import ast

    tree = ast.parse((ADAPTER_DIR / "_tasks.py").read_text())
    members: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and getattr(node.target, "id", "") == "COMPOSITE_SUITES"
        ):
            suites = ast.literal_eval(node.value)
            members += [t for tasks in suites.values() for t in tasks]
    return members


def build_table(lmeval_version: str) -> str:
    manifest = yaml.safe_load(MANIFEST.read_text())
    ids = [b["id"] for b in manifest["benchmarks"] if b["id"] not in NON_TASK_IDS]
    id_to_task = {i: CURATED.get(i, i) for i in ids}
    unknown_curated = sorted(
        i for i in id_to_task if i.startswith("lm-eval/") and i not in CURATED
    )
    if unknown_curated:
        raise SystemExit(f"curated IDs missing from CURATED: {unknown_curated}")

    index = TaskManager().task_index
    tasks = sorted({*id_to_task.values(), *_composite_members()})
    specs = {t: _spec(index, t) for t in tasks}
    for t in UNREGISTERED & specs.keys():
        specs[t]["registered"] = False

    lines = [
        '"""Benchmark classification table. GENERATED: do not edit by hand.',
        "",
        "Regenerate with ``python scripts/generate_benchmarks.py`` after changing the",
        "benchmark list in ``provider.yaml`` or the pinned lm-eval version. The facts",
        f"are read from lm-eval {lmeval_version}'s own task configs; the drift tests in",
        "``tests/test_adapter.py`` fail if this file no longer matches them.",
        "",
        "``scoring``     loglikelihood: needs /v1/completions with echo+logprobs;",
        "                generation: works on any endpoint.",
        "``rolling``     perplexity (loglikelihood_rolling) task.",
        "``unsafe_code`` the task executes model-generated code.",
        "``registered``  False when the pinned lm-eval does not ship the task.",
        "``primary_row`` results row that carries the benchmark's own score (None: none).",
        "``emits``       metric names lm-eval reports on that row.",
        "``lower``       the subset of ``emits`` where lower is better.",
        "``opaque``      metric names come from Python code and are not statically known.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import NamedTuple",
        "",
        f'LMEVAL_VERSION = "{lmeval_version}"',
        "",
        "",
        "class TaskSpec(NamedTuple):",
        '    """What the adapter needs to know about one lm-eval task."""',
        "",
        '    scoring: str  # "loglikelihood" | "generation"',
        "    rolling: bool",
        "    unsafe_code: bool",
        "    registered: bool",
        "    primary_row: str | None",
        "    emits: tuple[str, ...]",
        "    lower: tuple[str, ...]",
        "    opaque: bool",
        "",
        "",
        "#: lm-eval task (or group/tag) name -> classification.",
        "TASKS: dict[str, TaskSpec] = {",
    ]
    for t in tasks:
        s = specs[t]
        lines.append(
            f'    "{t}": TaskSpec({s["scoring"]!r}, {s["rolling"]}, {s["unsafe_code"]}, '
            f"{s['registered']}, {s['primary_row']!r}, {s['emits']!r}, {s['lower']!r}, "
            f"{s['opaque']}),"
        )
    lines += [
        "}",
        "",
        "#: Benchmark ID -> lm-eval task. Curated IDs carry the ``lm-eval/`` prefix.",
        "BENCHMARKS: dict[str, str] = {",
    ]
    for i in ids:
        lines.append(f'    "{i}": "{id_to_task[i]}",')
    lines += ["}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate _benchmarks.py")
    parser.add_argument(
        "--check", action="store_true", help="fail if _benchmarks.py is stale"
    )
    args = parser.parse_args()

    from importlib.metadata import version

    text = build_table(version("lm_eval"))
    if args.check:
        if OUTPUT.read_text() != text:
            print(
                f"{OUTPUT.name} is stale: run scripts/generate_benchmarks.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.name} is up to date")
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
