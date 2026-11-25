"""Build-time compatibility fixes for the pinned upstream Harness.

Keep task fixes in the library consumed by the CLI. Refuse to patch a new
version or unexpected source; remove these fixes when upstream ships them.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import distribution
from pathlib import Path

HARNESS_VERSION = "0.4.13"

# CAREQA fix: opendatahub-io/lm-evaluation-harness#251 (RHOAIENG-85393).
_EAGER_CAREQA = """try:
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", "bleurt-base-512", module_type="metric")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via pip install evaluate and pip install bert-score",
    )
except Exception as e:
    raise RuntimeError(
        f"Error loading evaluation metrics: {str(e)}. Please check your installation."
    )"""

_LAZY_CAREQA = '''@lru_cache(maxsize=1)
def _load_generative_metrics():
    """Load once when scoring generated answers, not when parsing task YAML."""
    try:
        import evaluate
    except ImportError as exc:
        raise ModuleNotFoundError(
            "CAREQA generation metrics require the evaluate package"
        ) from exc

    metrics = {}
    for name in ("bleu", "rouge", "bertscore", "bleurt"):
        try:
            if name == "bleurt":
                metrics[name] = evaluate.load(
                    "bleurt", "bleurt-base-512", module_type="metric"
                )
            else:
                metrics[name] = evaluate.load(name)
        except Exception as exc:
            raise RuntimeError(
                f"Error loading CAREQA generation metric '{name}': {exc}"
            ) from exc
    return metrics'''

_EAGER_TINY = """try:
    import tinyBenchmarks as tb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`tinyBenchmarks` is required for tinyBenchmarks task metric calculation, install via \\
`pip install git+https://github.com/felipemaiapolo/tinyBenchmarks`"
    )"""

_LAZY_TINY = """@lru_cache(maxsize=1)
def _tinybenchmarks():
    # Partial tinyTruthfulQA runs do not calculate an IRT estimate.
    try:
        import tinyBenchmarks
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tinyBenchmarks is required for complete tiny benchmark scoring"
        ) from exc
    return tinyBenchmarks"""

_TINY_FUNCTION = """def agg_gpirt_truthfulqa(items: List[float], benchmark: str = "truthfulqa") -> float:
    items = np.array(items)"""

_PARTIAL_TINY_FUNCTION = """def agg_gpirt_truthfulqa(items: List[float], benchmark: str = "truthfulqa") -> float | None:
    # RHAI-51139: the official estimate requires exactly 100 anchor outcomes.
    # None is deliberately unscored; the adapter reports count and reason.
    if len(items) != 100:
        return None
    items = np.array(items)"""

REPLACEMENTS = {
    "tasks/careqa/utils_open.py": (
        ("import numpy as np", "from functools import lru_cache\n\nimport numpy as np"),
        (_EAGER_CAREQA, _LAZY_CAREQA),
        (
            "def doc_eval(pred, refs):\n",
            (
                "def doc_eval(pred, refs):\n"
                "    metrics = _load_generative_metrics()\n"
                '    bleu = metrics["bleu"]\n'
                '    rouge = metrics["rouge"]\n'
                '    bertscore = metrics["bertscore"]\n'
                '    bleurt = metrics["bleurt"]\n\n'
            ),
        ),
    ),
    "tasks/tinyBenchmarks/agg_functions.py": (
        (
            "from typing import List",
            "from functools import lru_cache\nfrom typing import List",
        ),
        (_EAGER_TINY, _LAZY_TINY),
        (_TINY_FUNCTION, _PARTIAL_TINY_FUNCTION),
        ("tb.evaluate(", "_tinybenchmarks().evaluate("),
    ),
    # Keep an unscored child out of the score, weights and stderr inputs together.
    "api/group.py": (
        (
            "                    if metric_key in task_result:\n",
            (
                "                    if (\n"
                "                        metric_key in task_result\n"
                "                        and task_result[metric_key] is not None\n"
                "                    ):\n"
            ),
        ),
    ),
}

# SHA256 of the exact 0.4.13 source and the resulting patched source. This
# prevents a dependency bump or an upstream task change from silently losing
# the fixes. Already patched files are accepted for repeat local test runs.
CHECKSUMS = {
    "tasks/careqa/utils_open.py": (
        "b038f2f5f7626e4df14c90aeea5e82272e3ac14fa27ee34b1cdb22a80741a206",
        "7d07c78c6d424c6305aa6f6e5c1ebbd228c70269c3f9ed2a0d54133560d14703",
    ),
    "tasks/tinyBenchmarks/agg_functions.py": (
        "584214a413a32b3352f26fcb9f65c84cfe798453cbe42009993c7a4c2631315f",
        "fe966d99531e6988be2c3ac8313c78797abb8d6598a8ca6bc4785d35fa2a5bde",
    ),
    "api/group.py": (
        "e7cc6b6f1c8d34edc45d928d29b4940d60190da63e76192feaddbcf962718f52",
        "e29d49ac992951d18e8f17a9d486edaf8e24a1de1529ac7ac872fa84ca64b006",
    ),
}


def patched_source(relative_path: str, source: str) -> str:
    original, patched = CHECKSUMS[relative_path]
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == patched:
        return source
    if digest != original:
        raise RuntimeError(
            f"Unexpected Harness source in {relative_path}; review compatibility fixes"
        )
    for old, new in REPLACEMENTS[relative_path]:
        if old not in source:
            raise RuntimeError(f"Missing patch target in {relative_path}")
        source = source.replace(old, new)
    if hashlib.sha256(source.encode()).hexdigest() != patched:
        raise RuntimeError(f"Unexpected patched source in {relative_path}")
    compile(source, relative_path, "exec")
    return source


def apply_fixes(package_root: Path, version: str) -> None:
    if version != HARNESS_VERSION:
        raise RuntimeError(
            f"Harness {version} is not the reviewed {HARNESS_VERSION}; "
            "review or remove the compatibility fixes before upgrading"
        )
    # Validate every file before writing any changes.
    updates = [
        (package_root / name, patched_source(name, (package_root / name).read_text()))
        for name in REPLACEMENTS
    ]
    for path, source in updates:
        path.write_text(source)


if __name__ == "__main__":
    dist = distribution("lm_eval")
    apply_fixes(Path(dist.locate_file("lm_eval")), dist.version)
