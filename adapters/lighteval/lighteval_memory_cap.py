"""
Memory limit for the code lighteval grades.

lighteval grades LiveCodeBench answers by running each model-written solution in a
forked child process.  ``run_test`` in ``lighteval.tasks.tasks.lcb.codegen_metrics``
calls ``reliability_guard()`` without a limit (its comment says 4 GB, but nothing is
set), so a solution that allocates in a loop is stopped only by the container.  A
175-problem run generated every answer and was then OOMKilled at 512 GiB; sampled pod
memory reached about 250 GiB in another.  Only one grader child runs at a time, so such
a figure is one runaway solution, and it depends on what the model wrote, which is why
it looked random.

This wrapper sets ``RLIMIT_AS`` in the grader child to the address space the child
already inherited plus ``EVALHUB_LIGHTEVAL_CODE_MEMORY_MIB`` (default 2048, ``0``
turns the cap off).  The child is forked from a large process, so an absolute limit
would be either too small for the interpreter itself or too large to matter.  A solution
that exceeds the allowance fails that problem and the run continues.  Only the address
space is limited: ``RLIMIT_STACK`` stays as it was.

Applies on Linux under the ``fork`` start method (the default before Python 3.14), the
only one under which lighteval's grader runs.

``lighteval_logprob_patch.py``, which the adapter runs in place of bare ``lighteval``,
calls :func:`cap_code_grader_memory` before it hands over to lighteval's CLI.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_CODE_MEMORY_ENV = "EVALHUB_LIGHTEVAL_CODE_MEMORY_MIB"
_DEFAULT_CODE_MEMORY_MIB = 2048


def _address_space_bytes() -> int | None:
    """Current virtual size of this process (``VmSize``), or None where unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _code_memory_budget_bytes() -> int:
    """Memory a graded solution may add to what its process inherited; 0 means no cap."""
    raw = os.environ.get(_CODE_MEMORY_ENV, "")
    try:
        mib = int(raw) if raw.strip() else _DEFAULT_CODE_MEMORY_MIB
    except ValueError:
        logger.warning("ignoring %s=%r (not an integer)", _CODE_MEMORY_ENV, raw)
        mib = _DEFAULT_CODE_MEMORY_MIB
    return max(mib, 0) * 1024 * 1024


def cap_code_grader_memory() -> None:
    """Give lighteval's LiveCodeBench grader the memory limit it was meant to have.

    Wraps ``reliability_guard`` so the limit is set in the graded child.  The budget is
    read there, when a solution runs, not when the wrapper is installed, so the setting
    in force at that moment is the one that applies.  Safe to call more than once.
    """
    if _code_memory_budget_bytes() == 0:
        logger.info("%s=0: LiveCodeBench grader memory is not capped", _CODE_MEMORY_ENV)
        return
    try:
        import resource  # noqa: PLC0415 (reliability_guard hides it from the graded code)

        from lighteval.tasks.tasks.lcb import codegen_metrics  # noqa: PLC0415
    except ImportError as exc:
        logger.info("LiveCodeBench grader memory cap skipped: %s", exc)
        return

    original = codegen_metrics.reliability_guard
    if getattr(original, "_evalhub_capped", False):
        return

    def reliability_guard(maximum_memory_bytes=None):
        budget = _code_memory_budget_bytes()
        if maximum_memory_bytes is None and budget and (inherited := _address_space_bytes()) is not None:
            limit = inherited + budget
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError) as exc:
                logger.warning("could not cap grader memory at %d bytes: %s", limit, exc)
        return original(maximum_memory_bytes)

    reliability_guard._evalhub_capped = True  # type: ignore[attr-defined]
    reliability_guard._evalhub_original = original  # type: ignore[attr-defined]
    codegen_metrics.reliability_guard = reliability_guard
    logger.info(
        "LiveCodeBench grader memory capped at +%d MiB per solution",
        _code_memory_budget_bytes() // (1024 * 1024),
    )

