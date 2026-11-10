"""The memory cap on graded LiveCodeBench solutions (lighteval_memory_cap.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

CAP_PATH = Path(__file__).parent.parent / "lighteval_memory_cap.py"
LAUNCHER_PATH = Path(__file__).parent.parent / "lighteval_logprob_patch.py"
ENV = "EVALHUB_LIGHTEVAL_CODE_MEMORY_MIB"

_GRADER_SAMPLE = {
    "input_output": json.dumps({"inputs": ["3\n", "4\n"], "outputs": ["3\n", "4\n"], "fn_name": None})
}


def _load_cap_module():
    spec = importlib.util.spec_from_file_location(f"_memory_cap_{id(CAP_PATH)}", CAP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lcb_grader(monkeypatch):
    """lighteval's LiveCodeBench module with its own ``reliability_guard``, restored afterwards.

    Start from the function lighteval ships rather than whatever an earlier test left
    installed, so the tests do not depend on their order.
    """
    pytest.importorskip("lighteval")
    from lighteval.tasks.tasks.lcb import codegen_metrics

    shipped = getattr(codegen_metrics.reliability_guard, "_evalhub_original", codegen_metrics.reliability_guard)
    monkeypatch.setattr(codegen_metrics, "reliability_guard", shipped)
    return codegen_metrics


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs Linux /proc and RLIMIT_AS")
def test_cap_fails_a_solution_that_allocates_without_limit(lcb_grader, monkeypatch):
    """lighteval calls ``reliability_guard()`` with no limit, so a model-written solution
    that allocates in a loop is stopped only by the container: a 175-problem LiveCodeBench
    run was OOMKilled at 512 GiB after generating every answer.  With the cap, the
    solution fails and an ordinary one still passes.

    The oversized solution asks for 512 MiB once and then prints the right answer, so the
    test is cheap, and it passes (instead of failing) when the cap is missing.  The sizes
    are kept small because ``bytearray`` touches every page it allocates and the forked
    child sits on top of an already large test process.
    """
    monkeypatch.setenv(ENV, "128")
    greedy = "b = bytearray(512 * 1024 * 1024)\nprint(input())\n"
    modest = "b = bytearray(32 * 1024 * 1024)\nprint(input())\n"

    def pass_at_1(code):
        # The path the metric takes: a process pool, then a forked child per solution.
        metrics, _ = lcb_grader.codegen_metrics([_GRADER_SAMPLE], [[code]], k_list=[1], num_process_evaluate=1)
        return metrics["pass@1"]

    assert pass_at_1(greedy) == 1.0, "control: without the cap the oversized solution passes"

    _load_cap_module().cap_code_grader_memory()

    assert pass_at_1(greedy) == 0.0
    assert pass_at_1(modest) == 1.0


def test_cap_can_be_turned_off_and_ignores_bad_values(lcb_grader, monkeypatch):
    mod = _load_cap_module()
    original = lcb_grader.reliability_guard

    monkeypatch.setenv(ENV, "0")
    mod.cap_code_grader_memory()
    assert lcb_grader.reliability_guard is original

    monkeypatch.setenv(ENV, "lots")
    assert mod._code_memory_budget_bytes() == 1024 * 1024 * 1024
    monkeypatch.delenv(ENV)
    assert mod._code_memory_budget_bytes() == 1024 * 1024 * 1024
    monkeypatch.setenv(ENV, "512")
    assert mod._code_memory_budget_bytes() == 512 * 1024 * 1024


def test_default_allowance_fits_the_pod_memory_limit(monkeypatch):
    """The graded child adds up to the allowance on top of the LightEval process, which is
    resident at about 2.1 GiB (2070 MiB in a 175-problem run, 2121 MiB in a smaller one).  In a
    4 GiB container a 2048 MiB allowance left 12 MiB of headroom and, at 2150 MiB, the runaway
    solution was OOM-killed despite the cap, so the default and ``memory_limit`` in
    ``provider.yaml`` have to be changed together.  The cap is on address space, so it does not
    bound inherited pages the child dirties (copy-on-write): a child that ran ``gc.collect()``
    in the real process grew the container by about 280 MiB more than a plain solution."""
    import re

    import yaml

    monkeypatch.delenv(ENV, raising=False)
    allowance_mib = _load_cap_module()._code_memory_budget_bytes() // (1024 * 1024)

    provider_path = Path(__file__).parent.parent / "provider.yaml"
    provider = yaml.safe_load(provider_path.read_text())
    limit = str(provider["runtime"]["k8s"]["memory_limit"])
    parsed = re.fullmatch(r"(\d+)(Mi|Gi)", limit)
    assert parsed, f"memory_limit {limit!r} is not in Mi or Gi"
    limit_mib = int(parsed.group(1)) * (1024 if parsed.group(2) == "Gi" else 1)

    lighteval_process_mib = 2250  # measured 2070 to 2121 MiB, with some room
    copy_on_write_mib = 300  # measured about 280 MiB for a child running gc.collect()
    margin_mib = 256
    needed_mib = lighteval_process_mib + allowance_mib + copy_on_write_mib + margin_mib
    assert needed_mib <= limit_mib, (
        f"{lighteval_process_mib} MiB process + {allowance_mib} MiB allowance + {copy_on_write_mib} MiB "
        f"copy-on-write + {margin_mib} MiB margin = {needed_mib} MiB does not fit {limit}: "
        "lower the default allowance or raise memory_limit"
    )


def test_cap_is_installed_once_and_leaves_an_explicit_limit_alone(lcb_grader, monkeypatch):
    import resource

    mod = _load_cap_module()
    calls = []
    monkeypatch.setattr(lcb_grader, "reliability_guard", lambda maximum_memory_bytes=None: calls.append(maximum_memory_bytes))
    monkeypatch.setattr(mod, "_address_space_bytes", lambda: 1000)
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, value: limits.append((which, value)))
    monkeypatch.setenv(ENV, "1")

    mod.cap_code_grader_memory()
    wrapped = lcb_grader.reliability_guard
    mod.cap_code_grader_memory()
    assert lcb_grader.reliability_guard is wrapped, "a second call must not wrap the wrapper"

    wrapped()
    wrapped(5)
    assert limits == [(resource.RLIMIT_AS, (1000 + 1024 * 1024, 1000 + 1024 * 1024))]
    assert calls == [None, 5]


def test_budget_is_read_when_the_grader_runs_not_when_installed(lcb_grader, monkeypatch):
    """The wrapper is installed once per process, often before the setting is final.  The
    budget in force when a solution runs is the one that applies."""
    import resource

    mod = _load_cap_module()
    monkeypatch.setattr(lcb_grader, "reliability_guard", lambda maximum_memory_bytes=None: None)
    monkeypatch.setattr(mod, "_address_space_bytes", lambda: 0)
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, value: limits.append(value))

    monkeypatch.delenv(ENV, raising=False)
    mod.cap_code_grader_memory()  # installed with the default budget
    monkeypatch.setenv(ENV, "3")
    lcb_grader.reliability_guard()
    assert limits == [(3 * 1024 * 1024,) * 2]

    monkeypatch.setenv(ENV, "0")
    lcb_grader.reliability_guard()
    assert len(limits) == 1, "0 switches the cap off even after it was installed"


def test_patch_launcher_installs_the_cap_and_forwards_arguments(lcb_grader, monkeypatch):
    """``python lighteval_logprob_patch.py <args>``, how the adapter starts lighteval, installs
    the cap and then runs lighteval's app with the arguments it was given."""
    import lighteval.__main__ as lighteval_main

    seen = {}
    monkeypatch.setattr(lighteval_main, "app", lambda: seen.update(argv=list(sys.argv)))
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER_PATH), "endpoint", "litellm", "model_name=x", "gsm8k|0"])
    monkeypatch.delenv(ENV, raising=False)

    import runpy

    runpy.run_path(str(LAUNCHER_PATH), run_name="__main__")

    assert seen["argv"] == ["lighteval", "endpoint", "litellm", "model_name=x", "gsm8k|0"]
    assert getattr(lcb_grader.reliability_guard, "_evalhub_capped", False)
