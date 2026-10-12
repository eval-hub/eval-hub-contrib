"""Unit tests for ToolEval adapter (MUT/judge/MLflow + G1/G2/G3 modes)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from main import (
    ToolEvalAdapter,
    _calls_match,
    _first_api_name,
    _single_tool_catalog,
    _normalize_pass_label,
    _normalize_win_label,
    _parse_agent_action,
    _require_tool_server_healthy,
    _resolve_tool,
    _run_virtual_call,
    _structural_score,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock())


def _base_config(benchmark_id: str = "single-tool", **param_overrides: object) -> MagicMock:
    config = MagicMock()
    config.id = "job-1"
    config.benchmark_id = benchmark_id
    config.benchmark_index = 0
    config.provider_id = "tooleval"
    config.model.name = "test-model"
    config.model.url = "http://localhost:8080"
    config.model.auth = None
    config.exports = None
    config.parameters = {
        "tool_server_url": "http://toolbench:8080",
        "num_tasks": 1,
        "seed": 7,
        "enable_judge": False,
        "probe_models": False,
        **param_overrides,
    }
    return config


def _tools_catalog() -> list[dict]:
    return [
        {
            "category": "Tools",
            "tool_name": "echo",
            "path": "Tools/echo.json",
            "api_list": [{"name": "echo_message"}],
        },
        {
            "category": "Tools",
            "tool_name": "uppercase",
            "path": "Tools/uppercase.json",
            "api_list": [{"name": "uppercase_message"}],
        },
    ]


def test_unsupported_benchmark_fails() -> None:
    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    config = _base_config()
    config.benchmark_id = "unknown"
    with pytest.raises(ValueError, match="Unsupported benchmark_id"):
        adapter.run_benchmark_job(config, MagicMock())


def test_missing_tool_server_and_model_url() -> None:
    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    config = _base_config(tool_server_url="")
    with pytest.raises(ValueError, match="tool_server_url is required"):
        adapter.run_benchmark_job(config, MagicMock())

    config = _base_config()
    config.model.url = ""
    with pytest.raises(ValueError, match="model.url is required"):
        adapter.run_benchmark_job(config, MagicMock())


def test_helpers() -> None:
    assert _normalize_pass_label("Solved") == "solved"
    assert _normalize_win_label("WIN") == "win"
    action = _parse_agent_action(
        '{"action":"call","tool_name":"echo","api_name":"echo_message","tool_input":{"message":"hello"}}',
        catalog=_tools_catalog(),
    )
    assert action["action"] == "call"
    assert action["tool_name"] == "echo"
    finish = _parse_agent_action('{"action":"finish","final_answer":"done"}', catalog=_tools_catalog())
    assert finish["action"] == "finish"

    episode = {
        "predicted_calls": [
            {
                "category": "Tools",
                "tool_name": "echo",
                "api_name": "echo_message",
                "tool_input": {"message": "hello"},
            }
        ],
        "steps": [{"action": "call", "virtual_ok": True}],
        "any_virtual_ok": True,
    }
    ref = episode["predicted_calls"]
    assert _structural_score(episode, ref, mode="single_tool") == ("solved", "win")
    assert _calls_match(ref[0], ref[0]) is True

    echo_call = {
        "category": "Tools",
        "tool_name": "echo",
        "api_name": "echo_message",
        "tool_input": {"message": "hello"},
    }
    upper_call = {
        "category": "Tools",
        "tool_name": "uppercase",
        "api_name": "uppercase_message",
        "tool_input": {"message": "hello"},
    }
    ordered = {
        "predicted_calls": [echo_call, upper_call],
        "steps": [
            {"action": "call", "virtual_ok": True},
            {"action": "call", "virtual_ok": True},
        ],
    }
    reversed_calls = {
        "predicted_calls": [upper_call, echo_call],
        "steps": [
            {"action": "call", "virtual_ok": True},
            {"action": "call", "virtual_ok": True},
        ],
    }
    refs = [echo_call, upper_call]
    assert _structural_score(ordered, refs, mode="multi_step") == ("solved", "win")
    assert _structural_score(reversed_calls, refs, mode="multi_step") == ("unsolved", "lose")
    # multi_tool remains order-insensitive
    assert _structural_score(reversed_calls, refs, mode="multi_tool") == ("solved", "win")


def test_null_numeric_params_fall_back() -> None:
    from main import _param

    assert _param({"max_steps": None}, "max_steps", 3) == 3
    assert _param({"max_steps": 5}, "max_steps", 3) == 5
    assert _param({}, "max_steps", 3) == 3
    assert int(_param({"max_steps": None}, "max_steps", 3)) == 3


def test_require_healthy_and_virtual() -> None:
    client = MagicMock()
    client.base_url = "http://toolbench:8080"
    client.get.return_value = _FakeResponse(200, {"status": "ok"})
    _require_tool_server_healthy(client)
    client.post.return_value = _FakeResponse(200, {"error": "", "response": "hello"})
    assert _run_virtual_call(
        client,
        category="Tools",
        tool_name="echo",
        api_name="echo_message",
        tool_input={"message": "hello"},
        toolbench_key="local",
    )["error"] == ""


def test_single_tool_structural() -> None:
    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    config = _base_config(
        "single-tool",
        tool_name="echo",
        api_name="echo_message",
        tool_input={"message": "hello"},
        instruction="Echo hello",
    )
    callbacks = MagicMock()
    fake_http = MagicMock()
    fake_http.__enter__ = MagicMock(return_value=fake_http)
    fake_http.__exit__ = MagicMock(return_value=False)

    episode = {
        "steps": [
            {
                "step": 1,
                "action": "call",
                "call": {
                    "category": "Tools",
                    "tool_name": "echo",
                    "api_name": "echo_message",
                    "tool_input": {"message": "hello"},
                },
                "virtual_ok": True,
                "tool_response": {"error": "", "response": "hello"},
            }
        ],
        "final_answer": "",
        "any_virtual_ok": True,
        "predicted_calls": [
            {
                "category": "Tools",
                "tool_name": "echo",
                "api_name": "echo_message",
                "tool_input": {"message": "hello"},
            }
        ],
    }

    with (
        patch("main.httpx.Client", return_value=fake_http),
        patch("main._require_tool_server_healthy"),
        patch("main._list_tools", return_value=_tools_catalog()),
        patch("main._build_openai_client", return_value=MagicMock()),
        patch("main._resolve_mut_api_key", return_value="api-key:ref"),
        patch(
            "main._build_judge_client",
            return_value=(MagicMock(), "test-model", "http://localhost:8080"),
        ),
        patch("main._run_agent_episode", return_value=(episode, 1)),
    ):
        results = adapter.run_benchmark_job(config, callbacks)

    metrics = {r.metric_name: r.metric_value for r in results.results}
    assert metrics["pass_rate"] == 1.0
    assert metrics["win_rate"] == 1.0
    assert results.evaluation_metadata["mode"] == "single_tool"
    assert results.additional_info["framework"] == "tooleval"
    assert len(adapter.mlflow_artifacts) == 2
    callbacks.report_status.assert_called()


def test_multi_tool_and_multi_step_modes() -> None:
    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    callbacks = MagicMock()
    fake_http = MagicMock()
    fake_http.__enter__ = MagicMock(return_value=fake_http)
    fake_http.__exit__ = MagicMock(return_value=False)

    episode = {
        "steps": [
            {
                "step": 1,
                "action": "call",
                "call": {
                    "category": "Tools",
                    "tool_name": "echo",
                    "api_name": "echo_message",
                    "tool_input": {"message": "hello"},
                },
                "virtual_ok": True,
            },
            {
                "step": 2,
                "action": "call",
                "call": {
                    "category": "Tools",
                    "tool_name": "uppercase",
                    "api_name": "uppercase_message",
                    "tool_input": {"message": "hello"},
                },
                "virtual_ok": True,
            },
            {"step": 3, "action": "finish", "final_answer": "done"},
        ],
        "final_answer": "done",
        "any_virtual_ok": True,
        "predicted_calls": [
            {
                "category": "Tools",
                "tool_name": "echo",
                "api_name": "echo_message",
                "tool_input": {"message": "hello"},
            },
            {
                "category": "Tools",
                "tool_name": "uppercase",
                "api_name": "uppercase_message",
                "tool_input": {"message": "hello"},
            },
        ],
    }

    for bench, mode in (
        ("multi-tool", "multi_tool"),
        ("multi-step", "multi_step"),
    ):
        adapter.mlflow_artifacts = []
        config = _base_config(bench, num_tasks=1)
        with (
            patch("main.httpx.Client", return_value=fake_http),
            patch("main._require_tool_server_healthy"),
            patch("main._list_tools", return_value=_tools_catalog()),
            patch("main._build_openai_client", return_value=MagicMock()),
            patch("main._resolve_mut_api_key", return_value="api-key:ref"),
            patch(
                "main._build_judge_client",
                return_value=(MagicMock(), "test-model", "http://localhost:8080"),
            ),
            patch("main._run_agent_episode", return_value=(episode, 3)),
        ):
            results = adapter.run_benchmark_job(config, callbacks)
        assert results.evaluation_metadata["mode"] == mode
        assert {r.metric_name: r.metric_value for r in results.results}["pass_rate"] == 1.0


def test_with_judge_scoring() -> None:
    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    config = _base_config(
        "single-tool",
        enable_judge=True,
        tool_name="echo",
        api_name="echo_message",
        tool_input={"message": "hello"},
    )
    callbacks = MagicMock()
    fake_http = MagicMock()
    fake_http.__enter__ = MagicMock(return_value=fake_http)
    fake_http.__exit__ = MagicMock(return_value=False)
    episode = {
        "steps": [{"step": 1, "action": "call", "virtual_ok": True, "call": {}}],
        "final_answer": "",
        "any_virtual_ok": True,
        "predicted_calls": [
            {
                "category": "Tools",
                "tool_name": "echo",
                "api_name": "echo_message",
                "tool_input": {"message": "hello"},
            }
        ],
    }

    with (
        patch("main.httpx.Client", return_value=fake_http),
        patch("main._require_tool_server_healthy"),
        patch("main._list_tools", return_value=_tools_catalog()),
        patch("main._build_openai_client", return_value=MagicMock()),
        patch("main._resolve_mut_api_key", return_value="api-key:ref"),
        patch(
            "main._build_judge_client",
            return_value=(MagicMock(), "judge-model", "http://localhost:8080"),
        ),
        patch("main._run_agent_episode", return_value=(episode, 1)),
        patch("main._judge_pass_trajectory", return_value="solved"),
        patch("main._judge_win_trajectory", return_value="win"),
    ):
        results = adapter.run_benchmark_job(config, callbacks)

    assert results.evaluation_metadata["judge_calls"] == 2
    assert results.evaluation_metadata["judge_model"] == "judge-model"
    assert {r.metric_name: r.metric_value for r in results.results}["pass_rate"] == 1.0


def test_oci_export_when_configured(tmp_path) -> None:
    from evalhub.adapter import OCIArtifactResult, OCICoordinates

    adapter = ToolEvalAdapter.__new__(ToolEvalAdapter)
    adapter.mlflow_artifacts = []
    config = _base_config(
        "single-tool",
        tool_name="echo",
        api_name="echo_message",
        tool_input={"message": "hello"},
    )
    config.exports = MagicMock()
    config.exports.oci = MagicMock()
    config.exports.oci.coordinates = OCICoordinates(
        oci_host="quay.io",
        oci_repository="evalhub/tooleval-results",
        oci_tag="test",
    )
    callbacks = MagicMock()
    oci_result = OCIArtifactResult(
        digest="sha256:deadbeef",
        reference="oci://quay.io/evalhub/tooleval-results:test",
    )
    callbacks.create_oci_artifact.return_value = oci_result

    fake_http = MagicMock()
    fake_http.__enter__ = MagicMock(return_value=fake_http)
    fake_http.__exit__ = MagicMock(return_value=False)
    episode = {
        "steps": [{"step": 1, "action": "call", "virtual_ok": True, "call": {}}],
        "final_answer": "",
        "any_virtual_ok": True,
        "predicted_calls": [
            {
                "category": "Tools",
                "tool_name": "echo",
                "api_name": "echo_message",
                "tool_input": {"message": "hello"},
            }
        ],
    }

    with (
        patch("main.httpx.Client", return_value=fake_http),
        patch("main._require_tool_server_healthy"),
        patch("main._list_tools", return_value=_tools_catalog()),
        patch("main._build_openai_client", return_value=MagicMock()),
        patch("main._resolve_mut_api_key", return_value="api-key:ref"),
        patch(
            "main._build_judge_client",
            return_value=(MagicMock(), "test-model", "http://localhost:8080"),
        ),
        patch("main._run_agent_episode", return_value=(episode, 1)),
        patch("main._results_output_dir", return_value=tmp_path / "results"),
    ):
        results = adapter.run_benchmark_job(config, callbacks)

    assert results.oci_artifact == oci_result
    callbacks.create_oci_artifact.assert_called_once()
    assert (tmp_path / "results" / "trajectories.json").is_file()
    assert (tmp_path / "results" / "summary.json").is_file()


def test_debug_preview_truncates() -> None:
    from main import _DEBUG_IO_LIMIT, _configure_debug_io, _debug_preview, _log_debug_io

    assert _debug_preview("short") == "short"
    long = "x" * (_DEBUG_IO_LIMIT + 50)
    preview = _debug_preview(long)
    assert preview.startswith("x" * _DEBUG_IO_LIMIT)
    assert "truncated 50 chars" in preview

    _configure_debug_io({"debug_io": False})
    with patch("main.logger.info") as info:
        _log_debug_io("mut", request="hello")
        info.assert_not_called()

    _configure_debug_io({"debug_io": True})
    with patch("main.logger.info") as info:
        _log_debug_io("mut", request="hello")
        info.assert_called()
        assert info.call_args[0][1] == "mut"
        assert "hello" in info.call_args[0][2]


def test_virtual_debug_redacts_toolbench_key() -> None:
    from main import _configure_debug_io, _run_virtual_call

    _configure_debug_io({"debug_io": True})
    client = MagicMock()
    client.post.return_value = _FakeResponse(200, {"error": "", "response": "hello"})
    with patch("main.logger.info") as info:
        _run_virtual_call(
            client,
            category="Tools",
            tool_name="echo",
            api_name="echo_message",
            tool_input={"message": "hello"},
            toolbench_key="super-secret",
        )
    joined = " ".join(str(c) for c in info.call_args_list)
    assert "super-secret" not in joined
    assert "<redacted>" in joined

def test_first_api_name_from_api_list() -> None:
    assert _first_api_name({"api_list": [{"name": "search_flights"}]}) == "search_flights"
    assert _first_api_name({"api_list": [{"name": "  "}, {"name": "book"}]}) == "book"
    assert _first_api_name({}) == "echo_message"
    assert _first_api_name(None) == "echo_message"


def test_parse_agent_action_api_name_from_catalog() -> None:
    action = _parse_agent_action(
        '{"action":"call","tool_name":"uppercase","tool_input":{"message":"hi"}}',
        catalog=_tools_catalog(),
    )
    assert action["api_name"] == "uppercase_message"

    full_cache_catalog = [
        {
            "category": "Travel",
            "tool_name": "flights",
            "api_list": [{"name": "search_flights"}, {"name": "book_flight"}],
        }
    ]
    action = _parse_agent_action(
        '{"action":"call","tool_name":"flights","tool_input":{"from":"SFO"}}',
        catalog=full_cache_catalog,
    )
    assert action["api_name"] == "search_flights"


def test_resolve_tool_api_name_from_api_list() -> None:
    tools = [
        {
            "category": "Travel",
            "tool_name": "flights",
            "api_list": [{"name": "search_flights"}],
        }
    ]
    tool, api, cat = _resolve_tool(tools, category="Travel", tool_name="flights", api_name="")
    assert (tool, api, cat) == ("flights", "search_flights", "Travel")

    tool, api, cat = _resolve_tool(
        tools, category="Travel", tool_name="flights", api_name="book_flight"
    )
    assert api == "book_flight"


def test_single_tool_catalog_filters_and_falls_back() -> None:
    tools = [
        {"category": "Tools", "tool_name": "echo", "api_list": [{"name": "echo_message"}]},
        {
            "category": "Travel",
            "tool_name": "flights",
            "api_list": [{"name": "search_flights"}],
        },
    ]
    assert _single_tool_catalog(tools, tool_name="flights", category="Travel") == [tools[1]]
    assert _single_tool_catalog(tools, tool_name="missing", category="") == [tools[0]]

