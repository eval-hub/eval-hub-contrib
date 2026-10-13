"""Unit tests for tool-server path confinement (_safe_under)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_VENDOR = Path(__file__).resolve().parents[1] / "tool-server" / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from path_safety import _safe_under  # noqa: E402


def test_safe_under_allows_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    nested = root / "Travel" / "flights"
    nested.mkdir(parents=True)
    target = nested / "search.json"
    target.write_text("{}")

    resolved = _safe_under(str(root), "Travel", "flights", "search.json")
    assert resolved == os.path.realpath(target)


def test_safe_under_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("nope")

    assert _safe_under(str(root), "..") is None
    assert _safe_under(str(root), "Travel", "..") is None
    assert _safe_under(str(root), "Travel/../..") is None
    assert _safe_under(str(root), "Travel", "x/../y") is None


def test_safe_under_rejects_empty_and_absolute(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    assert _safe_under("", "Travel") is None
    assert _safe_under(str(root), "") is None
    assert _safe_under(str(root), ".") is None
    assert _safe_under(str(root), None) is None  # type: ignore[arg-type]
    assert _safe_under(str(root), "/etc") is None
    assert _safe_under(str(root), "a\\b") is None
    assert _safe_under(str(root), "a" + chr(0) + "b") is None


def test_safe_under_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.json").write_text("{}")
    link = root / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not available")

    assert _safe_under(str(root), "escape", "leak.json") is None
