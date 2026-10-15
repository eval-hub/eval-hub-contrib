"""Constrain request-controlled path segments under a root directory."""
from __future__ import annotations

import os


def _safe_under(root: str, *parts: str):
    """Resolve a path under root; return None if any segment escapes the root.

    Request fields (category / tool_name / api_name) must never be joined into
    filesystem paths without this check (CodeQL: uncontrolled path expression).
    """
    if not root:
        return None
    for part in parts:
        if part is None:
            return None
        text = str(part)
        if text in ("", ".", "..") or "/" in text or "\\" in text or "\x00" in text:
            return None
        if os.path.isabs(text):
            return None
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, *parts))
    try:
        common = os.path.commonpath([root_real, candidate])
    except ValueError:
        return None
    if common != root_real:
        return None
    return candidate

