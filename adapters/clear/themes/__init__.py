"""Red Hat theme assets for the CLEAR static HTML dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .red_hat_dashboard_patches import RED_HAT_DASHBOARD_JS_PATCHES

_THEMES_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_red_hat_clear_dashboard_css() -> str:
    return (_THEMES_DIR / "red_hat_clear_dashboard.css").read_text(encoding="utf-8")


def __getattr__(name: str) -> Any:
    if name == "RED_HAT_CLEAR_DASHBOARD_CSS":
        return _load_red_hat_clear_dashboard_css()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), "RED_HAT_CLEAR_DASHBOARD_CSS", "RED_HAT_DASHBOARD_JS_PATCHES"})


__all__ = ["RED_HAT_CLEAR_DASHBOARD_CSS", "RED_HAT_DASHBOARD_JS_PATCHES"]
