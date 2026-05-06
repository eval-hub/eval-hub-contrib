"""Red Hat theme assets for the CLEAR static HTML dashboard."""

from pathlib import Path

from .red_hat_dashboard_patches import RED_HAT_DASHBOARD_JS_PATCHES

_THEMES_DIR = Path(__file__).resolve().parent
RED_HAT_CLEAR_DASHBOARD_CSS: str = (_THEMES_DIR / "red_hat_clear_dashboard.css").read_text(
    encoding="utf-8"
)

__all__ = ["RED_HAT_CLEAR_DASHBOARD_CSS", "RED_HAT_DASHBOARD_JS_PATCHES"]
