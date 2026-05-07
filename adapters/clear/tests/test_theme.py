import re
from pathlib import Path

import main


def test_red_hat_clear_dashboard_css_file_exists_next_to_package():
    import themes

    css_path = Path(themes.__file__).resolve().parent / "red_hat_clear_dashboard.css"
    assert css_path.is_file(), f"missing theme CSS at {css_path}"


def test_themes_module_loads_css_from_disk():
    import themes

    assert isinstance(themes.RED_HAT_CLEAR_DASHBOARD_CSS, str)
    assert len(themes.RED_HAT_CLEAR_DASHBOARD_CSS) > 100
    assert "--primary:" in themes.RED_HAT_CLEAR_DASHBOARD_CSS


def test_clear_dashboard_theme_opt_out_logic():
    assert main._use_clear_default_dashboard_html(None) is False
    assert main._use_clear_default_dashboard_html("red_hat") is False
    assert main._use_clear_default_dashboard_html("redhat") is False
    assert main._use_clear_default_dashboard_html("clear") is True
    assert main._use_clear_default_dashboard_html("default") is True
    assert main._use_clear_default_dashboard_html("none") is True
    assert main._use_clear_default_dashboard_html("off") is True
    assert main._use_clear_default_dashboard_html("0") is True


def test_apply_clear_dashboard_theme_rewrites_html_by_default(tmp_path: Path):
    html = (
        "<!doctype html><html lang=\"en\"><head>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        "<title>Old</title>"
        "<style>body{background:#fff}</style>"
        "</head><body>hi</body></html>"
    )
    p = tmp_path / "x.html"
    p.write_text(html, encoding="utf-8")

    main._apply_clear_dashboard_theme([p], theme=None)
    updated = p.read_text(encoding="utf-8")

    assert "ibm-clear-adapter red_hat_theme" in updated
    assert "Agentic Workflow Dashboard" in updated
    assert "--primary:" in updated
    assert re.search(
        r"<style>\s*@import url\('https://fonts\.googleapis\.com",
        updated,
    )


def test_apply_clear_dashboard_theme_respects_opt_out(tmp_path: Path):
    html = (
        "<!doctype html><html lang=\"en\"><head>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
        "<title>Old</title>"
        "<style>body{background:#fff}</style>"
        "</head><body>hi</body></html>"
    )
    p = tmp_path / "x.html"
    p.write_text(html, encoding="utf-8")

    main._apply_clear_dashboard_theme([p], theme="clear")
    updated = p.read_text(encoding="utf-8")

    assert updated == html

