"""Web Apps tab — interactive HTML guides copied from leg-code's Manual Test Lab.

The bundled pages are static, self-contained files shown through
``components.html``. These tests pin: the catalog points at real files, no
bundled page reaches for the network, the theme hook only touches the root
element, and the tab renders every choice (or reports a missing file) without
taking the dashboard down.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit_app

APP_PATH = str(Path(__file__).with_name("streamlit_app.py"))
WEB_APPS_DIR = Path(__file__).with_name("web_apps")
WEB_APP_PAGES = sorted(WEB_APPS_DIR.glob("*.html"))

# Anything that would make a bundled page load code from, or send data to, a
# place other than its own file.
_EXTERNAL = re.compile(
    r"fetch\s*\(|XMLHttpRequest|WebSocket|EventSource|importScripts|"
    r"\bimport\s*\(|@import|<script[^>]+\bsrc\s*=|<link\b|<iframe\b|<form\b|"
    r"\b(?:src|href|action)\s*=\s*[\"']?(?:https?:)?//|"
    r"url\(\s*[\"']?(?:https?:)?//",
    re.IGNORECASE,
)
_FRAGMENT = "<title>x</title><p>y</p>"


def _run_app() -> AppTest:
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=20)
    return app


def _page(filename: str) -> str:
    return (WEB_APPS_DIR / filename).read_text(encoding="utf-8")


def test_catalog_points_at_bundled_files():
    assert streamlit_app.WEB_APPS_DIR == WEB_APPS_DIR.resolve()
    assert list(streamlit_app.WEB_APPS) == ["Rebalancing 101", "Rebalancing Playground"]
    for filename in streamlit_app.WEB_APPS.values():
        assert (WEB_APPS_DIR / filename).is_file(), filename


def test_there_are_bundled_pages_to_check():
    # An empty glob would make the parametrized checks below pass vacuously, and a
    # page that is not in the catalog would never be shown.
    assert {page.name for page in WEB_APP_PAGES} == set(streamlit_app.WEB_APPS.values())


@pytest.mark.parametrize("page", WEB_APP_PAGES, ids=lambda page: page.name)
def test_bundled_page_is_self_contained(page):
    html = page.read_text(encoding="utf-8")
    assert "<title>" in html and len(html) > 1000
    assert _EXTERNAL.search(html) is None


@pytest.mark.parametrize("theme", [None, "", "sepia", "auto"])
def test_apply_iframe_theme_leaves_the_page_alone_when_theme_is_unknown(theme):
    assert streamlit_app.apply_iframe_theme(_FRAGMENT, theme) == _FRAGMENT


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_apply_iframe_theme_wraps_a_fragment_without_adding_a_doctype(theme):
    themed = streamlit_app.apply_iframe_theme(_FRAGMENT, theme)

    assert themed == f'<html data-theme="{theme}">{_FRAGMENT}'
    assert "doctype" not in themed.lower()      # the page body stays byte-for-byte as shipped


def test_apply_iframe_theme_marks_only_the_existing_root_element():
    page = ('<!doctype html><html lang="th"><head><title>x</title></head>'
            "<body><p>&lt;html&gt; stays text</p></body></html>")

    themed = streamlit_app.apply_iframe_theme(page, "dark")

    assert themed.startswith('<!doctype html><html lang="th" data-theme="dark"><head>')
    assert themed.count("data-theme") == 1
    assert themed.endswith("</title></head><body><p>&lt;html&gt; stays text</p></body></html>")


def test_apply_iframe_theme_replaces_a_preset_theme():
    page = "<!doctype html><HTML data-theme='light' lang='th'><body></body></HTML>"

    themed = streamlit_app.apply_iframe_theme(page, "dark")

    assert themed.count("data-theme") == 1
    assert 'data-theme="dark"' in themed
    assert "light" not in themed
    assert "lang='th'" in themed


def test_tabs_are_listed_in_order():
    app = _run_app()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "📊 Live Dashboard", "🎓 Rebalancing 101", "🌐 Web Apps", "⚡ Cheat Sheet",
        "📝 Summary", "🔀 Diagrams"]            # the last two sit inside the Cheat Sheet tab, Summary first


def test_web_apps_tab_renders_every_choice_in_an_iframe():
    app = _run_app()
    radio = app.radio(key="web_app_choice")

    assert list(radio.options) == list(streamlit_app.WEB_APPS)
    assert radio.value == "Rebalancing 101"      # same default as the Manual Test Lab

    for label, filename in streamlit_app.WEB_APPS.items():
        app.radio(key="web_app_choice").set_value(label).run(timeout=20)

        assert not app.exception
        # AppTest reports no Streamlit theme, so the page must arrive untouched.
        assert [frame.proto.srcdoc for frame in app.get("iframe")] == [_page(filename)]


@pytest.mark.parametrize("filename", sorted(streamlit_app.WEB_APPS.values()))
def test_web_app_follows_the_streamlit_theme(monkeypatch, filename):
    monkeypatch.setattr(streamlit_app, "_streamlit_theme", lambda: "dark")

    def script(page_name):
        import streamlit_app
        streamlit_app.render_web_app(page_name)

    app = AppTest.from_function(script, args=(filename,))
    app.run(timeout=20)

    assert not app.exception
    srcdoc = app.get("iframe")[0].proto.srcdoc
    root = re.search(r"<html\b[^>]*>", srcdoc, re.IGNORECASE).group(0)
    assert 'data-theme="dark"' in root
    assert srcdoc.count("data-theme=") == _page(filename).count("data-theme=") + 1   # ours only


def test_missing_web_app_file_is_reported_instead_of_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(streamlit_app, "WEB_APPS_DIR", tmp_path)

    def script():
        import streamlit_app
        streamlit_app.render_web_app("gone.html")

    app = AppTest.from_function(script)
    app.run(timeout=20)

    assert not app.exception
    assert len(app.error) == 1
    assert "FileNotFoundError" in app.error[0].value
    assert not app.get("iframe")
