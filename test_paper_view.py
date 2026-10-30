"""PAPER view: the database of LEGO_MODE=paper, read beside PROD's.

fixtures/paper_session.json is one whole paper session produced by the lego backend's own
decision and worker (lego test_paper_mode.Session, broker doubled): paper holdings, paper
fills and the 17-column ledger exactly as lego-firebase-paper holds them.
"""
import json
from pathlib import Path

import firebase_admin
import firebase_admin.db  # noqa: F401  (the submodule the app reads through)
from streamlit.testing.v1 import AppTest

import lego_dash_core

ROOT = Path(__file__).parent
PAPER = json.loads((ROOT / "fixtures" / "paper_session.json").read_text(encoding="utf-8"))
PROD_URL = "https://lego-firebase-prod.asia-southeast1.firebasedatabase.app"
PAPER_URL = "https://lego-firebase-paper.asia-southeast1.firebasedatabase.app"


def databases(monkeypatch, prod, paper=None, calls=None):
    class Reference:
        def __init__(self, path, url=None):
            self.path, self.url = path, url
            if calls is not None:
                calls.append((path, url))

        def get(self):
            return ((paper if self.url == PAPER_URL else prod) or {}).get(self.path)

    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", Reference)


def app(**secrets):
    page = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    page.secrets.update({"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL": PROD_URL, **secrets})
    return page


def test_without_a_paper_database_the_page_is_prod_only(monkeypatch):
    databases(monkeypatch, prod={})
    page = app()
    page.run(timeout=30)
    assert not page.exception and not page.sidebar.radio
    assert any("**PROD**" in m.value for m in page.markdown)


def test_the_paper_view_reads_the_paper_database_and_says_so(monkeypatch):
    calls = []
    databases(monkeypatch, prod={}, paper=PAPER, calls=calls)
    page = app(FIREBASE_DB_URL_PAPER=PAPER_URL)
    page.run(timeout=30)
    assert page.sidebar.radio[0].value == "PROD"                    # PROD stays the default
    assert {url for _, url in calls} == {None}
    page.sidebar.radio[0].set_value("PAPER (จำลอง)").run(timeout=30)
    assert not page.exception
    assert {url for _, url in calls if url} == {PAPER_URL}
    texts = [m.value for m in page.markdown]
    assert any("**PAPER**" in t and "ไม่ใช่เงินจริง" in t for t in texts)
    assert not any("FIREBASE_DB_URL_PAPER" in e.value for e in page.error)
    # The backend's own ledger: the dashboard recomputes nothing and the equations hold.
    assert not any("engine เขียน ledger" in w.value for w in page.warning)
    assert any("ผ่านการตรวจสมการ" in s.value for s in page.success)
    assert {"สถานะตัดสินใจล่าสุด", "Pₙ (USD)", "Eₙ สะสม (USD)"}.issubset(
        {m.label for m in page.metric})


def test_a_paper_chain_is_marked_whichever_database_it_is_in(monkeypatch):
    databases(monkeypatch, prod=PAPER)
    page = app()
    page.run(timeout=30)
    assert not page.exception
    assert any("**PAPER**" in m.value and "chain นี้" in m.value for m in page.markdown)


def test_the_paper_ledger_needs_no_recompute():
    df = lego_dash_core.rows_to_df(PAPER["webull_lego_rows"])
    p0 = next(iter(PAPER["webull_lego_state"].values()))["p0"]
    assert len(df) == 26 and set(df["ฝั่ง"].dropna()) == {"BUY", "SELL"}
    fixed = lego_dash_core.recompute_gated_ledger(df, p0=p0)
    assert lego_dash_core.count_ledger_corrections(df, fixed) == 0
    report, ok = lego_dash_core.integrity_report(df, p0_hint=p0)
    assert ok, report
