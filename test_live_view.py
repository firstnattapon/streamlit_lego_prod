"""หน้า Live หลัง refactor: รู้ทันทีว่าดู Chain ไหน/สดหรือไม่ · 1 คลิกถึงข้อมูลของ Chain · ไม่ล่มทั้งหน้า

AppTest รันเต็มสคริปต์ทุก .run() (= อ่าน Firebase ใหม่ทุกครั้ง) และจำสถานะแท็บ/expander ที่ตั้งผ่าน
session_state ไว้แค่รอบเดียว — การสลับแท็บ/Chain แบบ rerun เฉพาะ fragment ตรวจในเบราว์เซอร์จริง
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from test_day_filter_total import (CHAIN, DAY2, P0, chain, chain_box, metrics, rtdb_rows,
                                   show_view, two_day_page, two_days)
from test_paper_view import PAPER, PAPER_URL, app, databases, open_integrity

LIVE_LABELS = {"สถานะตัดสินใจล่าสุด", "DNA step", "Pₙ (USD)", "Eₙ สะสม (USD)"}


def two_chains(**extra) -> dict:
    """CHAIN (updated_at ล่าสุด) และ QBTS_x — หน้า PROD ปกติที่มีหลาย chain"""
    return {"webull_lego_rows": rtdb_rows(two_days(), two_days("QBTS_x", "QBTS")),
            "webull_lego_state": {CHAIN: {"p0": P0, "updated_at": f"{DAY2}T14:05:00Z"},
                                  "QBTS_x": {"p0": P0, "updated_at": f"{DAY2}T14:04:00Z"}},
            "webull_lego_order_audit": {}, **extra}


def headers(page: AppTest) -> list[str]:
    return [h.value for h in page.subheader]


def status_line(page: AppTest) -> str:
    return next(m.value for m in page.markdown if "-badge[" in m.value)


def live_charts(page: AppTest) -> int:
    return len(page.get("plotly_chart")) - 1          # ลบกราฟของแท็บ Rebalancing 101


def hover_iframes(page: AppTest) -> int:
    return sum('data-tip="' in f.proto.srcdoc for f in page.get("iframe"))


# ---- รู้ทันทีว่าดู Chain ไหน / สดหรือไม่ ------------------------------------------

def test_one_chain_needs_no_picker_and_the_header_names_it(monkeypatch):
    databases(monkeypatch, prod=PAPER)
    page = app()
    page.run(timeout=30)
    assert not page.exception
    assert not [r for r in page.radio if r.label == "Chain"]
    assert "UBER · Live" in headers(page)
    line = status_line(page)
    assert line.startswith(":orange-badge[ค้าง] อัปเดต 2026-10-05 19:45 UTC")   # fixture เก่า -> ค้าง
    assert "chain `UBER_paper_session`" in line
    assert page.button(key="live_refresh").label == "รีเฟรช"


def test_many_chains_are_one_click_buttons_with_the_latest_first(monkeypatch):
    databases(monkeypatch, prod=two_chains())
    page = app()
    page.run(timeout=30)
    assert not page.exception and not page.warning
    box = chain_box(page)
    assert box.options == [CHAIN, "QBTS_x"] and box.value == CHAIN and box.horizontal
    assert "AAPL · Live" in headers(page)
    chain_box(page).set_value("QBTS_x").run(timeout=30)
    assert not page.exception and "QBTS · Live" in headers(page)
    assert "chain `QBTS_x`" in status_line(page)


def test_the_chain_comes_back_from_the_url_and_a_bad_one_is_ignored(monkeypatch):
    databases(monkeypatch, prod=two_chains())
    page = app()
    page.query_params["chain"] = "QBTS_x"
    page.run(timeout=30)
    assert not page.exception and chain_box(page).value == "QBTS_x"
    assert "QBTS · Live" in headers(page)

    page = app()
    page.query_params["chain"] = "no_such_chain"
    page.run(timeout=30)
    assert not page.exception and chain_box(page).value == CHAIN


def test_a_chain_left_over_from_the_other_database_falls_back_to_the_default(monkeypatch):
    paper = {"webull_lego_rows": rtdb_rows(two_days("P_a", "AAPL"), two_days("P_b", "UBER")),
             "webull_lego_state": {"P_a": {"p0": P0, "updated_at": f"{DAY2}T14:01:00Z"},
                                   "P_b": {"p0": P0, "updated_at": f"{DAY2}T14:09:00Z"}}}
    databases(monkeypatch, prod=two_chains(), paper=paper)
    page = app(FIREBASE_DB_URL_PAPER=PAPER_URL)
    page.run(timeout=30)
    chain_box(page).set_value("QBTS_x").run(timeout=30)
    page.sidebar.radio[0].set_value("PAPER (จำลอง)").run(timeout=30)
    assert not page.exception and chain_box(page).value == "P_b"   # ไม่มี QBTS_x ใน PAPER
    assert "UBER · Live" in headers(page)


# ---- ข้อมูล / กราฟ: รันเฉพาะแท็บที่เปิด -----------------------------------------------

def test_the_data_view_opens_first_and_the_chart_view_has_only_charts(monkeypatch):
    page, _ = two_day_page(monkeypatch)
    page.run(timeout=30)
    assert [t.label for t in page.tabs][:3] == ["📊 Live Dashboard", "ข้อมูล", "กราฟ"]
    assert LIVE_LABELS <= set(metrics(page)) and hover_iframes(page) == 1
    assert live_charts(page) == 0

    show_view(page, "กราฟ")
    assert not page.exception and not page.warning
    assert live_charts(page) == 2                    # Rₙ/Aₙ/Eₙ ของ chain + Eₙ ผลรวมทุกตัว
    assert not LIVE_LABELS & set(metrics(page)) and hover_iframes(page) == 0
    assert "Eₙ ผลรวมทุกตัว ตามเวลา" in headers(page)
    # หัวข้อ สถานะ ตัวกรองวัน อยู่เหนือแท็บ: เห็นทั้งสองมุมมอง
    assert "AAPL · Live" in headers(page) and "-badge[" in status_line(page)
    assert any(box.label == "วัน (UTC)" for box in page.selectbox)


def test_integrity_runs_only_while_its_expander_is_open(monkeypatch):
    page, _ = two_day_page(monkeypatch)
    page.run(timeout=30)
    assert not page.success and not any("ผ่าน" in d.value.columns for d in page.dataframe)
    open_integrity(page).run(timeout=30)
    assert not page.exception
    assert any("ผ่านการตรวจสมการ" in s.value for s in page.success)
    assert any("ผ่าน" in d.value.columns for d in page.dataframe)


def test_the_state_pointer_shows_only_this_chain(monkeypatch):
    databases(monkeypatch, prod=two_chains())
    page = app()
    page.run(timeout=30)
    state = next(d.value for d in page.dataframe if list(d.value.columns) == ["ค่า"])
    assert set(state.index) == {"p0", "updated_at"} and state.loc["updated_at", "ค่า"].startswith(DAY2)


# ---- ไม่ล่มทั้งหน้า --------------------------------------------------------------

def test_an_empty_database_says_so_under_the_header(monkeypatch):
    databases(monkeypatch, prod={})
    page = app()
    page.run(timeout=30)
    assert not page.exception and not page.error
    assert "Live" in headers(page) and status_line(page).startswith(":orange-badge[ค้าง]")
    assert any("ยังไม่มีแถว committed" in i.value for i in page.info)


def test_a_failed_read_keeps_the_other_tabs_and_offers_a_refresh(monkeypatch):
    class Broken:
        def __init__(self, path):
            self.path = path

        def get(self):
            raise ConnectionError("rtdb unreachable")

    import firebase_admin
    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", Broken)
    page = app()
    page.run(timeout=30)
    assert not page.exception
    assert any("โหลดข้อมูลไม่ได้" in e.value and "rtdb unreachable" in e.value for e in page.error)
    assert status_line(page).startswith(":red-badge[ผิดพลาด]")
    assert page.button(key="live_refresh").label == "รีเฟรช"
    assert len(page.tabs) == 6 and len(page.number_input) == 6       # อีก 3 แท็บยังอยู่ครบ


def test_an_optional_node_that_fails_is_a_warning_not_a_crash(monkeypatch):
    db = two_chains()

    class Flaky:
        def __init__(self, path, url=None):
            self.path = path

        def get(self):
            if self.path == "webull_lego_order_audit":
                raise TimeoutError("audit slow")
            return db.get(self.path)

    import firebase_admin
    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", Flaky)
    page = app()
    page.run(timeout=30)
    assert not page.exception and LIVE_LABELS <= set(metrics(page))
    assert any("webull_lego_order_audit" in w.value and "audit slow" in w.value for w in page.warning)


def test_a_chain_that_cannot_be_computed_leaves_the_picker_to_choose_another(monkeypatch):
    import lego_dash_core
    real = lego_dash_core.recompute_gated_ledger

    def recompute(df, *args, **kwargs):
        if (df.get("chain_key") == "QBTS_x").any():
            raise ValueError("bad cell")
        return real(df, *args, **kwargs)

    monkeypatch.setattr(lego_dash_core, "recompute_gated_ledger", recompute)
    databases(monkeypatch, prod=two_chains())
    page = app()
    page.query_params["chain"] = "QBTS_x"
    page.run(timeout=30)
    assert not page.exception
    assert any("แสดงหน้า Live ไม่ได้" in e.value and "bad cell" in e.value for e in page.error)
    assert len(page.tabs) == 6 and len(page.number_input) == 6           # อีก 3 แท็บยังอยู่
    chain_box(page).set_value(CHAIN).run(timeout=30)
    assert not page.exception and not page.error and LIVE_LABELS <= set(metrics(page))


def test_null_fields_show_a_dash_instead_of_crashing(monkeypatch):
    rows = chain("QBTS_x", "QBTS", n=5).assign(semantics="future_cashflow_v99")   # คงค่าที่บันทึก
    for col in ("ราคา Pₙ (USD)", "Eₙ ส่วนเกินสะสม (USD)", "DNA step", "สถานะ"):
        rows[col] = rows[col].astype(object)
        rows.loc[4, col] = None
    databases(monkeypatch, prod={"webull_lego_rows": rtdb_rows(rows),
                                 "webull_lego_state": {"QBTS_x": {"p0": P0}}})
    page = app()
    page.run(timeout=30)
    assert not page.exception
    shown = metrics(page)
    assert shown["Pₙ (USD)"] == shown["Eₙ สะสม (USD)"] == shown["DNA step"] == "—"
    assert shown["สถานะล่าสุด"] == "—"


def test_refresh_reads_firebase_again(monkeypatch):
    calls: list = []
    databases(monkeypatch, prod=two_chains(), calls=calls)
    page = app()
    page.run(timeout=30)
    assert sorted(path for path, _ in calls) == sorted(
        ["webull_lego_warnings", "webull_lego_state", "webull_lego_rows", "webull_lego_order_audit"])
    first = len(calls)
    page.button(key="live_refresh").click().run(timeout=30)
    assert not page.exception and len(calls) > first and len(calls) % 4 == 0


def test_a_normal_page_shows_no_warning_and_no_error(monkeypatch):
    databases(monkeypatch, prod=two_chains())
    page = app()
    page.run(timeout=30)
    assert not page.exception and not page.warning and not page.error
    assert isinstance(metrics(page)["Eₙ สะสม (USD)"], str)
    assert pd.notna(float(metrics(page)["Eₙ สะสม (USD)"]))


def test_hot_deploy_reloads_a_stale_live_data_module(monkeypatch):
    """lego_live_data ที่แคชไว้จากรุ่นก่อน (ไม่มี API ที่หน้านี้ใช้) ต้องถูกโหลดใหม่ ไม่ทำให้หน้าล่ม"""
    import lego_live_data
    monkeypatch.delattr(lego_live_data, "state_table")
    page = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    page.run(timeout=20)
    assert not page.exception and callable(lego_live_data.state_table)
