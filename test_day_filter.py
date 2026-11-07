"""ตัวกรองวัน (UTC) ของกราฟ · ตาราง 17 คอลัมน์ · Order audit และ metric "Eₙ สะสม (USD) ผลรวมทุกตัว"

ค่าคาดหวังมาจาก fixture ที่คำนวณแยกอิสระ (test_dash_core._mk_chain,
test_execution_confirmed_contract._execution_fixture และ fixtures/paper_session.json ที่ backend
เขียนเอง) ไม่ได้ย้อนจากโค้ด dashboard
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import firebase_admin
import pandas as pd
import pytest
import streamlit
import streamlit.components.v1 as components
from firebase_admin import db
from streamlit.testing.v1 import AppTest

import lego_dash_core
from lego_dash_core import (ALL_DAYS, LEDGER_COLS, excess_total_by_stock,
                            ledger_day_choices, ledger_day_rows, rows_to_df)
from test_dash_core import HOLD, PRICES, SIGNALS, _mk_chain
from test_execution_confirmed_contract import P0, _execution_fixture

R, DA, A, E = LEDGER_COLS
TIME = "เวลา (UTC)"
DAY_LABEL = "วัน (UTC)"
TOTAL_LABEL = "Eₙ สะสม (USD) ผลรวมทุกตัว"
OLD_DAY, NEW_DAY = "2026-07-31", "2026-08-01"

ROOT = Path(__file__).parent
PAPER = json.loads((ROOT / "fixtures" / "paper_session.json").read_text(encoding="utf-8"))
PAPER_CK = "UBER_paper_session"
AAPL_CK, OLD_AAPL_CK, APLS_CK = "AAPL_execution_contract", "AAPL_0ldc0nf1g000", "APLS_abc123def456"
PROD_URL = "https://lego-firebase-prod.asia-southeast1.firebasedatabase.app"
PAPER_URL = "https://lego-firebase-paper.asia-southeast1.firebasedatabase.app"


def _frame(times, index=None) -> pd.DataFrame:
    return pd.DataFrame({TIME: times}, index=index)


# ---- วัน (UTC) ---------------------------------------------------------------

def test_day_choices_are_unique_utc_days_latest_first():
    df = _frame(["2026-10-05T15:15:05+00:00", "2026-10-06T13:30:05Z",
                 "2026-10-05T19:45:05Z", "2026-10-07T14:00:00Z"])
    assert ledger_day_choices(df) == ["2026-10-07", "2026-10-06", "2026-10-05"]
    assert ledger_day_rows(df, "2026-10-05") == [0, 2]


def test_offsets_become_utc_before_the_day_is_taken():
    # 21:30 New York (−04:00) = 01:30 UTC ของวันถัดไป; ไม่มี offset = UTC ตามชื่อคอลัมน์
    # (แถวที่ไม่มี offset ต้องไม่ยืม −04:00 ของแถวก่อนหน้า แบบที่ pd.to_datetime ทั้งคอลัมน์ทำ)
    df = _frame(["2026-10-05T21:30:00-04:00", "2026-10-05 14:00:00",
                 pd.Timestamp("2026-10-05T23:59:59Z")])
    assert ledger_day_choices(df) == ["2026-10-06", "2026-10-05"]
    assert ledger_day_rows(df, "2026-10-06") == [0]
    assert ledger_day_rows(df, "2026-10-05") == [1, 2]


def test_unreadable_times_are_not_a_choice_and_show_only_under_all_days():
    df = _frame(["t0", None, float("nan"), "2026-10-05T15:00:00Z", "", pd.NaT])
    assert ledger_day_choices(df) == ["2026-10-05"]
    assert ledger_day_rows(df, "2026-10-05") == [3]
    assert ledger_day_rows(df, ALL_DAYS) == list(range(6))
    assert ledger_day_rows(df, None) == list(range(6))


def test_no_time_column_or_no_rows_gives_no_choices():
    no_time = pd.DataFrame({"x": [1, 2]})
    assert ledger_day_choices(no_time) == []
    assert ledger_day_rows(no_time, ALL_DAYS) == [0, 1]
    assert ledger_day_rows(no_time, "2026-10-05") == []
    assert ledger_day_choices(pd.DataFrame()) == []
    assert ledger_day_rows(pd.DataFrame(), ALL_DAYS) == []


def test_rows_are_positions_not_index_labels():
    # frame หลังกรอง chain มี index ไม่ต่อเนื่อง — ต้องได้ตำแหน่ง 0..n-1 ที่ใช้กับ tooltip ได้ตรง
    df = _frame(["2026-10-05T15:00:00Z", "2026-10-06T15:00:00Z", "2026-10-06T16:00:00Z"],
                index=[10, 3, 7])
    assert ledger_day_rows(df, "2026-10-06") == [1, 2]


# ---- Eₙ สะสม ผลรวมทุกตัว ----------------------------------------------------

def _payload(frame: pd.DataFrame) -> dict:
    return {str(row["run_id"]): row.to_dict() for _, row in frame.iterrows()}


def _aapl() -> pd.DataFrame:
    return _execution_fixture()


def _old_aapl() -> pd.DataFrame:
    """chain ของ config เก่าของ AAPL: ตัดจบที่ fill แรก (step 3) จึงมี Eₙ ล่าสุดคนละค่า"""
    old = _execution_fixture()
    old = old[old["DNA step"] <= 3].copy()
    old["chain_key"] = OLD_AAPL_CK
    old["run_id"] = "old" + old["run_id"].astype(str)
    return old


def _apls() -> pd.DataFrame:
    return rows_to_df(_mk_chain(PRICES, HOLD, SIGNALS))


E_AAPL = float(_aapl()[E].iloc[-1])
E_OLD_AAPL = float(_old_aapl()[E].iloc[-1])
E_APLS = float(_apls()[E].iloc[-1])
E_UBER = float(PAPER["webull_lego_state"][PAPER_CK]["execution_cashflow"]["excess"])


def _multi_stock() -> tuple[dict, dict]:
    rows = {**_payload(_aapl()), **_payload(_old_aapl()), **_payload(_apls()),
            **PAPER["webull_lego_rows"]}
    state = {
        AAPL_CK: {"p0": P0, "updated_at": "2026-08-01T15:00:00Z"},
        OLD_AAPL_CK: {"p0": P0, "updated_at": "2026-07-20T15:00:00Z"},
        APLS_CK: {"p0": PRICES[0], "updated_at": "2026-07-17T15:00:00Z"},
        PAPER_CK: json.loads(json.dumps(PAPER["webull_lego_state"][PAPER_CK])),
    }
    return rows, state


def test_the_fixtures_give_the_total_something_to_tell_apart():
    assert len({round(E_AAPL, 6), round(E_OLD_AAPL, 6), round(E_APLS, 6), round(E_UBER, 6)}) == 4
    assert PAPER["webull_lego_state"][PAPER_CK]["paper"] is True


def test_total_counts_the_latest_chain_of_each_stock_and_skips_paper_on_prod():
    rows, state = _multi_stock()
    out = excess_total_by_stock(rows_to_df(rows), state, include_paper=False)
    assert [(p["asset"], p["chain_key"]) for p in out["parts"]] == [("AAPL", AAPL_CK),
                                                                   ("APLS", APLS_CK)]
    assert out["total"] == pytest.approx(E_AAPL + E_APLS)
    assert out["paper"] == [PAPER_CK] and out["failed"] == []


def test_the_paper_view_counts_paper_chains_too():
    rows, state = _multi_stock()
    out = excess_total_by_stock(rows_to_df(rows), state, include_paper=True)
    assert [p["asset"] for p in out["parts"]] == ["AAPL", "APLS", "UBER"]
    assert out["total"] == pytest.approx(E_AAPL + E_APLS + E_UBER)
    assert out["paper"] == []


def test_the_newest_updated_at_decides_which_chain_stands_for_a_stock():
    rows, state = _multi_stock()
    state[OLD_AAPL_CK]["updated_at"] = "2026-08-02T15:00:00Z"   # กติกาเดียวกับ default_chain_index
    out = excess_total_by_stock(rows_to_df(rows), state, include_paper=False)
    assert ("AAPL", OLD_AAPL_CK) in [(p["asset"], p["chain_key"]) for p in out["parts"]]
    assert out["total"] == pytest.approx(E_OLD_AAPL + E_APLS)


def test_without_updated_at_the_chain_with_the_newest_row_wins():
    rows, state = _multi_stock()
    for ck in (AAPL_CK, OLD_AAPL_CK):
        del state[ck]["updated_at"]
    out = excess_total_by_stock(rows_to_df(rows), state, include_paper=False)
    assert ("AAPL", AAPL_CK) in [(p["asset"], p["chain_key"]) for p in out["parts"]]


def test_a_chain_that_cannot_be_computed_is_reported_and_left_out():
    rows, state = _multi_stock()
    state[APLS_CK]["p0"] = "not-a-number"
    out = excess_total_by_stock(rows_to_df(rows), state, include_paper=False)
    assert out["failed"] == [APLS_CK]
    assert out["total"] == pytest.approx(E_AAPL)


def test_rows_without_a_chain_key_are_one_chain_and_no_rows_is_zero():
    out = excess_total_by_stock(_apls().drop(columns="chain_key"), {}, include_paper=False)
    assert [(p["asset"], p["chain_key"]) for p in out["parts"]] == [("APLS", None)]
    assert out["total"] == pytest.approx(E_APLS)
    assert excess_total_by_stock(pd.DataFrame(), None, include_paper=False) == {
        "total": 0.0, "parts": [], "paper": [], "failed": []}


# ---- Streamlit ---------------------------------------------------------------

def _two_day_chain() -> pd.DataFrame:
    """execution fixture แบ่งเป็น 2 วัน: steps 0–3 (fill แรกที่ step 3) วันเก่า, steps 4–5 วันล่าสุด"""
    rows = _execution_fixture()
    rows[TIME] = [f"{OLD_DAY if step <= 3 else NEW_DAY}T14:{step:02d}:00Z"
                  for step in rows["DNA step"].astype(int)]
    return rows


def _run_id(rows: pd.DataFrame, step: int) -> str:
    return str(rows.loc[rows["DNA step"] == step, "run_id"].iloc[0])


def _page(monkeypatch, payloads: dict, secrets: dict | None = None):
    class Reference:
        def __init__(self, path, url=None):
            self.path = path

        def get(self):
            return payloads.get(self.path)

    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(db, "reference", Reference)
    tables: list[str] = []
    charts: list = []
    real_html, real_chart = components.html, streamlit.line_chart

    def html_spy(body, *args, **kwargs):
        tables.append(body)
        return real_html(body, *args, **kwargs)

    def chart_spy(data=None, *args, **kwargs):
        charts.append(data)
        return real_chart(data, *args, **kwargs)

    monkeypatch.setattr(components, "html", html_spy)
    monkeypatch.setattr(streamlit, "line_chart", chart_spy)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.secrets.update(secrets or {"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL": PROD_URL})
    return app, tables, charts


def _day_box(app):
    boxes = [box for box in app.selectbox if box.label == DAY_LABEL]
    assert len(boxes) == 1
    return boxes[0]


def _ledger_html(tables: list[str]) -> str:
    found = [page for page in tables if "<tbody>" in page and E in page]
    assert found, "ไม่พบตาราง 17 คอลัมน์แบบชี้เมาส์"
    return found[-1]


def _body_rows(page: str) -> list[str]:
    body = page.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    return body.split("<tr")[1:]


def _tips(row: str) -> list[str]:
    return [html.unescape(tip) for tip in re.findall(r'data-tip="([^"]*)"', row)]


def _ledger_steps(charts: list) -> list[int]:
    found = [c for c in charts if isinstance(c, pd.DataFrame) and A in c.columns]
    assert found, "ไม่พบกราฟ Recurrence"
    return [int(step) for step in found[-1].index]


def _frame_with(app, column: str) -> pd.DataFrame | None:
    found = [f.value for f in app.dataframe if column in f.value.columns]
    return found[-1] if found else None


def _two_day_page(monkeypatch, audit: dict):
    rows = _two_day_chain()
    payloads = {"webull_lego_rows": _payload(rows),
                "webull_lego_state": {AAPL_CK: {"p0": P0, "updated_at": "2026-08-01T15:00:00Z"}},
                "webull_lego_order_audit": audit}
    app, tables, charts = _page(monkeypatch, payloads)
    return rows, payloads, app, tables, charts


def _audit_of(rows: pd.DataFrame) -> dict:
    """order ของ fill วันเก่า (fee ยัง PENDING) และ fill วันล่าสุด"""
    return {"a_old": {"run_id": _run_id(rows, 3), "status": "FILLED", "broker_fee_status": "PENDING"},
            "a_new": {"run_id": _run_id(rows, 5), "status": "FILLED", "broker_fee_status": "FINAL"}}


def test_the_page_opens_on_the_latest_day_for_chart_table_and_audit(monkeypatch):
    rows = _two_day_chain()
    _, _, app, tables, charts = _two_day_page(monkeypatch, _audit_of(rows))
    app.run(timeout=30)
    assert not app.exception

    box = _day_box(app)
    assert box.value == NEW_DAY
    assert box.options == [f"{NEW_DAY} (ล่าสุด)", OLD_DAY, ALL_DAYS]
    assert any(f"แสดง 2 จาก 6 แถว (วัน {NEW_DAY})" in c.value for c in app.caption)

    assert _ledger_steps(charts) == [4, 5]

    body = _body_rows(_ledger_html(tables))
    assert len(body) == 2
    assert OLD_DAY not in "".join(body)
    for row, step in zip(body, (4, 5)):
        assert f"{NEW_DAY}T14:{step:02d}:00Z" in row
        titles = [tip.splitlines()[0] for tip in _tips(row)]
        assert titles and all(f"· DNA step {step} ·" in title for title in titles)   # tip ไม่เลื่อนแถว
    # ยอดสะสมต่อจากวันก่อน: Aₙ ที่ step 4 ยังเป็น 120 จาก fill ของวันก่อน และ fill ที่ step 5
    # ใช้ P_acted = 108 (fill ที่ step 3 ของวันก่อน) ไม่ใช่ quote แถวแรกของวัน
    fixture = rows.set_index("DNA step")
    assert f">{fixture.at[4, A]:.2f}<" in body[0] and f"{fixture.at[4, A]:.2f}" == "120.00"
    assert f">{fixture.at[5, DA]:.2f}<" in body[1] and f">{fixture.at[5, A]:.2f}<" in body[1]
    fill_tip = next(tip for tip in _tips(body[1]) if tip.startswith("ΔAₙ"))
    assert "108.0000" in fill_tip and "DNA step 3" in fill_tip

    audit = _frame_with(app, "broker_fee_status")
    assert list(audit["run_id"]) == [_run_id(rows, 5)]
    # fee PENDING อยู่ในวันเก่า แต่คำเตือน money fence ต้องยังขึ้นตอนดูวันล่าสุด
    assert any("broker fee ยังเป็น PENDING" in w.value for w in app.warning)


def test_another_day_or_all_days_only_changes_which_rows_are_shown(monkeypatch):
    rows = _two_day_chain()
    _, _, app, tables, charts = _two_day_page(monkeypatch, _audit_of(rows))
    app.run(timeout=30)
    latest_rows = _body_rows(_ledger_html(tables))

    _day_box(app).set_value(OLD_DAY).run(timeout=30)
    assert not app.exception
    assert _ledger_steps(charts) == [0, 1, 2, 3]
    assert len(_body_rows(_ledger_html(tables))) == 4
    assert list(_frame_with(app, "broker_fee_status")["run_id"]) == [_run_id(rows, 3)]

    _day_box(app).set_value(ALL_DAYS).run(timeout=30)
    assert not app.exception
    assert _ledger_steps(charts) == [0, 1, 2, 3, 4, 5]
    every_row = _body_rows(_ledger_html(tables))
    assert len(every_row) == 6
    assert every_row[4:] == latest_rows          # ตัวเลขและ tooltip เหมือนตอนดูทั้ง chain ทุกตัวอักษร
    assert sorted(_frame_with(app, "broker_fee_status")["run_id"]) == sorted(
        [_run_id(rows, 3), _run_id(rows, 5)])

    # โหมด st.dataframe ใช้แถวชุดเดียวกัน
    app.toggle(key="live_hover_table").set_value(False).run(timeout=30)
    assert len(_frame_with(app, "สินทรัพย์")) == 6
    _day_box(app).set_value(NEW_DAY).run(timeout=30)
    table = _frame_with(app, "สินทรัพย์")
    assert list(table[TIME]) == [f"{NEW_DAY}T14:04:00Z", f"{NEW_DAY}T14:05:00Z"]
    assert list(table[A].round(2)) == [round(float(v), 2) for v in rows[A].iloc[4:]]


def test_a_new_day_moves_the_filter_back_to_the_latest_day(monkeypatch):
    rows, payloads, app, _, _ = _two_day_page(monkeypatch, {})
    app.run(timeout=30)
    _day_box(app).set_value(OLD_DAY).run(timeout=30)
    app.toggle(key="live_hover_table").set_value(False).run(timeout=30)
    assert _day_box(app).value == OLD_DAY            # rerun ปกติจำวันที่เลือกไว้

    payloads["webull_lego_rows"][_run_id(rows, 5)][TIME] = "2026-08-02T14:05:00Z"
    app.run(timeout=30)
    assert not app.exception
    assert _day_box(app).value == "2026-08-02"       # มีวันใหม่ -> กลับไปวันล่าสุด
    assert _day_box(app).options[0] == "2026-08-02 (ล่าสุด)"


def test_a_day_without_orders_says_so_and_keeps_the_pending_fee_warning(monkeypatch):
    rows = _two_day_chain()
    audit = {"a_old": _audit_of(rows)["a_old"]}
    _, _, app, _, _ = _two_day_page(monkeypatch, audit)
    app.run(timeout=30)
    assert not app.exception
    assert _frame_with(app, "broker_fee_status") is None
    assert any(f"วัน {NEW_DAY} (UTC) ไม่มี order" in c.value for c in app.caption)
    assert any("broker fee ยังเป็น PENDING" in w.value for w in app.warning)


def test_one_chain_total_equals_the_per_stock_metric(monkeypatch):
    _, _, app, _, _ = _two_day_page(monkeypatch, {})
    app.run(timeout=30)
    metrics = {m.label: m for m in app.metric}
    assert metrics[TOTAL_LABEL].value == metrics["Eₙ สะสม (USD)"].value == f"{E_AAPL:.2f}"


def test_the_total_metric_sums_the_latest_chain_of_each_stock(monkeypatch):
    rows, state = _multi_stock()
    payloads = {"webull_lego_rows": rows, "webull_lego_state": state,
                "webull_lego_order_audit": {}}
    app, _, _ = _page(monkeypatch, payloads)
    app.run(timeout=30)
    assert not app.exception
    expected = f"{E_AAPL + E_APLS:.2f}"
    total = next(m for m in app.metric if m.label == TOTAL_LABEL)
    assert total.value == expected
    assert f"AAPL {E_AAPL:.2f} + APLS {E_APLS:.2f} = {expected}" in total.help
    assert any("ไม่รวม chain PAPER" in c.value and PAPER_CK in c.value for c in app.caption)

    chain_box = next(box for box in app.selectbox if box.label == "Chain")
    for chain, own in ((AAPL_CK, E_AAPL), (APLS_CK, E_APLS), (OLD_AAPL_CK, E_OLD_AAPL)):
        chain_box.set_value(chain).run(timeout=30)
        assert not app.exception
        metrics = {m.label: m.value for m in app.metric}
        assert metrics["Eₙ สะสม (USD)"] == f"{own:.2f}"        # รายตัว = chain ที่เลือก
        assert metrics[TOTAL_LABEL] == expected                 # ผลรวมไม่เปลี่ยนตาม chain ที่เลือก
        chain_box = next(box for box in app.selectbox if box.label == "Chain")


def test_the_paper_view_total_includes_the_paper_chain(monkeypatch):
    app, _, _ = _page(monkeypatch, PAPER,
                      secrets={"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL_PAPER": PAPER_URL})
    app.run(timeout=30)
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics[TOTAL_LABEL] == metrics["Eₙ สะสม (USD)"] == f"{E_UBER:.2f}"
    assert not any("ไม่รวม chain PAPER" in c.value for c in app.caption)


def test_hot_deploy_reloads_core_without_the_day_filter_api(monkeypatch):
    monkeypatch.delattr(lego_dash_core, "ledger_day_rows")
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert callable(getattr(lego_dash_core, "ledger_day_rows", None))
