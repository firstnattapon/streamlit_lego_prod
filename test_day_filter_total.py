"""ตัวกรองวัน (UTC) และ metric "Eₙ สะสม (USD) ผลรวมทุกตัว" — เปลี่ยนแค่ UI สูตรเหมือนเดิม

ค่าคาดหวังของ Eₙ มาจาก fixture ที่คำนวณแยกอิสระ (test_execution_confirmed_contract._execution_fixture
และ fixtures/paper_session.json ที่ backend เขียนเอง) ไม่ได้ย้อนจากโค้ด dashboard. ตัวกรองวันต้องแสดง
แถวของ ledger ที่คำนวณจากทั้ง chain ไม่ใช่ ledger ที่คำนวณใหม่จากวันนั้นวันเดียว
"""
from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import lego_dash_core
import streamlit_app
from lego_dash_core import (ALL_DAYS, LEDGER_COLS, MONEY_COLS, chain_p0_hint, day_options,
                            day_positions, latest_excess_total, order_columns,
                            recompute_gated_ledger, rows_to_df, table_tooltips, utc_days)
from lego_hover_table import build_hover_table_html
from test_dash_core import FIX_C as GATED_FIX_C
from test_dash_core import _mk_chain
from test_execution_confirmed_contract import P0, _execution_fixture
from test_paper_view import PAPER, PAPER_URL, ROOT, app, databases

E = LEDGER_COLS[3]
TOTAL = "Eₙ สะสม (USD) ผลรวมทุกตัว"
DAY1, DAY2 = "2026-08-01", "2026-08-02"
CHAIN = "AAPL_execution_contract"
PAPER_CHAIN = "UBER_paper_session"
STORED_E = list(_execution_fixture()[E])        # Eₙ ที่เทสต์สัญญา execution คำนวณแยกไว้ (แถว 0–5)


def chain(key: str, symbol: str, n: int = 6, days: tuple = (DAY1,) * 6) -> pd.DataFrame:
    """execution fixture n แถวแรกในชื่อ chain/หุ้นใหม่ — แถว i อยู่วัน days[i] (UTC)"""
    rows = _execution_fixture().iloc[:n].copy()
    rows["chain_key"], rows["สินทรัพย์"] = key, symbol
    rows["run_id"] = [f"{key}-{i}" for i in range(n)]
    rows["เวลา (UTC)"] = [f"{days[i]}T14:{i:02d}:00Z" for i in range(n)]
    return rows


def two_days(key: str = CHAIN, symbol: str = "AAPL") -> pd.DataFrame:
    """แถว 0–2 วันที่ 1 ส.ค. และแถว 3–5 วันที่ 2 ส.ค. (UTC)"""
    return chain(key, symbol, days=(DAY1,) * 3 + (DAY2,) * 3)


def rtdb_rows(*frames: pd.DataFrame) -> dict:
    """แถวแบบที่ RTDB คืน: dict ตาม run_id, NaN เป็น None, ตัวเลข numpy เป็น Python"""
    return {str(r["run_id"]): {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                               for k, v in r.to_dict().items()}
            for frame in frames for _, r in frame.iterrows()}


def prod_db() -> dict:
    """AAPL สอง chain (chain เก่าหยุดก่อน), QBTS หนึ่ง chain และ chain PAPER ที่หลงมาอยู่ใน DB ของ PROD"""
    rows = rtdb_rows(chain("AAPL_old", "AAPL", n=4, days=("2026-07-30",) * 4),
                     chain("AAPL_new", "AAPL"), chain("QBTS_x", "QBTS", n=5))
    rows.update(PAPER["webull_lego_rows"])
    state = {"AAPL_old": {"p0": P0, "updated_at": "2026-07-30T14:03:00Z"},
             "AAPL_new": {"p0": P0, "updated_at": "2026-08-01T14:05:00Z"},
             "QBTS_x": {"p0": P0, "updated_at": "2026-08-01T14:04:00Z"},
             **PAPER["webull_lego_state"]}
    return {"webull_lego_rows": rows, "webull_lego_state": state, "webull_lego_order_audit": {}}


# ---- ตัวกรองวัน: helper ล้วน ------------------------------------------------

def test_utc_days_cut_each_row_at_utc_midnight():
    df = pd.DataFrame({"เวลา (UTC)": [
        "2026-08-01T14:00:00Z", "2026-08-01T23:59:59+00:00",
        "2026-08-02T06:30:00+07:00",      # = 2026-08-01T23:30Z
        "2026-08-02 00:00:00",            # ไม่มี offset = UTC (ไม่ติด +07:00 ของแถวก่อน)
        "t3", None, 5]})
    assert utc_days(df) == [DAY1, DAY1, DAY1, DAY2, None, None, None]
    assert utc_days(pd.DataFrame({"x": [1, 2]})) == [None, None]


def test_the_latest_day_comes_first_and_the_whole_chain_last():
    df = two_days()
    assert day_options(df) == [DAY2, DAY1, ALL_DAYS]
    assert day_positions(df, DAY2) == [3, 4, 5]
    assert day_positions(df, DAY1) == [0, 1, 2]
    assert day_positions(df, ALL_DAYS) == day_positions(df, None) == list(range(6))
    assert day_options(df.assign(**{"เวลา (UTC)": "x"})) == []     # ไม่มีเวลาที่อ่านได้ -> ไม่มีตัวกรอง
    one_bad = df.copy()
    one_bad.loc[5, "เวลา (UTC)"] = "x"
    assert day_positions(one_bad, DAY2) == [3, 4]
    assert day_positions(one_bad, ALL_DAYS) == list(range(6))       # แถวเวลาเสียไม่หายจากทั้ง chain


# ---- ผลรวม Eₙ: helper ล้วน ---------------------------------------------------

def test_chain_p0_hint_reads_the_state_pointer_like_before():
    state = {"c": {"p0": 65.49}, "s": {"p0": "10.19"}, "empty": {}, "bad": "x"}
    assert chain_p0_hint(state, "c") == 65.49 and chain_p0_hint(state, "s") == 10.19
    assert chain_p0_hint(state, "empty") is None and chain_p0_hint(state, "bad") is None
    assert chain_p0_hint(state, "missing") is None and chain_p0_hint(state, None) is None
    assert chain_p0_hint(None, "c") is None


def test_the_total_adds_the_latest_chain_of_each_symbol_and_prod_skips_paper():
    db = prod_db()
    summary = latest_excess_total(rows_to_df(db["webull_lego_rows"]), db["webull_lego_state"])
    assert [(i["symbol"], i["chain_key"]) for i in summary["items"]] == [("AAPL", "AAPL_new"),
                                                                        ("QBTS", "QBTS_x")]
    assert [i["excess"] for i in summary["items"]] == pytest.approx([STORED_E[5], STORED_E[4]])
    assert summary["total"] == pytest.approx(STORED_E[5] + STORED_E[4])
    assert summary["older"] == ["AAPL_old"] and summary["paper"] == [PAPER_CHAIN]
    text = summary["text"]
    for value in (STORED_E[5], STORED_E[4], STORED_E[5] + STORED_E[4]):
        assert f"{value:,.4f}" in text
    assert "`AAPL_new`" in text and "`QBTS_x`" in text
    assert "chain เก่า `AAPL_old`" in text and f"chain PAPER `{PAPER_CHAIN}`" in text


def test_the_paper_page_counts_its_paper_chain():
    rows, state = rows_to_df(PAPER["webull_lego_rows"]), PAPER["webull_lego_state"]
    expected = float(rows[E].iloc[-1])      # ledger ของ backend เอง (ไม่ต้อง recompute และไม่ rebase)
    on_paper = latest_excess_total(rows, state, paper_page=True)
    assert on_paper["total"] == pytest.approx(expected) and on_paper["paper"] == []
    on_prod = latest_excess_total(rows, state)
    assert on_prod["total"] is None and on_prod["items"] == [] and on_prod["paper"] == [PAPER_CHAIN]
    assert "ไม่มี chain ที่นับได้" in on_prod["text"]


def test_a_chain_is_paper_by_its_state_pointer_or_by_its_rows():
    rows = rtdb_rows(chain("QBTS_x", "QBTS"))
    plain = rows_to_df(rows)
    assert latest_excess_total(plain, {"QBTS_x": {"p0": P0}})["total"] == pytest.approx(STORED_E[5])
    assert latest_excess_total(plain, {"QBTS_x": {"p0": P0, "paper": True}})["paper"] == ["QBTS_x"]
    flagged = rows_to_df({k: {**v, "paper": True} for k, v in rows.items()})
    assert latest_excess_total(flagged, {"QBTS_x": {"p0": P0}})["paper"] == ["QBTS_x"]


def test_without_a_state_pointer_the_latest_row_picks_the_chain():
    df = rows_to_df(rtdb_rows(chain("AAPL_a", "AAPL", n=4, days=(DAY2,) * 4),
                              chain("AAPL_b", "AAPL", days=(DAY1,) * 6)))
    summary = latest_excess_total(df, {})
    assert [i["chain_key"] for i in summary["items"]] == ["AAPL_a"] and summary["older"] == ["AAPL_b"]
    assert summary["total"] == pytest.approx(STORED_E[3])


def test_an_unreadable_excess_withholds_the_total():
    legacy = chain("QBTS_x", "QBTS", n=5).assign(semantics="future_cashflow_v99")  # คงค่าที่บันทึก
    legacy.loc[4, E] = float("nan")
    df = rows_to_df(rtdb_rows(chain("AAPL_new", "AAPL"), legacy))
    summary = latest_excess_total(df, {"AAPL_new": {"p0": P0}, "QBTS_x": {"p0": P0}})
    assert summary["total"] is None
    assert "อ่านไม่ได้" in summary["text"] and f"{STORED_E[5]:,.4f}" in summary["text"]


def test_a_total_that_rounds_to_zero_shows_no_minus_sign(monkeypatch):
    monkeypatch.setattr(streamlit_app, "latest_excess_total",
                        lambda *args, **kwargs: {"total": -1e-12, "text": "t"})
    assert streamlit_app.excess_total_metric(pd.DataFrame(), {}, False) == ("0.00", "t")


# ---- แอป --------------------------------------------------------------------

def day_box(page: AppTest):
    return next(box for box in page.selectbox if box.label == "วัน (UTC)")


def chain_box(page: AppTest):
    return next(box for box in page.selectbox if box.label == "Chain")


def metrics(page: AppTest) -> dict:
    return {m.label: m.value for m in page.metric}


def hover_html(page: AppTest) -> str:
    return next(f.proto.srcdoc for f in page.get("iframe") if 'data-tip="' in f.proto.srcdoc)


def ledger_table(page: AppTest) -> pd.DataFrame:
    return next(d.value for d in page.dataframe if "Aₙ สะสม (USD)" in d.value.columns)


def audit_table(page: AppTest) -> pd.DataFrame | None:
    return next((d.value for d in page.dataframe if "status" in d.value.columns), None)


@pytest.fixture
def charts(monkeypatch) -> list:
    """ข้อมูลที่ Live Dashboard ส่งเข้า st.line_chart (กราฟของแท็บ 101 ไม่นับ)"""
    seen: list = []
    real = st.line_chart

    def spy(data=None, *args, **kwargs):
        if hasattr(data, "columns") and "Aₙ สะสม (USD)" in data.columns:
            seen.append(data.copy())
        return real(data, *args, **kwargs)

    monkeypatch.setattr(st, "line_chart", spy)
    return seen


def two_day_page(monkeypatch, audited=(1, 3, 5)) -> tuple[AppTest, pd.DataFrame]:
    rows = two_days()
    audit = {f"{CHAIN}-{i}": {"run_id": f"{CHAIN}-{i}", "status": "FILLED"} for i in audited}
    databases(monkeypatch, prod={
        "webull_lego_rows": rtdb_rows(rows),
        "webull_lego_state": {CHAIN: {"p0": P0, "updated_at": f"{DAY2}T14:05:00Z"}},
        "webull_lego_order_audit": audit})
    return app(), rows


def whole_chain(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    persisted = rows_to_df(rtdb_rows(rows))
    return persisted, recompute_gated_ledger(persisted, p0=P0, genesis_baseline=True)


def test_the_day_filter_opens_on_the_latest_day_and_shows_rows_of_the_whole_chain(
        monkeypatch, charts):
    page, rows = two_day_page(monkeypatch)
    page.run(timeout=30)
    assert not page.exception and not page.warning
    box = day_box(page)
    assert box.options == [DAY2, DAY1, ALL_DAYS] and box.value == DAY2
    assert any("แสดง 3 จาก 6 แถว" in c.value for c in page.caption)

    persisted, full = whole_chain(rows)
    # ตาราง: แถว 3–5 ของ ledger ทั้ง chain ทั้งค่าและ tooltip
    show = order_columns(full.iloc[3:6].copy())
    for col in MONEY_COLS:
        show[col] = show[col].astype(float).round(2)
    tips, columns = table_tooltips(persisted, p0=P0, genesis_baseline=True)
    assert hover_html(page) == build_hover_table_html(show, tips[3:6], columns, MONEY_COLS)
    # ถ้ากรองก่อนคำนวณ แถวแรกของวันจะกลายเป็น "แถวแรกของช่วงที่โหลด" — ตารางต้องไม่เป็นแบบนั้น
    window_tips, _ = table_tooltips(persisted.iloc[3:6], p0=P0, genesis_baseline=True)
    assert "แถวแรกของช่วงที่โหลด" in window_tips[0]["ΔAₙ ต่อสเต็ป (USD)"]
    assert "แถวแรกของช่วงที่โหลด" not in hover_html(page)
    # กราฟ: DNA step ของวันล่าสุดเท่านั้น ค่าจาก ledger ทั้ง chain
    assert list(charts[-1].index) == [3, 4, 5]
    assert charts[-1][E].tolist() == pytest.approx(full[E].iloc[3:6].tolist())
    # Order audit: เฉพาะ order ของแถววันล่าสุด เรียงล่าสุดก่อน
    assert list(audit_table(page)["run_id"]) == [f"{CHAIN}-5", f"{CHAIN}-3"]

    before = metrics(page)
    day_box(page).set_value(ALL_DAYS).run(timeout=30)
    assert not page.exception
    assert metrics(page) == before                  # metric ใช้ทั้ง chain ไม่ขึ้นกับตัวกรอง
    assert list(charts[-1].index) == list(range(6))
    assert list(audit_table(page)["run_id"]) == [f"{CHAIN}-5", f"{CHAIN}-3", f"{CHAIN}-1"]
    tips_all = table_tooltips(persisted, p0=P0, genesis_baseline=True)[0]
    show_all = order_columns(full.copy())
    for col in MONEY_COLS:
        show_all[col] = show_all[col].astype(float).round(2)
    assert hover_html(page) == build_hover_table_html(show_all, tips_all, columns, MONEY_COLS)


def test_the_plain_table_follows_the_day_and_keeps_it_across_other_widgets(monkeypatch):
    page, rows = two_day_page(monkeypatch)
    page.run(timeout=30)
    page.toggle(key="live_hover_table").set_value(False).run(timeout=30)
    _, full = whole_chain(rows)
    table = ledger_table(page)
    assert list(table["DNA step"]) == [3, 4, 5]
    for col in LEDGER_COLS:
        assert table[col].tolist() == pytest.approx(full[col].iloc[3:6].round(2).tolist())

    day_box(page).set_value(DAY1).run(timeout=30)
    assert list(ledger_table(page)["DNA step"]) == [0, 1, 2]
    assert list(audit_table(page)["run_id"]) == [f"{CHAIN}-1"]
    page.toggle(key="live_hover_table").set_value(True).run(timeout=30)
    assert not page.exception and day_box(page).value == DAY1    # widget อื่นไม่รีเซ็ตวันที่เลือก


def test_a_day_view_shows_the_whole_chain_correction_not_a_fresh_window(monkeypatch):
    """engine เขียน ΔAₙ ผิดบนแถว PASS ของวันแรกแล้วสะสมต่อ — แถวของวันถัดไปต้องเป็นค่าที่ recompute
    ทั้ง chain แก้แล้ว (ถ้ากรองก่อนคำนวณ แถวแรกของวันจะคงค่าผิดที่บันทึกไว้เป็นฐาน)"""
    data = _mk_chain([10.0, 12.0, 11.0, 9.0], [150.0, 150.0, 136.36364, 136.36364], [1, 1, 1, 1])
    ordered = sorted(data.values(), key=lambda row: row["version"])
    bad_dA = GATED_FIX_C * (11.0 / 12.0 - 1.0)
    for n, row in enumerate(ordered):
        row["เวลา (UTC)"] = f"{DAY1 if n < 3 else DAY2}T14:{n:02d}:00Z"
        if n == 2:                                  # PASS_THRESHOLD ถูกเขียนแบบ act
            row["ΔAₙ ต่อสเต็ป (USD)"] = bad_dA
        if n >= 2:                                  # แล้วสะสมค่าผิดต่อ
            row["Aₙ สะสม (USD)"] += bad_dA
            row["Eₙ ส่วนเกินสะสม (USD)"] += bad_dA
    key = ordered[0]["chain_key"]
    databases(monkeypatch, prod={
        "webull_lego_rows": data,
        "webull_lego_state": {key: {"p0": 10.0, "updated_at": f"{DAY2}T14:03:00Z"}},
        "webull_lego_order_audit": {}})
    page = app()
    page.run(timeout=30)
    page.toggle(key="live_hover_table").set_value(False).run(timeout=30)
    assert not page.exception and day_box(page).value == DAY2

    persisted = rows_to_df(data)
    whole = recompute_gated_ledger(persisted, p0=10.0, genesis_baseline=True)
    window = recompute_gated_ledger(persisted.iloc[3:], p0=10.0, genesis_baseline=True)
    shown = ledger_table(page)
    assert list(shown["DNA step"]) == [3]
    for col in ("Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"):
        assert shown[col].iloc[0] == pytest.approx(round(whole[col].iloc[3], 2))
        assert shown[col].iloc[0] != pytest.approx(round(window[col].iloc[0], 2))


def test_a_day_without_orders_says_so(monkeypatch):
    page, _ = two_day_page(monkeypatch, audited=(1,))
    page.run(timeout=30)
    assert not page.exception and audit_table(page) is None
    assert any("ไม่มี order ในวันที่เลือก" in c.value for c in page.caption)
    day_box(page).set_value(ALL_DAYS).run(timeout=30)
    assert list(audit_table(page)["run_id"]) == [f"{CHAIN}-1"]


def fee_page(monkeypatch, audited) -> AppTest:
    """fill แรก (แถว 3) อยู่วันก่อนและ fee ยัง PENDING; fill ที่สอง (แถว 5) อยู่วันล่าสุด fee FINAL"""
    rows = chain(CHAIN, "AAPL", days=(DAY1,) * 4 + (DAY2,) * 2)
    fees = {3: "PENDING", 5: "FINAL"}
    audit = {f"{CHAIN}-{i}": {"run_id": f"{CHAIN}-{i}", "status": "FILLED",
                              "broker_fee_status": fees[i]} for i in audited}
    databases(monkeypatch, prod={
        "webull_lego_rows": rtdb_rows(rows),
        "webull_lego_state": {CHAIN: {"p0": P0, "updated_at": f"{DAY2}T14:05:00Z"}},
        "webull_lego_order_audit": audit})
    return app()


def pending_fee_warnings(page: AppTest) -> list[str]:
    return [w.value for w in page.warning if "broker fee ยังเป็น PENDING" in w.value]


def test_a_pending_broker_fee_from_an_earlier_day_still_warns(monkeypatch):
    """คำเตือน money fence นับจากทั้ง chain — ต้องไม่หายเพราะตัวกรองวัน"""
    page = fee_page(monkeypatch, audited=(3, 5))
    page.run(timeout=30)
    assert not page.exception and day_box(page).value == DAY2
    assert list(audit_table(page)["run_id"]) == [f"{CHAIN}-5"]      # ตาราง: order ของวันล่าสุด
    warnings = pending_fee_warnings(page)
    assert len(warnings) == 1 and "มี 1 fill" in warnings[0] and "นับทั้ง chain" in warnings[0]

    day_box(page).set_value(ALL_DAYS).run(timeout=30)
    warnings = pending_fee_warnings(page)
    assert len(warnings) == 1 and "นับทั้ง chain" not in warnings[0]   # ไม่กรอง = ข้อความเดิม


def test_a_day_without_orders_still_shows_the_pending_fee_warning(monkeypatch):
    page = fee_page(monkeypatch, audited=(3,))
    page.run(timeout=30)
    assert not page.exception and audit_table(page) is None
    assert any("ไม่มี order ในวันที่เลือก" in c.value for c in page.caption)
    assert len(pending_fee_warnings(page)) == 1


def test_switching_chain_returns_the_filter_to_the_latest_day(monkeypatch):
    databases(monkeypatch, prod={
        "webull_lego_rows": rtdb_rows(two_days(), two_days("QBTS_x", "QBTS")),
        "webull_lego_state": {CHAIN: {"p0": P0, "updated_at": f"{DAY2}T14:05:00Z"},
                              "QBTS_x": {"p0": P0, "updated_at": f"{DAY2}T14:04:00Z"}},
        "webull_lego_order_audit": {}})
    page = app()
    page.run(timeout=30)
    assert chain_box(page).value == CHAIN and day_box(page).value == DAY2
    day_box(page).set_value(ALL_DAYS).run(timeout=30)
    assert day_box(page).value == ALL_DAYS
    chain_box(page).set_value("QBTS_x").run(timeout=30)
    assert not page.exception
    assert day_box(page).value == DAY2          # chain ใหม่ -> วันล่าสุด แม้มีวันชุดเดียวกัน


def test_the_total_metric_is_the_sum_of_what_each_chain_shows(monkeypatch):
    databases(monkeypatch, prod=prod_db())
    page = app()
    page.run(timeout=30)
    assert not page.exception
    expected = f"{STORED_E[5] + STORED_E[4]:.2f}"
    total = next(m for m in page.metric if m.label == TOTAL)
    assert total.value == expected
    assert "`AAPL_new`" in total.help and "`QBTS_x`" in total.help
    assert "chain เก่า `AAPL_old`" in total.help and f"chain PAPER `{PAPER_CHAIN}`" in total.help
    # Eₙ ของแต่ละตัว = ค่าที่ metric "Eₙ สะสม (USD)" แสดงเมื่อเลือก chain นั้น; ผลรวมไม่ขึ้นกับ chain ที่เลือก
    for key, stored in (("AAPL_new", STORED_E[5]), ("QBTS_x", STORED_E[4]),
                        ("AAPL_old", STORED_E[3])):
        chain_box(page).set_value(key).run(timeout=30)
        assert not page.exception
        assert metrics(page)["Eₙ สะสม (USD)"] == f"{stored:.2f}"
        assert metrics(page)[TOTAL] == expected


def test_prod_skips_a_stray_paper_chain_and_the_paper_page_counts_it(monkeypatch):
    paper_e = f"{float(rows_to_df(PAPER['webull_lego_rows'])[E].iloc[-1]):.2f}"
    databases(monkeypatch, prod=PAPER)          # chain PAPER หลงอยู่ใน DB ของ PROD
    page = app()
    page.run(timeout=30)
    assert not page.exception
    total = next(m for m in page.metric if m.label == TOTAL)
    assert total.value == "—" and f"chain PAPER `{PAPER_CHAIN}`" in total.help
    assert metrics(page)["Eₙ สะสม (USD)"] == paper_e      # metric ของ chain ยังแสดงเหมือนเดิม

    databases(monkeypatch, prod={}, paper=PAPER)
    page = app(FIREBASE_DB_URL_PAPER=PAPER_URL)
    page.run(timeout=30)
    page.sidebar.radio[0].set_value("PAPER (จำลอง)").run(timeout=30)
    assert not page.exception
    assert metrics(page)[TOTAL] == paper_e == metrics(page)["Eₙ สะสม (USD)"]
    assert day_box(page).key.startswith("live_day:PAPER:")


def test_a_total_that_cannot_be_computed_does_not_break_the_page(monkeypatch):
    page, _ = two_day_page(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("sum exploded")

    monkeypatch.setattr(lego_dash_core, "latest_excess_total", boom)
    page.run(timeout=30)
    assert not page.exception
    total = next(m for m in page.metric if m.label == TOTAL)
    assert total.value == "—" and "sum exploded" in total.help
    assert "Eₙ สะสม (USD)" in metrics(page)


def test_hot_deploy_reloads_a_core_without_the_day_filter_or_the_total(monkeypatch):
    """core ที่แคชไว้ก่อนมีตัวกรองวัน/ผลรวม ต้องถูกโหลดใหม่ ไม่ทำให้หน้าล่ม"""
    for name in ("latest_excess_total", "day_options"):
        monkeypatch.delattr(lego_dash_core, name)
    page = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    page.run(timeout=20)
    assert not page.exception
    assert callable(lego_dash_core.latest_excess_total) and callable(lego_dash_core.day_options)
