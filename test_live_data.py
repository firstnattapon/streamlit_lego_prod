"""lego_live_data — ชั้นข้อมูลของแท็บ Live (pure: ไม่มี streamlit / firebase_admin)

snapshot อ่าน Firebase ครั้งเดียวแล้ว memo ผลคำนวณบนตัวเอง · สถานะ สด/ค้าง/ผิดพลาด จากเวลาในข้อมูล ·
ตัวเลขบนการ์ดไม่ล่มเมื่อเป็น null — ค่าคาดหวังมาจาก fixture ชุดเดียวกับเทสต์หน้าเว็บ (test_day_filter_total)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import lego_dash_core
import lego_live_data as live
from test_day_filter_total import STORED_E, chain, prod_db, rtdb_rows
from test_paper_view import PAPER

NOW = pd.Timestamp("2026-08-03T15:00:00Z")          # จันทร์ 11:00 ET — ตลาด US เปิด


def reader(payloads: dict, calls: list | None = None, fail: tuple = ()):
    def read(path):
        if calls is not None:
            calls.append(path)
        if path in fail:
            raise ConnectionError(f"{path} offline")
        return payloads.get(path)
    return read


# ---- fetch ------------------------------------------------------------------

def test_one_fetch_reads_each_node_once_and_keeps_the_committed_rows():
    calls: list = []
    snap = live.fetch_snapshot(reader(prod_db(), calls), now=NOW)
    assert sorted(calls) == sorted(live.READ_ORDER) and len(calls) == 4
    assert len(snap.rows) == len(lego_dash_core.rows_to_df(prod_db()["webull_lego_rows"]))
    assert snap.fetched_at == NOW and snap.notes == [] and snap.audit == {}
    assert snap.health.empty and not snap.alarm


@pytest.mark.parametrize("path", [live.ROWS_PATH, live.STATE_PATH])
def test_a_required_node_that_cannot_be_read_is_a_snapshot_error(path):
    with pytest.raises(live.SnapshotError, match=path):
        live.fetch_snapshot(reader(prod_db(), fail=(path,)))


def test_optional_nodes_that_cannot_be_read_become_notes():
    snap = live.fetch_snapshot(reader(prod_db(), fail=(live.AUDIT_PATH, live.WARNINGS_PATH)))
    assert len(snap.notes) == 2 and all("offline" in note for note in snap.notes)
    assert snap.audit == {} and snap.health.empty and not snap.rows.empty


def test_state_and_audit_that_are_not_dicts_read_as_empty():
    db = {**prod_db(), "webull_lego_state": ["x"], "webull_lego_order_audit": "junk"}
    snap = live.fetch_snapshot(reader(db))
    assert snap.state == {} and snap.audit == {}
    assert snap.entry("AAPL_new") == {} and snap.chain_view("AAPL_new").audit_rows().empty


def test_an_active_warning_raises_the_alarm():
    db = {**prod_db(), "webull_lego_warnings": {"auth": {"kind": "AUTH_BACKOFF", "active": True}}}
    snap = live.fetch_snapshot(reader(db), now=NOW)
    assert snap.alarm
    fresh = snap.freshness("AAPL_new", NOW)
    assert fresh.status == live.ERROR and fresh.label == "ผิดพลาด" and "Authentication" in fresh.reason


# ---- chains + memo ----------------------------------------------------------

def prod_only() -> dict:
    """prod_db() ไม่มี chain PAPER ที่หลงมา: AAPL_old · AAPL_new (updated_at ล่าสุด) · QBTS_x"""
    db = prod_db()
    paper = set(PAPER["webull_lego_state"])
    db["webull_lego_rows"] = {k: v for k, v in db["webull_lego_rows"].items()
                              if v.get("chain_key") not in paper}
    db["webull_lego_state"] = {k: v for k, v in db["webull_lego_state"].items() if k not in paper}
    return db


def test_chains_are_sorted_and_the_default_is_the_latest_updated():
    snap = live.fetch_snapshot(reader(prod_only()))
    assert snap.chains() == ["AAPL_new", "AAPL_old", "QBTS_x"]
    assert snap.default_chain() == "AAPL_new"          # updated_at ล่าสุด ไม่ใช่ตัวท้ายตามอักษร
    assert snap.symbol("QBTS_x") == "QBTS"
    assert len(snap.chain_rows("QBTS_x")) == 5


def test_a_single_chain_uses_the_whole_frame_and_the_symbol_falls_back():
    rows = chain("ONLY", "", n=3)
    rows.loc[2, "chain_key"] = None                     # แถวที่ไม่มี chain_key ยังอยู่ (พฤติกรรมเดิม)
    snap = live.fetch_snapshot(reader({"webull_lego_rows": rtdb_rows(rows),
                                       "webull_lego_state": {"ONLY": {"symbol": "UBER"}}}))
    assert snap.chains() == ["ONLY"] and len(snap.chain_rows("ONLY")) == 3
    assert snap.symbol("ONLY") == "UBER"
    assert live.fetch_snapshot(reader({"webull_lego_rows": rtdb_rows(rows)})).symbol("ONLY") == "ONLY"


def test_every_result_is_computed_once_per_snapshot(monkeypatch):
    seen = {"integrity": 0, "tips": 0}
    real_integrity, real_tips = lego_dash_core.integrity_report, lego_dash_core.table_tooltips

    def integrity(*args, **kwargs):
        seen["integrity"] += 1
        return real_integrity(*args, **kwargs)

    def tips(*args, **kwargs):
        seen["tips"] += 1
        return real_tips(*args, **kwargs)

    monkeypatch.setattr(lego_dash_core, "integrity_report", integrity)
    monkeypatch.setattr(lego_dash_core, "table_tooltips", tips)
    snap = live.fetch_snapshot(reader(prod_db()))
    view = snap.chain_view("AAPL_new")
    assert snap.chain_view("AAPL_new") is view
    assert seen == {"integrity": 0, "tips": 0}          # งานหนักไม่ถูกคำนวณตอนสร้าง view
    for _ in range(3):
        view.integrity(), view.tooltips()
    assert seen == {"integrity": 1, "tips": 1}
    assert view.integrity()[1] is True
    # ตัวเลขชุดเดียวกับที่หน้าเว็บแสดงเดิม
    assert float(view.ledger["Eₙ ส่วนเกินสะสม (USD)"].iloc[-1]) == pytest.approx(STORED_E[5])
    calls = []
    assert snap.memo("k", lambda: calls.append(1) or 7) == 7 == snap.memo("k", lambda: 8)
    assert calls == [1]


def test_a_broken_chain_does_not_break_the_snapshot(monkeypatch):
    real = lego_dash_core.recompute_gated_ledger

    def recompute(df, *args, **kwargs):
        if (df.get("chain_key") == "QBTS_x").any():
            raise ValueError("bad cell")
        return real(df, *args, **kwargs)

    monkeypatch.setattr(lego_dash_core, "recompute_gated_ledger", recompute)
    snap = live.fetch_snapshot(reader(prod_db()))
    with pytest.raises(ValueError):
        snap.chain_view("QBTS_x")
    assert not snap.chain_view("AAPL_new").ledger.empty
    assert "QBTS_x" in snap.chains()


def test_an_unreadable_p0_falls_back_to_the_genesis_price_with_a_note():
    db = prod_db()
    db["webull_lego_state"]["QBTS_x"]["p0"] = "not a price"
    view = live.fetch_snapshot(reader(db)).chain_view("QBTS_x")
    assert view.p0_hint is None and len(view.notes) == 1 and "p0" in view.notes[0]


def test_the_paper_stamp_comes_from_the_state_pointer():
    snap = live.fetch_snapshot(reader(PAPER))
    key = snap.chains()[0]
    assert snap.chain_view(key).paper_marked
    assert not live.fetch_snapshot(reader(prod_db())).chain_view("AAPL_new").paper_marked


# ---- สด / ค้าง / ผิดพลาด -------------------------------------------------------

def frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"เวลา (UTC)": times})


def test_the_slot_length_is_read_from_the_rows():
    paper = lego_dash_core.rows_to_df(PAPER["webull_lego_rows"])
    assert live.slot_seconds(paper) == 900                       # fixture: 15 นาที/slot
    assert live.slot_seconds(frame(["2026-08-03T14:00:00Z", "2026-08-03T14:05:00Z",
                                    "2026-08-04T14:00:00Z", "2026-08-04T14:05:00Z"])) == 300
    assert live.slot_seconds(frame([])) == live.SLOT_FALLBACK_S
    assert live.slot_seconds(pd.DataFrame({"x": [1]})) == live.SLOT_FALLBACK_S


def test_us_market_hours_follow_new_york_time():
    assert live.market_open(pd.Timestamp("2026-08-03T13:30:00Z"))      # จันทร์ 09:30 EDT
    assert not live.market_open(pd.Timestamp("2026-08-03T13:29:00Z"))
    assert not live.market_open(pd.Timestamp("2026-08-03T20:00:00Z"))  # 16:00 EDT ปิดแล้ว
    assert live.market_open(pd.Timestamp("2026-11-09T14:30:00Z"))      # จันทร์ 09:30 EST
    assert not live.market_open(pd.Timestamp("2026-11-08T16:00:00Z"))  # อาทิตย์


def test_fresh_within_two_slots_then_stale_and_why():
    rows = frame([f"2026-08-03T14:{m:02d}:00Z" for m in (0, 15, 30)])     # slot 15 นาที
    live_now = live.freshness(rows, {}, pd.Timestamp("2026-08-03T14:55:00Z"))
    assert live_now.status == live.LIVE and live_now.label == "สด" and live_now.age_s == 25 * 60
    stale = live.freshness(rows, {}, pd.Timestamp("2026-08-03T15:05:00Z"))
    assert stale.status == live.STALE and stale.reason == "ไม่มีแถวใหม่"
    closed = live.freshness(rows, {}, pd.Timestamp("2026-08-08T15:00:00Z"))     # เสาร์
    assert closed.status == live.STALE and closed.reason == "ตลาดปิด" and closed.label == "ค้าง"
    # state.updated_at ที่ใหม่กว่าแถวล่าสุดนับเป็นการอัปเดต
    bumped = live.freshness(rows, {"updated_at": "2026-08-03T15:00:00Z"},
                            pd.Timestamp("2026-08-03T15:05:00Z"))
    assert bumped.status == live.LIVE and bumped.last_update == pd.Timestamp("2026-08-03T15:00:00Z")
    unknown = live.freshness(frame(["x"]), {"updated_at": 5}, NOW)
    assert unknown.status == live.STALE and unknown.last_update is None
    assert live.freshness(rows, {}, NOW, alarm=True).status == live.ERROR


def test_the_status_line_reads_like_a_clock():
    rows = frame(["2026-08-03T14:05:00Z"])
    fresh = live.freshness(rows, {}, pd.Timestamp("2026-08-03T14:08:30Z"))
    assert live.describe(fresh, pd.Timestamp("2026-08-03T14:08:30Z")) == "อัปเดต 14:05 UTC · 3 นาทีที่แล้ว"
    later = pd.Timestamp("2026-08-08T15:00:00Z")
    text = live.describe(live.freshness(rows, {}, later), later)
    assert text == "อัปเดต 2026-08-03 14:05 UTC · 5 วันที่แล้ว · ตลาดปิด"
    assert live.fmt_age(5) == "เมื่อครู่" and live.fmt_age(7200) == "2 ชม.ที่แล้ว"
    assert live.describe(live.Freshness(live.ERROR, None, None, "โหลดไม่ได้"), later) == "โหลดไม่ได้"


def test_the_live_snapshot_is_stale_on_the_fixture_dates():
    snap = live.fetch_snapshot(reader(prod_db()))
    fresh = snap.freshness("AAPL_new", pd.Timestamp("2026-11-08T04:00:00Z"))
    assert fresh.status == live.STALE and fresh.last_update == pd.Timestamp("2026-08-01T14:05:00Z")


# ---- ตัวเลขบนการ์ด / ตาราง State -------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, "—"), (float("nan"), "—"), (float("inf"), "—"), ("x", "—"), ({}, "—"),
    (12.345, "12.35"), ("12.5", "12.50"), (-1e-12, "0.00"), (0, "0.00")])
def test_fmt_num_never_raises(value, expected):
    assert live.fmt_num(value) == expected


def test_fmt_num_digits_and_fmt_text():
    assert live.fmt_num(5.0, 0) == "5" and live.fmt_num(None, 0) == "—"
    assert live.fmt_text(None) == live.fmt_text(float("nan")) == live.fmt_text("  ") == "—"
    assert live.fmt_text("READY_BUY") == "READY_BUY"


def test_state_table_shows_one_chain_as_text():
    table = live.state_table({"p0": 65.49, "paper": True, "nested": {"a": 1}, "gone": None})
    assert list(table.index) == ["gone", "nested", "p0", "paper"]
    assert table["ค่า"].tolist() == ["—", '{"a": 1}', "65.49", "True"]
    assert live.state_table("x").empty


def test_the_module_stays_free_of_streamlit_and_firebase():
    tree = ast.parse(Path(live.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert not imported & {"streamlit", "firebase_admin"}

