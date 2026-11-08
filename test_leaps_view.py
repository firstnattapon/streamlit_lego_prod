"""LEAPS แทนหุ้น (LEGO_LEAPS): ตาราง 17 คอลัมน์แสดงค่าเดียวกับที่ bot เขียน + hover แยกหุ้นจริง/LEAPS

fixtures/paper_session_leaps.json is one paper session produced by the lego backend's own decision
and worker (lego test_leaps_stock_replacement.switch over test_paper_mode.Session, broker doubled,
fills finalized at their dispatch time): LEAPS of 20 share-equivalents for 10 slots, 60 for 3 (the
LEAPS alone are above FIX_C, so every SELL would need more than the real shares and the slot is
PASS_SELL_EXCEEDS_REAL), then 20 again. Column 7 is real + LEAPS; the split rides on flat fields.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from lego_dash_core import (COLUMN_ORDER, HOLDINGS_COL, LEDGER_COLS, META_COLS, MONEY_COLS,
                            PASS_SELL_EXCEEDS_REAL, count_ledger_corrections,
                            holdings_explanations, integrity_report, leaps_summary,
                            ledger_explanations, order_columns, recompute_gated_ledger,
                            rows_to_df, table_tooltips)
from lego_hover_table import build_hover_table_html
from test_paper_view import PAPER, PAPER_URL, app, databases, open_integrity

ROOT = Path(__file__).parent
LEAPS = json.loads((ROOT / "fixtures" / "paper_session_leaps.json").read_text(encoding="utf-8"))
CHAIN = "UBER_paper_leaps"
P0 = LEAPS["webull_lego_state"][CHAIN]["p0"]
LEAPS_FIELDS = ("real_holdings", "leaps_shares", "leaps_shares_changed_from",
                "post_execution_real_holdings")


def frame() -> pd.DataFrame:
    return rows_to_df(LEAPS["webull_lego_rows"])


def raw_rows() -> list[dict]:
    return sorted(LEAPS["webull_lego_rows"].values(), key=lambda r: r["version"])


def test_the_17_columns_show_exactly_what_the_bot_wrote():
    shown = order_columns(frame())
    assert list(shown.columns)[:17] == COLUMN_ORDER
    for i, row in enumerate(raw_rows()):
        for col in COLUMN_ORDER:
            assert shown.iloc[i][col] == row[col], (row["version"], col)
        # Column 7 is the position the decision valued: real + LEAPS, summed exactly.
        assert row[HOLDINGS_COL] == pytest.approx(row["real_holdings"] + row["leaps_shares"],
                                                  abs=1e-12)
    # The split is metadata after the 17 columns, never a column of its own among them.
    columns = list(shown.columns)
    assert all(columns.index(name) >= 17 for name in LEAPS_FIELDS)
    assert set(LEAPS_FIELDS) <= set(META_COLS)


def test_the_ledger_checks_pass_with_leaps_rows():
    df = frame()
    report, ok = integrity_report(df, p0_hint=P0)
    assert ok, report.to_string()
    assert count_ledger_corrections(df, recompute_gated_ledger(df, p0=P0)) == 0
    passes = df[df["สถานะ"] == PASS_SELL_EXCEEDS_REAL]
    assert len(passes) == 3
    assert (passes["จำนวนสั่ง (หุ้น)"] == 0).all() and (passes["leaps_shares"] == 60).all()
    assert passes["เหตุผล"].str.startswith(PASS_SELL_EXCEEDS_REAL + ":").all()
    # The real shares never move while the LEAPS alone are above FIX_C.
    assert passes["real_holdings"].nunique() == 1


def test_the_holdings_tooltip_splits_real_and_leaps_like_the_table_shows():
    df = frame()
    tips = holdings_explanations(df)
    assert len(tips) == len(df) and all(set(tip) == {HOLDINGS_COL} for tip in tips)
    changes = 0
    for i, tip in enumerate(tips):
        row, text = df.iloc[i], tips[i][HOLDINGS_COL]
        assert f"หุ้นจริง {row['real_holdings']} + LEAPS {row['leaps_shares']} = " \
               f"{row[HOLDINGS_COL]}" in text                      # str(value): the cell text
        assert f"SELL ได้ไม่เกินหุ้นจริง {row['real_holdings']} หุ้น" in text
        if pd.notna(row.get("leaps_shares_changed_from")):
            changes += 1
            assert f"LEAPS เปลี่ยนจาก {row['leaps_shares_changed_from']} เป็น " \
                   f"{row['leaps_shares']}" in text and "ไม่ rebase" in text
        if row["สถานะ"] == PASS_SELL_EXCEEDS_REAL:
            assert "แถวนี้ PASS" in text and "หลัง fill" not in text
        if row["cashflow_status"] == "FINALIZED":
            assert f"หลัง fill: หุ้นจริง {row['post_execution_real_holdings']} + LEAPS " \
                   f"{row['leaps_shares']} = {row['post_execution_holdings']}" in text
    assert changes == 2
    summary = leaps_summary(df)
    last = df.iloc[-1]
    assert summary["changes"] == 2 and summary["leaps_shares"] == 20.0
    assert summary["text"] == (f"หุ้นจริง {last['real_holdings']} + LEAPS {last['leaps_shares']} = "
                               f"{last[HOLDINGS_COL]}")


def test_without_leaps_nothing_changes():
    plain = rows_to_df(PAPER["webull_lego_rows"])
    assert holdings_explanations(plain) == [{}] * len(plain)
    assert leaps_summary(plain) is None
    assert not set(LEAPS_FIELDS) & set(plain.columns)
    p0 = next(iter(PAPER["webull_lego_state"].values()))["p0"]
    tips, columns = table_tooltips(plain, p0=p0)
    assert columns == LEDGER_COLS and tips == ledger_explanations(plain, p0=p0)
    show = order_columns(plain.copy())
    assert build_hover_table_html(show, tips, columns, MONEY_COLS) == \
        build_hover_table_html(show, ledger_explanations(plain, p0=p0), LEDGER_COLS, MONEY_COLS)


def test_the_hover_table_puts_the_split_on_column_7_only_where_leaps_exist():
    df = frame()
    tips, columns = table_tooltips(df, p0=P0)
    assert columns == [*LEDGER_COLS, HOLDINGS_COL]
    ledger = ledger_explanations(df, p0=P0)
    assert all({k: v for k, v in tip.items() if k != HOLDINGS_COL} == base
               for tip, base in zip(tips, ledger))                 # Rₙ/ΔAₙ/Aₙ/Eₙ untouched
    out = build_hover_table_html(order_columns(df.copy()), tips, columns, MONEY_COLS)
    assert out.count("data-tip=") == len(df) * (len(LEDGER_COLS) + 1)
    assert "หุ้นจริง 24.48 + LEAPS 60.0 = 84.48" in out


def test_the_paper_view_explains_the_leaps(monkeypatch):
    databases(monkeypatch, prod={}, paper=LEAPS)
    page = app(FIREBASE_DB_URL_PAPER=PAPER_URL)
    page.run(timeout=30)
    page.sidebar.radio[0].set_value("PAPER (จำลอง)")
    open_integrity(page).run(timeout=30)
    assert not page.exception
    captions = [c.value for c in page.caption]
    assert any("LEAPS แทนหุ้น" in c and "หุ้นจริง 25.86 + LEAPS 20.0 = 45.86" in c
               and "2 ครั้ง" in c for c in captions), captions
    assert any("จำนวนถือครอง (หุ้นจริง + LEAPS)" in c for c in captions)
    assert any("ผ่านการตรวจสมการ" in m.value for m in page.success)
