"""แท็บ 🧪 ประวัติเป้า (ทดสอบ) — สูตร (lego_goal_history) และหน้าเว็บ (AppTest)
ตัวเลขทั้งหมดเป็นตัวอย่างสมมติ (symbol AAA/BBB) ไม่ใช่ข้อมูลบัญชี"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lego_goal as goal
import lego_goal_history as gh
import lego_positions as positions
from test_goal import aaa_book, bbb_book, inputs_of
from test_positions import clear_streamlit_cache, goal_app  # noqa: F401

CHAIN = "AAA_fake01"                     # chain ปัจจุบันของ AAA ใน fixture (P₀ 10 · FIX_C 1500 · K 7 · LEAPS 100)


def row(time, price, shares, excess, leaps=100.0, chain=CHAIN, **extra) -> dict:
    return {gh.TIME: time, gh.PRICE: price, gh.HOLDINGS: shares, gh.LEAPS: leaps, gh.EXCESS: excess,
            gh.CHAIN: chain, "committed": True, **extra}


ROWS_NODE = {
    "r1": row("2026-11-03T14:30:00Z", 10.0, 50.0, 0.0),
    "r2": row("2026-11-05T15:00:00Z", 9.5, 55.0, 6.0),
    "r3": row("2026-11-10T16:00:00Z", 9.2, 58.0, 12.0),
    "r4": row("2026-11-16T20:45:00Z", 9.0, 60.0, 20.0),        # = สถานะของแท็บ Goal ใน fixture (ราคา 9 · หุ้น 60)
    "pending": row("2026-11-17T14:30:00Z", 8.0, 60.0, 25.0, committed=False),
    "broken": row("2026-11-06T15:00:00Z", None, 55.0, 6.0),
    "old_chain": row("2026-10-01T15:00:00Z", 12.0, 1.0, 1.0, chain="AAA_older"),
}


def aaa_history() -> pd.DataFrame:
    rows, _ = gh.chain_rows(ROWS_NODE, CHAIN)
    return gh.history(inputs_of(aaa_book()), rows)


# ---------------------------------------------------------------- สูตร

def test_chain_rows_keep_committed_rows_of_the_current_chain_in_time_order():
    rows, notes = gh.chain_rows(ROWS_NODE, CHAIN)
    assert rows["price"].tolist() == [10.0, 9.5, 9.2, 9.0]
    assert rows["time"].is_monotonic_increasing and rows["time"].dt.tz is not None
    assert notes == [f"{CHAIN}: ข้าม 1 แถวที่ค่าไม่ครบ"]
    assert gh.chain_rows(None, CHAIN)[0].empty and gh.chain_rows({"x": "bad"}, CHAIN)[0].empty


def test_each_point_is_what_the_goal_tab_shows_at_that_time():
    df = aaa_history()
    now = goal.goal_at(inputs_of(aaa_book()))               # แถวล่าสุด = ราคา 9 · หุ้น 60 · LEAPS 100 · Eₙ 20
    for key in goal.VALUE_KEYS:
        assert df[key].iloc[-1] == pytest.approx(now[key])
    first = df.iloc[0]                                       # ราคา = P₀ → ทุกเส้นเป็น 0 และ Eₙ = 0
    assert first["goal"] == first["payoff"] == first["gap_e"] == 0.0
    second = df.iloc[1]
    assert second["goal"] == pytest.approx(1500 * math.log(0.95))
    assert second["stock_leg"] == pytest.approx(55 * 9.5 * math.log(0.95))
    assert second["leaps_leg"] == pytest.approx(100 * ((9.5 - 7) - (10 - 7)))
    assert second["gap_e"] == pytest.approx(second["payoff"] + 6.0 - second["goal"])


def test_leaps_follow_the_row_and_an_unknown_strike_fails_closed():
    inputs = inputs_of(aaa_book())
    half = gh.history(inputs, gh.chain_rows({"a": row("2026-11-05T15:00:00Z", 9.5, 55.0, 6.0, leaps=50.0)},
                                            CHAIN)[0])
    assert half["leaps_leg"].iloc[0] == pytest.approx(50 * ((9.5 - 7) - (10 - 7)))
    no_leaps_now = inputs_of(bbb_book()).__class__(**{**inputs.__dict__, "calls": ()})
    rows, _ = gh.chain_rows({"a": row("2026-11-05T15:00:00Z", 9.5, 55.0, 6.0, leaps=100.0),
                             "b": row("2026-11-06T15:00:00Z", 9.5, 55.0, 6.0, leaps=0.0)}, CHAIN)
    out = gh.history(no_leaps_now, rows)
    assert np.isnan(out["payoff_e"].iloc[0]) and out["leaps_leg"].iloc[1] == 0.0


def test_trend_compares_with_the_last_row_at_least_seven_days_older():
    df = aaa_history()
    t = gh.trend(df, "excess")
    days = (pd.Timestamp("2026-11-16T20:45Z") - pd.Timestamp("2026-11-05T15:00Z")) / pd.Timedelta(days=1)
    assert (t["ref"], t["last"], t["change"], t["partial"]) == (6.0, 20.0, 14.0, False)
    assert t["days"] == pytest.approx(days) and t["rate"] == pytest.approx(14.0 / days)
    whole = gh.trend(df, "excess", days=36500)                # ทั้ง chain → เทียบแถวแรก
    assert whole["partial"] and whole["ref"] == 0.0 and whole["change"] == 20.0
    assert gh.trend(df.iloc[:1], "excess") is None


def test_gap_change_says_whether_the_gap_to_the_goal_narrowed_or_widened():
    df = aaa_history()
    change = gh.gap_change(df)
    ref = df["gap_e"].iloc[1]
    assert change["width_change"] == pytest.approx(abs(df["gap_e"].iloc[-1]) - abs(ref))
    assert change["word"] == ("แคบลง" if change["width_change"] < 0 else "กว้างขึ้น")
    assert change["side"] == ("เหนือเป้า" if df["gap_e"].iloc[-1] > 0 else "ใต้เป้า")


def test_the_portfolio_carries_each_symbol_forward_and_starts_when_all_have_rows():
    a = aaa_history()
    b = a.copy()
    b["time"] = b["time"] + pd.Timedelta(minutes=5)          # อีกตัวบันทึกช้ากว่า 5 นาทีทุกรอบ
    b[["excess", *goal.VALUE_KEYS]] = b[["excess", *goal.VALUE_KEYS]] * 2
    total = gh.portfolio_history([a, b])
    assert total["time"].iloc[0] == b["time"].iloc[0]         # เริ่มเมื่อ b มีแถวแรก
    assert len(total) == 2 * len(a) - 1
    at_b2 = total.set_index("time").loc[b["time"].iloc[1]]
    assert at_b2["excess"] == pytest.approx(a["excess"].iloc[1] + 2 * a["excess"].iloc[1])
    after_a2 = total.set_index("time").loc[a["time"].iloc[1]]   # a ใหม่ · b ยังใช้ค่ารอบก่อน
    assert after_a2["gap_e"] == pytest.approx(a["gap_e"].iloc[1] + 2 * a["gap_e"].iloc[0])
    assert gh.portfolio_history([]).empty


def test_the_daily_table_keeps_the_last_row_of_each_day():
    table = gh.daily_table(aaa_history())
    assert table["วัน (UTC)"].tolist() == ["2026-11-03", "2026-11-05", "2026-11-10", "2026-11-16"]
    assert table["Eₙ สะสม"].tolist() == [0.0, 6.0, 12.0, 20.0]
    assert table["Eₙ เพิ่มในวัน"].tolist()[1:] == [6.0, 6.0, 8.0] and np.isnan(table["Eₙ เพิ่มในวัน"].iloc[0])


def test_the_charts_open_with_gap_plus_e_and_skip_closed_market_hours():
    df = aaa_history()
    fig = gh.gap_history_figure(df, theme="dark", title="AAA")
    assert [(t.name, t.visible) for t in fig.data] == [(goal.NAMES["gap"], "legendonly"), (goal.NAMES["gap_e"], True)]
    assert {"▲ เหนือเป้า", "▼ ใต้เป้า"} <= {a.text for a in fig.layout.annotations}
    breaks = [dict(b.to_plotly_json()) for b in fig.layout.xaxis.rangebreaks]
    assert {"bounds": ["sat", "mon"]} in breaks and {"bounds": [21, 13], "pattern": "hour"} in breaks
    excess = gh.excess_history_figure(df)
    assert len(excess.data) == 1 and excess.layout.showlegend is False
    for f in (fig, excess):                                  # d3 format ที่ขึ้นต้นด้วย "+" ทำให้ hover ทศนิยมเต็ม
        assert not f.layout.yaxis.hoverformat.startswith("+")


def test_the_module_stays_isolated():
    tree = ast.parse(Path(gh.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"streamlit", "lego_positions", "lego_webull_read"}


# ---------------------------------------------------------------- หน้าเว็บ (AppTest)

def history_tab(app):
    return next(t for t in app.tabs if t.label == "🧪 ประวัติเป้า")


def test_the_history_tab_shows_the_trend_cards_and_two_charts(monkeypatch):
    app = goal_app(monkeypatch, rows_node=ROWS_NODE)
    app.run(timeout=30)
    assert not app.exception
    tab = history_tab(app)
    assert not tab.error, [e.value for e in tab.error]
    metrics = {m.label: m.value for m in tab.metric}
    df = aaa_history()
    assert metrics["ห่างจากเป้า + Eₙ (ล่าสุด)"] == f"{df['gap_e'].iloc[-1]:+,.2f}"
    assert metrics["Eₙ สะสม (ล่าสุด)"] == "+20.00"
    rate = gh.trend(df, "excess")["rate"]
    assert metrics["Eₙ ต่อวัน (7 วัน)"] == f"{rate:+,.2f}"
    assert metrics["ช่องว่างกับเป้า (7 วัน)"].split()[0] in {"แคบลง", "กว้างขึ้น", "เท่าเดิม"}
    assert len(tab.get("plotly_chart")) == 2
    assert app.button_group(key="hist_symbol").value == "AAA"


def test_without_rows_the_history_tab_explains_why(monkeypatch):
    app = goal_app(monkeypatch)
    app.run(timeout=30)
    assert not app.exception
    tab = history_tab(app)
    assert not tab.error and any(i.value.startswith("ยังไม่มีประวัติที่คำนวณได้") for i in tab.info)
    assert any("AAA: ไม่มีแถวของ chain ปัจจุบันใน webull_lego_rows" in c.value for c in tab.caption)
    import streamlit_app
    assert streamlit_app.goal_history.HISTORY_SCHEMA >= 1
