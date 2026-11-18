"""แท็บ 🎯 Goal — สูตร (lego_goal) และหน้าเว็บ (AppTest) · ตัวเลขทั้งหมดเป็นตัวอย่างสมมติ ไม่ใช่ข้อมูลบัญชี"""
from __future__ import annotations

import ast
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lego_goal as goal
import lego_payoff as payoff
from test_payoff import NOW, ROWS, STATE, clear_streamlit_cache, payoff_app, snapshot  # noqa: F401


def leg(kind, qty, *, strike=None, delta=None, multiplier=1.0):
    return SimpleNamespace(kind=kind, qty=qty, strike=strike, delta=delta, multiplier=multiplier,
                           expiry=date(2027, 5, 21) if kind != "stock" else None)


def book(legs, *, spot=9.0, fix_c=1500.0, p0=9.0, excess=None, chain=True, usable=True, symbol="AAA"):
    chain_obj = SimpleNamespace(fix_c=fix_c, p0=p0, excess=excess) if chain else None
    return SimpleNamespace(symbol=symbol, spot=spot, legs=legs, chain=chain_obj,
                           usable_chain=chain_obj if usable else None)


# ตัวอย่าง 60:40 ในแชต: FIX_C 1500 · P₀ = ราคาตอนนี้ = 9 · หุ้น 600 USD (40%) + LEAPS 1 สัญญา K 6 (60%)
SIXTY_FORTY = book([leg("stock", 600 / 9), leg("call", 1, strike=6.0, delta=1.0, multiplier=100.0)])


def inputs_of(b) -> goal.GoalInputs:
    out, issue = goal.goal_inputs(b)
    assert issue is None
    return out


def aaa_book():
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, _ = payoff.build_books(snapshot(), chains, issues, NOW)
    return books[0]


# ---------------------------------------------------------------- สูตร

def test_the_sixty_forty_example_matches_the_chat_table():
    inputs = inputs_of(SIXTY_FORTY)
    assert inputs.stock_weight == pytest.approx(0.4) and inputs.leaps_weight == pytest.approx(0.6)
    assert inputs.coverage == pytest.approx(1.0)
    df = goal.goal_curves(inputs, [4, 6, 8, 9, 11, 13, 16])
    assert df["payoff"].round(1).tolist() == [-786.6, -543.3, -170.7, 0.0, 320.4, 620.6, 1045.2]
    assert df["goal"].round(1).tolist()[-1] == 863.0
    assert (df["payoff"] == df["stock_leg"] + df["leaps_leg"]).all()
    assert np.allclose(df["gap"], df["payoff"] - df["goal"])


def test_with_full_coverage_the_payoff_touches_the_goal_at_p0_and_stays_above():
    inputs = inputs_of(SIXTY_FORTY)
    df = goal.goal_curves(inputs, goal.goal_grid(inputs, 25, 300))
    assert df["gap"].min() >= -1e-9                          # x − 1 ≥ ln x (ไม่คิด premium)
    h = 1e-6                                                 # ความชันเท่ากันที่ P₀ (ฝั่งขวา)
    right = goal.goal_curves(inputs, [9.0, 9.0 + h])
    slope = right.diff().iloc[1] / h
    assert slope["payoff"] == pytest.approx(slope["goal"], rel=1e-4) == pytest.approx(1500 / 9, rel=1e-4)


def test_the_real_shaped_fixture_uses_webull_holdings_and_the_bot_chain():
    inputs = inputs_of(aaa_book())
    assert (inputs.fix_c, inputs.p0, inputs.spot, inputs.shares) == (1500.0, 10.0, 9.0, 60.0)
    assert inputs.stock_weight == pytest.approx(0.36) and inputs.leaps_weight == pytest.approx(0.48)
    assert inputs.coverage == pytest.approx(0.84)            # = exposure จริง ของแท็บ 📐 Payoff
    now = goal.goal_at(inputs)
    assert now["goal"] == pytest.approx(1500 * math.log(0.9))
    assert now["stock_leg"] == pytest.approx(540 * math.log(0.9))
    assert now["leaps_leg"] == pytest.approx(100 * ((9 - 7) - (10 - 7)))
    assert round(now["payoff"], 2) == -156.89 and round(now["gap"], 2) == 1.15
    assert inputs.needed_leaps_shares == pytest.approx(96.0) and inputs.option_text() == "1×C7"


def test_the_leaps_leg_is_flat_below_the_strike_and_zero_at_p0():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, [3.0, 5.0, 7.0, 10.0, 12.0])
    assert df["leaps_leg"].tolist() == [-300.0, -300.0, -300.0, 0.0, 200.0]


def test_missing_delta_hides_only_the_weights():
    inputs = inputs_of(book([leg("stock", 60), leg("call", 1, strike=7.0, delta=None, multiplier=100.0)]))
    assert inputs.leaps_weight is None and inputs.coverage is None
    assert np.isfinite(goal.goal_curves(inputs, [8.0, 9.0])["payoff"]).all()
    assert "leaps_weight = Σ delta · leaps_shares · spot / fix_c = (0) · 9 / 1,500 = —" in goal.equation_text(inputs)


@pytest.mark.parametrize("b, words", [
    (book([leg("stock", 60)], chain=False), "ไม่มี chain ของ bot"),
    (book([leg("stock", 60)], usable=False), "Webull กับ bot ไม่ตรงกัน"),
    (book([leg("stock", 60)], p0=None), "ไม่มี FIX_C หรือ P₀"),
    (book([leg("stock", 60)], fix_c=0.0), "ไม่มี FIX_C หรือ P₀"),
    (book([leg("stock", 60)], spot=None), "ไม่มีราคาหุ้น"),
    (book([leg("stock", -5)]), "หุ้น short"),
    (book([leg("put", 1, strike=7.0, delta=-0.3, multiplier=100.0)]), "ไม่ใช่ long call"),
    (book([leg("call", -1, strike=7.0, delta=0.8, multiplier=100.0)]), "ไม่ใช่ long call"),
])
def test_inputs_fail_closed_with_a_reason(b, words):
    inputs, issue = goal.goal_inputs(b)
    assert inputs is None and words in issue and issue.startswith("AAA: ")


def test_the_grid_starts_at_one_percent_and_keeps_the_key_prices():
    inputs = inputs_of(aaa_book())
    grid = goal.goal_grid(inputs, 0, 200)
    assert grid.min() == pytest.approx(0.09) and grid.max() == pytest.approx(18.0)
    assert {9.0, 10.0, 7.0} <= set(grid.round(9).tolist())
    assert (np.diff(grid) > 0).all()


def test_the_equations_use_names_and_real_values():
    text = goal.equation_text(inputs_of(aaa_book()))
    for line in ("stock_weight = shares · spot / fix_c = 60.00 · 9 / 1,500 = 0.360",
                 "coverage     = stock_weight + leaps_weight = 84.0%",
                 "goal      = fix_c · ln(x / p0)  = 1,500 · -0.105361 = -158.04",
                 "leaps_leg = Σ leaps_shares · [max(0, x − K) − max(0, p0 − K)] = 100 · [2 − 3] = -100.00",
                 "payoff    = stock_leg + leaps_leg = -156.89",
                 "gap       = payoff − goal = +1.15",
                 "excess    = Eₙ สะสมของ bot (chain) = +20.00",
                 "payoff_e  = payoff + excess = -136.89",
                 "gap_e     = payoff_e − goal = +21.15",
                 "= 96.0 หุ้น · ถืออยู่ 100 หุ้น"):
        assert line in text


def test_the_overview_lists_usable_symbols_and_reasons_for_the_rest():
    other = book([leg("stock", 10)], chain=False, symbol="BBB")
    table, issues = goal.goal_overview([aaa_book(), other])
    assert table["ตัว"].tolist() == ["AAA"] and table["ครอบคลุม %"].round(1).tolist() == [84.0]
    assert issues == ["BBB: ไม่มี chain ของ bot ใน Firebase PROD (ต้องใช้ FIX_C และ P₀) — ไม่แสดงในแท็บนี้"]


def test_the_figures_name_every_line_and_mark_k_p0_and_now():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, goal.goal_grid(inputs, 25, 200))
    fig = goal.goal_figure(df, inputs, theme="dark", title="AAA")
    assert [t.name for t in fig.data if t.showlegend is not False] == [goal.NAMES[k] for k in goal.LINE_KEYS]
    texts = [a.text for a in fig.layout.annotations]
    assert {"K 7", "P₀ 10", "ตอนนี้ 9"} <= set(texts)
    assert any(t.startswith("ตอนนี้: Payoff <b>-157</b> · +Eₙ -137 · เป้า -158 · ห่าง +1") for t in texts)
    gap = goal.gap_figure(df, inputs)
    assert {"▲ เหนือเป้า", "▼ ใต้เป้า"} <= {a.text for a in gap.layout.annotations}
    assert [t.name for t in gap.data] == [goal.NAMES["gap"], goal.NAMES["gap_e"]]
    for f in (fig, gap):                       # d3 format ที่ขึ้นต้นด้วย "+" ทำให้ hover ทศนิยมเต็ม
        assert not f.layout.yaxis.hoverformat.startswith("+")


def test_payoff_plus_e_shifts_the_payoff_by_the_bot_excess():
    inputs = inputs_of(aaa_book())
    assert inputs.excess == 20.0                             # Eₙ สะสมของ chain ใน fixture
    df = goal.goal_curves(inputs, goal.goal_grid(inputs, 25, 200))
    assert np.allclose(df["payoff_e"], df["payoff"] + 20.0) and np.allclose(df["gap_e"], df["gap"] + 20.0)
    now = goal.goal_at(inputs)
    assert round(now["payoff_e"], 2) == -136.89 and round(now["gap_e"], 2) == 21.15
    table, _ = goal.goal_overview([aaa_book()])
    assert table.loc[0, "Eₙ (bot)"] == 20.0 and round(table.loc[0, "Payoff + Eₙ ตอนนี้"], 2) == -136.89


def test_the_excess_band_sits_between_payoff_and_payoff_plus_e():
    inputs = inputs_of(aaa_book())
    fig = goal.goal_figure(goal.goal_curves(inputs, goal.goal_grid(inputs, 25, 200)), inputs)
    names = [t.name for t in fig.data]
    i = names.index(goal.NAMES["payoff_e"])
    assert names[i - 1] == goal.NAMES["payoff"] and fig.data[i].fill == "tonexty"
    ends = [a.text.split(" <b>")[0] for a in fig.layout.annotations if a.xanchor == "left" and a.xshift == 8]
    assert ends == ["เป้า", "Payoff", "Payoff+Eₙ", "LEAPS"]          # ป้ายปลายเส้น ≤ 4


def test_without_excess_there_is_no_payoff_plus_e_line():
    inputs = inputs_of(book([leg("stock", 60), leg("call", 1, strike=7.0, delta=0.8, multiplier=100.0)]))
    assert inputs.excess is None
    df = goal.goal_curves(inputs, [8.0, 9.0])
    assert df["payoff_e"].isna().all() and goal.goal_at(inputs)["payoff_e"] is None
    fig = goal.goal_figure(df, inputs)
    assert goal.NAMES["payoff_e"] not in [t.name for t in fig.data]
    gap = goal.gap_figure(df, inputs)
    assert len(gap.data) == 1 and gap.data[0].showlegend is False
    assert "payoff_e  = payoff + excess = —" in goal.equation_text(inputs)


def test_the_module_stays_isolated_from_the_payoff_tab():
    tree = ast.parse(Path(goal.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "lego_payoff" not in imported and "streamlit" not in imported and "lego_webull_read" not in imported


# ---------------------------------------------------------------- หน้าเว็บ (AppTest)

def test_the_goal_tab_shows_weights_charts_and_equations(monkeypatch):
    app = payoff_app(monkeypatch)
    app.run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["ขาหุ้น (% FIX_C)"] == "36.0%" and metrics["ขา LEAPS (% FIX_C)"] == "48.0%"
    assert metrics["ครอบคลุม FIX_C"] == "84.0%" and metrics["ราคาตอนนี้"] == "9.00"
    assert metrics["Payoff ตอนนี้"] == "-156.89" and metrics["ห่างจากเป้าตอนนี้"] == "+1.15"
    assert metrics["Eₙ สะสม (bot)"] == "+20.00" and metrics["Payoff + Eₙ ตอนนี้"] == "-136.89"
    assert app.slider(key="goal_range").value == (25, 200)
    assert any("stock_weight = shares · spot / fix_c" in c.value for c in app.code)
    assert app.button_group(key="goal_symbol").value == "AAA"
    assert [t.label for t in app.tabs][4] == "🎯 Goal"
    import streamlit_app
    assert streamlit_app.goal.GOAL_SCHEMA >= 2                       # reload guard รู้จักเส้น Payoff + Eₙ


def test_without_bot_state_the_goal_tab_explains_why(monkeypatch):
    app = payoff_app(monkeypatch, firebase=False)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert any(i.value.startswith("ยังไม่มีตัวที่คำนวณได้") for i in app.info)
    assert any("AAA: ไม่มี chain ของ bot ใน Firebase PROD (ต้องใช้ FIX_C และ P₀)" in c.value for c in app.caption)
