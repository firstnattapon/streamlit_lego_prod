"""แท็บ 🎯 Payoff — สูตร (lego_goal) และหน้าเว็บ (AppTest) · ตัวเลขทั้งหมดเป็นตัวอย่างสมมติ ไม่ใช่ข้อมูลบัญชี"""
from __future__ import annotations

import ast
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import lego_goal as goal
import lego_positions as positions
from test_positions import (NOW, ROWS, STATE, STOCK, clear_streamlit_cache, goal_app,  # noqa: F401
                            snapshot)


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
    chains, issues = positions.lego_chains(STATE, ROWS)
    books, _ = positions.build_books(snapshot(), chains, issues, NOW)
    return books[0]


def bbb_book(excess=10.0):
    """ตัวที่สอง (60:40 · P₀ 9 · ราคาตอนนี้ 9) สำหรับรวมพอร์ต"""
    return book(SIXTY_FORTY.legs, excess=excess, symbol="BBB")


def visible(fig) -> dict[str, object]:
    return {t.name: t.visible for t in fig.data if t.showlegend is not False}


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
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    assert df["gap"].min() >= -1e-9                          # x − 1 ≥ ln x (ไม่คิด premium)
    h = 1e-6                                                 # ความชันเท่ากันที่ P₀ (ฝั่งขวา)
    slope = goal.goal_curves(inputs, [9.0, 9.0 + h]).diff().iloc[1] / h
    assert slope["payoff"] == pytest.approx(slope["goal"], rel=1e-4) == pytest.approx(1500 / 9, rel=1e-4)


def test_the_real_shaped_fixture_uses_webull_holdings_and_the_bot_chain():
    inputs = inputs_of(aaa_book())
    assert (inputs.fix_c, inputs.p0, inputs.spot, inputs.shares, inputs.excess) == (1500.0, 10.0, 9.0, 60.0, 20.0)
    assert inputs.stock_weight == pytest.approx(0.36) and inputs.leaps_weight == pytest.approx(0.48)
    assert inputs.coverage == pytest.approx(0.84) and inputs.pct_now == pytest.approx(90.0)
    now = goal.goal_at(inputs)
    assert now["goal"] == pytest.approx(1500 * math.log(0.9))
    assert now["stock_leg"] == pytest.approx(540 * math.log(0.9))
    assert now["leaps_leg"] == pytest.approx(100 * ((9 - 7) - (10 - 7)))
    assert round(now["payoff"], 2) == -156.89 and round(now["gap"], 2) == 1.15
    assert round(now["payoff_e"], 2) == -136.89 and round(now["gap_e"], 2) == 21.15
    assert inputs.needed_leaps_shares == pytest.approx(96.0) and inputs.option_text() == "1×C7"


def test_payoff_plus_e_shifts_the_whole_payoff_by_the_bot_excess():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    assert np.allclose(df["payoff_e"], df["payoff"] + 20.0) and np.allclose(df["gap_e"], df["gap"] + 20.0)


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


def test_the_price_axis_is_zero_to_one_and_a_half_p0():
    inputs = inputs_of(aaa_book())
    grid = goal.goal_grid(inputs)
    assert grid.min() == pytest.approx(0.1) and grid.max() == pytest.approx(15.0)     # 1% … 150% ของ P₀ 10
    assert {9.0, 10.0, 7.0} <= set(grid.round(9).tolist()) and (np.diff(grid) > 0).all()
    axis = goal.symbol_axis(inputs)
    assert (axis.col, axis.x_max, axis.y_fit_from) == (goal.PRICE, 15.0, 2.5)
    pct = goal.pct_grid([inputs])
    assert pct.min() == pytest.approx(1.0) and pct.max() == pytest.approx(150.0) and 90.0 in pct


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


def test_collect_keeps_usable_symbols_and_reasons_for_the_rest():
    other = book([leg("stock", 10)], chain=False, symbol="BBB")
    items, issues = goal.collect([aaa_book(), other])
    assert [i.symbol for i in items] == ["AAA"]
    assert issues == ["BBB: ไม่มี chain ของ bot ใน Firebase PROD (ต้องใช้ FIX_C และ P₀) — ไม่แสดงในแท็บนี้"]
    table = goal.overview_table(items)
    assert table["ครอบคลุม %"].round(1).tolist() == [84.0] and table["ตอนนี้ % ของ P₀"].tolist() == [90.0]
    assert table.loc[0, "Eₙ (bot)"] == 20.0 and round(table.loc[0, "Payoff + Eₙ ตอนนี้"], 2) == -136.89


# ---------------------------------------------------------------- รวมพอร์ต

def test_the_portfolio_sums_each_symbol_at_the_same_percent_of_its_own_p0():
    items = [inputs_of(aaa_book()), inputs_of(bbb_book())]
    df = goal.portfolio_curves(items, [50.0, 100.0, 150.0])
    parts = [goal.goal_curves(i, [i.p0 * 0.5, i.p0, i.p0 * 1.5]) for i in items]
    for key in goal.VALUE_KEYS:
        assert np.allclose(df[key], parts[0][key].to_numpy() + parts[1][key].to_numpy())
    assert df.loc[1, "goal"] == pytest.approx(0.0) and df.loc[1, "payoff"] == pytest.approx(0.0)
    assert np.allclose(df["payoff_e"], df["payoff"] + 30.0)                     # Eₙ 20 + 10
    assert np.allclose(df["goal"], 3000 * np.log(df[goal.PCT] / 100))


def test_at_the_portfolio_now_percent_the_goal_line_equals_the_sum_of_todays_goals():
    items = [inputs_of(aaa_book()), inputs_of(book(SIXTY_FORTY.legs, spot=7.2, excess=10.0, symbol="BBB"))]
    now = goal.portfolio_pct_now(items)                              # 90% กับ 80% (FIX_C เท่ากัน)
    assert now == pytest.approx(100 * math.sqrt(0.9 * 0.8))
    at = goal.portfolio_curves(items, [now]).iloc[0]
    total = goal.summary(items)
    assert at["goal"] == pytest.approx(total["goal"])                 # ตรงพอดีตามนิยาม (Payoff ไม่ใช่ ln ล้วน → ใกล้เคียง)
    assert now in goal.pct_grid(items)
    text = goal.portfolio_equation_text(items)
    assert f"ตอนนี้ (เทียบเท่า) pct* = 100 · exp(Σ fix_c_i · ln(pct_i / 100) / Σ fix_c_i) = {now:.2f}%" in text


def test_one_symbol_without_excess_removes_payoff_plus_e_from_the_portfolio():
    items = [inputs_of(aaa_book()), inputs_of(bbb_book(excess=None))]
    df = goal.portfolio_curves(items, [80.0, 120.0])
    assert df["payoff_e"].isna().all() and df["payoff"].notna().all()
    assert goal.summary(items)["payoff_e"] is None and goal.summary(items)["excess"] is None


def test_the_portfolio_summary_adds_today_values_and_weights_by_total_fix_c():
    aaa, bbb = inputs_of(aaa_book()), inputs_of(bbb_book())
    total = goal.summary([aaa, bbb])
    assert total["fix_c"] == 3000.0 and total["excess"] == 30.0
    assert total["stock_weight"] == pytest.approx((540 + 600) / 3000)
    assert total["leaps_weight"] == pytest.approx((0.8 * 100 * 9 + 1.0 * 100 * 9) / 3000)
    for key in goal.VALUE_KEYS:
        assert total[key] == pytest.approx(goal.goal_at(aaa)[key] + goal.goal_at(bbb)[key])
    axis = goal.portfolio_axis([aaa, bbb])
    assert (axis.col, axis.x_max, axis.suffix) == (goal.PCT, 150.0, "%")
    now = goal.portfolio_pct_now([aaa, bbb])
    assert now == pytest.approx(100 * math.sqrt(0.9))                 # FIX_C เท่ากัน → เฉลี่ยเรขาคณิตของ 90% กับ 100%
    assert axis.marks == ((100.0, "P₀ (100%)", "dash", False), (now, "ตอนนี้ 94.9%", "solid", True))
    text = goal.portfolio_equation_text([aaa, bbb])
    assert text.startswith("รวมพอร์ต (2 ตัว)") and "goal_รวม(pct)     = Σ fix_c_i · ln(pct / 100) = 3,000" in text
    assert "payoff_e_รวม(pct) = payoff_รวม(pct) + Σ excess_i = payoff_รวม(pct) +30.00" in text


# ---------------------------------------------------------------- กราฟ

def test_by_default_only_the_goal_and_payoff_plus_e_are_shown():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    fig = goal.goal_figure(df, goal.symbol_axis(inputs), now=(inputs.spot, goal.goal_at(inputs)), theme="dark")
    assert visible(fig) == {goal.NAMES["goal"]: True, goal.NAMES["stock_leg"]: "legendonly",
                            goal.NAMES["leaps_leg"]: "legendonly", goal.NAMES["payoff"]: "legendonly",
                            goal.NAMES["payoff_e"]: True}
    band = next(t for t in fig.data if t.name == "แถบ Eₙ")          # ซ่อน/โชว์พร้อมเส้น Payoff
    payoff_line = next(t for t in fig.data if t.name == goal.NAMES["payoff"])
    assert band.legendgroup == payoff_line.legendgroup == "payoff" and band.visible == "legendonly"
    assert band.fill == "toself"
    ends = [a.text.split(" <b>")[0] for a in fig.layout.annotations if a.xanchor == "left" and a.xshift == 8]
    assert ends == ["เป้า", "Payoff+Eₙ"]
    texts = [a.text for a in fig.layout.annotations]
    assert {"K 7", "P₀ 10", "ตอนนี้ 9"} <= set(texts)
    assert "ตอนนี้: Payoff + Eₙ <b>-137</b> · เป้า -158 · ห่าง +21" in texts
    assert list(fig.layout.xaxis.range) == [0.0, 15.0]


def test_the_y_axis_fits_from_a_quarter_of_p0_so_the_log_tail_does_not_flatten_the_chart():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    fig = goal.goal_figure(df, goal.symbol_axis(inputs))
    low = fig.layout.yaxis.range[0]
    assert df["goal"].min() < low < 1500 * math.log(0.25)          # ปลายซ้ายสุดออกนอกกรอบ (ปุ่ม Autoscale)
    assert "autoScale2d" not in goal.CHART_CONFIG["modeBarButtonsToRemove"]


def test_the_gap_chart_keeps_both_lines_and_shows_only_gap_plus_e_at_first():
    inputs = inputs_of(aaa_book())
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    gap = goal.gap_figure(df, goal.symbol_axis(inputs))
    assert [(t.name, t.visible) for t in gap.data] == [(goal.NAMES["gap"], "legendonly"), (goal.NAMES["gap_e"], True)]
    assert {"▲ เหนือเป้า", "▼ ใต้เป้า"} <= {a.text for a in gap.layout.annotations}
    assert not gap.layout.yaxis.hoverformat.startswith("+")          # "+" ทำให้ hover ทศนิยมเต็ม


def test_without_excess_the_payoff_is_shown_instead():
    inputs = inputs_of(book([leg("stock", 60), leg("call", 1, strike=7.0, delta=0.8, multiplier=100.0)]))
    df = goal.goal_curves(inputs, goal.goal_grid(inputs))
    assert df["payoff_e"].isna().all() and goal.goal_at(inputs)["payoff_e"] is None
    fig = goal.goal_figure(df, goal.symbol_axis(inputs), now=(inputs.spot, goal.goal_at(inputs)))
    shown = visible(fig)
    assert goal.NAMES["payoff_e"] not in shown and shown[goal.NAMES["payoff"]] is True
    assert not any(t.name == "แถบ Eₙ" for t in fig.data)
    assert any(a.text.startswith("ตอนนี้: Payoff <b>") for a in fig.layout.annotations)
    gap = goal.gap_figure(df, goal.symbol_axis(inputs))
    assert len(gap.data) == 1 and gap.data[0].showlegend is False and gap.data[0].visible is True
    assert "payoff_e  = payoff + excess = —" in goal.equation_text(inputs)


def test_the_portfolio_chart_uses_percent_of_p0_and_a_band_for_today():
    items = [inputs_of(aaa_book()), inputs_of(bbb_book())]
    axis = goal.portfolio_axis(items)
    fig = goal.goal_figure(goal.portfolio_curves(items, goal.pct_grid(items)), axis)
    assert list(fig.layout.xaxis.range) == [0.0, 150.0] and fig.layout.xaxis.ticksuffix == "%"
    texts = {a.text for a in fig.layout.annotations}
    assert {"P₀ (100%)", "ตอนนี้ 94.9%"} <= texts
    assert len(fig.layout.shapes) == 2                                # เส้น P₀ + เส้นตอนนี้


def test_the_module_stays_isolated():
    tree = ast.parse(Path(goal.__file__).read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"lego_positions", "lego_payoff", "streamlit", "lego_webull_read"}


# ---------------------------------------------------------------- หน้าเว็บ (AppTest)

BBB_STOCK = dict(STOCK, symbol="BBB", position_id="P-9", quantity="50", last_price="20.00",
                 unrealized_profit_loss="-10.00")
BBB_STATE = {"symbol": "BBB", "updated_at": "2026-11-16T20:45:04Z", "p0": 25.0, "prev_holdings": 50.0,
             "leaps_shares": 0.0, "last_run_id": "run_fake_9",
             "execution_cashflow": {"actual_cumulative": -50.0, "excess": 5.0, "r_basis": -55.0,
                                    "last_action_price": 20.0}}
BBB_ROW = {positions.VALUE_COL: 990.0, positions.GAP_COL: -10.0}             # FIX_C 1000


def test_the_goal_tab_shows_weights_charts_and_equations(monkeypatch):
    app = goal_app(monkeypatch)
    app.run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["ขาหุ้น (% FIX_C)"] == "36.0%" and metrics["ขา LEAPS (% FIX_C)"] == "48.0%"
    assert metrics["ครอบคลุม FIX_C"] == "84.0%" and metrics["ราคาตอนนี้"] == "9.00"
    assert metrics["Payoff ตอนนี้"] == "-156.89" and metrics["ห่างจากเป้าตอนนี้"] == "+1.15"
    assert metrics["Eₙ สะสม (bot)"] == "+20.00" and metrics["Payoff + Eₙ ตอนนี้"] == "-136.89"
    assert all(s.key != "goal_range" for s in app.slider)              # ไม่มีสไลเดอร์ช่วงราคาแล้ว
    assert any("stock_weight = shares · spot / fix_c" in c.value for c in app.code)
    picker = app.button_group(key="goal_symbol")
    assert picker.value == "AAA" and list(picker.options) == ["AAA"]  # ตัวเดียว → ไม่มีรวมพอร์ต
    assert [t.label for t in app.tabs][:5] == ["📊 Live Dashboard", "🎯 Payoff", "🎓 Rebalancing 101", "🌐 Web Apps",
                                               "⚡ Cheat Sheet"]          # 🎯 Payoff ต่อจาก Live
    assert len(app.get("plotly_chart")) == 4          # 101 + 🎯 Payoff (กราฟหลัก + ห่างจากเป้า + กราฟเวลา)


def test_the_goal_tab_has_a_portfolio_choice(monkeypatch):
    state, rows = dict(STATE, BBB_fake09=BBB_STATE), dict(ROWS, BBB_fake09=BBB_ROW)
    app = goal_app(monkeypatch, snap=snapshot(positions=snapshot()["positions"] + [BBB_STOCK]),
                   state=state, rows=rows)
    app.run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    picker = app.button_group(key="goal_symbol")
    assert list(picker.options) == ["AAA", "BBB", goal.ALL] and picker.value == goal.ALL     # ค่าเริ่มต้น = รวมพอร์ต
    metrics = {m.label: m.value for m in app.metric}
    now = 100 * math.exp((1500 * math.log(0.9) + 1000 * math.log(0.8)) / 2500)          # ถ่วงด้วย FIX_C
    assert metrics["ตอนนี้ (% ของ P₀)"] == f"{now:.1f}%" == "85.9%" and metrics["ครอบคลุม FIX_C"] == "90.4%"
    expected = 540 * math.log(0.9) - 100 + 1000 * math.log(0.8)                       # AAA + BBB ที่ราคาตอนนี้
    assert metrics["Payoff ตอนนี้"] == f"{expected:+,.2f}" == "-380.04"
    assert metrics["Eₙ สะสม (bot)"] == "+25.00" and metrics["Payoff + Eₙ ตอนนี้"] == "-355.04"
    assert any(c.value.startswith("รวมพอร์ต (2 ตัว)") for c in app.code)


def test_without_bot_state_the_goal_tab_explains_why(monkeypatch):
    app = goal_app(monkeypatch, firebase=False)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert any(i.value.startswith("ยังไม่มีตัวที่คำนวณได้") for i in app.info)
    assert any("AAA: ไม่มี chain ของ bot ใน Firebase PROD (ต้องใช้ FIX_C และ P₀)" in c.value for c in app.caption)


# ---------------------------------------------------------------- กราฟเวลา: Eₙ · เฉลี่ยต่อวัน · ต้นทุน option (ev) · Cash Burn

T_NOW = pd.Timestamp(NOW)
DAY = pd.Timedelta(days=1)


def timed(symbol="AAA", *, t0_days=4.0, fills=(), calls=(), excess=None, source="row"):
    """GoalInputs ที่มี t0 = NOW − t0_days และ fill (วันหลัง t0, Aₙ, Eₙ)"""
    t0 = None if t0_days is None else T_NOW - t0_days * DAY
    rows = tuple((t0 + d * DAY, a, e) for d, a, e in fills)
    last = rows[-1][2] if rows else 0.0
    return goal.GoalInputs(symbol, 1500.0, 10.0, 9.0, 60.0, tuple(calls), last if excess is None else excess,
                           rows, t0, source if t0 is not None else None)


CALL_C7 = goal.LeapsCall(7.0, 100.0, 0.8, date(2027, 5, 21), 4.90)


def test_cash_burn_holds_the_deepest_negative_actual():
    """ตัวอย่างของผู้ใช้: Aₙ 0 −500 −400 −300 −1200 −500 → Cash Burn 0 −500 −500 −500 −1200 −1200"""
    actual = [-500, -400, -300, -1200, -500]
    df = goal.cash_timeline([timed(fills=[(0.5 * (n + 1), a, 1.0) for n, a in enumerate(actual)])], T_NOW)
    assert df["actual"].tolist() == [0, -500, -400, -300, -1200, -500, -500]            # + จุดตอนนี้
    assert df["cash_burn"].tolist() == [0, -500, -500, -500, -1200, -1200, -1200]
    up = goal.cash_timeline([timed(fills=[(1, 300.0, 1.0), (2, -50.0, 2.0)])], T_NOW)
    assert up["cash_burn"].tolist() == [0, 0, -50, -50]                                   # Aₙ บวก → 0 (ไม่เกิน 0)


def test_portfolio_cash_burn_is_the_running_min_of_the_summed_actual():
    x = timed("X", t0_days=4, fills=[(1, -100.0, 3.0), (2, 0.0, 5.0)])
    y = timed("Y", t0_days=1, fills=[(0.5, -80.0, 2.0)])                  # เริ่มทีหลัง: ก่อน t0 = 0
    df = goal.cash_timeline([x, y], T_NOW)
    assert df["actual"].tolist() == [0, -100, 0, 0, -80, -80]
    assert df["excess"].tolist() == [0, 3, 5, 5, 7, 7]                     # Eₙ ผลรวมทุกตัว ณ เวลา t
    assert df["cash_burn"].min() == -100                                   # ไม่ใช่ −100 + −80 ของแต่ละตัว
    assert df[goal.TIME].iloc[-1] == T_NOW


def test_option_ev_counts_only_time_value_from_p0():
    inputs = inputs_of(aaa_book())                                         # C7 · cost 4.90 · P₀ 10
    assert inputs.calls[0].cost == 4.9 and inputs.option_ev == pytest.approx(100 * (4.90 - (10 - 7))) == 190.0
    otm = timed(calls=[goal.LeapsCall(12.0, 200.0, 0.4, None, 1.5)])     # ต่ำกว่า strike ที่ P₀ → ทั้งก้อนเป็น ev
    assert otm.option_ev == pytest.approx(300.0)
    missing = timed(calls=[goal.LeapsCall(7.0, 100.0, 0.8, None, None)], fills=[(1, -10.0, 4.0)])
    assert missing.option_ev is None
    summary = goal.cash_summary([missing], T_NOW)
    assert summary["ev_cost"] is None and summary["status"] is None
    assert any("ไม่ส่ง cost_price" in n for n in summary["notes"])
    assert any("ไม่มี LEAPS" in n for n in goal.cash_summary([timed(fills=[(1, 0.0, 1.0)])], T_NOW)["notes"])


def test_the_daily_average_line_meets_the_option_cost_at_the_breakeven_day():
    inputs = timed(t0_days=4, fills=[(1, -60.0, 8.0), (3, -100.0, 20.0)], calls=[CALL_C7])
    s = goal.cash_summary([inputs], T_NOW)
    assert s["excess"] == 20.0 and s["per_day"] == pytest.approx(5.0) and s["ev_cost"] == pytest.approx(190.0)
    assert s["cash_burn"] == -100.0 and s["days"] == pytest.approx(4.0)
    assert goal.trend_value(s, T_NOW) == pytest.approx(20.0)              # เส้นเฉลี่ยผ่าน Eₙ ตอนนี้
    assert goal.trend_value(s, T_NOW - 4 * DAY) == 0.0                     # เริ่มที่ 0 ที่ t0
    assert s["status"] == "projected" and s["breakeven"] == T_NOW + 34 * DAY   # (190 − 20) / 5
    assert goal.trend_value(s, s["breakeven"]) == pytest.approx(190.0) and s["end"] == s["breakeven"]
    assert goal.breakeven_text(s) == "คาด 2026-12-21 (อีก 34 วัน)"


def test_breakeven_states():
    covered = goal.cash_summary([timed(fills=[(1, 0.0, 250.0)], calls=[CALL_C7])], T_NOW)
    assert covered["status"] == "covered" and covered["end"] > T_NOW
    slow = goal.cash_summary([timed(t0_days=400, fills=[(1, 0.0, 1.0)], calls=[CALL_C7])], T_NOW)
    assert slow["status"] == "after_expiry" and slow["end"] == pd.Timestamp("2027-05-21 20:00", tz="UTC")
    assert "ไม่ทันLEAPS หมดอายุ 2027-05-21" in goal.breakeven_text(slow)
    losing = goal.cash_summary([timed(fills=[(1, 0.0, -3.0)], calls=[CALL_C7])], T_NOW)
    assert losing["status"] == "never" and losing["end"] == T_NOW
    fresh = goal.cash_summary([timed(t0_days=0.01, calls=[CALL_C7])], T_NOW)   # < 1 ชม.
    assert fresh["per_day"] is None and any("ไม่ถึง 1 ชั่วโมง" in n for n in fresh["notes"])


def test_the_portfolio_average_sums_each_chain_from_its_own_t0():
    x = timed("X", t0_days=10, fills=[(1, 0.0, 50.0)])                  # 5 / วัน
    y = timed("Y", t0_days=2, fills=[(1, 0.0, 10.0)])                   # 5 / วัน
    s = goal.cash_summary([x, y], T_NOW)
    assert s["per_day"] == pytest.approx(10.0) and s["t0"] == T_NOW - 10 * DAY
    assert goal.trend_value(s, T_NOW) == pytest.approx(60.0) == s["excess"]
    assert goal.trend_value(s, T_NOW - 2 * DAY) == pytest.approx(40.0)    # ก่อน t0 ของ Y นับแค่ X


def test_no_t0_and_no_fill_means_no_time_chart():
    lonely = timed(t0_days=None)
    assert goal.cash_timeline([lonely], T_NOW).empty
    assert any("ไม่มีเวลาเริ่ม chain" in n for n in goal.cash_summary([lonely], T_NOW)["notes"])
    stale = timed(fills=[(1, 0.0, 5.0)], excess=7.0, source="fill")
    notes = goal.cash_summary([stale], T_NOW)["notes"]
    assert any("t0 = เวลา fill แรก" in n for n in notes) and any("≠ Eₙ ใน state" in n for n in notes)


def test_the_cash_figure_has_three_lines_without_the_en_line():
    inputs = timed(t0_days=4, fills=[(1, -60.0, 8.0), (2, -150.0, 14.0), (3, -100.0, 20.0)], calls=[CALL_C7])
    s = goal.cash_summary([inputs], T_NOW)
    df = goal.cash_timeline([inputs], T_NOW)
    fig = goal.cash_figure(df, s, theme="light", title="t")
    named = {t.name: t for t in fig.data if t.showlegend is not False}
    assert list(named) == ["Eₙ เฉลี่ย 5.00 USD/วัน (ตั้งแต่ t0 P₀)",              # ไม่มีเส้น Eₙ ขั้นบันไดแล้ว
                           "ต้นทุน option (ev ไม่คิด intrinsic)", "Cash Burn สะสม (Aₙ ต่ำสุด, ≤ 0)"]
    assert np.allclose(named["ต้นทุน option (ev ไม่คิด intrinsic)"].y, 190.0)       # แนวนอนฝั่งบวก
    burn = named["Cash Burn สะสม (Aₙ ต่ำสุด, ≤ 0)"]
    assert max(burn.y) <= 0 and list(burn.y) == [0, -60, -150, -150, -150] and burn.line.shape == "hv"
    projected = next(t for t in fig.data if t.name.endswith("· คาด"))
    assert projected.line.dash == "dash" and projected.y[-1] == pytest.approx(190.0)
    texts = [a.text for a in fig.layout.annotations]
    assert "คืนทุน ev 2026-12-21" in texts and "ตอนนี้" in texts and "t0 · P₀" in texts
    assert "Cash Burn <b>-150</b>" in texts and "ต้นทุน ev <b>+190</b>" in texts
    assert not any(t.startswith("Eₙ <b>") for t in texts)                    # ไม่มีป้ายปลายเส้น Eₙ
    assert fig.layout.xaxis.type == "date"


def test_the_daily_average_line_is_dark_green_in_both_themes():
    inputs = timed(t0_days=4, fills=[(1, -60.0, 8.0), (3, -100.0, 20.0)], calls=[CALL_C7])
    s, df = goal.cash_summary([inputs], T_NOW), goal.cash_timeline([inputs], T_NOW)
    for theme, color in (("light", "#006400"), ("dark", "#1b7a34"), (None, "#006400")):
        fig = goal.cash_figure(df, s, theme=theme)
        trend = [t for t in fig.data if t.legendgroup == "trend" and t.mode == "lines"]
        assert len(trend) == 2 and {t.line.color for t in trend} == {color}        # ทึบ + ประ (คาด) สีเดียวกัน
        assert fig.data[-1].marker.line.color == color                               # วงคืนทุน ev


def test_the_projection_reaches_the_leaps_expiry_only_when_switched_on():
    inputs = timed(t0_days=4, fills=[(1, -60.0, 8.0), (3, -100.0, 20.0)], calls=[CALL_C7])
    expiry = pd.Timestamp("2027-05-21 20:00", tz="UTC")                          # 16:00 New York
    assert goal.first_expiry([inputs]) == expiry and goal.first_expiry([timed()]) is None
    off = goal.cash_summary([inputs], T_NOW)                                     # ปิด = ต่อถึงวันคืนทุนแบบเดิม
    on = goal.cash_summary([inputs], T_NOW, to_expiry=True)
    assert off["to_expiry"] is False and off["end"] == off["breakeven"]
    assert on["to_expiry"] is True and on["end"] == expiry
    assert (on["status"], on["breakeven"]) == (off["status"], off["breakeven"])     # วันคืนทุนไม่เปลี่ยน
    df = goal.cash_timeline([inputs], T_NOW)
    fig = goal.cash_figure(df, on, theme="light")
    projected = next(t for t in fig.data if t.name.endswith("· คาด"))
    assert pd.Timestamp(projected.x[-1]) == expiry
    assert projected.y[-1] == pytest.approx(goal.trend_value(on, expiry)) == pytest.approx(5.0 * (4 + (expiry - T_NOW) / DAY))
    texts = [a.text for a in fig.layout.annotations]
    assert "LEAPS หมดอายุ 2027-05-21" in texts and "คืนทุน ev 2026-12-21" in texts
    shift = {a.text: a.yshift or 0 for a in fig.layout.annotations}
    assert shift["t0 · P₀"] == 0 and shift["ตอนนี้"] < 0                        # ชิดกันเมื่อแกนยาว → ป้ายแถวที่ 2
    assert shift["คืนทุน ev 2026-12-21"] == 0 == shift["LEAPS หมดอายุ 2027-05-21"]
    assert "LEAPS หมดอายุ 2027-05-21" not in [a.text for a in goal.cash_figure(df, off, theme="light").layout.annotations]
    assert "ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ 2027-05-21 20:00 UTC: เส้นเฉลี่ย = +" in goal.cash_equation_text([inputs], on)
    assert "ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ" not in goal.cash_equation_text([inputs], off)


def test_the_expiry_switch_extends_every_state_and_falls_back_without_an_expiry():
    losing = timed(fills=[(1, 0.0, -3.0)], calls=[CALL_C7])                     # Ē < 0 → ต่อเส้นลงจนหมดอายุ
    s = goal.cash_summary([losing], T_NOW, to_expiry=True)
    assert s["status"] == "never" and s["end"] == goal.first_expiry([losing]) and s["to_expiry"]
    covered = goal.cash_summary([timed(fills=[(1, 0.0, 250.0)], calls=[CALL_C7])], T_NOW, to_expiry=True)
    assert covered["status"] == "covered" and covered["end"] == goal.first_expiry([losing])
    no_date = timed(fills=[(1, 0.0, 8.0)], calls=[goal.LeapsCall(7.0, 100.0, 0.8, None, 4.9)])
    plain, switched = goal.cash_summary([no_date], T_NOW), goal.cash_summary([no_date], T_NOW, to_expiry=True)
    assert switched["end"] == plain["end"] and not switched["to_expiry"]
    assert "ไม่มีวันหมดอายุ LEAPS — ต่อเส้นเฉลี่ยแบบเดิม" in switched["notes"]
    expired = timed(fills=[(1, 0.0, 8.0)], calls=[goal.LeapsCall(7.0, 100.0, 0.8, date(2026, 11, 1), 4.9)])
    gone = goal.cash_summary([expired], T_NOW, to_expiry=True)
    assert not gone["to_expiry"] and any("LEAPS หมดอายุแล้ว (2026-11-01)" in n for n in gone["notes"])


def test_cash_equations_show_the_real_numbers():
    inputs = inputs_of(aaa_book())
    text = goal.cash_equation_text([inputs], goal.cash_summary([inputs], T_NOW))
    assert "ev = 100 · (4.9 − max(0, 10 − 7)) = +190.00" in text
    assert "t0 2026-11-10 15:00 UTC (fill แรก)" in text
    assert "Ē = +20.00 / 7.00 วัน = +2.86 USD/วัน" in text
    assert "cash_burn -150.00" in text


def test_the_goal_tab_draws_the_time_chart_from_the_first_row(monkeypatch):
    from test_positions import ALL_ROWS
    app = goal_app(monkeypatch, all_rows=ALL_ROWS)
    app.run(timeout=30)
    assert not app.exception
    assert any(m.value == "**AAA · Eₙ เทียบต้นทุน option (ev) และ Cash Burn ตามเวลา**" for m in app.markdown)
    text = next(c.value for c in app.code if "กราฟเวลา" in c.value)
    assert "t0 2026-11-10 14:30 UTC (แถวแรก)" in text                 # แถว version 1 ไม่ใช่ fill แรก
    assert "ev = 100 · (4.9 − max(0, 10 − 7)) = +190.00" in text and "cash_burn -150.00" in text
    assert not any("t0 = เวลา fill แรก" in c.value for c in app.caption)


def test_the_goal_tab_expiry_switch_starts_on_and_can_be_turned_off(monkeypatch):
    from test_positions import ALL_ROWS
    app = goal_app(monkeypatch, all_rows=ALL_ROWS)
    app.run(timeout=30)
    assert not app.exception and not app.error
    switch = app.toggle(key="goal_cash_to_expiry")
    assert switch.value is True and not switch.disabled                     # ค่าเริ่มต้นเปิด
    assert any("ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ 2027-05-21 20:00 UTC" in c.value for c in app.code)
    switch.set_value(False).run(timeout=30)
    assert not app.exception and not app.error
    assert app.toggle(key="goal_cash_to_expiry").value is False
    assert not any("ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ" in c.value for c in app.code)   # ปิด = ต่อถึงวันคืนทุน


def test_without_a_leaps_expiry_the_switch_is_disabled_and_adds_no_note(monkeypatch):
    from test_positions import ALL_ROWS
    monkeypatch.setattr(goal, "first_expiry", lambda items: None)      # เหมือนไม่มี LEAPS ที่มีวันหมดอายุ
    app = goal_app(monkeypatch, all_rows=ALL_ROWS)
    app.run(timeout=30)
    assert not app.exception and not app.error
    switch = app.toggle(key="goal_cash_to_expiry")
    assert switch.disabled and switch.value is True                     # กดไม่ได้ · ไม่ส่ง to_expiry ให้สูตร
    assert not any("ไม่มีวันหมดอายุ LEAPS" in c.value for c in app.caption)
    assert not any("ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ" in c.value for c in app.code)
    assert any("กราฟเวลา" in c.value for c in app.code)


def test_without_the_first_row_t0_falls_back_to_the_first_fill(monkeypatch):
    app = goal_app(monkeypatch)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert any("AAA: อ่านเวลาแถวแรกของ chain ไม่ได้ — t0 = เวลา fill แรก" in c.value for c in app.caption)
    assert "t0 2026-11-10 15:00 UTC (fill แรก)" in next(c.value for c in app.code if "กราฟเวลา" in c.value)


def test_the_portfolio_time_chart_sums_every_symbol(monkeypatch):
    state, rows = dict(STATE, BBB_fake09=BBB_STATE), dict(ROWS, BBB_fake09=BBB_ROW)
    app = goal_app(monkeypatch, snap=snapshot(positions=snapshot()["positions"] + [BBB_STOCK]),
                   state=state, rows=rows)
    app.run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert any(m.value.startswith("**รวมพอร์ต · Eₙ เทียบต้นทุน option") for m in app.markdown)
    assert any("BBB: ไม่มีเวลาเริ่ม chain (t0) และยังไม่มี fill" in c.value for c in app.caption)
    assert len(app.get("plotly_chart")) == 4
