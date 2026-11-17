"""แท็บ 📐 Payoff — สูตร (lego_payoff) · Webull อ่านอย่างเดียว (lego_webull_read) · หน้าเว็บ (AppTest)

ข้อมูลทั้งหมดเป็นของสมมติ (symbol AAA/BBB, ค่าลับขึ้นต้นด้วย FAKE_TEST_) — repo เป็น public ห้ามมีตัวเลขบัญชีจริง
HTTP ปลอม: patch ``Response.get_response_object`` (จุดเดียวที่ SDK ยิง requests) จึงใช้ request + signing ของ SDK จริง
"""
from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import io
import json
import logging
import math
import os
import pkgutil
from datetime import date, datetime, timezone
from pathlib import Path

import firebase_admin
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import lego_payoff as payoff
import lego_webull_read as webull_read
from lego_webull_read import READ_ONLY_ALLOWLIST, ReadOnlyGuard, WebullReadError, redact

ROOT = Path(__file__).parent
APP_PATH = str(ROOT / "streamlit_app.py")
NOW = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
FAKE_KEY, FAKE_SECRET, FAKE_TOKEN = "FAKE_TEST_KEY_0001", "FAKE_TEST_SECRET_0001", "FAKE_TEST_TOKEN_000"
TOKEN_TXT = f"\n{FAKE_TOKEN}\n1790000000000\nNORMAL\n"
OCC = "AAA270521C00007000"

STOCK = {"currency": "USD", "quantity": "60.00000", "position_id": "P-1", "symbol": "AAA",
         "instrument_type": "EQUITY", "cost_price": "10.00", "last_price": "9.00",
         "unrealized_profit_loss": "-60.00"}
CALL = {"currency": "USD", "quantity": "1.0000000000", "position_id": "P-2", "symbol": "AAA",
        "option_strategy": "SINGLE", "instrument_type": "OPTION", "cost_price": "4.90",
        "last_price": "3.30", "unrealized_profit_loss": "-160.00",
        "items": [{"side": "BUY", "quantity": "1.0000000000", "symbol": "AAA", "cost_price": "4.90",
                   "option_type": "CALL", "option_expire_date": "2027-05-21",
                   "option_exercise_price": "7.0000000000",
                   "option_contract_multiplier": "100.0000000000"}]}
POSITIONS = [STOCK, CALL]
OPTION_QUOTES = {OCC: {"imp_vol": "0.80", "delta": "0.80", "bid": "3.10", "ask": "3.50"}}
STATE = {
    "AAA_fake01": {"symbol": "AAA", "updated_at": "2026-11-16T20:45:04Z", "p0": 10.0,
                   "prev_holdings": 60.0, "leaps_shares": 100.0, "last_run_id": "run_fake_1",
                   "execution_cashflow": {"actual_cumulative": -100.0, "excess": 20.0,
                                          "r_basis": -120.0, "last_action_price": 9.0}},
    "AAA_older": {"symbol": "AAA", "updated_at": "2026-10-01T15:00:00Z", "p0": 12.0,
                  "prev_holdings": 1.0, "leaps_shares": 0.0, "last_run_id": "run_fake_0",
                  "execution_cashflow": {"actual_cumulative": 1.0, "excess": 1.0,
                                         "r_basis": 0.0, "last_action_price": 12.0}},
}
ROWS = {"AAA_fake01": {payoff.VALUE_COL: 1468.27, payoff.GAP_COL: -31.73},
        "AAA_older": {payoff.VALUE_COL: 900.0, payoff.GAP_COL: -100.0}}

# ตัวอย่างในแผน (fixture AAA): ราคา → (① T+0, ② หมดอายุ, ③ rebalance)
EXAMPLE = {4.5: (-752.75, -820.00, -1072.47), 6.0: (-598.95, -730.00, -727.15),
           9.0: (-220.00, -350.00, -220.00), 12.0: (213.51, 130.00, 165.03),
           15.0: (671.87, 610.00, 478.11), 20.0: (1457.59, 1410.00, 963.83)}


def legs() -> list[payoff.Leg]:
    out, problems = payoff.parse_positions(POSITIONS, OPTION_QUOTES)
    assert not problems
    return out


def chain() -> payoff.LegoChain:
    chains, issues = payoff.lego_chains(STATE, ROWS)
    assert not issues
    return chains["AAA"]


# ---------------------------------------------------------------- Black-Scholes

def test_put_call_parity_and_the_example_time():
    T = payoff.years_to_expiry(date(2027, 5, 21), NOW)
    assert T == pytest.approx(0.50742, abs=1e-5)          # 16:00 New York (EDT) = 20:00 UTC
    c = payoff.bs_price(9, 7, T, 0.04, 0.8, "call")
    p = payoff.bs_price(9, 7, T, 0.04, 0.8, "put")
    assert c - p == pytest.approx(9 - 7 * math.exp(-0.04 * T), abs=1e-9)
    assert c - p == pytest.approx(2.140645, abs=1e-6)
    assert payoff.bs_price(9, 7, T, 0.04, 0.8, "call") == pytest.approx(3.0405, abs=1e-4)


def test_no_time_or_no_volatility_means_intrinsic():
    for S in (5.0, 7.0, 9.5):
        assert payoff.bs_price(S, 7, 0.0, 0.04, 0.8, "call") == max(0.0, S - 7)
        assert payoff.bs_price(S, 7, 0.0, 0.04, 0.8, "put") == max(0.0, 7 - S)
        assert payoff.bs_price(S, 7, 0.5, 0.04, 0.0, "call") == max(0.0, S - 7)
        assert payoff.bs_price(S, 7, 0.5, 0.04, None, "put") == max(0.0, 7 - S)


def test_implied_vol_round_trips_and_refuses_impossible_prices():
    T = 0.5
    for kind in ("call", "put"):
        price = payoff.bs_price(9, 7, T, 0.04, 0.8, kind)
        assert payoff.implied_vol(price, 9, 7, T, 0.04, kind) == pytest.approx(0.8, abs=1e-4)
    assert payoff.implied_vol(1.0, 9, 7, T, 0.04, "call") is None      # ต่ำกว่า intrinsic: ไม่เดา
    assert payoff.implied_vol(None, 9, 7, T, 0.04, "call") is None
    assert payoff.implied_vol(3.0, 9, 7, 0.0, 0.04, "call") is None


# ---------------------------------------------------------------- อ่าน positions / token

def test_positions_are_parsed_like_the_webull_payload():
    out = legs()
    stock, call = out
    assert (stock.symbol, stock.kind, stock.qty, stock.multiplier, stock.mark, stock.unrealized) == \
        ("AAA", "stock", 60.0, 1.0, 9.0, -60.0)
    assert (call.kind, call.qty, call.multiplier, call.strike, call.expiry, call.mark) == \
        ("call", 1.0, 100.0, 7.0, date(2027, 5, 21), 3.30)
    assert call.occ == OCC == payoff.occ_symbol("AAA", date(2027, 5, 21), "call", 7.0)
    assert (call.iv, call.delta, call.iv_source) == (0.8, 0.8, "webull")
    assert payoff.option_contracts(POSITIONS) == [OCC]


def test_bad_positions_are_skipped_with_field_names_only():
    broken = [{"instrument_type": "EQUITY", "symbol": "AAA", "cost_price": "123.45"},
              {"instrument_type": "OPTION", "symbol": "AAA"},
              {"instrument_type": "CRYPTO", "symbol": "BTC"}]
    out, problems = payoff.parse_positions(broken)
    assert out == []
    assert problems[0] == "EQUITY: ไม่พบ field quantity, last_price, unrealized_profit_loss"
    assert problems[1] == "OPTION: ไม่พบ legs/items, unrealized_profit_loss (มี field: instrument_type, symbol)"
    assert "CRYPTO" in problems[2]
    assert not any("123.45" in p for p in problems)                  # ไม่มีค่าบัญชีในข้อความ


def test_sold_options_are_negative_and_a_combo_keeps_its_pl_on_the_first_leg():
    combo = dict(CALL, unrealized_profit_loss="50.00", option_strategy="VERTICAL", items=[
        CALL["items"][0],
        dict(CALL["items"][0], side="SELL", option_exercise_price="9.0000000000")])
    quotes = {OCC: {"bid": "3.0", "ask": "3.4"}, "AAA270521C00009000": {"bid": "1.8", "ask": "2.2"}}
    long_leg, short_leg = payoff.parse_positions([combo], quotes)[0]
    assert (long_leg.qty, short_leg.qty) == (1.0, -1.0)
    assert (long_leg.unrealized, short_leg.unrealized) == (50.0, 0.0)
    assert (long_leg.mark, short_leg.mark) == (pytest.approx(3.2), pytest.approx(2.0))


# รูปแบบทางการของ Webull OpenAPI v3 (GET /trading/assets/positions/list): option อยู่ใน "legs" และไม่มี "side"
# (เอกสาร developer.webull.co.th — Account Positions) · รูปแบบของ MCP ที่ใช้ตอนพัฒนาคือ "items" + "side"
V3_CALL = {"position_id": "P-2", "currency": "USD", "quantity": "1", "symbol": "AAA", "option_strategy": "SINGLE",
           "instrument_type": "OPTION", "cost_price": "4.90", "last_price": "3.30",
           "unrealized_profit_loss": "-160.00",
           "legs": [{"leg_id": "L-1", "symbol": "AAA", "quantity": "1", "option_type": "CALL",
                     "option_expire_date": "2027-05-21", "option_exercise_price": "7",
                     "option_contract_multiplier": "100", "option_contract_deliverable": "100",
                     "expiration_type": "AM"}]}
V3_POSITIONS = [STOCK, V3_CALL]


def _shape(leg: payoff.Leg) -> tuple:
    return (leg.symbol, leg.kind, leg.qty, leg.multiplier, leg.mark, leg.unrealized, leg.strike, leg.expiry,
            leg.occ, leg.iv, leg.delta)


def test_v3_legs_parse_like_the_mcp_items():
    v3, problems = payoff.parse_positions(V3_POSITIONS, OPTION_QUOTES)
    assert not problems
    assert [_shape(leg) for leg in v3] == [_shape(leg) for leg in legs()]
    assert v3[1].cost == 4.90                                         # legs ไม่มีทุน → ใช้ของ position (ขาเดียว)
    assert payoff.option_contracts(V3_POSITIONS) == [OCC]
    assert payoff.underlying_symbols(V3_POSITIONS) == ["AAA"]


def test_occ_symbols_on_the_leg_or_position_give_the_underlying_and_the_contract():
    assert payoff.parse_occ(OCC) == {"symbol": "AAA", "expiry": date(2027, 5, 21), "kind": "call", "strike": 7.0}
    assert payoff.parse_occ("BRK.B 270115P00450000")["symbol"] == "BRK.B"
    assert payoff.parse_occ("AAA") is None and payoff.parse_occ("AAA271399C00007000") is None
    leg_occ = dict(V3_CALL, legs=[{"leg_id": "L-1", "symbol": OCC, "quantity": "1"}])     # ไม่มี field option อื่น
    position_occ = {k: v for k, v in V3_CALL.items() if k != "legs"} | {"symbol": OCC}     # ไม่มี legs เลย
    for shape in (leg_occ, position_occ):
        out, problems = payoff.parse_positions([STOCK, shape], OPTION_QUOTES)
        assert not problems and _shape(out[1]) == _shape(legs()[1])
        assert payoff.underlying_symbols([STOCK, shape]) == ["AAA"]
        assert payoff.option_contracts([shape]) == [OCC]


def test_v3_short_legs_use_the_sign_of_the_quantity():
    short = dict(V3_CALL, legs=[dict(V3_CALL["legs"][0], quantity="-2")])
    assert payoff.parse_positions([short])[0][0].qty == -2.0
    short_total = dict(V3_CALL, quantity="-1")
    assert payoff.parse_positions([short_total])[0][0].qty == -1.0
    assert payoff.parse_positions([V3_CALL])[0][0].qty == 1.0                    # ไม่บอกอะไร = ซื้อ


def test_an_unreadable_leg_reports_field_names_only():
    broken = dict(V3_CALL, legs=[{"leg_id": "L-9", "symbol": "AAA", "quantity": "1", "cost": "777.77"}])
    out, problems = payoff.parse_positions([broken])
    assert out == [] and problems == [
        "OPTION: ขาอ่านไม่ได้ ต้องมี option_type / option_expire_date / option_exercise_price "
        "(หรือ symbol แบบ OCC) และ quantity (มี field: cost, leg_id, quantity, symbol)"]
    assert "777.77" not in problems[0]


def test_v3_positions_build_full_books_with_the_bot_chain():
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, notes = payoff.build_books({"positions": V3_POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    book = books[0]
    assert not notes and book.chain_issue is None                     # LEAPS 1 × 100 = bot 100
    assert book.unrealized == pytest.approx(-220.0)                   # หุ้น + option
    df = payoff.book_curves(book, NOW, np.array(sorted(EXAMPLE)))
    for _, row in df.iterrows():
        assert row["rebalance"] == pytest.approx(EXAMPLE[row["price"]][2], abs=0.01)
    assert df[list(payoff.LINE_KEYS)].notna().all().all()


def test_token_txt_needs_three_lines():
    assert payoff.parse_token_txt(TOKEN_TXT) == (FAKE_TOKEN, 1790000000000, "NORMAL")
    for bad in ("", FAKE_TOKEN, f"{FAKE_TOKEN}\nNORMAL", f"{FAKE_TOKEN}\nsoon\nNORMAL"):
        with pytest.raises(ValueError, match="3 บรรทัด"):
            payoff.parse_token_txt(bad)


# ---------------------------------------------------------------- เส้นของมุมมองบัญชีจริง

def test_the_three_lines_match_the_worked_example():
    df = payoff.symbol_curves(legs(), 9.0, NOW, chain(), np.array(sorted(EXAMPLE)))
    for _, row in df.iterrows():
        t0, expiry, rebalance = EXAMPLE[row["price"]]
        assert row["t0"] == pytest.approx(t0, abs=0.01)
        assert row["expiry"] == pytest.approx(expiry, abs=0.01)
        assert row["rebalance"] == pytest.approx(rebalance, abs=0.01)
        assert row["rebalance_e"] == pytest.approx(rebalance + 20.0, abs=0.01)    # ④ = ③ + Eₙ


def test_t0_and_rebalance_pass_through_today_and_expiry_drops_the_time_value():
    df = payoff.symbol_curves(legs(), 9.0, NOW, chain(), np.array([9.0]))
    U0 = -220.0
    assert df["t0"].iloc[0] == pytest.approx(U0, abs=1e-9)
    assert df["rebalance"].iloc[0] == pytest.approx(U0, abs=1e-9)
    assert df["expiry"].iloc[0] == pytest.approx(U0 - 100 * (3.30 - (9 - 7)), abs=1e-9)   # −350
    assert df["rebalance_e"].iloc[0] == pytest.approx(U0 + chain().excess, abs=1e-9)     # −200


def test_rebalance_is_lego_delta_r_plus_the_leaps_gap_below_u_star():
    grid = np.linspace(4.0, 15.0, 23)
    df = payoff.symbol_curves(legs(), 9.0, NOW, chain(), grid)
    options_change = df["t0"] - (-220.0) - 60 * (grid - 9.0)
    expected = -220.0 + 1500 * np.log(grid / 9.0) - 100 * (grid - 9.0) + options_change
    assert np.allclose(df["rebalance"], expected, atol=1e-6)


def test_the_stock_part_of_rebalance_is_flat_above_u_star():
    c = chain()
    assert c.u_star == pytest.approx(15.0)
    F = payoff._log_floor
    assert F(c.fix_c, c.leaps_shares, 20.0) == pytest.approx(F(c.fix_c, c.leaps_shares, 15.0))
    assert F(c.fix_c, c.leaps_shares, 12.0) < F(c.fix_c, c.leaps_shares, 15.0)


def test_the_expiry_line_uses_the_nearest_expiry_and_values_later_options_by_bs():
    later = dict(CALL, position_id="P-3", last_price="2.00", unrealized_profit_loss="10.00",
                 items=[dict(CALL["items"][0], option_expire_date="2027-12-17",
                             option_exercise_price="8.0000000000")])
    quotes = dict(OPTION_QUOTES, AAA271217C00008000={"imp_vol": "0.70"})
    out, _ = payoff.parse_positions([STOCK, CALL, later], quotes)
    grid = np.array([6.0, 9.0, 12.0])
    df = payoff.symbol_curves(out, 9.0, NOW, None, grid)
    T_left = payoff.years_to_expiry(date(2027, 12, 17), payoff._expiry_close(date(2027, 5, 21)))
    for S, value in zip(grid, df["expiry"]):
        first = 100 * (max(0.0, S - 7) - 3.30)
        second = 100 * (payoff.bs_price(S, 8, T_left, payoff.RISK_FREE, 0.7, "call") - 2.00)
        assert value == pytest.approx(-210.0 + 60 * (S - 9) + first + second, abs=1e-9)


def test_lines_fail_closed_when_data_is_missing():
    no_chain = payoff.symbol_curves(legs(), 9.0, NOW, None, np.array([8.0, 9.0, 10.0]))
    assert no_chain["rebalance"].isna().all() and no_chain["t0"].notna().all()
    assert no_chain["rebalance_e"].isna().all()                       # ④ ต้องมี chain เหมือน ③
    no_iv = [payoff.Leg(**{**leg.__dict__, "iv": None}) if leg.is_option else leg for leg in legs()]
    df = payoff.symbol_curves(no_iv, 9.0, NOW, chain(), np.array([8.0, 9.0, 10.0]))
    assert df["t0"].isna().all() and df["rebalance"].isna().all() and df["expiry"].notna().all()
    stock_only = payoff.symbol_curves(legs()[:1], 9.0, NOW, None, np.array([8.0, 9.0]))
    assert stock_only["expiry"].isna().all()                          # ไม่มีวันหมดอายุให้วาด


def test_portfolio_at_zero_percent_is_the_sum_of_todays_pl():
    other = [dict(STOCK, symbol="BBB", quantity="10", last_price="20", unrealized_profit_loss="15.5")]
    snap = {"positions": POSITIONS + other, "options": OPTION_QUOTES, "quotes": {}}
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, _ = payoff.build_books(snap, chains, issues, NOW)
    df = payoff.portfolio_curves(books, np.array([90.0, 100.0, 110.0]), NOW)
    at_zero = df[df[payoff.PCT_COL] == 100.0].iloc[0]
    assert at_zero["t0"] == pytest.approx(-220.0 + 15.5)
    assert np.isnan(at_zero["rebalance"])                             # BBB ไม่มี chain → เส้นรวมหาย
    assert np.isnan(at_zero["rebalance_e"])


def test_break_evens_interpolate_zero_crossings():
    x = np.linspace(0, 10, 11)
    assert payoff.break_evens(x, x - 3) == [3.0]
    assert payoff.break_evens(x, np.abs(x - 5) - 2.5) == pytest.approx([2.5, 7.5])   # เส้นตรงเป็นท่อน
    assert payoff.break_evens(x, np.full_like(x, np.nan)) == []


def test_curve_table_interpolates_and_hides_missing_lines():
    df = payoff.symbol_curves(legs(), 9.0, NOW, None, payoff.price_grid(9.0, 40, 200))
    table = payoff.curve_table(df, payoff.PRICE_COL, {k: payoff.LINE_NAMES[k] for k in payoff.LINE_KEYS},
                               [9.0, 12.0, 100.0])
    assert list(table[payoff.PRICE_COL]) == [9.0, 12.0]                # 100 อยู่นอกช่วง
    assert table[payoff.LINE_NAMES["t0"]].iloc[0] == pytest.approx(-220.0, abs=1e-6)
    assert table[payoff.LINE_NAMES["rebalance"]].isna().all()


def test_price_grid_contains_today_strikes_and_u_star():
    grid = payoff.price_grid(9.0, 40, 200, extra=[7.0, 15.0, None, math.inf, 99.0])
    assert {9.0, 7.0, 15.0} <= set(grid) and 99.0 not in grid
    assert grid.min() == pytest.approx(3.6) and grid.max() == pytest.approx(18.0)


def test_a_range_from_zero_starts_the_data_at_one_percent_and_the_axis_at_zero():
    grid = payoff.price_grid(9.0, 0, 200, extra=[7.0, 15.0])
    assert grid.min() == pytest.approx(0.09) and grid.max() == pytest.approx(18.0)       # 1% ถึง 2 × S₀
    df = payoff.symbol_curves(legs(), 9.0, NOW, chain(), grid)
    assert np.isfinite(df[list(payoff.LINE_KEYS)].to_numpy()).all()                       # ไม่มี ln(0) = −∞
    fig = payoff.payoff_figure(df, spot=9.0, now_pl=-220.0, x_start=0.0)
    assert fig.layout.xaxis.range[0] == 0.0 and fig.layout.xaxis.rangeslider.range[0] == 0.0
    assert fig.layout.updatemenus[0].buttons[-1].args[0]["xaxis.range"] == [0.0, pytest.approx(18.0)]
    assert {t.name for t in fig.data} >= {payoff.LINE_NAMES[k] for k in payoff.LINE_KEYS}   # ทุกเส้นยังวาด
    pcts = payoff.pct_grid(0, 200)
    assert pcts.min() == payoff.MIN_GRID_PCT and 100.0 in pcts and pcts.max() == 200.0
    lego = payoff.lego_ledger_figure(payoff.lego_view_curves(chain(), grid), x_start=0.0)
    assert lego.layout.xaxis.range[0] == 0.0


# ---------------------------------------------------------------- state ของ bot (LEGO)

def test_lego_chains_pick_the_latest_chain_and_read_v3_fields():
    c = chain()
    assert (c.chain_key, c.fix_c, c.leaps_shares, c.real_holdings) == ("AAA_fake01", pytest.approx(1500.0),
                                                                     100.0, 60.0)
    assert (c.p_acted, c.actual, c.r_basis, c.excess, c.p0) == (9.0, -100.0, -120.0, 20.0, 10.0)


def test_lego_chains_fail_closed():
    state = {"AAA_x": {**STATE["AAA_fake01"], "execution_cashflow": {}, "prev_price": None}}
    chains, issues = payoff.lego_chains(state, {"AAA_x": ROWS["AAA_fake01"]})
    assert chains == {} and "P_acted" in issues["AAA"]
    chains, issues = payoff.lego_chains({"AAA_x": STATE["AAA_fake01"]}, {})          # ไม่มีแถว → ไม่มี FIX_C
    assert chains == {} and "FIX_C" in issues["AAA"]
    v2 = {"AAA_x": {**STATE["AAA_fake01"],
                    "execution_cashflow": {**STATE["AAA_fake01"]["execution_cashflow"], "excess": 5.0}}}
    chains, issues = payoff.lego_chains(v2, {"AAA_x": ROWS["AAA_fake01"]})
    assert chains == {} and "R_basis" in issues["AAA"]


def test_lego_chains_fall_back_to_the_state_pointer_fields():
    pointer = {k: v for k, v in STATE["AAA_fake01"].items() if k != "execution_cashflow"}
    pointer.update(prev_price=9.0, prev_actual=-100.0, r_basis=-120.0, prev_excess=20.0)
    chains, issues = payoff.lego_chains({"AAA_x": pointer}, {"AAA_x": ROWS["AAA_fake01"]})
    assert not issues and chains["AAA"].p_acted == 9.0 and chains["AAA"].excess == 20.0


def test_lego_view_matches_the_state_at_p_acted():
    c = chain()
    df = payoff.lego_view_curves(c, np.array([c.p_acted]))
    assert df["A_floor"].iloc[0] == pytest.approx(c.actual, abs=1e-9)
    assert df["R"].iloc[0] == pytest.approx(c.r_basis, abs=1e-9)
    assert df["gap"].iloc[0] == pytest.approx(c.excess, abs=1e-9)


def test_lego_view_freezes_a_above_u_star_while_r_keeps_rising():
    c = chain()
    df = payoff.lego_view_curves(c, np.array([6.0, 12.0, 15.0, 20.0, 25.0]))
    assert np.allclose(df["gap"].iloc[:3], c.excess)                   # ถึง u* ช่องห่าง = Eₙ
    assert df["A_floor"].iloc[2] == df["A_floor"].iloc[3] == df["A_floor"].iloc[4]
    assert df["R"].is_monotonic_increasing and df["gap"].iloc[4] < df["gap"].iloc[3] < c.excess
    no_leaps = payoff.LegoChain(**{**c.__dict__, "leaps_shares": 0.0})
    assert math.isinf(no_leaps.u_star)
    free = payoff.lego_view_curves(no_leaps, np.array([20.0]))
    assert free["A_floor"].iloc[0] == pytest.approx(c.actual + c.fix_c * math.log(20 / c.p_acted))


def test_consistency_between_webull_and_the_bot():
    c = chain()
    assert payoff.check_consistency(legs(), c) is None
    near = [payoff.Leg(**{**legs()[0].__dict__, "qty": 60.005}), legs()[1]]
    assert payoff.check_consistency(near, c) is None
    off = [payoff.Leg(**{**legs()[0].__dict__, "qty": 59.5}), legs()[1]]
    assert "หุ้นจริงใน Webull 59.50 ≠ bot 60.00" in payoff.check_consistency(off, c)
    two = [legs()[0], payoff.Leg(**{**legs()[1].__dict__, "qty": 2.0})]
    assert "LEAPS ใน Webull 200 หุ้น ≠ bot 100" in payoff.check_consistency(two, c)


def test_exposure_uses_the_option_delta():
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, _ = payoff.build_books({"positions": POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    assert payoff.exposure_ratio(books[0]) == pytest.approx((60 + 0.8 * 100) * 9 / 1500)    # 0.84


def test_build_books_marks_missing_chain_and_mismatch_and_solves_iv_from_mark():
    books, notes = payoff.build_books({"positions": POSITIONS, "options": None}, {}, {}, NOW)
    book = books[0]
    assert book.usable_chain is None and "ไม่มี chain ของ bot" in book.chain_issue
    assert any("คำนวณจาก mark" in n for n in notes)
    call = book.options[0]
    assert call.iv_source == "mark" and payoff.bs_price(
        9.0, 7.0, payoff.years_to_expiry(call.expiry, NOW), payoff.RISK_FREE, call.iv, "call") \
        == pytest.approx(3.30, abs=1e-6)
    cheap = [STOCK, dict(CALL, last_price="0.50")]                      # ต่ำกว่า intrinsic: แก้ IV ไม่ได้
    books, _ = payoff.build_books({"positions": cheap, "options": None}, {}, {}, NOW)
    assert not books[0].options_priced and any("ไม่พอคำนวณ IV" in n for n in books[0].notes)
    mismatch = {"AAA_fake01": {**STATE["AAA_fake01"], "prev_holdings": 61.0}}
    chains, issues = payoff.lego_chains(mismatch, ROWS)
    books, _ = payoff.build_books({"positions": POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    assert books[0].usable_chain is None and "≠ bot 61.00" in books[0].chain_issue


def test_portfolio_lego_sums_usable_chains_only():
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, _ = payoff.build_books({"positions": POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    df = payoff.portfolio_lego(books, np.array([100.0]))
    single = payoff.lego_view_curves(chain(), np.array([9.0]))
    assert df["A_floor"].iloc[0] == pytest.approx(single["A_floor"].iloc[0])
    assert payoff.portfolio_lego([], np.array([100.0])) is None


# ---------------------------------------------------------------- แบบ 11 สมการ (แอป React)

def _eleven_book(state=STATE, positions=POSITIONS):
    chains, issues = payoff.lego_chains(state, ROWS)
    books, _ = payoff.build_books({"positions": positions, "options": OPTION_QUOTES}, chains, issues, NOW)
    return books[0]


def test_eleven_params_use_p0_and_the_dynamic_delta1():
    params, issue = payoff.eleven_params(_eleven_book())
    assert issue is None
    assert (params.x01, params.delta2, params.fix_c) == (10.0, 1.0, pytest.approx(1500.0))
    assert (params.option_value, params.stock_value) == (330.0, 540.0)       # 1 × 100 × 3.30 · 60 × 9
    assert params.delta1 == pytest.approx((330 + 540) / 1500)                  # 0.58


def test_eleven_lines_follow_equations_5_and_6():
    params, _ = payoff.eleven_params(_eleven_book())
    grid = np.array([5.0, 9.0, 10.0, 12.0, 20.0])
    df = payoff.eleven_curves(params, grid)
    y6 = 1500 * np.log(grid / 10.0)
    assert np.allclose(df["y6"], y6 * 1.0)                                     # y₆ = FIX_C·ln(x/x₀₁)·δ₂
    below = grid < 10.0
    assert np.allclose(df["y5"][below], y6[below] * params.delta1)             # ต่ำกว่า x₀₁ → δ₁
    assert np.allclose(df["y5"][~below], y6[~below] * 1.0)                     # x₀₁ ขึ้นไป → δ₂
    assert df.loc[df["price"] == 10.0, "y5"].iloc[0] == 0.0                    # ต่อเนื่องที่ x₀₁ (bias 0)


def test_eleven_net_is_the_sum_of_active_lines_without_the_benchmark():
    params, _ = payoff.eleven_params(_eleven_book())
    grid = np.array([6.0, 10.0, 14.0])
    on = payoff.eleven_curves(params, grid, active=["y5"])
    assert np.allclose(on["y3"], on["y5"])                                     # y₃ = Σ Active = y₅
    off = payoff.eleven_curves(params, grid, active=[])
    assert np.allclose(off["y3"], 0.0)                                         # ไม่มี Active → 0
    bench = payoff.eleven_curves(params, grid, active=["y5", "y6"])
    assert np.allclose(bench["y3"], bench["y5"])                               # y₆ ไม่ถูกรวมเสมอ


def test_eleven_fails_closed_without_the_bot_chain_or_p0():
    no_chain = _eleven_book(state={})
    assert payoff.eleven_params(no_chain)[0] is None
    no_p0 = {"AAA_fake01": {**STATE["AAA_fake01"], "p0": None}}
    params, issue = payoff.eleven_params(_eleven_book(state=no_p0))
    assert params is None and "P₀" in issue


def test_eleven_portfolio_sums_each_symbol_with_its_own_parameters():
    book = _eleven_book()
    df, skipped = payoff.eleven_portfolio([book], np.array([50.0, 100.0, 150.0]))
    params, _ = payoff.eleven_params(book)
    single = payoff.eleven_curves(params, book.spot * np.array([0.5, 1.0, 1.5]))
    assert not skipped and np.allclose(df["y5"], single["y5"]) and np.allclose(df["y3"], single["y3"])
    none, why = payoff.eleven_portfolio([_eleven_book(state={})], np.array([100.0]))
    assert none is None and why


def test_eleven_figure_draws_benchmark_lines_and_net_without_zones():
    params, _ = payoff.eleven_params(_eleven_book())
    df = payoff.eleven_curves(params, payoff.price_grid(9.0, 0, 200, extra=[10.0]))
    fig = payoff.eleven_figure(df, x01=10.0, spot=9.0, x_start=0.0, theme="dark")
    by_rank = sorted(fig.data, key=lambda trace: trace.legendrank)
    assert [t.name for t in by_rank] == [payoff.ELEVEN_NAMES[k] for k in ("y6", "y5", "y3")]   # legend
    assert [t.name for t in fig.data][-2:] == [payoff.ELEVEN_NAMES["y3"], payoff.ELEVEN_NAMES["y5"]]  # y₅ ทับ y₃
    assert fig.data[0].line.dash == payoff.REFERENCE_DASH                      # Benchmark เป็นเส้นประ
    assert not [shape for shape in fig.layout.shapes if shape.type == "rect"]  # model: ไม่มีโซนกำไร/ขาดทุน
    assert any("x₀₁ = P₀ 10.00" in (a.text or "") for a in fig.layout.annotations)
    hidden = payoff.eleven_figure(df, active=[], show_benchmark=False)
    assert [t.name for t in hidden.data] == [payoff.ELEVEN_NAMES["y3"]]


# ---------------------------------------------------------------- กราฟ

def test_payoff_figure_draws_only_finite_selected_lines():
    df = payoff.symbol_curves(legs(), 9.0, NOW, None, payoff.price_grid(9.0, 40, 200, extra=[7.0]))
    fig = payoff.payoff_figure(df, show=["t0", "expiry", "rebalance"], spot=9.0, now_pl=-220.0,
                               strikes=[7.0], u_star=15.0, theme="dark")
    names = [trace.name for trace in fig.data]
    assert payoff.LINE_NAMES["rebalance"] not in names                 # NaN ทั้งเส้น → ไม่วาด
    assert {payoff.LINE_NAMES["t0"], payoff.LINE_NAMES["expiry"], "P/L ตอนนี้"} <= set(names)
    assert fig.layout.paper_bgcolor == payoff.theme_tokens("dark")["surface"]
    only_t0 = payoff.payoff_figure(df, show=["t0"], theme="light")
    assert [t.name for t in only_t0.data if t.mode == "lines" and t.showlegend is not False] == \
        [payoff.LINE_NAMES["t0"]]


def _texts(fig) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def test_payoff_figure_is_interactive_and_self_explaining():
    c = chain()
    df = payoff.symbol_curves(legs(), 9.0, NOW, c, payoff.price_grid(9.0, 40, 200, extra=[7.0, 15.0]))
    fig = payoff.payoff_figure(df, spot=9.0, now_pl=-220.0, strikes=[7.0], u_star=15.0, theme="light",
                               title="AAA · Payoff ณ ตอนนี้", subtitle="S₀ 9.00 USD")
    layout = fig.layout
    # หัวการ์ด hover + แถว "ห่างจากราคาตอนนี้" (ไม่อยู่ใน legend) + ค่านำหน้าชื่อเส้น
    assert layout.hovermode == "x unified" and "ราคา" in layout.xaxis.unifiedhovertitle.text
    helper = fig.data[0]
    assert helper.showlegend is False and "ห่างจากราคาตอนนี้" in helper.hovertemplate
    assert helper.customdata[list(helper.x).index(9.0)] == pytest.approx(0.0)
    assert fig.data[1].hovertemplate.startswith("<b>%{y:,.2f}</b> USD  ①")
    # ปุ่มซูม: ±10/25/50% รอบ S₀ ตั้งทั้งแกน x และ y · y พอดีกับค่าที่เห็นในช่วงนั้น
    menu = layout.updatemenus[0]
    assert [b.label for b in menu.buttons] == ["±10%", "±25%", "±50%", "ทั้งหมด"]
    ten = menu.buttons[0].args[0]
    assert ten["xaxis.range"] == pytest.approx([8.1, 9.9])
    window = (df["price"] >= 8.1) & (df["price"] <= 9.9)
    visible = df.loc[window, list(payoff.LINE_KEYS)].to_numpy()
    assert ten["yaxis.range"][0] < visible.min() and ten["yaxis.range"][1] > max(0.0, visible.max())
    assert menu.buttons[-1].args[0]["xaxis.range"] == list(layout.xaxis.range)
    assert layout.xaxis.rangeslider.visible is True and list(layout.xaxis.rangeslider.range) == list(layout.xaxis.range)
    templates = [t.hovertemplate for t in fig.data] + [layout.yaxis.hoverformat]
    assert not any(":+" in (t or "") or (t or "").startswith("+") for t in templates)   # d3 + Plotly: ห้าม "+"
    # โซนกำไร/ขาดทุน (สี status) คู่ป้าย ▲/▼ · ป้ายจุดคุ้มทุนและ P/L ตอนนี้ · title/subtitle ในภาพ
    fills = {shape.fillcolor for shape in layout.shapes if shape.type == "rect"}
    assert fills == {payoff._rgba(payoff.STATUS["good"], payoff.ZONE_ALPHA),
                     payoff._rgba(payoff.STATUS["critical"], payoff.ZONE_ALPHA)}
    texts = _texts(fig)
    assert "▲ กำไร" in texts and "▼ ขาดทุน" in texts
    assert any(t.startswith("คุ้มทุน ①") for t in texts) and "ตอนนี้ <b>-220.00</b>" in texts
    assert layout.title.text == "<b>AAA · Payoff ณ ตอนนี้</b>" and layout.title.subtitle.text == "S₀ 9.00 USD"


def test_portfolio_figure_zooms_in_percent_points():
    chains, issues = payoff.lego_chains(STATE, ROWS)
    books, _ = payoff.build_books({"positions": POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    df = payoff.portfolio_curves(books, payoff.pct_grid(40, 200), NOW)
    fig = payoff.payoff_figure(df, x_col=payoff.PCT_COL, spot=100.0, now_pl=-220.0)
    assert fig.layout.updatemenus[0].buttons[1].args[0]["xaxis.range"] == pytest.approx([75.0, 125.0])
    assert not any(t.name == "Δ จาก S₀" for t in fig.data)          # แกนเป็น % อยู่แล้ว
    assert "%" in fig.layout.xaxis.unifiedhovertitle.text


def test_chart_config_names_the_png_and_resets_on_double_click():
    config = payoff.chart_config("payoff_AAA")
    assert config["toImageButtonOptions"] == {"format": "png", "scale": 2, "filename": "payoff_AAA"}
    assert config["doubleClick"] == "reset" and config["displaylogo"] is False


def test_line_four_is_line_three_plus_excess_with_the_band_between():
    c = chain()
    df = payoff.symbol_curves(legs(), 9.0, NOW, c, payoff.price_grid(9.0, 40, 200, extra=[7.0, 15.0]))
    assert np.allclose(df["rebalance_e"] - df["rebalance"], c.excess)
    fig = payoff.payoff_figure(df, show=payoff.LINE_KEYS, spot=9.0, now_pl=-220.0, theme="light")
    lines = {t.name: t for t in fig.data if t.mode == "lines"}
    four, three = lines[payoff.LINE_NAMES["rebalance_e"]], lines[payoff.LINE_NAMES["rebalance"]]
    assert four.line.color == three.line.color and four.line.dash == "dot"      # ตัวตนเดียวกับ ③
    assert four.fill == "tonexty" and three.fill is None                        # แถบ Eₙ ระหว่าง ③ กับ ④
    alone = payoff.payoff_figure(df, show=["t0", "rebalance_e"], theme="dark")
    assert next(t for t in alone.data if t.name == payoff.LINE_NAMES["rebalance_e"]).fill is None
    assert len([a for a in fig.layout.annotations if a.text.startswith(("①", "②", "③", "④"))]) == 4
    assert fig.layout.legend.traceorder == "normal"                             # ① ② ③ ④ ตามลำดับ


def test_zoom_buttons_stay_readable_in_dark_mode():
    df = payoff.symbol_curves(legs(), 9.0, NOW, chain(), payoff.price_grid(9.0))
    for theme in ("light", "dark"):
        menu = payoff.payoff_figure(df, spot=9.0, theme=theme).layout.updatemenus[0]
        # Plotly ทาสีปุ่มที่เลือก/ชี้เป็น #F4FAFF ตายตัว → ตัวหนังสือต้องเข้มบนพื้นอ่อนทุกธีม
        assert menu.font.color == payoff.MENU_COLORS["text"] == "#0b0b0b"
        assert menu.bgcolor == payoff.MENU_COLORS["bg"]


def test_lego_figure_has_r_and_a():
    c = chain()
    fig = payoff.lego_ledger_figure(payoff.lego_view_curves(c, payoff.price_grid(9.0)),
                                    p_acted=c.p_acted, u_star=c.u_star, spot=9.5)
    assert [t.name for t in fig.data] == [payoff.LEGO_NAMES["R"], payoff.LEGO_NAMES["A"]]
    assert not [shape for shape in fig.layout.shapes if shape.type == "rect"]   # model ledger ≠ P/L: ไม่มีโซน
    assert fig.layout.updatemenus[0].buttons[0].args[0]["xaxis.range"] == pytest.approx([8.55, 10.45])


# ---------------------------------------------------------------- Webull อ่านอย่างเดียว (HTTP ปลอม)

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeHttp:
    """แทน Response.get_response_object: ตอบตาม (method, path) และจดทุก request ที่ SDK ส่ง"""

    def __init__(self, routes=None, error: Exception | None = None):
        self.routes = {
            ("GET", "/trading/accounts/list"): (200, [{"account_id": "ACC-FAKE-1", "account_number": "N-1"}]),
            ("GET", "/trading/assets/positions/list"): (200, [dict(p, account_id="ACC-FAKE-1") for p in POSITIONS]),
            ("GET", "/market-data/stocks/snapshots/list"): (
                200, [{"symbol": "AAA", "price": "9.05", "trade_status": "TRADING", "quote_time": 1}]),
            ("GET", "/market-data/options/snapshots/list"): (200, [dict(OPTION_QUOTES[OCC], symbol=OCC)]),
            **(routes or {})}
        self.error = error
        self.calls: list[tuple[str, str, dict, str]] = []

    def __call__(self, response):
        url = response.get_url()
        method, path = response.get_method(), url.split("?")[0]
        self.calls.append((method, path, dict(response.get_headers()), url))
        if self.error is not None:
            raise self.error
        status, body = self.routes.get((method, path), (404, {"error_code": "NOT_FOUND"}))
        return status, {"x-request-id": "req-fake"}, json.dumps(body).encode(), _Resp(status, body)


@pytest.fixture
def http(monkeypatch):
    from webull.core.http.response import Response
    fake = FakeHttp()
    monkeypatch.setattr(Response, "get_response_object", lambda self: fake(self))
    monkeypatch.delenv("WEBULL_API_DEBUG", raising=False)
    return fake


def fetch(**kwargs):
    return webull_read.fetch_account_snapshot(FAKE_KEY, FAKE_SECRET, "th", TOKEN_TXT, now=NOW, **kwargs)


def test_fetch_reads_four_get_endpoints_with_the_token_header(http):
    snap = fetch()
    assert [(m, p) for m, p, _, _ in http.calls] == [
        ("GET", "/trading/accounts/list"), ("GET", "/trading/assets/positions/list"),
        ("GET", "/market-data/stocks/snapshots/list"), ("GET", "/market-data/options/snapshots/list")]
    assert {(m, p) for m, p, _, _ in http.calls} <= READ_ONLY_ALLOWLIST
    for _, _, headers, _ in http.calls:
        assert headers["x-access-token"] == FAKE_TOKEN and headers["x-app-key"] == FAKE_KEY
    assert snap["account_count"] == 1 and snap["as_of"] == "2026-11-17T15:00:00+00:00"
    assert all("account_id" not in p and "account_number" not in p for p in snap["positions"])
    assert snap["quotes"]["AAA"]["trade_status"] == "TRADING"
    assert snap["options"][OCC]["imp_vol"] == "0.80"


def test_fetch_with_v3_positions_quotes_the_underlying_and_the_option(http):
    http.routes[("GET", "/trading/assets/positions/list")] = (
        200, [dict(STOCK), dict(V3_CALL, symbol=OCC, legs=[dict(V3_CALL["legs"][0], symbol=OCC)])])
    snap = fetch()
    urls = {p: url for _, p, _, url in http.calls}
    assert "symbols=AAA&" in urls["/market-data/stocks/snapshots/list"]          # ไม่ส่ง OCC ไปขอราคาหุ้น
    assert f"symbols={OCC}&" in urls["/market-data/options/snapshots/list"]
    books, notes = payoff.build_books(snap, {}, {}, NOW)
    assert [leg.kind for leg in books[0].legs] == ["stock", "call"] and books[0].options[0].iv == 0.8


def _sdk_requests():
    """request ทุกคลาสของ SDK ที่สร้างได้โดยไม่ต้องมี argument (order v1–v3, account, data, token)"""
    from webull.core.request import ApiRequest
    found = {}
    for package in ("webull.trade.request", "webull.data.request", "webull.core.http.initializer"):
        module = importlib.import_module(package)
        for info in pkgutil.walk_packages(module.__path__, package + "."):
            mod = importlib.import_module(info.name)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, ApiRequest) and cls is not ApiRequest and cls.__module__ == mod.__name__:
                    try:
                        request = cls()
                    except TypeError:
                        continue
                    found[(str(request.get_method()).upper(), request.get_action_name())] = request
    return found


def test_r1_every_post_and_every_non_allowlisted_request_is_blocked_before_the_network(http):
    from webull.core.client import ApiClient
    guard = ReadOnlyGuard(ApiClient(FAKE_KEY, FAKE_SECRET, "th"))
    requests = _sdk_requests()
    posts = {key for key in requests if key[0] == "POST"}
    for must in [("POST", "/trading/orders/place"), ("POST", "/trading/orders/preview"),
                 ("POST", "/trading/orders/replace"), ("POST", "/trading/orders/cancel"),
                 ("POST", "/trading/orders/batch-place"), ("POST", "/openapi/trade/stock/order/place"),
                 ("POST", "/openapi/trade/option/order/place"), ("POST", "/trade/order/place"),
                 ("POST", "/auth/tokens/create"), ("POST", "/auth/tokens/check"),
                 ("POST", "/openapi/auth/token/refresh")]:
        assert must in posts, must
    blocked = 0
    for key, request in requests.items():
        if key in READ_ONLY_ALLOWLIST:
            continue
        with pytest.raises(WebullReadError, match="blocked"):
            guard.get_response(request)
        blocked += 1
    assert blocked >= len(posts) > 20
    assert http.calls == []                                            # ไม่มีอะไรออก network เลย
    assert ("GET", "/trading/orders/open-orders/list") in requests     # GET อื่นก็โดนกันด้วย
    assert not hasattr(guard, "_implementation_of_do_action") and not hasattr(guard, "__dict__")


def test_r2_the_wrapper_imports_no_order_or_token_module():
    source = (ROOT / "lego_webull_read.py").read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module}.{alias.name}" for alias in node.names]
    forbidden = ("trade_client", "data_client", "quotes_client", "token_manager", "token_operation",
                 "token_storage", "client_initializer", "order")
    assert not [name for name in imported if any(word in name.lower() for word in forbidden)], imported
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert '"POST"' not in code and "'POST'" not in code
    assert all(method == "GET" for method, _ in READ_ONLY_ALLOWLIST) and len(READ_ONLY_ALLOWLIST) == 4


def test_r3_no_token_is_created_or_checked(http, monkeypatch):
    from webull.core.http.initializer.client_initializer import ClientInitializer
    from webull.core.http.initializer.token.token_manager import TokenManager
    from webull.core.http.initializer.token.token_operation import TokenOperation
    from webull.core.http.initializer.token.token_storage import TokenStorage

    def forbidden(*args, **kwargs):
        raise AssertionError("token path must never run")

    for target, name in [(ClientInitializer, "initializer"), (ClientInitializer, "init_token"),
                         (TokenManager, "__init__"), (TokenManager, "init_token"),
                         (TokenOperation, "create_token"), (TokenOperation, "refresh_token"),
                         (TokenOperation, "check_token"), (TokenStorage, "__init__")]:
        monkeypatch.setattr(target, name, forbidden)
    assert fetch()["positions"]
    assert not any("/auth/" in path for _, path, _, _ in http.calls)


def test_r4_nothing_is_written_to_disk(http, monkeypatch, tmp_path):
    work, token_dir = tmp_path / "cwd", tmp_path / "token-dir"
    work.mkdir()
    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"write attempt: {file}")
        return real_open(file, mode, *args, **kwargs)

    def no_mkdir(*args, **kwargs):
        raise AssertionError("mkdir attempt")

    with monkeypatch.context() as m:
        m.chdir(work)
        m.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(token_dir))
        m.setattr(builtins, "open", read_only_open)
        m.setattr(io, "open", read_only_open)
        m.setattr(Path, "mkdir", no_mkdir)
        m.setattr(os, "mkdir", no_mkdir)
        m.setattr(os, "makedirs", no_mkdir)
        assert fetch()["positions"]
    assert list(work.iterdir()) == [] and not token_dir.exists()


def test_r5_a_rejected_token_stops_without_leaking_secrets(http, caplog):
    http.routes[("GET", "/trading/accounts/list")] = (401, {"error_code": "TOKEN_EXPIRED",
                                                            "message": f"bad token {FAKE_TOKEN}"})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(WebullReadError) as info:
        fetch()
    message = str(info.value)
    assert "token ใช้ไม่ได้ (HTTP 401 TOKEN_EXPIRED)" in message and "ไม่สร้าง token ใหม่" in message
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (FAKE_TOKEN, FAKE_KEY, FAKE_SECRET):
        assert secret not in message and secret not in logged
    assert [p for _, p, _, _ in http.calls] == ["/trading/accounts/list"]   # หยุดทันที ไม่ลองอย่างอื่น


def test_unexpected_errors_are_redacted(http):
    http.error = RuntimeError(f"boom {FAKE_TOKEN} x-app-key: {FAKE_KEY} app_secret={FAKE_SECRET}")
    with pytest.raises(WebullReadError) as info:
        fetch()
    message = str(info.value)
    assert "boom" in message and all(s not in message for s in (FAKE_TOKEN, FAKE_KEY, FAKE_SECRET))


def test_positions_error_and_schema_change_are_reported(http):
    http.routes[("GET", "/trading/assets/positions/list")] = (403, {"error_code": "FORBIDDEN"})
    with pytest.raises(WebullReadError, match="HTTP 403 \\(FORBIDDEN\\) ตอนอ่านpositions"):
        fetch()
    http.routes[("GET", "/trading/assets/positions/list")] = (
        200, [{"instrument_type": "EQUITY", "symbol": "AAA", "cost_price": "321.09"}])
    with pytest.raises(WebullReadError) as info:
        fetch()
    assert "รูปแบบข้อมูล Webull เปลี่ยน" in str(info.value) and "321.09" not in str(info.value)


def test_option_and_stock_snapshots_are_optional(http):
    http.routes[("GET", "/market-data/options/snapshots/list")] = (403, {"error_code": "NO_PERMISSION"})
    http.routes[("GET", "/market-data/stocks/snapshots/list")] = (500, {"error_code": "DOWN"})
    snap = fetch()
    assert snap["options"] is None and snap["quotes"] == {} and snap["positions"]
    assert any("ไม่มีสถานะตลาด" in n for n in snap["notes"])
    books, notes = payoff.build_books(snap, {}, {}, NOW)
    assert books[0].options[0].iv_source == "mark" and any("คำนวณจาก mark" in n for n in notes)


def test_guards_before_any_client_is_built(http, monkeypatch):
    with pytest.raises(WebullReadError, match="WEBULL_ENVIRONMENT = prod"):
        fetch(environment="uat")
    with pytest.raises(WebullReadError, match="3 บรรทัด"):
        webull_read.fetch_account_snapshot(FAKE_KEY, FAKE_SECRET, "th", FAKE_TOKEN)
    monkeypatch.setenv("WEBULL_API_DEBUG", "true")
    with pytest.raises(WebullReadError, match="WEBULL_API_DEBUG"):
        fetch()
    assert http.calls == []


def test_the_signer_uses_the_given_key_not_the_environment(http, monkeypatch):
    monkeypatch.setenv("WEBULL_APP_KEY_ID", "FAKE_TEST_OTHER_KEY")
    monkeypatch.setenv("WEBULL_APP_KEY_SECRET", "FAKE_TEST_OTHER_SECRET")
    fetch()
    assert {headers["x-app-key"] for _, _, headers, _ in http.calls} == {FAKE_KEY}


def test_sdk_logs_do_not_reach_the_root_logger(http):
    fetch()
    log = logging.getLogger("webull")
    assert log.propagate is False and log.level == logging.CRITICAL


def test_redact_hides_values_and_named_fields():
    text = (f"key {FAKE_KEY} secret {FAKE_SECRET} {{'x-access-token': 'abc123', 'x-signature': 'sig=='}} "
            "token=zzz999 แท็บนี้ไม่สร้าง token ใหม่")
    out = redact(text, [FAKE_KEY, FAKE_SECRET, "", "abc"])
    assert FAKE_KEY not in out and FAKE_SECRET not in out
    assert "abc123" not in out and "sig==" not in out and "zzz999" not in out
    assert "ไม่สร้าง token ใหม่" in out


# ---------------------------------------------------------------- หน้าเว็บ (AppTest)

WEBULL_SECRETS = {"WEBULL_APP_KEY": FAKE_KEY, "WEBULL_APP_SECRET": FAKE_SECRET,
                  "WEBULL_REGION_ID": "th", "WEBULL_ENVIRONMENT": "prod", "WEBULL_TOKEN_TXT": TOKEN_TXT}


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def snapshot(**extra) -> dict:
    return {"as_of": "2026-11-17T15:00:00+00:00", "positions": POSITIONS, "options": OPTION_QUOTES,
            "quotes": {"AAA": {"price": "9.05", "trade_status": "TRADING"}}, "account_count": 1,
            "notes": [], **extra}


def payoff_app(monkeypatch, *, snap=None, error=None, firebase=True) -> AppTest:
    def fake_fetch(*args, **kwargs):
        if error is not None:
            raise error
        return snap if snap is not None else snapshot()

    monkeypatch.setattr(webull_read, "fetch_account_snapshot", fake_fetch)
    secrets = dict(WEBULL_SECRETS)
    if firebase:
        payloads = {"webull_lego_state": STATE, "webull_lego_rows/run_fake_1": ROWS["AAA_fake01"],
                    "webull_lego_rows/run_fake_0": ROWS["AAA_older"], "webull_lego_rows": {},
                    "webull_lego_order_audit": {}}

        class _Reference:
            def __init__(self, path, url=None):
                self.path = path

            def get(self):
                return payloads.get(self.path)

        monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
        monkeypatch.setattr(firebase_admin.db, "reference", _Reference)
        secrets.update({"FIREBASE_SA_JSON": "{}",
                        "FIREBASE_DB_URL": "https://lego-firebase-prod.mock.test"})
    app = AppTest.from_file(APP_PATH)
    app.secrets.update(secrets)
    return app


def test_without_webull_secrets_the_tab_only_explains():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    assert not app.exception and not app.error and not app.warning
    assert any("WEBULL_APP_KEY" in info.value for info in app.info)
    assert len(app.get("plotly_chart")) == 1                         # มีแค่กราฟของแท็บ 101


def test_the_tab_draws_real_positions_and_the_lego_view(monkeypatch):
    app = payoff_app(monkeypatch)
    app.run(timeout=30)
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert len(app.get("plotly_chart")) == 2                         # 101 + Payoff
    labels = {m.label for m in app.metric}
    assert {"S₀ (USD)", "P/L ตอนนี้", "หุ้นจริง", "FIX_C (bot)", "exposure จริง"} <= labels
    assert next(m for m in app.metric if m.label == "exposure จริง").value == "84.0%"
    assert app.slider(key="payoff_range").value == (0, 200)                              # ค่าเริ่มต้น 0 ถึง 2 × S₀
    assert next(m for m in app.metric if m.label == "P/L ตอนนี้").value == "-220.00"
    assert any(c.value.startswith("④ = ③ + Eₙ สะสมของ bot (+20.00 USD)") for c in app.caption)
    assert len(app.button_group(key="payoff_lines").options) == 4
    app.button_group(key="payoff_view").set_value(payoff_view_lego()).run(timeout=30)
    assert not app.exception and not app.error
    lego = {m.label: m.value for m in app.metric}
    assert lego["Aₙ สะสม"] == "-100.00" and lego["R_basis"] == "-120.00" and lego["Eₙ สะสม"] == "+20.00"
    assert len(app.tabs) == 7                                        # 5 แท็บ + Summary/Diagrams


def payoff_view_lego() -> str:
    import streamlit_app
    return streamlit_app.PAYOFF_VIEW_LEGO


def test_the_portfolio_view_sums_every_symbol(monkeypatch):
    other = dict(STOCK, symbol="BBB", quantity="10", last_price="20", unrealized_profit_loss="15.50")
    app = payoff_app(monkeypatch, snap=snapshot(positions=POSITIONS + [other]))
    app.run(timeout=30)
    app.button_group(key="payoff_symbol").set_value("รวมพอร์ต").run(timeout=30)
    assert not app.exception and not app.error
    assert next(m for m in app.metric if m.label == "P/L ตอนนี้ (รวม)").value == "-204.50"
    assert any("BBB: ไม่มี chain ของ bot" in c.value for c in app.caption)


def test_v3_snapshot_shows_the_option_and_all_lines(monkeypatch):
    app = payoff_app(monkeypatch, snap=snapshot(positions=V3_POSITIONS))
    app.run(timeout=30)
    assert not app.exception and not app.error and not app.warning
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Option"] == "1×C7" and metrics["P/L ตอนนี้"] == "-220.00"
    assert metrics["FIX_C (bot)"] == "1500.00" and metrics["exposure จริง"] == "84.0%"
    assert not any("ซ่อนเส้น" in c.value for c in app.caption)


def test_a_skipped_position_is_a_warning_at_the_top(monkeypatch):
    broken = {k: v for k, v in V3_CALL.items() if k != "legs"}
    app = payoff_app(monkeypatch, snap=snapshot(positions=[STOCK, broken]))
    app.run(timeout=30)
    assert not app.exception
    assert len(app.warning) == 1 and app.warning[0].value.startswith("P/L ไม่ครบ: ข้าม position ที่อ่านไม่ได้")
    assert "มี field: cost_price, currency" in app.warning[0].value and "-160.00" not in app.warning[0].value


def test_the_reload_guard_knows_the_v3_parser():
    import streamlit_app
    assert {"parse_occ", "underlying_symbols", "SKIP_PREFIX"} <= set(streamlit_app._REQUIRED_PAYOFF_API)
    assert webull_read.SNAPSHOT_SCHEMA >= 2


def test_the_eleven_equation_view(monkeypatch):
    import streamlit_app
    app = payoff_app(monkeypatch)
    app.run(timeout=30)
    app.button_group(key="payoff_view").set_value(streamlit_app.PAYOFF_VIEW_ELEVEN).run(timeout=30)
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["x₀₁ = P₀"] == "10.00" and metrics["δ₁ (ขาลง)"] == "0.580" and metrics["δ₂ (ขาขึ้น)"] == "1.00"
    expected = 1500 * math.log(9.0 / 10.0) * 0.58
    assert metrics["y₃ ที่ราคาตอนนี้"] == f"{expected:+,.2f}"
    assert len(app.get("plotly_chart")) == 2                                    # 101 + แบบ 11 สมการ
    assert any(c.value.startswith("สมการที่ 3 (Net): y₃ = Σ(เส้นที่ Active)") for c in app.caption)


def test_a_webull_error_stays_inside_the_tab(monkeypatch):
    app = payoff_app(monkeypatch, error=WebullReadError("token ใช้ไม่ได้ (HTTP 401 X) — แท็บนี้ไม่สร้าง token ใหม่"))
    app.run(timeout=30)
    assert not app.exception
    assert [e.value for e in app.error] == ["ดึงข้อมูล Webull ไม่ได้ — token ใช้ไม่ได้ (HTTP 401 X) — "
                                            "แท็บนี้ไม่สร้าง token ใหม่"]
    assert len(app.tabs) >= 7 and len(app.number_input) == 6          # แท็บอื่นยังอยู่ครบ


def test_an_unexpected_error_shows_only_its_type(monkeypatch):
    app = payoff_app(monkeypatch, error=RuntimeError(f"leak {FAKE_TOKEN}"))
    app.run(timeout=30)
    assert [e.value for e in app.error] == ["ดึงข้อมูล Webull ไม่ได้ (RuntimeError)"]


def test_without_bot_state_line_three_is_hidden_with_a_reason(monkeypatch):
    app = payoff_app(monkeypatch, firebase=False)
    app.run(timeout=30)
    assert not app.exception and not app.error
    captions = [c.value for c in app.caption]
    assert any("ซ่อนเส้น ③" in c for c in captions)
    assert any("ไม่มี chain ของ bot" in c for c in captions)
    assert next(m for m in app.metric if m.label == "FIX_C (bot)").value == "—"
    assert any("ซ่อนเส้น ③, ④" in c for c in captions)
    assert not any(c.startswith("④ = ③ + Eₙ") for c in captions)
