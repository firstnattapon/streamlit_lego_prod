"""แถว n = 0 ต้องเริ่มจาก 0: R₀ = ΔA₀ = A₀ = E₀ = 0 และแถวถัดไปต่อจาก 0 นั้น

fill แรกของ chain มี slippage (quote → fill). ถ้านับเป็น ΔA₀ = FIX_C × (fill / quote − 1) จะกลายเป็น "กำไร"
ที่ค้างอยู่ใน Aₙ/Eₙ ทุกแถว — backend ใส่ baseline ศูนย์ (initial_funding_zero_v1) เฉพาะบัญชี holdings = 0
จึงมี chain ที่เริ่มด้วย LEAPS (holdings ≠ 0) ที่ DB บันทึก ΔA₀ ≠ 0 ไว้ (fixtures/paper_session_leaps.json ของ
backend เองก็เป็นแบบนี้). dashboard "แสดง" n = 0 เริ่มจาก 0 (genesis_baseline=True) โดยไม่แก้ DB และไม่เปลี่ยน
โหมดดิบที่ integrity_report ใช้.

ค่าคาดหวังคำนวณแยกอิสระในไฟล์นี้ (recurrence ของ backend v3 จาก events) ไม่ได้ย้อนจากโค้ด dashboard.
ข้อมูลทั้งหมดสังเคราะห์: ไม่มี run_id/chain_key จริงของบัญชี (repo เป็น public และ gitleaks จับชื่อ chain แบบ symbol+hex)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import firebase_admin
import firebase_admin.db  # noqa: F401  (submodule ที่แอปอ่านผ่าน)
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import lego_dash_core
import test_dash_core as dash
import test_execution_confirmed_contract as ec
from lego_dash_core import (COLUMN_ORDER, EXECUTION_CONFIRMED_SEMANTICS,
                            EXECUTION_TERMINAL_FROZEN_V2, EXECUTION_TERMINAL_FUNDING_V3,
                            FUNDING_BASELINE_POLICY, LEDGER_COLS, MONEY_COLS,
                            count_ledger_corrections, genesis_baseline_info, integrity_report,
                            ledger_explanations, normalize_genesis_baseline, order_columns,
                            recompute_gated_ledger, rows_to_df, table_tooltips)
from lego_hover_table import build_hover_table_html

ROOT = Path(__file__).parent
R, DA, A, E = LEDGER_COLS
FIX_C, P0, FILL = 1500.0, 10.19, 10.245
SLIP = FIX_C * (FILL / P0 - 1.0)            # ΔA₀ ที่ backend บันทึก (นับ slippage เป็นกำไร) ≈ 8.0962
BASE = pd.Timestamp("2026-11-03T14:30:09Z")
DELAY = pd.Timedelta(seconds=56)           # สังเกต decision → fill finalize (ตามรูปแบบ export จริง)
CHAIN = "SYN_leaps_chain"


def _f(x: float) -> str:
    return f"{x:,.4f}"


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def v3_chain(events, *, semantics=EXECUTION_TERMINAL_FUNDING_V3, step0=1040) -> pd.DataFrame:
    """chain ย่อรูปเดียวกับ export จริง: LEAPS 100 หุ้นตั้งแต่แรก (holdings ≠ 0 จึงไม่เข้า initial funding),
    แถวแรกคือ BUY ที่ fill เกิน quote (slippage) แล้วบันทึก ΔA₀ = FIX_C × (fill / P₀ − 1)

    ``events``: ``[{"price": quote, "fill": ราคา fill หรือ None, "at": วินาทีนับจาก BASE}]``
    ledger ที่ persist = recurrence ของ backend v3 (fill ใหม่ใช้ราคา fill ก่อนหน้าเป็น P_acted;
    แถวไม่ fill แช่แข็งที่ A ของ fill ล่าสุดที่ finalize แล้วตอนสังเกต)
    """
    obs = [BASE + pd.Timedelta(seconds=e["at"]) for e in events]
    fills, prev_exec, actual = [], P0, 0.0
    for k, e in enumerate(events):
        if e["fill"] is not None:
            delta = FIX_C * (e["fill"] / prev_exec - 1.0)
            fills.append({"k": k, "fill": e["fill"], "prev_exec": prev_exec, "prev_A": actual,
                          "delta": delta, "A": actual + delta, "at": obs[k] + DELAY,
                          "R": FIX_C * math.log(e["price"] / P0)})
            actual += delta
            prev_exec = e["fill"]

    rows, holdings, seq = [], 100.0, 0
    for k, e in enumerate(events):
        price = e["price"]
        value = holdings * price
        gap = value - FIX_C
        ref = FIX_C * math.log(price / P0)
        own = next((f for f in fills if f["k"] == k), None)
        seen = [f for f in fills if f["at"] <= obs[k] and f["k"] != k]
        if own:
            a_row, basis, delta = own["A"], own["R"], own["delta"]
            side, qty = ("BUY", round(-gap / price, 2)) if gap < 0 else ("SELL", round(gap / price, 2))
            status, action = f"READY_{side}", "TRIGGER_ACTION"
        else:
            a_row, basis, delta = (seen[-1]["A"], seen[-1]["R"], 0.0) if seen else (0.0, 0.0, 0.0)
            side, qty, status, action = "", 0.0, "PASS_THRESHOLD", "PASS"
        row = dict(zip(COLUMN_ORDER, [
            _iso(obs[k]), "SYN", status, step0 + k, 1, price, holdings, action, side, status, qty,
            value, gap, ref, delta, a_row, a_row - basis]))
        row.update(
            run_id=f"syn{k + 1:029d}", chain_key=CHAIN, version=k + 1, committed=True, semantics=semantics,
            market_slot_id=f"2026-11-03:{k}", market_ordinal=step0 + k, clock_mode="market", schema_version=2,
            ledger_version_at_observation=semantics,
            E_mark_at_observation=(seen[-1]["A"] if seen else 0.0) - ref, R_basis=basis,
            real_holdings=holdings - 100.0, leaps_shares=100.0,
            cashflow_status="FINALIZED" if own else "NO_ACTION")
        if own:
            seq += 1
            holdings += qty if side == "BUY" else -qty
            row.update(finalized_seq=seq, execution_price=own["fill"], execution_quantity=qty,
                       cashflow_finalized_at=_iso(own["at"]), previous_action_price=own["prev_exec"],
                       previous_actual_cumulative=own["prev_A"], post_execution_holdings=holdings,
                       post_execution_real_holdings=holdings - 100.0)
        rows.append(row)
    return rows_to_df({r_["run_id"]: r_ for r_ in rows})


def pass_events(prices, start=900, step=900):
    return [{"price": p, "fill": None, "at": start + i * step} for i, p in enumerate(prices)]


GENESIS = {"price": P0, "fill": FILL, "at": 0}
PASS_PRICES = [10.47, 10.355, 10.42, 10.32, 10.33, 10.35, 10.38]
PLAIN = [GENESIS, *pass_events(PASS_PRICES)]                      # เหมือน export จริง: BUY แล้ว PASS 7 แถว
WITH_SELL = [GENESIS, *pass_events([10.47]),                      # PASS แล้ว SELL (fill ต่ำกว่า quote) แล้ว PASS
             {"price": 10.62, "fill": 10.59, "at": 2700}, *pass_events([10.4, 10.5], start=3600)]


def _stored(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _display(df: pd.DataFrame, p0: float | None = P0) -> pd.DataFrame:
    return recompute_gated_ledger(df, p0=p0, genesis_baseline=True)


# ---- ค่า: n = 0 เริ่มจาก 0 และแถวถัดไปต่อจาก 0 -----------------------------

def test_the_fixture_chain_is_a_consistent_v3_chain_that_books_the_slippage():
    df = v3_chain(PLAIN)
    assert df.loc[0, DA] == pytest.approx(SLIP) and SLIP == pytest.approx(8.096173, abs=1e-6)
    assert df[A].astype(float).tolist() == pytest.approx([SLIP] * len(df))    # PASS ทุกแถวค้างที่ค่านี้
    assert df[E].astype(float).tolist() == pytest.approx([SLIP] * len(df))
    report, ok = integrity_report(df, p0_hint=P0)
    assert ok, report.to_string()                       # ข้อมูลสอดคล้องกันเองตามสัญญา v3: ไม่ใช่ข้อมูลเสีย


def test_raw_mode_is_unchanged_and_integrity_reads_the_raw_rows():
    df = v3_chain(WITH_SELL)
    before = _stored(df)
    raw = recompute_gated_ledger(df, p0=P0)
    assert raw.loc[0, A] == pytest.approx(SLIP)         # โหมดดิบคงค่าที่ DB บันทึก
    assert count_ledger_corrections(df, raw) == 0
    pd.testing.assert_frame_equal(df, before)           # ไม่แก้ input
    assert integrity_report(df, p0_hint=P0)[1]
    info = genesis_baseline_info(df, P0)
    assert info["shift"] == pytest.approx(SLIP) and info["frozen"] is True
    assert (info["p_quote"], info["p_fill"], info["fix_c"]) == (P0, FILL, FIX_C)


def test_display_mode_starts_every_ledger_column_at_zero_and_carries_on_from_it():
    df = v3_chain(PLAIN)
    before = _stored(df)
    shown = _display(df)
    pd.testing.assert_frame_equal(df, before)           # dashboard เป็น read-only
    assert [shown.loc[0, c] for c in (R, DA, A, E)] == [0.0, 0.0, 0.0, 0.0]
    for i in range(1, len(shown)):                      # PASS ทั้งหมด: ไม่มี act ใหม่ จึงยังเป็นศูนย์
        assert shown.loc[i, DA] == 0.0 and shown.loc[i, A] == 0.0 and shown.loc[i, E] == 0.0
        assert shown.loc[i, R] == pytest.approx(FIX_C * math.log(df.loc[i, "ราคา Pₙ (USD)"] / P0))
    # ตารางปัด 2 ตำแหน่ง: ต้องไม่มี "-0.00"
    for col in (DA, A, E):
        assert {f"{float(v):.2f}" for v in shown[col]} == {"0.00"}


def test_e_mark_stays_a_minus_r_after_the_baseline_is_removed():
    shown = _display(v3_chain(PLAIN))
    assert shown.loc[0, "E_mark_at_observation"] == 0.0          # แถว 0 สังเกตก่อน fill finalize
    for i in range(1, len(shown)):
        # E_mark = A ที่มองเห็นตอนสังเกต (= A ของแถวก่อนหน้า) − R ของแถวนี้
        assert shown.loc[i, "E_mark_at_observation"] == pytest.approx(shown.loc[i - 1, A] - shown.loc[i, R])


def test_later_fills_keep_their_own_deltas_and_accumulate_from_zero():
    df = v3_chain(WITH_SELL)
    sell = int(df.index[df["สถานะ"] == "READY_SELL"][0])
    shown = _display(df)
    expected_delta = FIX_C * (10.59 / FILL - 1.0)       # P_acted = ราคา fill แรก (10.245) ไม่ใช่ quote
    assert shown.loc[sell, DA] == pytest.approx(expected_delta) == pytest.approx(df.loc[sell, DA])
    assert shown.loc[sell, A] == pytest.approx(expected_delta)  # สะสมจาก 0 ไม่รวม slippage แถวแรก
    assert shown.loc[sell, "previous_actual_cumulative"] == 0.0
    r_sell = FIX_C * math.log(10.62 / P0)
    assert shown.loc[sell, E] == pytest.approx(expected_delta - r_sell)          # E = A − R ณ fill
    for i in range(sell + 1, len(shown)):               # PASS หลัง fill: A ค้าง, E = A − R_basis (R ที่แถว fill)
        assert shown.loc[i, A] == pytest.approx(expected_delta)
        assert shown.loc[i, E] == pytest.approx(expected_delta - r_sell)
    for i in range(1, sell):
        assert shown.loc[i, A] == 0.0 and shown.loc[i, E] == 0.0


def test_a_row_observed_before_the_first_fill_finalizes_is_not_shifted():
    # แถว 1 สังเกตที่ +30 วินาที ก่อน fill แรก finalize (+56 วินาที): A ที่บันทึก = 0 ไม่รวม slip อยู่แล้ว
    events = [GENESIS, {"price": 10.30, "fill": None, "at": 30}, *pass_events([10.47])]
    df = v3_chain(events)
    assert df.loc[1, A] == 0.0 and df.loc[2, A] == pytest.approx(SLIP)
    assert integrity_report(df, p0_hint=P0)[1]
    shown = _display(df)
    assert shown.loc[1, A] == 0.0 and shown.loc[1, E] == 0.0   # ห้ามกลายเป็น −SLIP
    assert shown.loc[2, A] == 0.0 and shown.loc[2, E] == 0.0
    assert shown.loc[1, "E_mark_at_observation"] == df.loc[1, "E_mark_at_observation"]   # ยังไม่เห็น fill
    assert shown.loc[2, "E_mark_at_observation"] == pytest.approx(
        df.loc[2, "E_mark_at_observation"] - SLIP)


def test_v2_frozen_semantics_is_rebased_the_same_way():
    df = v3_chain(PLAIN, semantics=EXECUTION_TERMINAL_FROZEN_V2)
    shown = _display(df)
    assert [shown.loc[0, c] for c in (DA, A, E)] == [0.0, 0.0, 0.0]
    assert (shown[A] == 0.0).all() and (shown[E] == 0.0).all()


# ---- guard: rebase เฉพาะสิ่งที่อธิบายได้ ----------------------------------

def test_guards_leave_unexplained_or_already_funded_or_clipped_rows_alone():
    base = v3_chain(PLAIN)

    wrong = base.copy()
    wrong.loc[0, DA] = SLIP + 1.0                                # ΔA₀ ไม่ตรงสูตร slippage -> ให้ integrity จับ
    assert genesis_baseline_info(wrong, P0) is None
    assert normalize_genesis_baseline(wrong, P0)[0] is wrong
    assert _display(wrong).loc[0, DA] == pytest.approx(SLIP + 1.0)

    funded = base.copy()                                         # แถวแรกเป็น initial funding แล้ว: ΔA₀ = 0
    funded["initial_funding"] = [True] + [False] * (len(funded) - 1)
    funded["model_baseline_policy"] = FUNDING_BASELINE_POLICY
    funded["funding_reference_offset"] = 0.0
    funded.loc[0, [DA, A, E]] = 0.0
    funded.loc[1:, [A, E]] = funded.loc[1:, [A, E]].astype(float) - SLIP
    funded.loc[1:, "previous_actual_cumulative"] = funded.loc[1:, "previous_actual_cumulative"]
    assert genesis_baseline_info(funded, P0) is None
    assert normalize_genesis_baseline(funded, P0)[0] is funded

    no_time = base.copy()
    no_time = no_time.drop(columns=["cashflow_finalized_at"])    # ไม่รู้ว่าแถวไหนเห็น fill แรกแล้ว
    assert genesis_baseline_info(no_time, P0) is None

    clipped = base.iloc[3:].reset_index(drop=True)               # ตัดหน้า: แถวแรกไม่ใช่ genesis
    assert genesis_baseline_info(clipped, P0) is None

    bad_seed = genesis_baseline_info(base, P0 + 0.5)             # previous_action_price ≠ P₀ ของ state
    assert bad_seed is None


def test_a_chain_funded_from_flat_is_untouched_in_both_modes():
    paper = json.loads((ROOT / "fixtures" / "paper_session.json").read_text(encoding="utf-8"))
    df = rows_to_df(paper["webull_lego_rows"])
    p0 = next(iter(paper["webull_lego_state"].values()))["p0"]
    assert genesis_baseline_info(df, p0) is None
    assert normalize_genesis_baseline(df, p0)[0] is df
    pd.testing.assert_frame_equal(_display(df, p0), recompute_gated_ledger(df, p0=p0))


# ---- fixture LEAPS ของ backend: รูปเดียวกับ chain จริง -------------------------

LEAPS = json.loads((ROOT / "fixtures" / "paper_session_leaps.json").read_text(encoding="utf-8"))
LEAPS_P0 = next(iter(LEAPS["webull_lego_state"].values()))["p0"]


def test_backend_leaps_fixture_books_the_first_fill_slippage_and_is_rebased():
    df = rows_to_df(LEAPS["webull_lego_rows"])
    slip = 3000.0 * (65.51 / 65.49 - 1.0)
    assert "initial_funding" not in df.columns or df["initial_funding"].isna().all()
    assert df.loc[0, DA] == pytest.approx(slip)         # backend เองบันทึก slippage ของ fill แรกเป็นกำไร
    info = genesis_baseline_info(df, LEAPS_P0)
    assert info["shift"] == pytest.approx(slip) and info["fix_c"] == pytest.approx(3000.0)

    shown = _display(df, LEAPS_P0)
    assert [shown.loc[0, c] for c in (R, DA, A, E)] == [0.0, 0.0, 0.0, 0.0]
    # A′ = ผลรวม ΔA ของ fill ที่ไม่ใช่ fill แรก (คำนวณแยกจากคอลัมน์ ΔA ที่บันทึก); E′ = A′ − R_basis
    running = df[DA].astype(float).copy()
    running.iloc[0] = 0.0
    assert shown[A].astype(float).tolist() == pytest.approx(running.cumsum().tolist(), abs=1e-9)
    assert shown[E].astype(float).tolist() == pytest.approx(
        (shown[A].astype(float) - df["R_basis"].astype(float)).tolist(), abs=1e-9)
    assert shown[DA].astype(float).iloc[1:].tolist() == pytest.approx(df[DA].astype(float).iloc[1:].tolist())
    for i in range(1, len(shown)):
        assert shown.loc[i, "E_mark_at_observation"] == pytest.approx(shown.loc[i - 1, A] - shown.loc[i, R], abs=1e-9)
    # ข้อมูลที่ backend เขียนยังผ่านการตรวจสมการ (อ่านแถวดิบ) และโหมดดิบไม่แก้อะไร
    assert integrity_report(df, p0_hint=LEAPS_P0)[1]
    assert count_ledger_corrections(df, recompute_gated_ledger(df, p0=LEAPS_P0)) == 0


# ---- genesis ตัดสินด้วย version == 1 ไม่ใช่ DNA step == 0 อย่างเดียว -------------

def test_a_market_clock_chain_starting_at_a_nonzero_step_is_a_genesis_not_a_clipped_window():
    df = rows_to_df(dash._mk_chain(dash.PRICES, dash.HOLD, dash.SIGNALS))
    df["DNA step"] = df["DNA step"] + 133                       # นาฬิกาตลาด: step เริ่ม 133 แต่ version เริ่ม 1
    df[R] = 0.0                                                 # จำลอง engine ที่แช่ Rₙ ไว้
    fixed = recompute_gated_ledger(df)                          # ไม่มี p0 จาก state: P₀ ต้องมาจากแถว genesis
    expected = [dash.FIX_C * math.log(p / dash.PRICES[0]) for p in dash.PRICES]
    assert fixed[R].astype(float).tolist() == pytest.approx(expected)
    tips = ledger_explanations(df)
    assert "แถว genesis" in tips[1][R]
    assert "ถูกตัดหน้า" not in tips[0][DA] and "genesis" in tips[0][DA]
    # ชุดที่ถูกตัดหน้าจริง (version > 1) ยังไม่ใช่ genesis
    clipped = df.iloc[2:].reset_index(drop=True)
    assert "ถูกตัดหน้า" in ledger_explanations(clipped, p0=dash.PRICES[0])[0][DA]


def _exec_v1_chain(first_step: int = 0) -> pd.DataFrame:
    """execution_confirmed_v1 (ไม่ frozen): BUY แรก fill 101 เทียบ quote 100 แล้ว PASS แล้ว fill ที่สอง"""
    fix_c, p0, fill0 = ec.FIX_C, ec.P0, 101.0
    d0 = fix_c * (fill0 / p0 - 1.0)                             # backend v1 นับ slippage ที่ genesis
    sem = EXECUTION_CONFIRMED_SEMANTICS
    p1, p2, fill2 = 103.0, 104.5, 104.0
    d2 = fix_c * (fill2 / fill0 - 1.0)
    r2 = fix_c * math.log(p2 / p0)
    rows = [
        ec._row(first_step, 100.0, 10.0, 1, semantics=sem, R=0.0, dA=d0, A=d0, E=d0,
                cashflow_status="FINALIZED", execution_price=fill0, execution_quantity=5.0),
        ec._row(first_step + 1, p1, 15.0, 0, semantics=sem, R=fix_c * math.log(p1 / p0), dA=0.0, A=d0,
                E=d0 - fix_c * math.log(fill0 / p0), cashflow_status="NO_ACTION"),
        ec._row(first_step + 2, p2, 15.0, 1, semantics=sem, R=r2, dA=d2, A=d0 + d2, E=d0 + d2 - r2,
                cashflow_status="FINALIZED", execution_price=fill2, execution_quantity=1.0),
    ]
    df = pd.DataFrame(rows)
    df["version"] = [1, 2, 3]
    return df


@pytest.mark.parametrize("first_step", [0, 133])
def test_execution_confirmed_genesis_fill_books_no_pnl_in_display_mode(first_step):
    df = _exec_v1_chain(first_step)
    fix_c, p0, fill0 = ec.FIX_C, ec.P0, 101.0
    raw = recompute_gated_ledger(df, p0=p0)
    assert raw.loc[0, DA] == pytest.approx(fix_c * (fill0 / p0 - 1.0))       # โหมดดิบ = ที่ backend บันทึก
    assert count_ledger_corrections(df, raw) == 0
    info = genesis_baseline_info(df, p0)
    assert info["frozen"] is False and info["shift"] == pytest.approx(fix_c * 0.01)

    shown = _display(df, p0)
    assert [shown.loc[0, c] for c in (R, DA, A, E)] == [0.0, 0.0, 0.0, 0.0]
    assert shown.loc[1, A] == 0.0                                            # PASS แช่แข็งที่ 0
    assert shown.loc[1, E] == pytest.approx(-fix_c * math.log(fill0 / p0))   # P_acted = ราคา fill (101)
    d2 = fix_c * (104.0 / fill0 - 1.0)                                       # fill ถัดไปเทียบราคา fill แรก
    assert shown.loc[2, DA] == pytest.approx(d2) and shown.loc[2, A] == pytest.approx(d2)
    assert shown.loc[2, E] == pytest.approx(d2 - fix_c * math.log(104.5 / p0))
    tips = ledger_explanations(df, p0=p0, genesis_baseline=True)
    assert f"{_f(fix_c)} × ({_f(100.0)} / {_f(100.0)} − 1)" in tips[0][DA]
    assert "fill จริง 101.0000" in tips[0][DA] and "ค่าที่ DB บันทึก" in tips[0][A]


# ---- tooltip: สูตรแทนค่าที่ผู้ใช้ขอ + ค่าที่ DB บันทึก ---------------------------

def test_genesis_tooltips_show_the_p0_over_p0_formulas_and_what_was_removed():
    df = v3_chain(WITH_SELL)
    tips = ledger_explanations(df, p0=P0, genesis_baseline=True)
    formula = f"{_f(FIX_C)} × ({_f(P0)} / {_f(P0)} − 1)"        # 1,500.0000 × (10.1900 / 10.1900 − 1)
    assert formula in tips[0][DA] and formula in tips[0][A]
    assert f"ln({_f(P0)} / {_f(P0)})" in tips[0][R]             # 1,500.0000 × ln(10.1900 / 10.1900)
    assert "Eₙ = Aₙ − Rₙ" in tips[0][E] and "Aₙ = 0 — เริ่มนับสะสมจากศูนย์" in tips[0][A]
    assert f"fill จริง {_f(FILL)}" in tips[0][DA] and f"P_acted ของ act ถัดไป = {_f(FILL)}" in tips[0][DA]
    assert f"= {_f(SLIP)} (นับ slippage เป็นกำไร)" in tips[0][DA]   # ที่มาของค่าที่ DB บันทึก
    for col in (DA, A, E):
        assert f"ค่าที่ DB บันทึก {_f(SLIP)} − slippage ของ fill แรกที่ตัดออก {_f(SLIP)} = {_f(0.0)}" in tips[0][col]
    assert "ค่าที่ DB บันทึก" not in tips[1][DA]                # ΔA ของแถวที่ไม่ใช่ genesis ไม่ถูกแตะ
    assert "ค่าที่ DB บันทึก" in tips[1][A] and "(rebase n = 0)" in tips[1][E]


def test_every_tooltip_contains_the_value_the_display_shows():
    for events in (PLAIN, WITH_SELL):
        df = v3_chain(events)
        shown = _display(df)
        tips = ledger_explanations(df, p0=P0, genesis_baseline=True)
        assert len(tips) == len(df)
        for i in range(len(df)):
            for col in LEDGER_COLS:
                value, text = float(shown.at[i, col]), tips[i][col]
                assert text.strip() and (_f(value) in text or (value == 0.0 and "= 0" in text)), (i, col, text)
                assert "ถูกตัดหน้า" not in text
                for clash in (" − -", " + -", "× -"):           # ติดลบต้องมีวงเล็บเสมอ
                    assert clash not in text, (i, col, text)


def test_tooltip_rows_after_a_later_fill_use_the_rebased_operands():
    df = v3_chain(WITH_SELL)
    sell = int(df.index[df["สถานะ"] == "READY_SELL"][0])
    delta = FIX_C * (10.59 / FILL - 1.0)
    tips = ledger_explanations(df, p0=P0, genesis_baseline=True)
    a_text = tips[sell][A]
    assert "Aₙ = previous_actual_cumulative + ΔAₙ" in a_text
    assert f"= {_f(0.0)} + {_f(delta)}" in a_text and f"= {_f(delta)}" in a_text   # prev′ = 0 ไม่ใช่ 8.0962


def test_default_mode_tooltips_are_exactly_what_they_were():
    df = v3_chain(PLAIN)
    tips = ledger_explanations(df, p0=P0)
    assert tips == ledger_explanations(df, p0=P0, genesis_baseline=False)
    assert f"({_f(FILL)} / {_f(P0)} − 1)" in tips[0][DA] and "ค่าที่ DB บันทึก" not in tips[0][DA]
    # chain LEAPS เติม tooltip คอลัมน์ 7 (หุ้นจริง + LEAPS) เพิ่ม; ส่วน Rₙ/ΔAₙ/Aₙ/Eₙ ต้องเหมือนเดิมทุกตัวอักษร
    merged = table_tooltips(df, p0=P0)[0]
    assert [{k: v for k, v in tip.items() if k in LEDGER_COLS} for tip in merged] == tips


def test_the_hover_table_shows_a_zero_first_row_without_negative_zero():
    df = v3_chain(PLAIN)
    shown = _display(df)
    tips, columns = table_tooltips(df, p0=P0, genesis_baseline=True)
    show = order_columns(shown.copy())
    for col in MONEY_COLS:
        show[col] = show[col].astype(float).round(2)
    page = build_hover_table_html(show, tips, columns, MONEY_COLS)
    assert "-0.00" not in page
    assert page.count('data-tip="') >= len(df) * len(LEDGER_COLS)
    assert "1,500.0000 × (10.1900 / 10.1900 − 1)" in page


# ---- แอป: ป้ายแจ้ง + ตัวเลขเริ่มจาก 0 ----------------------------------------

def _app(monkeypatch, df: pd.DataFrame) -> AppTest:
    rows = {str(r["run_id"]): {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                               for k, v in r.to_dict().items()} for _, r in df.iterrows()}
    payloads = {"webull_lego_rows": rows,
                "webull_lego_state": {CHAIN: {"p0": P0, "updated_at": "2026-11-03T16:30:00Z"}},
                "webull_lego_order_audit": {}}

    class _Reference:
        def __init__(self, path):
            self.path = path

        def get(self):
            return payloads.get(self.path)

    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", _Reference)
    page = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    page.secrets.update({"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL": "https://mock.firebaseio.test"})
    return page


def test_the_page_says_the_table_was_rebased_and_starts_the_numbers_from_zero(monkeypatch):
    page = _app(monkeypatch, v3_chain(PLAIN))
    page.run(timeout=30)
    assert not page.exception
    banners = [i.value for i in page.info if "แสดงแบบเริ่มจาก 0" in i.value]
    assert len(banners) == 1
    assert f"ΔA₀ = {SLIP:,.4f}" in banners[0] and "10.2450" in banners[0] and "10.1900" in banners[0]
    assert "ไม่แก้ DB" in banners[0] and "Integrity check" in banners[0]
    assert {m.label: m.value for m in page.metric}["Eₙ สะสม (USD)"] == "0.00"
    # ledger ที่ DB บันทึกยังสอดคล้องกันเอง: integrity ผ่าน และไม่ขึ้นคำเตือน "engine เขียนผิด"
    assert any("ผ่านการตรวจสมการ" in s.value for s in page.success)
    assert not any("engine เขียน ledger" in w.value for w in page.warning)


def test_a_chain_funded_from_flat_shows_no_rebase_banner(monkeypatch):
    paper = json.loads((ROOT / "fixtures" / "paper_session.json").read_text(encoding="utf-8"))
    df = rows_to_df(paper["webull_lego_rows"])
    page = _app(monkeypatch, df)
    page.run(timeout=30)
    assert not page.exception
    assert not any("แสดงแบบเริ่มจาก 0" in i.value for i in page.info)


def test_hot_deploy_reloads_a_core_that_predates_the_genesis_baseline(monkeypatch):
    """core ที่แคชไว้ก่อนมี genesis_baseline_info ต้องถูกโหลดใหม่ ไม่ทำให้หน้าล่ม"""
    assert callable(lego_dash_core.genesis_baseline_info)
    monkeypatch.delattr(lego_dash_core, "genesis_baseline_info")
    page = AppTest.from_file(str(ROOT / "streamlit_app.py"))
    page.run(timeout=20)
    assert not page.exception
    assert callable(getattr(lego_dash_core, "genesis_baseline_info", None))
