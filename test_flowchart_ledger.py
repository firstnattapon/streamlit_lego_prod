"""ΔAₙ / Aₙ / Eₙ ต้องตามผังรอบเดียว และไม่มีคอลัมน์ "ΔAₙ เงินจริง (USD)" อีกต่อไป

ผัง (ขั้นตอน 5–6):  ΔA = Fix_c × (VWAP / P_acted − 1);  A_new = A_old + ΔA;
E_new = A_new − R;  P_acted = VWAP;  PASS -> A, E, P_acted คงเดิม

เงินสด broker (qty × price − fee) เก็บแยกที่ webull_lego_broker_cashflow และไม่เข้าสูตรนี้
แถวเก่าใน RTDB อาจยังมีคีย์ของคอลัมน์ที่เลิกใช้ค้างอยู่ — dashboard ต้องตัดทิ้งและไม่ใช้
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from lego_dash_core import (COLUMN_ORDER, GATED_SEMANTICS, LEGACY_DROPPED_COLS,
                            MONEY_COLS, integrity_report, order_columns,
                            recompute_gated_ledger, rows_to_df)

FIX_C = 1500.0
P0 = 100.0
RETIRED = "ΔAₙ เงินจริง (USD)"


def _row(step, price, status, qty, holdings=15.0, **extra):
    value = holdings * price
    row = {
        "เวลา (UTC)": f"2026-07-23T18:{step:02d}:00Z", "สินทรัพย์": "AAPL",
        "สถานะ": status, "DNA step": step, "DNA signal": 1,
        "ราคา Pₙ (USD)": price, "จำนวนถือครอง (หุ้น)": holdings,
        "คำสั่ง": "TRIGGER_ACTION" if qty else "PASS",
        "ฝั่ง": "SELL" if status == "READY_SELL" else "",
        "เหตุผล": status, "จำนวนสั่ง (หุ้น)": qty,
        "มูลค่าพอร์ต (USD)": value, "ส่วนต่างเป้าหมาย (USD)": value - FIX_C,
        "Rₙ อ้างอิง (USD)": 0.0, "ΔAₙ ต่อสเต็ป (USD)": 0.0,
        "Aₙ สะสม (USD)": 0.0, "Eₙ ส่วนเกินสะสม (USD)": 0.0,
        "run_id": f"r{step}", "chain_key": "AAPL_x", "version": step + 1,
        "committed": True, "semantics": GATED_SEMANTICS,
    }
    row.update(extra)
    return row


def _chain(**row1_extra):
    # genesis @100 -> act @102 (ตัวอย่างในผัง: ΔA = 1500 × 0.02 = 30) -> PASS @101
    return [
        _row(0, 100.0, "PASS_THRESHOLD", 0.0),
        _row(1, 102.0, "READY_SELL", 0.2941, **row1_extra),
        _row(2, 101.0, "PASS_THRESHOLD", 0.0),
    ]


def test_contract_has_17_columns_and_no_cash_column():
    assert len(COLUMN_ORDER) == 17
    assert RETIRED not in COLUMN_ORDER
    assert RETIRED not in MONEY_COLS
    assert LEGACY_DROPPED_COLS == (RETIRED,)


def test_act_then_pass_follows_flowchart_step_6():
    fixed = recompute_gated_ledger(pd.DataFrame(_chain()), p0=P0)
    dA = fixed["ΔAₙ ต่อสเต็ป (USD)"].astype(float).tolist()
    A = fixed["Aₙ สะสม (USD)"].astype(float).tolist()
    E = fixed["Eₙ ส่วนเกินสะสม (USD)"].astype(float).tolist()
    R = fixed["Rₙ อ้างอิง (USD)"].astype(float).tolist()

    assert dA == pytest.approx([0.0, 30.0, 0.0])
    assert A == pytest.approx([0.0, 30.0, 30.0])
    # act: E = A_new − R ที่ราคา act
    assert E[1] == pytest.approx(30.0 - FIX_C * math.log(102.0 / P0))
    # PASS: A, E (และ P_acted) คงเดิม แม้ R ยังวิ่งตามราคา
    assert E[2] == pytest.approx(E[1])
    assert R[2] == pytest.approx(FIX_C * math.log(101.0 / P0))


def test_cash_column_left_on_an_old_row_never_reaches_a_or_e():
    # เงินสดจริงของแถว act = 30.00 − fee 0.25 ; ต้องไม่ถูกสะสมเป็น A
    with_cash = pd.DataFrame(_chain(**{RETIRED: 29.75}))
    without = pd.DataFrame(_chain())
    a = recompute_gated_ledger(with_cash, p0=P0)
    b = recompute_gated_ledger(without, p0=P0)
    for col in ("ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"):
        assert a[col].astype(float).tolist() == pytest.approx(b[col].astype(float).tolist())
    assert a["Aₙ สะสม (USD)"].astype(float).iloc[1] == pytest.approx(30.0)


def test_integrity_e5_accumulates_the_model_delta_only():
    persisted = recompute_gated_ledger(pd.DataFrame(_chain()), p0=P0)
    persisted[RETIRED] = [0.0, 29.75, 0.0]              # เงินสดต่างจาก ΔA เพราะ fee
    report, _ = integrity_report(persisted, p0_hint=P0)
    checks = {r["ข้อ"]: bool(r["ผ่าน"]) for r in report.to_dict(orient="records")}
    assert checks["E4"] and checks["E5"] and checks["E6"]


def test_readers_drop_the_retired_column_from_old_rows():
    rows = {f"k{n}": r for n, r in enumerate(_chain(**{RETIRED: 29.75}))}
    for r in rows.values():
        r.setdefault(RETIRED, 0.0)
    df = rows_to_df(rows)
    assert RETIRED not in df.columns

    dirty = pd.DataFrame(_chain(**{RETIRED: 29.75}))
    dirty[RETIRED] = dirty[RETIRED].fillna(0.0)
    assert RETIRED in dirty.columns
    shown = order_columns(dirty)
    assert RETIRED not in shown.columns
    assert list(shown.columns)[:len(COLUMN_ORDER)] == COLUMN_ORDER
