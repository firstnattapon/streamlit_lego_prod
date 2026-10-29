import math

import pandas as pd
import pytest

from lego_dash_core import (COLUMN_ORDER, EXECUTION_TERMINAL_FROZEN_V2,
                            EXECUTION_TERMINAL_FUNDING_V3, FUNDING_BASELINE_POLICY, integrity_report,
                            recompute_gated_ledger)


def funding_row():
    row = dict(zip(COLUMN_ORDER, [
        "2026-09-15T14:16:09Z", "XYZ", "READY_BUY", 133, 1,
        27.34, 0, "TRIGGER_ACTION", "BUY", "READY_BUY", 182,
        0, -5000, 0, 0, 0, 0]))
    row.update(version=1, committed=True, market_ordinal=133,
               semantics=EXECUTION_TERMINAL_FROZEN_V2, cashflow_status="FINALIZED",
               execution_price=27.37, execution_quantity=182,
               finalized_seq=1, R_basis=0, previous_action_price=27.34,
               previous_actual_cumulative=0, initial_funding=True,
               model_baseline_policy=FUNDING_BASELINE_POLICY,
               funding_reference_offset=0)
    return row


def test_funding_zero_and_next_terminal_fill_pass_independent_check():
    first = funding_row()
    second = dict(first)
    delta = 5000 * (28 / 27.37 - 1)
    reference = 5000 * math.log(28 / 27.34)
    second.update({"version": 2, "DNA step": 134, "market_ordinal": 134,
                   "ราคา Pₙ (USD)": 28, "จำนวนถือครอง (หุ้น)": 182,
                   "มูลค่าพอร์ต (USD)": 5096, "ส่วนต่างเป้าหมาย (USD)": 96,
                   "สถานะ": "READY_SELL", "ฝั่ง": "SELL", "จำนวนสั่ง (หุ้น)": 3,
                   "Rₙ อ้างอิง (USD)": reference, "ΔAₙ ต่อสเต็ป (USD)": delta,
                   "Aₙ สะสม (USD)": delta, "Eₙ ส่วนเกินสะสม (USD)": delta - reference,
                   "execution_price": 28, "execution_quantity": 3, "finalized_seq": 2,
                   "R_basis": reference, "initial_funding": False,
                   "previous_action_price": 27.37})
    assert integrity_report(pd.DataFrame([first, second]))[1]


@pytest.mark.parametrize("field,value", [
    ("ΔAₙ ต่อสเต็ป (USD)", 5.48646671543529), ("Aₙ สะสม (USD)", 99),
    ("Eₙ ส่วนเกินสะสม (USD)", 99), ("execution_quantity", 0),
    ("execution_price", float("nan")), ("R_basis", 9),
    ("initial_funding", False), ("previous_action_price", -1),
    ("model_baseline_policy", "unknown"),
])
def test_corrupted_persisted_funding_row_cannot_pass_by_comparing_to_itself(field, value):
    row = funding_row()
    row[field] = value
    df = pd.DataFrame([row])
    assert not integrity_report(df)[1]


def test_historical_first_buy_drift_is_reported_and_not_rewritten():
    row = funding_row()
    for key in ("initial_funding", "model_baseline_policy", "funding_reference_offset",
                "previous_action_price", "previous_actual_cumulative"):
        row.pop(key)
    for col in ("ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"):
        row[col] = 5000 * (27.37 / 27.34 - 1)
    df = pd.DataFrame([row])
    assert not integrity_report(df)[1]
    assert recompute_gated_ledger(df, p0=27.34)["Aₙ สะสม (USD)"].iloc[0] == row["Aₙ สะสม (USD)"]


def test_truncated_history_without_execution_basis_is_not_certified():
    row = funding_row()
    row.update(version=23, initial_funding=False)
    row.pop("previous_action_price")
    row.pop("previous_actual_cumulative")
    assert not integrity_report(pd.DataFrame([row]), p0_hint=27.34)[1]


def test_unfilled_row_with_nonzero_delta_is_detected():
    row = funding_row()
    row.update({"สถานะ": "PASS_THRESHOLD", "จำนวนสั่ง (หุ้น)": 0,
                "cashflow_status": "NO_ACTION", "ΔAₙ ต่อสเต็ป (USD)": 1})
    row.pop("execution_price")
    row.pop("execution_quantity")
    assert not integrity_report(pd.DataFrame([row]))[1]


def test_v3_funding_contract_is_recognized_and_preserved():
    row = funding_row()
    row["semantics"] = EXECUTION_TERMINAL_FUNDING_V3
    df = pd.DataFrame([row])
    assert integrity_report(df)[1]
    pd.testing.assert_frame_equal(recompute_gated_ledger(df, p0=27.34), df, check_dtype=False)


def test_pass_cannot_move_a_and_e_together_even_when_basis_still_matches():
    first = funding_row()
    passed = dict(first)
    passed.update({"version": 2, "DNA step": 134, "market_ordinal": 134,
                   "สถานะ": "PASS_THRESHOLD", "จำนวนสั่ง (หุ้น)": 0,
                   "cashflow_status": "NO_ACTION", "Aₙ สะสม (USD)": 10, "Eₙ ส่วนเกินสะสม (USD)": 10})
    passed.pop("execution_price")
    passed.pop("execution_quantity")
    assert not integrity_report(pd.DataFrame([first, passed]))[1]
