"""Cross-repo contract fixtures for lego-firebase ``execution_confirmed_v1``.

Fixture shape mirrors the 17-column rows plus metadata written by backend commit
9959c5b175e93441fb9e1112eb035f17873e4d36. Expected cashflow is calculated
independently here so dashboard code cannot make its own tests pass by rewriting
both the input and expectation with the same helper.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dna_engine import dna_fingerprint
from lego_dash_core import (COLUMN_ORDER, EXECUTION_CONFIRMED_SEMANTICS,
                            GATED_SEMANTICS, integrity_report, order_columns,
                            recompute_gated_ledger)


FIX_C = 1500.0
P0 = 100.0
DNA_CONTRACT = "26021034252903219354832053493"


def _decision(price: float, holdings: float, signal: int):
    value = holdings * price
    gap = value - FIX_C
    if signal == 0:
        return "PASS_DNA_ZERO", "PASS", "", 0.0, value, gap
    if abs(gap) <= 0.0:
        return "PASS_THRESHOLD", "PASS", "", 0.0, value, gap
    side = "SELL" if gap > 0 else "BUY"
    return (f"READY_{side}", "TRIGGER_ACTION", side,
            round(abs(gap) / price, 5), value, gap)


def _row(step: int, price: float, holdings: float, signal: int, *,
         semantics: str, R: float, dA: float, A: float, E: float,
         cashflow_status: str | None = None,
         execution_price: float | None = None,
         execution_quantity: float | None = None) -> dict:
    status, action, side, qty, value, gap = _decision(price, holdings, signal)
    row = {
        "เวลา (UTC)": f"2026-08-01T14:{step:02d}:00Z",
        "สินทรัพย์": "AAPL",
        "สถานะ": status,
        "DNA step": step,
        "DNA signal": signal,
        "ราคา Pₙ (USD)": price,
        "จำนวนถือครอง (หุ้น)": holdings,
        "คำสั่ง": action,
        "ฝั่ง": side,
        "เหตุผล": status,
        "จำนวนสั่ง (หุ้น)": qty,
        "มูลค่าพอร์ต (USD)": value,
        "ส่วนต่างเป้าหมาย (USD)": gap,
        "Rₙ อ้างอิง (USD)": R,
        "ΔAₙ ต่อสเต็ป (USD)": dA,
        "Aₙ สะสม (USD)": A,
        "Eₙ ส่วนเกินสะสม (USD)": E,
        "run_id": f"exec{step:028d}",
        "chain_key": "AAPL_execution_contract",
        "version": step + 1,
        "committed": True,
        "semantics": semantics,
    }
    if cashflow_status is not None:
        row["cashflow_status"] = cashflow_status
    if execution_price is not None:
        row["execution_price"] = execution_price
    if execution_quantity is not None:
        row["execution_quantity"] = execution_quantity
    return row


def _execution_fixture() -> pd.DataFrame:
    """NO_ACTION, two pending rows, two fills and quote/fill slippage."""
    rows = []
    actual = 0.0
    last_execution_price = P0

    # Genesis PASS: quote reference and all cashflow values are zero.
    rows.append(_row(
        0, 100.0, 15.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=0.0, dA=0.0, A=0.0, E=0.0, cashflow_status="NO_ACTION"))

    # READY intent not filled: pending freezes at P0.
    p = 105.0
    rows.append(_row(
        1, p, 10.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=FIX_C * math.log(p / P0), dA=0.0, A=actual,
        E=actual - FIX_C * math.log(last_execution_price / P0),
        cashflow_status="PENDING_EXECUTION"))

    # A rejected broker order is recorded in audit/outbox. The committed model
    # row remains PENDING_EXECUTION and frozen because it has no finalized fill.
    p = 106.0
    rows.append(_row(
        2, p, 10.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=FIX_C * math.log(p / P0), dA=0.0, A=actual,
        E=actual - FIX_C * math.log(last_execution_price / P0),
        cashflow_status="PENDING_EXECUTION"))

    # Quote is 110 but actual fill is 108: recurrence must use 108.
    p, fill = 110.0, 108.0
    delta = FIX_C * (fill / last_execution_price - 1.0)
    actual += delta
    last_execution_price = fill
    reference = FIX_C * math.log(p / P0)
    rows.append(_row(
        3, p, 10.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=reference, dA=delta, A=actual, E=actual - reference,
        cashflow_status="FINALIZED", execution_price=fill,
        execution_quantity=2.5))

    # PASS after fill freezes against last execution price 108, while R follows quote 95.
    p = 95.0
    rows.append(_row(
        4, p, 15.0, 0, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=FIX_C * math.log(p / P0), dA=0.0, A=actual,
        E=actual - FIX_C * math.log(last_execution_price / P0),
        cashflow_status="NO_ACTION"))

    # Second confirmed fill uses prior execution price, not prior quote.
    p, fill = 90.0, 92.0
    delta = FIX_C * (fill / last_execution_price - 1.0)
    actual += delta
    last_execution_price = fill
    reference = FIX_C * math.log(p / P0)
    rows.append(_row(
        5, p, 20.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=reference, dA=delta, A=actual, E=actual - reference,
        cashflow_status="FINALIZED", execution_price=fill,
        execution_quantity=1.0))
    return pd.DataFrame(rows)


def test_execution_fixture_recomputes_without_changing_17_column_contract():
    source = _execution_fixture()
    fixed = recompute_gated_ledger(source, p0=P0)
    for col in ("Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)",
                "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"):
        assert np.allclose(fixed[col].astype(float), source[col].astype(float), atol=1e-9)
    assert list(order_columns(fixed).columns)[:len(COLUMN_ORDER)] == COLUMN_ORDER
    for col in ("สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)"):
        assert list(fixed[col]) == list(source[col])


def test_execution_integrity_checks_v3_rows_instead_of_skipping_them_green():
    report, ok = integrity_report(_execution_fixture(), p0_hint=P0)
    assert ok, report.to_string(index=False)
    e4 = report.loc[report["ข้อ"] == "E4"].iloc[0]
    assert "FINALIZED execution-price" in e4["สมการ/กฎ"]
    assert "ข้าม" not in str(e4["หมายเหตุ"])


def test_missing_or_unknown_semantics_fail_closed_instead_of_skipping_green():
    variants = [
        _execution_fixture().drop(columns=["semantics"]),
        _execution_fixture().assign(semantics="future_cashflow_v99"),
    ]
    for source in variants:
        report, ok = integrity_report(source, p0_hint=P0)
        assert not ok, report.to_string(index=False)
        for check in ("E4", "E6"):
            row = report.loc[report["ข้อ"] == check].iloc[0]
            assert bool(row["ผ่าน"]) is False
            assert "fail closed" in str(row["หมายเหตุ"])


def test_pending_rows_freeze_and_corruption_is_detected():
    source = _execution_fixture()
    broken = source.copy(deep=True)
    broken.loc[1, "ΔAₙ ต่อสเต็ป (USD)"] = 75.0
    broken.loc[1, "Aₙ สะสม (USD)"] = 75.0
    broken.loc[2, "ΔAₙ ต่อสเต็ป (USD)"] = 15.0
    broken.loc[2, "Aₙ สะสม (USD)"] = 90.0
    assert not integrity_report(broken, p0_hint=P0)[1]

    fixed = recompute_gated_ledger(broken, p0=P0)
    assert list(fixed.loc[1:2, "ΔAₙ ต่อสเต็ป (USD)"].astype(float)) == [0.0, 0.0]
    assert list(fixed.loc[1:2, "Aₙ สะสม (USD)"].astype(float)) == [0.0, 0.0]


def test_unknown_row_cashflow_status_fails_closed():
    broken = _execution_fixture()
    broken.loc[2, "cashflow_status"] = "REJECTED"

    report, ok = integrity_report(broken, p0_hint=P0)

    assert not ok, report.to_string(index=False)
    e4 = report.loc[report["ข้อ"] == "E4"].iloc[0]
    assert "execution provenance ผิด" in str(e4["หมายเหตุ"])


@pytest.mark.parametrize(("row_index", "cashflow_status"), [
    (0, "PENDING_EXECUTION"),  # PASS must be NO_ACTION
    (1, "NO_ACTION"),         # READY must be pending or finalized
])
def test_cashflow_status_must_match_committed_decision(
        row_index, cashflow_status):
    broken = _execution_fixture()
    broken.loc[row_index, "cashflow_status"] = cashflow_status

    report, ok = integrity_report(broken, p0_hint=P0)

    assert not ok, report.to_string(index=False)
    assert "execution provenance ผิด" in str(
        report.loc[report["ข้อ"] == "E4", "หมายเหตุ"].iloc[0])


def test_cashflow_semantics_downgrade_fails_closed():
    broken = _execution_fixture()
    broken.loc[4:, "semantics"] = GATED_SEMANTICS
    broken = recompute_gated_ledger(broken, p0=P0)

    report, ok = integrity_report(broken, p0_hint=P0)

    assert not ok, report.to_string(index=False)
    assert "semantics เดินถอยหลัง" in str(
        report.loc[report["ข้อ"] == "E4", "หมายเหตุ"].iloc[0])


def test_finalized_requires_status_quantity_and_price_fail_closed():
    source = _execution_fixture()
    for column, value in (("execution_price", None),
                          ("execution_quantity", 0.0),
                          ("cashflow_status", "PENDING_EXECUTION")):
        broken = source.copy(deep=True)
        broken.loc[3, column] = value
        report, ok = integrity_report(broken, p0_hint=P0)
        assert not ok, f"{column} must fail provenance:\n{report}"
        assert "execution provenance ผิด" in str(
            report.loc[report["ข้อ"] == "E4", "หมายเหตุ"].iloc[0])


def test_finalized_slippage_uses_execution_price_but_R_uses_quote():
    fixed = recompute_gated_ledger(_execution_fixture(), p0=P0)
    row = fixed.iloc[3]
    expected_delta = FIX_C * (108.0 / 100.0 - 1.0)
    decision_price_delta = FIX_C * (110.0 / 100.0 - 1.0)
    assert math.isclose(float(row["ΔAₙ ต่อสเต็ป (USD)"]), expected_delta)
    assert not math.isclose(float(row["ΔAₙ ต่อสเต็ป (USD)"]), decision_price_delta)
    assert math.isclose(float(row["Rₙ อ้างอิง (USD)"]), FIX_C * math.log(110.0 / P0))
    assert float(row["Eₙ ส่วนเกินสะสม (USD)"]) < 0.0  # slippage may make E negative


def test_mixed_historical_gated_then_execution_resets_A_and_keeps_action_anchor():
    gated0 = _row(
        0, 100.0, 15.0, 1, semantics=GATED_SEMANTICS,
        R=0.0, dA=0.0, A=0.0, E=0.0)
    p = 104.0
    d = FIX_C * (p / P0 - 1.0)
    R = FIX_C * math.log(p / P0)
    gated1 = _row(
        1, p, 10.0, 1, semantics=GATED_SEMANTICS,
        R=R, dA=d, A=d, E=d - R)

    # Migration resets A to zero, but P_acted seed remains last gated act (104).
    p = 106.0
    execution2 = _row(
        2, p, 10.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=FIX_C * math.log(p / P0), dA=0.0, A=0.0,
        E=-FIX_C * math.log(104.0 / P0), cashflow_status="PENDING_EXECUTION")
    p, fill = 110.0, 108.0
    delta = FIX_C * (fill / 104.0 - 1.0)
    execution3 = _row(
        3, p, 10.0, 1, semantics=EXECUTION_CONFIRMED_SEMANTICS,
        R=FIX_C * math.log(p / P0), dA=delta, A=delta,
        E=delta - FIX_C * math.log(p / P0), cashflow_status="FINALIZED",
        execution_price=fill, execution_quantity=2.0)
    source = pd.DataFrame([gated0, gated1, execution2, execution3])

    fixed = recompute_gated_ledger(source, p0=P0)
    assert np.allclose(fixed["Aₙ สะสม (USD)"], source["Aₙ สะสม (USD)"], atol=1e-9)
    report, ok = integrity_report(source, p0_hint=P0)
    assert ok, report.to_string(index=False)
    assert "รอยต่อ" in str(report.loc[report["ข้อ"] == "E5", "หมายเหตุ"].iloc[0])


def test_numpy_and_dna_fingerprint_match_backend_provenance():
    requirements = Path(__file__).with_name("requirements.txt").read_text(encoding="utf-8")
    assert "numpy==2.4.6" in requirements
    assert "google-cloud-firestore==2.21.0" in requirements
    assert "protobuf==5.29.6" in requirements
    assert np.__version__ == "2.4.6"
    assert dna_fingerprint(DNA_CONTRACT) == "5cd66c244b3a72ce"

