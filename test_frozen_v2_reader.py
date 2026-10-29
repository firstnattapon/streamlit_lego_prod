import pandas as pd

from lego_dash_core import (COLUMN_ORDER, EXECUTION_TERMINAL_FROZEN_V2,
                            pending_broker_fee_count, recompute_gated_ledger)


def test_v2_reader_never_rewrites_frozen_e_or_late_finalization():
    row = {column: 0 for column in COLUMN_ORDER}
    row.update({
        "ราคา Pₙ (USD)": 112.0,
        "สถานะ": "PASS_THRESHOLD",
        "DNA step": 2,
        "DNA signal": 1,
        "จำนวนสั่ง (หุ้น)": 0.0,
        "มูลค่าพอร์ต (USD)": 1500.0,
        "ส่วนต่างเป้าหมาย (USD)": 0.0,
        "Rₙ อ้างอิง (USD)": 169.999,
        "ΔAₙ ต่อสเต็ป (USD)": 0.0,
        "Aₙ สะสม (USD)": 165.0,
        "Eₙ ส่วนเกินสะสม (USD)": 22.0347302935,
        "semantics": EXECUTION_TERMINAL_FROZEN_V2,
        "cashflow_status": "NO_ACTION",
        "E_mark_at_observation": -4.999,
    })
    source = pd.DataFrame([row])
    fixed = recompute_gated_ledger(source, p0=100.0)
    assert fixed.loc[0, "ΔAₙ ต่อสเต็ป (USD)"] == 0.0
    assert fixed.loc[0, "Aₙ สะสม (USD)"] == 165.0
    assert fixed.loc[0, "Eₙ ส่วนเกินสะสม (USD)"] == 22.0347302935


def test_reader_surfaces_unknown_broker_fee_as_pending_not_zero():
    audit = pd.DataFrame([
        {"run_id": "a", "broker_fee_status": "PENDING",
         "broker_cash_cumulative": "-200.25"},
        {"run_id": "b", "broker_fee_status": "KNOWN",
         "filled_fee": 0.0},
    ])
    assert pending_broker_fee_count(audit) == 1

