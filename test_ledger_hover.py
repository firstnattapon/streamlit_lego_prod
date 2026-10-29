"""ที่มาของตัวเลข Rₙ/ΔAₙ/Aₙ/Eₙ ตอนชี้เมาส์: ข้อความต้องตรงกับค่าที่ตารางแสดงและสูตรจริง

ค่าคาดหวังมาจาก fixture ที่คำนวณแยกอิสระ (test_dash_core._mk_chain และ
test_execution_confirmed_contract._execution_fixture) ไม่ได้ย้อนจากโค้ด dashboard
"""
from __future__ import annotations

import math
from pathlib import Path

import firebase_admin
import pandas as pd
import pytest
from firebase_admin import db
from streamlit.testing.v1 import AppTest

from lego_dash_core import (EXECUTION_CONFIRMED_SEMANTICS, LEDGER_COLS,
                            ledger_explanations, recompute_gated_ledger, rows_to_df)
from lego_hover_table import (MAX_HEIGHT, MIN_HEIGHT, build_hover_table_html,
                              hover_table_height)
from test_dash_core import FIX_C, HOLD, PRICES, SIGNALS, _mk_chain
from test_execution_confirmed_contract import P0, _execution_fixture
from test_v2_integrity_evidence import funding_row

R, DA, A, E = LEDGER_COLS


def _f(x: float) -> str:
    return f"{x:,.4f}"


def _o(x: float) -> str:
    """ตัวถูกดำเนินการติดลบถูกครอบด้วยวงเล็บ"""
    return f"({_f(x)})" if x < 0 else _f(x)


def _gated() -> pd.DataFrame:
    return rows_to_df(_mk_chain(PRICES, HOLD, SIGNALS))


def _text(tips, i, col) -> str:
    return tips[i][col]


def test_trace_does_not_change_recompute_result():
    for df, p0 in ((_gated(), PRICES[0]), (_execution_fixture(), P0)):
        plain = recompute_gated_ledger(df, p0=p0)
        traced_into: list = []
        traced = recompute_gated_ledger(df, p0=p0, trace=traced_into)
        pd.testing.assert_frame_equal(plain, traced)
        assert len(traced_into) == len(df)


@pytest.mark.parametrize("df,p0", [(_gated(), PRICES[0]), (_execution_fixture(), P0)])
def test_every_cell_text_contains_the_value_the_table_shows(df, p0):
    fixed = recompute_gated_ledger(df, p0=p0)
    tips = ledger_explanations(df, p0=p0)
    assert len(tips) == len(df)
    for i in range(len(df)):
        assert set(tips[i]) == set(LEDGER_COLS)
        for col in LEDGER_COLS:
            value = float(fixed.at[i, col])
            text = _text(tips, i, col)
            assert text.strip(), (i, col)
            assert _f(value) in text or (value == 0.0 and "= 0" in text), (i, col, text)
            for clash in (" − -", " + -", "× -"):                 # ติดลบต้องมีวงเล็บเสมอ
                assert clash not in text, (i, col, text)


def test_gated_act_row_substitutes_the_independently_computed_numbers():
    tips = ledger_explanations(_gated(), p0=PRICES[0])
    # step 3: READY_BUY ที่ 9.5 เทียบ P_acted = 12 (act ล่าสุดที่ step 1) ไม่ใช่ราคา step 2 (11)
    delta = FIX_C * (9.5 / 12.0 - 1.0)
    text = _text(tips, 3, DA)
    assert "FIX_C × (P_fill / P_acted − 1)" in text
    assert f"({_f(9.5)} / {_f(12.0)} − 1)" in text
    assert _f(delta) in text
    assert "DNA step 1" in text            # P_acted มาจากแถว act ล่าสุด
    assert "gated_theoretical_v2" in text  # P_fill = ราคาตัดสินใจ Pₙ


def test_gated_pass_row_freezes_and_explains_why():
    tips = ledger_explanations(_gated(), p0=PRICES[0])
    delta_text = _text(tips, 2, DA)         # PASS_DNA_ZERO
    assert "ΔAₙ = 0" in delta_text and "PASS_DNA_ZERO" in delta_text
    assert "P_acted ค้างที่ 12.0000" in delta_text and "DNA step 1" in delta_text
    a_prev = FIX_C * (12.0 / 10.0 - 1.0)
    assert f"= {_f(a_prev)}" in _text(tips, 2, A)
    ref = FIX_C * math.log(12.0 / PRICES[0])
    e_text = _text(tips, 2, E)
    assert "ln(P_acted / P₀)" in e_text and f"ln({_f(12.0)} / {_f(PRICES[0])})" in e_text
    assert f"= {_f(a_prev)} − {_f(ref)}" in e_text        # ค่ากลางทางต้องตรง ไม่ใช่แค่ผลลัพธ์
    assert f"= {_f(a_prev - ref)}" in e_text


def test_act_row_excess_is_a_minus_r_at_the_row_quote():
    tips = ledger_explanations(_gated(), p0=PRICES[0])
    a = FIX_C * (12.0 / 10.0 - 1.0) + FIX_C * (9.5 / 12.0 - 1.0)      # step 3 หลัง act สองครั้ง
    r = FIX_C * math.log(9.5 / PRICES[0])
    text = _text(tips, 3, E)
    assert "Eₙ = Aₙ − Rₙ" in text
    assert r < 0 and f"= {_f(a)} − {_o(r)}" in text and f"= {_f(a - r)}" in text


def test_genesis_pass_threshold_row_is_zero_and_seeds_p_acted_from_p0():
    tips = ledger_explanations(_gated(), p0=PRICES[0])
    assert "genesis" in _text(tips, 0, DA) and "P₀" in _text(tips, 0, DA)
    assert "Aₙ = 0" in _text(tips, 0, A)
    assert "Eₙ = 0" in _text(tips, 0, E)


def test_reference_uses_quote_every_row_and_names_p0_source():
    tips = ledger_explanations(_gated(), p0=PRICES[0])
    for i, price in enumerate(PRICES):
        text = _text(tips, i, R)
        assert f"ln({_f(price)} / {_f(PRICES[0])})" in text
        assert _f(FIX_C * math.log(price / PRICES[0])) in text
        assert f"{math.log(price / PRICES[0]):.6f}" in text   # ln(Pₙ/P₀) กลางทาง
        assert "state pointer" in text
    # ไม่ส่ง p0 -> P₀ มาจากราคาแถว genesis
    assert "แถว genesis" in _text(ledger_explanations(_gated()), 1, R)


def test_execution_fill_uses_execution_price_and_pending_rows_say_intent():
    tips = ledger_explanations(_execution_fixture(), p0=P0)
    pending = _text(tips, 2, DA)
    assert "PENDING_EXECUTION" in pending and "intent" in pending
    fill = _text(tips, 3, DA)
    assert f"({_f(108.0)} / {_f(100.0)} − 1)" in fill
    assert "execution_price" in fill and _f(110.0) in fill      # แสดง quote เทียบด้วย
    assert "(110" not in fill.split("P_fill")[0]                # สูตรไม่ใช้ quote
    freeze_e = _text(tips, 4, E)                                 # PASS หลัง fill: P_acted = 108
    assert f"ln({_f(108.0)} / {_f(P0)})" in freeze_e and "DNA step 3" in _text(tips, 4, DA)
    a_after_fill = FIX_C * (108.0 / 100.0 - 1.0)
    ref108 = FIX_C * math.log(108.0 / P0)
    assert f"= {_f(a_after_fill)} − {_f(ref108)}" in freeze_e
    assert f"= {_f(a_after_fill)} + " not in freeze_e


def test_semantics_boundary_reset_is_called_out():
    rows = _execution_fixture()
    rows.loc[3:, "semantics"] = EXECUTION_CONFIRMED_SEMANTICS
    rows.loc[:2, "semantics"] = "gated_theoretical_v2"
    rows.loc[:2, "cashflow_status"] = None
    tips = ledger_explanations(rows, p0=P0)
    assert "รีเซ็ตเป็น 0" in _text(tips, 3, A)


def test_unknown_semantics_shows_stored_value_without_guessing_a_formula():
    df = _execution_fixture()
    df["semantics"] = "future_cashflow_v99"
    tips = ledger_explanations(df, p0=P0)
    for i in range(1, len(df)):
        for col in (R, DA, A, E):
            text = _text(tips, i, col)
            assert "ค่าที่บันทึกไว้" in text and "ไม่รู้จัก" in text
            assert "FIX_C ×" not in text


def test_missing_columns_falls_back_to_stored_values():
    df = _execution_fixture().drop(columns=["ส่วนต่างเป้าหมาย (USD)"])
    tips = ledger_explanations(df, p0=P0)
    assert len(tips) == len(df)
    assert all("ค่าที่บันทึกไว้" in tips[1][c] for c in LEDGER_COLS)


def test_empty_frame_has_no_explanations():
    assert ledger_explanations(pd.DataFrame()) == []


def test_frozen_terminal_funding_and_next_fill_explain_persisted_fields():
    import math as m
    first = funding_row()
    second = dict(first)
    delta = 5000 * (28 / 27.37 - 1)
    reference = 5000 * m.log(28 / 27.34)
    second.update({"version": 2, "DNA step": 134, "market_ordinal": 134,
                   "ราคา Pₙ (USD)": 28, "จำนวนถือครอง (หุ้น)": 182,
                   "มูลค่าพอร์ต (USD)": 5096, "ส่วนต่างเป้าหมาย (USD)": 96,
                   "สถานะ": "READY_SELL", "ฝั่ง": "SELL", "จำนวนสั่ง (หุ้น)": 3,
                   "Rₙ อ้างอิง (USD)": reference, "ΔAₙ ต่อสเต็ป (USD)": delta,
                   "Aₙ สะสม (USD)": delta, "Eₙ ส่วนเกินสะสม (USD)": delta - reference,
                   "execution_price": 28, "execution_quantity": 3, "finalized_seq": 2,
                   "R_basis": reference, "initial_funding": False,
                   "previous_action_price": 27.37, "previous_actual_cumulative": 0})
    df = pd.DataFrame([first, second])
    tips = ledger_explanations(df, p0=27.34)
    assert "initial funding" in _text(tips, 0, DA)
    assert "initial_funding_zero_v1" in _text(tips, 0, DA)
    d = _text(tips, 1, DA)
    assert f"({_f(28.0)} / {_f(27.37)} − 1)" in d and _f(delta) in d
    assert "previous_actual_cumulative + ΔAₙ" in _text(tips, 1, A)
    e = _text(tips, 1, E)
    assert "R_basis" in e and _f(delta - reference) in e
    # ค่า ledger ของ frozen terminal ต้องไม่ถูกคำนวณทับ
    fixed = recompute_gated_ledger(df, p0=27.34)
    assert float(fixed.at[1, DA]) == pytest.approx(delta)


def test_unfilled_frozen_terminal_row_says_no_finalized_fill():
    first = funding_row()
    passed = dict(first)
    passed.update({"version": 2, "DNA step": 134, "market_ordinal": 134,
                   "สถานะ": "PASS_THRESHOLD", "จำนวนสั่ง (หุ้น)": 0,
                   "cashflow_status": "NO_ACTION"})
    passed.pop("execution_price")
    passed.pop("execution_quantity")
    tips = ledger_explanations(pd.DataFrame([first, passed]), p0=27.34)
    assert "ยังไม่มี fill ที่ FINALIZED" in _text(tips, 1, DA)
    assert "NO_ACTION" in _text(tips, 1, DA)


# ---- ตาราง HTML ------------------------------------------------------------

def _frame() -> pd.DataFrame:
    return pd.DataFrame({"สถานะ": ["READY_BUY", "<b>x</b>"],
                         "ราคา Pₙ (USD)": [10.123, float("nan")],
                         "ΔAₙ ต่อสเต็ป (USD)": [1.0, 2.0],
                         "DNA step": [0, 1]})


def test_html_puts_tooltips_only_on_requested_columns_and_rounds_money():
    tips = [{"ΔAₙ ต่อสเต็ป (USD)": "หัวข้อ\nบรรทัดสอง"}, {"ΔAₙ ต่อสเต็ป (USD)": "อีกอัน"}]
    out = build_hover_table_html(_frame(), tips, ["ΔAₙ ต่อสเต็ป (USD)"],
                                 ["ราคา Pₙ (USD)", "ΔAₙ ต่อสเต็ป (USD)"])
    assert out.count("data-tip=") == 2
    assert 'th class="tip"' in out and out.count('th class="tip"') == 1
    assert ">10.12<" in out and ">1.00<" in out        # money round 2dp
    assert 'tabindex="0"' in out                       # โฟกัสด้วยคีย์บอร์ดได้
    assert "หัวข้อ\nบรรทัดสอง" in out                   # newline คงอยู่ใน attribute


def test_html_escapes_cell_and_tooltip_text():
    evil = '"><script>alert(1)</script>'
    frame = pd.DataFrame({"สถานะ": [evil], "ΔAₙ ต่อสเต็ป (USD)": [1.0]})
    out = build_hover_table_html(frame, [{"ΔAₙ ต่อสเต็ป (USD)": evil}],
                                 ["ΔAₙ ต่อสเต็ป (USD)"], [])
    assert "<script>alert(1)" not in out
    assert out.count("<script>") == 1                  # มีเฉพาะสคริปต์ของเราเอง
    assert "innerHTML" not in out                      # tooltip ใส่ด้วย textContent เท่านั้น


def test_html_theme_attribute_and_height_bounds():
    assert 'data-theme="dark"' in build_hover_table_html(_frame(), [], [], [], theme="dark")
    assert "data-theme" not in build_hover_table_html(_frame(), [], [], [], theme="neon").split(
        "<style>")[0]
    assert hover_table_height(0) == MIN_HEIGHT
    assert hover_table_height(10_000) == MAX_HEIGHT
    assert MIN_HEIGHT <= hover_table_height(8) <= MAX_HEIGHT


# ---- Streamlit -------------------------------------------------------------

@pytest.fixture
def live_app(monkeypatch):
    rows = _execution_fixture()
    chain = str(rows.iloc[0]["chain_key"])
    payloads = {
        "webull_lego_rows": {str(r["run_id"]): r.to_dict() for _, r in rows.iterrows()},
        "webull_lego_state": {chain: {"p0": P0, "updated_at": "2026-08-01T15:00:00Z"}},
        "webull_lego_order_audit": {},
    }

    class _Reference:
        def __init__(self, path):
            self.path = path

        def get(self):
            return payloads.get(self.path)

    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(db, "reference", _Reference)
    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    app.secrets.update({"FIREBASE_SA_JSON": "{}",
                        "FIREBASE_DB_URL": "https://mock.firebaseio.test"})
    return app


def test_live_table_defaults_to_hover_mode_and_toggle_restores_dataframe(live_app):
    live_app.run(timeout=30)
    assert not live_app.exception and not live_app.warning
    toggle = live_app.toggle(key="live_hover_table")
    assert toggle.value is True
    hover_frames = len(live_app.dataframe)

    toggle.set_value(False).run(timeout=30)
    assert not live_app.exception and not live_app.warning
    assert len(live_app.dataframe) == hover_frames + 1   # ตาราง 17 คอลัมน์กลับเป็น st.dataframe


def test_hover_failure_falls_back_to_dataframe_with_a_warning(live_app, monkeypatch):
    import lego_dash_core

    def boom(*args, **kwargs):
        raise RuntimeError("explain exploded")

    monkeypatch.setattr(lego_dash_core, "ledger_explanations", boom)
    live_app.run(timeout=30)
    assert not live_app.exception
    assert any("explain exploded" in w.value for w in live_app.warning)
    assert len(live_app.dataframe) >= 4
