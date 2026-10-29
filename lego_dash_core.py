"""lego_dash_core.py — pure logic ของ dashboard (ไม่มี streamlit/firebase I/O)

แยกจาก streamlit_app.py เพื่อให้ test ได้ตรง ๆ:
  - rows_to_df       : RTDB dict -> DataFrame เฉพาะ committed==True เรียง version (fail closed)
  - order_columns    : บังคับลำดับสัญญา 17 คอลัมน์ (RTDB ไม่การันตีลำดับ key) + meta ต่อท้าย
  - integrity_report : ตรวจสมการ LEGO ต่อแถวจากค่า full precision (E1–E8)
  - recompute_gated_ledger : derive recurrence ใหม่ตามหลักการแช่แข็ง ก่อนแสดงผลเสมอ

รองรับ cashflow contract ตาม ``semantics`` ของแต่ละแถว:
  - gated_theoretical_v2: act จาก decision READY_* + signal/quantity ตาม historical contract
  - execution_confirmed_v1: act เฉพาะ broker-confirmed fill ที่มี execution qty/price > 0
  - ทุกกรณีอื่นแช่แข็ง: ΔAₙ = 0, Aₙ ค้าง, P_acted ค้าง
  - ไม่แช่แข็ง Rₙ เด็ดขาด — Rₙ = FIX_C × ln(Pₙ/P₀) จาก quote Pₙ ทุกแถว
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from dna_engine import decode_dna

# mirror ของ lego_one_row.COLUMN_ORDER (สัญญา 17 คอลัมน์ ลำดับตายตัว)
COLUMN_ORDER = [
    "เวลา (UTC)",             # 1
    "สินทรัพย์",              # 2
    "สถานะ",                  # 3
    "DNA step",              # 4
    "DNA signal",            # 5
    "ราคา Pₙ (USD)",          # 6
    "จำนวนถือครอง (หุ้น)",    # 7
    "คำสั่ง",                 # 8
    "ฝั่ง",                   # 9
    "เหตุผล",                 # 10
    "จำนวนสั่ง (หุ้น)",        # 11
    "มูลค่าพอร์ต (USD)",      # 12
    "ส่วนต่างเป้าหมาย (USD)",  # 13
    "Rₙ อ้างอิง (USD)",       # 14
    "ΔAₙ ต่อสเต็ป (USD)",     # 15
    "Aₙ สะสม (USD)",          # 16
    "Eₙ ส่วนเกินสะสม (USD)",  # 17
]
# คอลัมน์ที่ contract เลิกใช้แล้ว: แถวเก่าใน RTDB ยังมีคีย์นี้ค้างอยู่ จึงตัดทิ้งตอนอ่าน
# (ไม่เคยเข้าสูตร Aₙ/Eₙ — เงินสด broker เก็บแยกที่ webull_lego_broker_cashflow)
LEGACY_DROPPED_COLS = ("ΔAₙ เงินจริง (USD)",)
META_COLS = ["run_id", "chain_key", "version", "committed", "semantics",
             "market_slot_id", "market_ordinal", "clock_mode", "schema_version",
             "ledger_version_at_observation", "E_mark_at_observation",
             "R_basis", "finalized_seq"]

# mirror ของ lego_state.CASHFLOW_SEMANTICS — ledger ทฤษฎีแบบ gated (ตาม gated demo):
#   ledger คีย์ที่ "เทรดจริงหรือไม่" (การตัดสินใจ) ไม่ใช่ DNA signal ดิบ:
#   act (เทรดจริง READY_BUY/READY_SELL, signal = 1, จำนวนสั่ง > 0):
#       ΔAₙ = FIX_C×(Pₙ/P_acted − 1) โดย P_acted = ราคาแถว act ล่าสุด
#   pass (ไม่เทรด): ΔAₙ = 0, Aₙ ค้าง, P_acted แช่แข็ง, Eₙ smooth
#       *รวม PASS_DNA_ZERO (signal=0) และ PASS_THRESHOLD (signal=1 แต่ |gap| ≤ DIFF):
#        ทั้งคู่ไม่ยิง order จึงต้องแช่แข็ง ledger เหมือนกัน (ΔAₙ = 0)
GATED_SEMANTICS = "gated_theoretical_v2"
# semantics เก่า (ก่อน v2): ΔAₙ = กำไร realized จากรอบ Buy↔Sell ที่จับคู่ปิด
LEGACY_REALIZED_SEMANTICS = "cycle_realized_v1"
# semantics ปัจจุบันของ lego-firebase: decision เป็นเพียง intent; broker fill เท่านั้นที่ act
EXECUTION_CONFIRMED_SEMANTICS = "execution_confirmed_v1"
EXECUTION_TERMINAL_FROZEN_V2 = "execution_terminal_frozen_v2"
EXECUTION_TERMINAL_FUNDING_V3 = "execution_terminal_funding_v3"
FROZEN_TERMINAL_SEMANTICS = (EXECUTION_TERMINAL_FROZEN_V2, EXECUTION_TERMINAL_FUNDING_V3)
CASHFLOW_SEMANTICS_HISTORY = (
    LEGACY_REALIZED_SEMANTICS, GATED_SEMANTICS, EXECUTION_CONFIRMED_SEMANTICS,
    *FROZEN_TERMINAL_SEMANTICS,
)
CASHFLOW_SEMANTICS_RANK = {
    name: rank for rank, name in enumerate(CASHFLOW_SEMANTICS_HISTORY)
}
CASHFLOW_NO_ACTION = "NO_ACTION"
CASHFLOW_PENDING_EXECUTION = "PENDING_EXECUTION"
CASHFLOW_FINALIZED = "FINALIZED"
FUNDING_BASELINE_POLICY = "initial_funding_zero_v1"
CASHFLOW_STATUSES = frozenset({
    CASHFLOW_NO_ACTION, CASHFLOW_PENDING_EXECUTION, CASHFLOW_FINALIZED,
})

# คอลัมน์เงิน 7 ตัว (6, 12–17) — round 2dp เฉพาะตอนแสดง (ตรง columns_presented ฝั่ง engine)
MONEY_COLS = ["ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)",
              "Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)",
              "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]
LEDGER_COLS = ["Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)",
               "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]
# คอลัมน์ที่ recompute ต้องมีครบ มิฉะนั้นไม่แตะ (fail safe)
RECOMPUTE_REQUIRED = ["ราคา Pₙ (USD)", "สถานะ", "DNA step", "มูลค่าพอร์ต (USD)",
                      "ส่วนต่างเป้าหมาย (USD)", "Rₙ อ้างอิง (USD)",
                      "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]
# คอลัมน์ตัดสินใจที่ทำให้ตัดสิน act/pass ได้เข้มตามนโยบาย
POLICY_COLS = ["สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)"]

PASS_DNA_ZERO = "PASS_DNA_ZERO"
PASS_THRESHOLD = "PASS_THRESHOLD"
READY_BUY = "READY_BUY"
READY_SELL = "READY_SELL"


def _drop_legacy_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ตัดคอลัมน์ที่ contract เลิกใช้ (ถ้ามี) — ไม่แตะคอลัมน์อื่น"""
    legacy = [c for c in LEGACY_DROPPED_COLS if c in df.columns]
    return df.drop(columns=legacy) if legacy else df


def rows_to_df(data) -> pd.DataFrame:
    """RTDB payload -> DataFrame เฉพาะแถว committed==True เรียงตาม version

    fail closed: ไม่มีข้อมูล / ไม่มี flag committed -> DataFrame ว่าง
    (orphan/pending ต้องไม่โผล่บน dashboard ตามสัญญา Step 18)
    """
    if not data:
        return pd.DataFrame()
    items = list(data.values()) if isinstance(data, dict) else [x for x in data if x]
    df = pd.DataFrame(items)
    if df.empty or "committed" not in df.columns:
        return pd.DataFrame()
    df = df[df["committed"] == True]  # noqa: E712 — กรอง orphan/pending
    if df.empty:
        return pd.DataFrame()
    if "version" in df.columns:
        df = df.sort_values("version")
    return _drop_legacy_cols(df).reset_index(drop=True)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """RTDB คืน key ไม่การันตีลำดับ -> บังคับ 17 คอลัมน์ตามสัญญา แล้วต่อ meta/อื่น ๆ"""
    df = _drop_legacy_cols(df)
    cols = [c for c in COLUMN_ORDER if c in df.columns]
    cols += [c for c in META_COLS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def default_chain_index(chains: list, state: dict | None) -> int:
    """chain ที่ active ล่าสุดตาม state[ck].updated_at (ISO string เทียบอักษร = ลำดับเวลา)
    ไม่มีข้อมูล state/updated_at -> ตัวสุดท้ายของ list (พฤติกรรมเดิม)"""
    if not chains:
        return 0
    if not isinstance(state, dict) or not state:
        return len(chains) - 1
    stamps = {ck: str((state.get(ck) or {}).get("updated_at", "")) for ck in chains}
    latest = max(chains, key=lambda ck: stamps[ck])
    return chains.index(latest) if stamps[latest] else len(chains) - 1


def filter_audit_rows(audit: dict | None, run_ids) -> pd.DataFrame:
    """audit เฉพาะ order ของ chain ที่เลือก — ผูกด้วย run_id (audit 1 รายการ/แถว)
    payload เก่าที่ไม่มี run_id เลย -> คืนทั้งหมด (ไม่ตัดข้อมูลที่กรองไม่ได้)"""
    if not audit:
        return pd.DataFrame()
    adf = pd.DataFrame([v for v in audit.values() if isinstance(v, dict)])
    if adf.empty or "run_id" not in adf.columns:
        return adf
    return adf[adf["run_id"].isin(set(run_ids))].reset_index(drop=True)


def audit_with_ledger_time(audit_rows: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """เติม "เวลา (UTC)" จากตาราง 17 คอลัมน์ (ผูกด้วย run_id) แล้วเรียง run_id ล่าสุดอยู่บนสุด
    ลำดับ "ล่าสุด" = ตำแหน่งแถวใน ledger (เรียง version แล้ว) เพราะ run_id ไม่ได้เรียงตามเวลา
    audit ที่ไม่มี run_id ใน ledger -> เวลาว่างและอยู่ท้ายตาราง; payload เก่าไม่มี run_id -> คืนเดิม"""
    time_col = COLUMN_ORDER[0]
    if (audit_rows.empty or "run_id" not in audit_rows.columns
            or "run_id" not in ledger.columns or time_col not in ledger.columns):
        return audit_rows
    latest = ledger.drop_duplicates("run_id", keep="last")
    times = dict(zip(latest["run_id"], latest[time_col]))
    rank = {rid: pos for pos, rid in enumerate(ledger["run_id"])}
    adf = audit_rows.copy()
    adf[time_col] = adf["run_id"].map(times)
    order = adf["run_id"].map(rank).fillna(-1).sort_values(ascending=False, kind="stable")
    return adf.loc[order.index].reset_index(drop=True)


def pending_broker_fee_count(audit_rows: pd.DataFrame) -> int:
    """Count fills whose broker fee is still unknown, never inferred as zero."""
    if audit_rows.empty or "broker_fee_status" not in audit_rows.columns:
        return 0
    return int(audit_rows["broker_fee_status"].fillna("").astype(str)
               .str.upper().eq("PENDING").sum())


def _max_abs(s: pd.Series) -> float:
    s = s.dropna()
    return 0.0 if s.empty else float(s.abs().max())


def has_policy_columns(df: pd.DataFrame) -> bool:
    """ตัดสิน act/pass แบบเข้มได้ก็ต่อเมื่อมีคอลัมน์ตัดสินใจครบทั้งสาม"""
    return all(col in df.columns for col in POLICY_COLS)


def _traded_flags(df: pd.DataFrame) -> list[bool]:
    """act = เทรดจริงเท่านั้น: signal = 1 **และ** READY_BUY/READY_SELL **และ** qty > 0

    signal 0, PASS ทุกชนิด (รวม PASS_THRESHOLD ที่ signal=1 แต่ |gap| ≤ DIFF), สถานะแปลก
    และแถว READY ที่ผิดรูป (qty = 0 หรือ signal = 0) ล้วนเป็น pass ต้องแช่แข็ง ledger
    frame เก่าที่ไม่มีคอลัมน์ตัดสินใจครบ -> ถอยไปดูสถานะอย่างเดียว (พฤติกรรมรุ่นก่อน)
    """
    ready = df["สถานะ"].astype(str).isin([READY_BUY, READY_SELL])
    if not has_policy_columns(df):
        return ready.tolist()
    return (ready
            & df["DNA signal"].astype(int).eq(1)
            & df["จำนวนสั่ง (หุ้น)"].astype(float).gt(0)).tolist()


def _semantics(df: pd.DataFrame) -> pd.Series:
    """คืน semantics ต่อแถว; missing/unknown คงเป็น legacy ที่ dashboard ห้ามเดา."""
    if "semantics" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df["semantics"].fillna("").astype(str)


def _execution_flags(df: pd.DataFrame) -> list[bool]:
    return _semantics(df).isin([
        EXECUTION_CONFIRMED_SEMANTICS, *FROZEN_TERMINAL_SEMANTICS]).tolist()


def _positive_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """ค่าบวก finite; missing/อ่านไม่ได้เป็น False เพื่อ fail closed."""
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    values = pd.to_numeric(df[column], errors="coerce")
    return values.gt(0) & np.isfinite(values)


def _confirmed_execution_flags(df: pd.DataFrame) -> list[bool]:
    """act จริง = semantics v3 + FINALIZED + execution quantity/price ที่ใช้ได้.

    READY_*, ordered quantity, SUBMITTED, PENDING, REJECTED และ EXPIRED ไม่ใช่
    execution evidence และต้องแช่แข็ง ledger ทั้งหมด.
    """
    execution = pd.Series(_execution_flags(df), index=df.index)
    if "cashflow_status" not in df.columns:
        return [False] * len(df)
    finalized = df["cashflow_status"].fillna("").astype(str).eq(CASHFLOW_FINALIZED)
    valid_qty = _positive_numeric(df, "execution_quantity")
    valid_price = _positive_numeric(df, "execution_price")
    return (execution & finalized & valid_qty & valid_price).tolist()


def _gated_flags(df: pd.DataFrame) -> list[bool]:
    """แถวไหนอยู่ใต้ semantics gated_theoretical_v2 (แถวเก่าห้ามคำนวณด้วยสมการนี้)

    ไม่มีคอลัมน์ semantics เลย -> ทั้ง frame มาจากก่อนมีฟิลด์นี้ (เก่ากว่า v1 ด้วยซ้ำ)
    ถือว่าไม่ gated ทั้งหมด (fail safe เหมือน semantics เก่าอื่น ๆ) ไม่ใช่ถือว่า gated
    """
    return _semantics(df).eq(GATED_SEMANTICS).tolist()


def _known_cashflow_flags(df: pd.DataFrame) -> list[bool]:
    semantics = _semantics(df)
    return semantics.isin([
        GATED_SEMANTICS, EXECUTION_CONFIRMED_SEMANTICS,
        *FROZEN_TERMINAL_SEMANTICS]).tolist()


def _reference_context(df: pd.DataFrame, p0: float | None) -> tuple[float, float] | None:
    """หา (P₀, FIX_C) สำหรับเส้นอ้างอิง โดยไม่พึ่งสถานะการเทรดเลย"""
    required = {"ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)"}
    if df.empty or not required.issubset(df.columns):
        return None

    resolved_p0 = p0
    if resolved_p0 is None and "DNA step" in df.columns:
        try:
            if int(df["DNA step"].iloc[0]) == 0:
                resolved_p0 = float(df["ราคา Pₙ (USD)"].iloc[0])
        except (TypeError, ValueError):
            return None
    if resolved_p0 is None:
        return None

    resolved_p0 = float(resolved_p0)
    fix_c = float(df["มูลค่าพอร์ต (USD)"].astype(float).iloc[0]
                  - df["ส่วนต่างเป้าหมาย (USD)"].astype(float).iloc[0])
    if not np.isfinite(resolved_p0) or resolved_p0 <= 0:
        return None
    if not np.isfinite(fix_c):
        return None
    return resolved_p0, fix_c


def _apply_reference_column(out: pd.DataFrame, source: pd.DataFrame,
                            p0: float | None) -> pd.DataFrame:
    """Rₙ คือเส้นอ้างอิงตลาด จึงต้องขยับทั้งแถว act และแถว pass ที่แช่แข็ง"""
    if "Rₙ อ้างอิง (USD)" not in out.columns:
        return out
    context = _reference_context(source, p0)
    if context is None:
        return out

    resolved_p0, fix_c = context
    prices = source["ราคา Pₙ (USD)"].astype(float)
    if bool((~np.isfinite(prices) | prices.le(0)).any()):
        return out

    reference = fix_c * np.log(prices / resolved_p0)
    known = pd.Series(_known_cashflow_flags(source), index=source.index)
    out.loc[known.to_numpy(), "Rₙ อ้างอิง (USD)"] = reference.loc[known].to_numpy()
    return out


def _frozen_v2_errors(df: pd.DataFrame, fix_c: float, p0: float | None,
                      scale: float) -> list[str]:
    """Validate persisted v2 arithmetic independently; never compare a copy to itself.

    A clipped window without enough basis evidence is incomplete, not green.
    New backend rows carry the previous execution basis; old complete chains
    can reconstruct it from confirmed fills. Funding drift remains visible.
    """
    errors = []
    previous_price = p0 if df.iloc[0].get("version") == 1 else None
    previous_A = 0.0 if df.iloc[0].get("version") == 1 else None
    previous_E = 0.0 if df.iloc[0].get("version") == 1 else None
    complete_origin = df.iloc[0].get("version") == 1
    confirmed_history = []
    for _, row in df.iterrows():
        if row.get("semantics") not in FROZEN_TERMINAL_SEMANTICS:
            previous_price = previous_A = previous_E = None
            continue
        def number(key, default=None):
            value = row.get(key, default)
            if pd.isna(value):
                value = default
            if value is None:
                raise ValueError(f"missing {key}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {key}")
            return value
        try:
            delta = number("ΔAₙ ต่อสเต็ป (USD)")
            actual = number("Aₙ สะสม (USD)")
            excess = number("Eₙ ส่วนเกินสะสม (USD)")
            policy = row.get("model_baseline_policy")
            if pd.isna(policy):
                policy = None
            if policy not in (None, FUNDING_BASELINE_POLICY):
                raise ValueError("unknown model baseline policy")
            finalized = row.get("cashflow_status") == CASHFLOW_FINALIZED
            if finalized:
                price = number("execution_price")
                if price <= 0 or number("execution_quantity") <= 0:
                    raise ValueError("invalid execution facts")
                prior_price = number("previous_action_price", previous_price)
                prior_A = number("previous_actual_cumulative", previous_A)
                if prior_price <= 0:
                    raise ValueError("invalid previous action price")
                funding = row.get("initial_funding") is True
                # Historical v2 initialized from a quote, booking slippage as
                # a return on the first flat BUY. Flag it; do not hide/rewrite it.
                flat_first_buy = (row.get("version") == 1
                                  and row.get("ฝั่ง") == "BUY"
                                  and number("จำนวนถือครอง (หุ้น)") == 0)
                if funding:
                    if (policy != FUNDING_BASELINE_POLICY
                            or number("finalized_seq") != 1
                            or row.get("ฝั่ง") != "BUY"
                            or number("จำนวนถือครอง (หุ้น)") != 0 or prior_A != 0):
                        raise ValueError("invalid initial funding provenance")
                    expected_delta = 0.0
                else:
                    expected_delta = fix_c * (price / prior_price - 1)
                basis = number("R_basis")
                offset = number("funding_reference_offset", 0)
                reference = number("Rₙ อ้างอิง (USD)")
                expected_basis = reference - offset
                if funding:
                    expected_basis = 0.0
                    if abs(reference - offset) > scale:
                        raise ValueError("funding reference offset mismatch")
                residuals = [delta - expected_delta, actual - (prior_A + delta),
                             excess - (actual - basis), basis - expected_basis]
                if flat_first_buy:
                    residuals.extend([delta, actual, excess])
                if max(abs(x) for x in residuals) > scale:
                    raise ValueError("funding/terminal arithmetic mismatch")
                previous_price, previous_A, previous_E = price, actual, excess
                confirmed_history.append((row.get("cashflow_finalized_at"), actual, excess))
            else:
                if abs(delta) > scale:
                    raise ValueError("unfilled row moved delta")
                basis = number("R_basis", (previous_A - previous_E)
                               if previous_A is not None and previous_E is not None else None)
                if abs(excess - (actual - basis)) > scale:
                    raise ValueError("frozen excess disagrees with persisted basis")
                # Baseline for a leading frozen row of a truncated window can
                # be carried, but cannot prove its own prior execution history.
                if previous_A is None:
                    raise ValueError("incomplete prior execution history")
                observed_at = pd.to_datetime(row.get("เวลา (UTC)"), utc=True, errors="coerce")
                visible = []
                for finalized_at, final_A, final_E in confirmed_history:
                    timestamp = pd.to_datetime(finalized_at, utc=True, errors="coerce")
                    if pd.isna(timestamp) or pd.isna(observed_at) or timestamp <= observed_at:
                        visible.append((final_A, final_E))
                if visible:
                    expected_A, expected_E = visible[-1]
                elif complete_origin:
                    expected_A = expected_E = 0.0
                else:
                    raise ValueError("incomplete observation baseline")
                if max(abs(actual - expected_A), abs(excess - expected_E)) > scale:
                    raise ValueError("unfilled row moved cumulative ledger")
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"v{row.get('version', '?')}: {exc}")
    return errors


def integrity_report(df: pd.DataFrame, p0_hint: float | None = None,
                     tol: float = 1e-6) -> tuple[pd.DataFrame, bool]:
    """ตรวจสมการ LEGO กับแถว committed ของ chain เดียว (เรียง version แล้ว)

    ใช้ค่า full precision จาก RTDB (ห้ามใช้ค่า round 2dp)
    residual ผ่านเมื่อ <= tol × max(1, FIX_C)

      E1  FIX_C คงที่:   Vₙ + gapₙ = FIX_C ทุกแถว   (นิยาม gap = FIX_C − Vₙ)
      E2  มูลค่าพอร์ต:    Vₙ = holdingsₙ × Pₙ
      E3  อ้างอิง:        Rₙ = FIX_C × ln(Pₙ / P₀)
      E4  ต่อสเต็ป:       ΔAₙ = FIX_C × (P_fill / P_acted − 1); v2 act จาก decision price,
                         v3 act เฉพาะ FINALIZED และใช้ execution_price (VWAP);
                         PASS/pending/rejected -> ΔAₙ = 0
      E5  สะสม:          Aₙ = Aₙ₋₁ + ΔAₙ (ΔAₙ ต่อสเต็ปของโมเดล ไม่ใช่เงินสด broker) —
                         ข้ามเฉพาะรอยต่อเปลี่ยน semantics (baseline Aₙ รีเซ็ตเป็น 0)
      E6  ส่วนเกิน (smooth): act -> Eₙ = Aₙ − Rₙ(row quote) ;
                         frozen -> Eₙ = Aₙ − FIX_C × ln(P_acted / P₀)
      E7  โครงสร้าง:      step เพิ่มตาม market slot (มี market_ordinal -> Δstep = Δordinal ≥ 1;
                         ไม่มี -> step +1 แบบเดิม), version +1 ทุกแถว, signal ∈ {0,1}
      E8  decision:      signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง
    """
    p = df["ราคา Pₙ (USD)"].astype(float)
    h = df["จำนวนถือครอง (หุ้น)"].astype(float)
    v = df["มูลค่าพอร์ต (USD)"].astype(float)
    gap = df["ส่วนต่างเป้าหมาย (USD)"].astype(float)
    R = df["Rₙ อ้างอิง (USD)"].astype(float)
    dA = df["ΔAₙ ต่อสเต็ป (USD)"].astype(float)
    A = df["Aₙ สะสม (USD)"].astype(float)
    E = df["Eₙ ส่วนเกินสะสม (USD)"].astype(float)
    step = df["DNA step"].astype(int)
    sig = df["DNA signal"].astype(int)
    qty = df["จำนวนสั่ง (หุ้น)"].astype(float)
    status = df["สถานะ"].astype(str)
    fixc_series = v - gap
    fix_c = float(fixc_series.iloc[0])
    scale = tol * max(1.0, abs(fix_c))
    genesis = bool(df.iloc[0].get("version") == 1 or step.iloc[0] == 0)

    p0 = p0_hint
    if p0 is None and genesis:
        p0 = float(p.iloc[0])   # แถว genesis: P₀ = P ของแถวแรก

    checks: list[tuple[str, str, float | None, bool, str]] = []

    def add(cid: str, eq: str, residual: float | None, ok: bool, note: str = ""):
        checks.append((cid, eq, residual, ok, note))

    r1 = _max_abs(fixc_series - fix_c)
    add("E1", "Vₙ − gapₙ = FIX_C (คงที่)", r1, r1 <= scale, f"FIX_C ≈ {fix_c:.2f}")

    r2 = _max_abs(v - h * p)
    add("E2", "Vₙ = holdingsₙ × Pₙ", r2, r2 <= scale)

    if p0 is not None and p0 > 0:
        r3 = _max_abs(R - fix_c * np.log(p / p0))
        add("E3", "Rₙ = FIX_C × ln(Pₙ/P₀)", r3, r3 <= scale, f"P₀ = {p0}")
    else:
        add("E3", "Rₙ = FIX_C × ln(Pₙ/P₀)", None, True, "ข้าม — ไม่รู้ P₀ (ไม่มีแถว genesis/state)")

    # สร้าง expected ledger จาก contract เดียวกับที่ dashboard ใช้แสดง แล้วเทียบ
    # ค่า persist ตรง ๆ. execution_confirmed_v1 จึงถูกตรวจทุกแถว ไม่ถูกจัดเป็น
    # "legacy" แล้ว skip จนรายงานเขียวทั้งชุดเหมือนเวอร์ชันก่อน.
    semantics = _semantics(df)
    known = semantics.isin([
        GATED_SEMANTICS, EXECUTION_CONFIRMED_SEMANTICS,
        *FROZEN_TERMINAL_SEMANTICS])
    legacy = semantics.eq(LEGACY_REALIZED_SEMANTICS)
    unknown = ~(known | legacy)
    execution = semantics.isin([EXECUTION_CONFIRMED_SEMANTICS,
                               *FROZEN_TERMINAL_SEMANTICS])
    frozen_v2 = semantics.isin(FROZEN_TERMINAL_SEMANTICS)
    v2_errors = _frozen_v2_errors(df, fix_c, p0, scale) if frozen_v2.any() else []
    semantic_ranks = semantics.map(CASHFLOW_SEMANTICS_RANK)
    semantics_downgrade = semantic_ranks.diff().lt(0)
    semantics_downgrade_count = int(semantics_downgrade.fillna(False).sum())
    expected = recompute_gated_ledger(df, p0=p0)
    expected_dA = expected["ΔAₙ ต่อสเต็ป (USD)"].astype(float)
    expected_E = expected["Eₙ ส่วนเกินสะสม (USD)"].astype(float)

    resid4 = (dA - expected_dA).abs()[known]
    r4 = _max_abs(resid4) if bool(known.any()) else None

    # Provenance fence: FINALIZED ที่ขาด fill qty/price และ fill fields บนสถานะ
    # ที่ยังไม่ FINALIZED ล้วนผิด contract แม้ตัวเลข recurrence บังเอิญเป็นศูนย์.
    cashflow_status = (df["cashflow_status"].fillna("").astype(str)
                       if "cashflow_status" in df.columns
                       else pd.Series([""] * len(df), index=df.index))
    finalized = cashflow_status.eq(CASHFLOW_FINALIZED)
    execution_qty = _positive_numeric(df, "execution_quantity")
    execution_price = _positive_numeric(df, "execution_price")
    confirmed = finalized & execution_qty & execution_price
    decision_acted = pd.Series(_traded_flags(df), index=df.index, dtype=bool)
    status_matches_decision = (
        (decision_acted & cashflow_status.isin([
            CASHFLOW_PENDING_EXECUTION, CASHFLOW_FINALIZED]))
        | (~decision_acted & cashflow_status.eq(CASHFLOW_NO_ACTION))
    )
    invalid_execution = execution & (
        ~cashflow_status.isin(CASHFLOW_STATUSES)
        | (finalized & ~confirmed)
        | (~finalized & (execution_qty | execution_price))
        | ~status_matches_decision
    )
    invalid_execution_count = int(invalid_execution.sum())
    legacy_count = int(legacy.sum())
    unknown_count = int(unknown.sum())
    notes4 = []
    if legacy_count:
        notes4.append(f"ข้าม {legacy_count} แถว semantics เก่าที่รู้จัก")
    if unknown_count:
        notes4.append(f"semantics ว่าง/ไม่รู้จัก {unknown_count} แถว — fail closed")
    if semantics_downgrade_count:
        notes4.append(
            f"cashflow semantics เดินถอยหลัง {semantics_downgrade_count} จุด — fail closed")
    if invalid_execution_count:
        notes4.append(f"execution provenance ผิด {invalid_execution_count} แถว")
    if v2_errors:
        notes4.append("; ".join(v2_errors[:5]))
    note4 = " · ".join(notes4)
    equation4 = ("v2 decision-price / v3 FINALIZED execution-price; frozen ΔAₙ = 0"
                 if bool(execution.any()) else
                 "gated: act ΔAₙ = FIX_C × (Pₙ/P_acted − 1) ; pass ΔAₙ = 0")
    ok4 = (((r4 is None) or r4 <= scale)
           and invalid_execution_count == 0 and unknown_count == 0
           and semantics_downgrade_count == 0 and not v2_errors)
    add("E4", equation4, r4, ok4,
        note4 or ("ไม่มีแถวตรวจได้" if r4 is None else ""))

    n = len(df)
    if n > 1:
        boundary = semantics.ne(semantics.shift(1))       # exact semantics boundary
        boundary.iloc[0] = False
        resid5 = A - (A.shift(1) + dA)
        # v2 finalization can occur after later observation rows were committed.
        # Its per-fill previous_actual_cumulative is checked independently above.
        r5 = _max_abs(resid5[~boundary & ~frozen_v2 & ~frozen_v2.shift(1, fill_value=False)])
        note5 = ("" if int(boundary.sum()) == 0 else
                 "ข้ามรอยต่อเปลี่ยน semantics (baseline Aₙ รีเซ็ตเป็น 0)")
        add("E5", "Aₙ = Aₙ₋₁ + ΔAₙ", r5, r5 <= scale, note5)
    else:
        add("E5", "Aₙ = Aₙ₋₁ + ΔAₙ", None, True, "แถวเดียว — ไม่มีคู่เทียบ")

    resid6 = (E - expected_E).abs()[known]
    r6 = _max_abs(resid6) if bool(known.any()) else 0.0
    notes6 = []
    if legacy_count:
        notes6.append(f"ข้าม {legacy_count} แถว semantics เก่าที่รู้จัก")
    if unknown_count:
        notes6.append(f"semantics ว่าง/ไม่รู้จัก {unknown_count} แถว — fail closed")
    note6 = " · ".join(notes6)
    add("E6", "act Eₙ = Aₙ − Rₙ(row quote); frozen Eₙ = Aₙ − FIX_C × ln(P_acted/P₀)",
        r6, r6 <= scale and unknown_count == 0 and not v2_errors, note6)

    # DNA เดินตาม market slot: ปกติ scheduler ไม่พลาด -> step +1 เหมือนเดิมทุกประการ
    # แต่ถ้าพลาด slot step ต้องกระโดดเท่ากับ market_ordinal ที่ข้ามไป (ห้ามย้อน/ซ้ำ)
    ok_step, note_step = True, ""
    if len(df) > 1:
        step_delta = step.diff().iloc[1:]
        if "market_ordinal" in df.columns and df["market_ordinal"].notna().all():
            slot_delta = df["market_ordinal"].astype(int).diff().iloc[1:]
            ok_step = bool((step_delta == slot_delta).all() and (slot_delta >= 1).all())
            skipped = int((slot_delta > 1).sum())
            if ok_step and skipped:
                note_step = f"ข้าม {skipped} ช่วง (scheduler พลาด slot — DNA เดินตามเวลาตลาด)"
        else:
            ok_step = bool((step_delta == 1).all())   # แถวเก่าที่ไม่มี slot provenance
    ok_ver = True
    if "version" in df.columns and len(df) > 1:
        ok_ver = bool((df["version"].astype(int).diff().iloc[1:] == 1).all())
    ok_sig = bool(sig.isin([0, 1]).all())
    ok7 = ok_step and ok_ver and ok_sig
    add("E7", "step ตาม market slot / version +1 / signal ∈ {0,1}", None, ok7,
        note_step if ok7 else "ลำดับ step/version ขาด หรือ signal นอก {0,1}")

    # E8 ตรวจ decision contract. ใต้ v3 READY_* ยังเป็น intent; E4 เป็นผู้ตรวจ
    # หลักฐาน execution แยกต่างหากเพื่อไม่เรียก decision ว่า fill จริง.
    ready_buy = status.eq(READY_BUY)
    ready_sell = status.eq(READY_SELL)
    ready = ready_buy | ready_sell
    valid_trade = ((ready_buy & sig.eq(1) & gap.lt(0) & qty.gt(0))
                   | (ready_sell & sig.eq(1) & gap.gt(0) & qty.gt(0)))
    valid_frozen = ~ready & qty.eq(0)
    bad = int((~(valid_trade | valid_frozen)).sum())
    equation8 = ("decision ต้อง signal=1 + READY + order qty>0; "
                 "act จริงต้อง FINALIZED + execution qty/price>0"
                 if bool(execution.any()) else
                 "signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง")
    add("E8", equation8, None, bad == 0,
        "" if bad == 0 else f"ผิด {bad} แถว")

    report = pd.DataFrame(checks, columns=["ข้อ", "สมการ/กฎ", "residual สูงสุด", "ผ่าน", "หมายเหตุ"])
    return report, bool(report["ผ่าน"].all())


def recompute_gated_ledger(df: pd.DataFrame, p0: float | None = None,
                           trace: list | None = None) -> pd.DataFrame:
    """คำนวณ recurrence ใหม่ตาม semantics ของแถว โดยไม่แก้ decision evidence.

    ``gated_theoretical_v2`` ใช้ decision price เมื่อ READY_* เป็น act ตามสัญญาเดิม.
    ``execution_confirmed_v1`` ใช้ ``execution_price`` และ act เฉพาะแถวที่
    ``cashflow_status=FINALIZED`` พร้อม ``execution_quantity>0`` และราคา > 0;
    pending/rejected/expired/PASS ล้วนแช่แข็ง. Rₙ ยังใช้ quote Pₙ ทุกแถว.
    semantics เก่าหรือไม่รู้จักคงค่าที่เก็บไว้ ไม่เดาสูตรใหม่ให้หลักฐานเก่า.

    ``trace`` (ไม่บังคับ): list ที่ถูกเติมทีละแถว (1 dict/แถว) ว่าแถวนั้นเดินสูตรไหนด้วยค่าอะไร
    — ใช้อธิบายที่มาของตัวเลขตอนชี้เมาส์ (``ledger_explanations``) ไม่มีผลต่อค่าที่คืน
    """
    def note(**fields) -> None:
        if trace is not None:
            trace.append(fields)

    if df.empty or any(c not in df.columns for c in RECOMPUTE_REQUIRED):
        # fail safe: ไม่ครบคอลัมน์ -> ไม่คำนวณ ledger เลย
        # คืน copy เสมอ: dashboard เป็น read-only จึงห้ามเขียนทับ frame ของผู้เรียก
        return _apply_reference_column(df.copy(), df, p0)

    source = df.reset_index(drop=True)
    out = source.copy()
    frozen_v2 = _semantics(source).isin(FROZEN_TERMINAL_SEMANTICS)
    p = out["ราคา Pₙ (USD)"].astype(float)
    fix_c = float((out["มูลค่าพอร์ต (USD)"].astype(float)
                   - out["ส่วนต่างเป้าหมาย (USD)"].astype(float)).iloc[0])
    step = out["DNA step"].astype(int)
    genesis = bool(step.iloc[0] == 0)
    if p0 is None and genesis:
        p0 = float(p.iloc[0])                          # แถว genesis: P₀ = ราคาแถวแรก
    can_ref = p0 is not None and p0 > 0
    semantics = _semantics(out).tolist()
    gated = _gated_flags(out)
    execution = _execution_flags(out)
    gated_acted = _traded_flags(out)
    execution_acted = _confirmed_execution_flags(out)
    if "execution_price" in out.columns:
        execution_prices = pd.to_numeric(out["execution_price"], errors="coerce")
    else:
        execution_prices = pd.Series([np.nan] * len(out), index=out.index)

    R = out["Rₙ อ้างอิง (USD)"].astype(float).tolist()
    dA = out["ΔAₙ ต่อสเต็ป (USD)"].astype(float).tolist()
    A = out["Aₙ สะสม (USD)"].astype(float).tolist()
    E = out["Eₙ ส่วนเกินสะสม (USD)"].astype(float).tolist()

    acted: float | None = None
    acted_idx: int | None = None        # แถวที่ตั้ง P_acted ล่าสุด (ไว้อธิบายตอนชี้เมาส์)
    A_prev = 0.0
    previous_semantics: str | None = None
    for i in range(len(out)):
        Pi = float(p.iloc[i])
        known = gated[i] or execution[i]
        if known and can_ref:
            R[i] = fix_c * math.log(Pi / p0)

        if i == 0:
            if genesis and known:
                # P_acted seed ของ chain คือ quote P₀. v2 genesis ยังเป็นศูนย์;
                # v3 genesis อาจถูก worker patch ภายหลังด้วย fill ที่มี slippage.
                acted, A_prev = Pi, 0.0
                acted_idx = 0
                if execution[i] and execution_acted[i]:
                    action_price = float(execution_prices.iloc[i])
                    d = fix_c * (action_price / acted - 1.0)
                    note(kind="genesis_fill", p_fill=action_price, acted_before=acted,
                         acted_idx_before=0, A_prev=0.0)
                    A_prev += d
                    dA[i], A[i], E[i] = d, A_prev, A_prev - R[i]
                    acted = action_price
                else:
                    note(kind="genesis", acted_before=acted, acted_idx_before=0,
                         A_prev=0.0)
                    dA[i], A[i], E[i] = 0.0, 0.0, 0.0
            else:
                # ชุดข้อมูลตัดหน้า chain: ไม่มี P_acted/A ก่อนแถวแรก จึงรักษา
                # baseline ที่ persist มา แล้วเริ่มตรวจ/recompute จากแถวถัดไป.
                A_prev = float(A[i])
                if execution[i] and execution_acted[i]:
                    acted = float(execution_prices.iloc[i])
                else:
                    acted = Pi
                acted_idx = 0
                note(kind="window_start", acted_before=acted, acted_idx_before=0,
                     A_prev=A_prev)
            previous_semantics = semantics[i]
            continue

        semantics_changed = semantics[i] != previous_semantics
        if not known:                                 # legacy/unknown: คงค่าเดิม
            acted, A_prev = Pi, float(A[i])
            acted_idx = i
            note(kind="stored")
            previous_semantics = semantics[i]
            continue

        if semantics_changed:
            # backend reset A baseline เมื่อ cashflow semantics เปลี่ยน แต่คง
            # P_acted จาก contract ก่อนหน้าไว้เป็น seed ของก้าวแรก.
            A_prev = 0.0
        if acted is None:                             # ไม่มี genesis ในชุด -> ใช้แถวก่อนหน้า
            acted, A_prev = float(p.iloc[i - 1]), float(A[i - 1])
            acted_idx = i - 1

        action_price: float | None = None
        if gated[i] and gated_acted[i]:
            action_price = Pi
        elif execution[i] and execution_acted[i]:
            action_price = float(execution_prices.iloc[i])

        step_note = dict(acted_before=acted, acted_idx_before=acted_idx,
                         A_prev=A_prev, reset=semantics_changed,
                         prev_semantics=previous_semantics)
        if action_price is not None:                  # act ที่ contract นั้นยืนยัน
            d = fix_c * (action_price / acted - 1.0)
            note(kind="act", p_fill=action_price,
                 fill_from="execution" if execution[i] else "quote", **step_note)
            A_prev += d
            dA[i], A[i], E[i] = d, A_prev, A_prev - R[i]
            acted = action_price
            acted_idx = i
        else:                                         # PASS/pending/rejected: แช่แข็ง
            note(kind="frozen", **step_note)
            dA[i], A[i] = 0.0, A_prev
            if can_ref:
                E[i] = A_prev - fix_c * math.log(acted / p0)
        previous_semantics = semantics[i]

    out["Rₙ อ้างอิง (USD)"] = R
    out["ΔAₙ ต่อสเต็ป (USD)"] = dA
    out["Aₙ สะสม (USD)"] = A
    out["Eₙ ส่วนเกินสะสม (USD)"] = E
    if bool(frozen_v2.any()):
        cols_to_preserve = [c for c in LEDGER_COLS if c in source.columns]
        out.loc[frozen_v2, cols_to_preserve] = source.loc[frozen_v2, cols_to_preserve]
    return _apply_reference_column(out, source, p0)


def count_ledger_corrections(stored: pd.DataFrame, fixed: pd.DataFrame,
                             tol: float = 1e-6) -> int:
    """นับแถวที่ recompute_gated_ledger แก้ค่า recurrence (stored ≠ fixed)

    รวม Rₙ ด้วย เพราะ Rₙ ต้องไม่เคยถูกแช่แข็ง — engine ที่เขียน Rₙ ค้างไว้ก็คือแถวที่ผิด
    ใช้เตือนบน dashboard ว่า engine เขียน ledger ผิดกี่แถว (ควรไปแก้ engine ต้นทาง)
    """
    if stored.empty or fixed.empty:
        return 0
    cols = [c for c in LEDGER_COLS if c in stored.columns and c in fixed.columns]
    if not cols:
        return 0
    a = stored.reset_index(drop=True)[cols].astype(float)
    b = fixed.reset_index(drop=True)[cols].astype(float)
    return int(((a - b).abs().max(axis=1) > tol).sum())


# ----------------------------------------------------------------------------
# ที่มาของตัวเลข Rₙ / ΔAₙ / Aₙ / Eₙ ต่อเซลล์ (ข้อความ tooltip ตอนชี้เมาส์บนตาราง)
# ข้อความมาจาก trace ของ recompute_gated_ledger เอง — สูตรกับตัวเลขที่แทนค่าจึงเป็นชุด
# เดียวกับที่คำนวณจริง ไม่ใช่การเดาย้อนจากค่าที่แสดง
# ----------------------------------------------------------------------------
_EXECUTION_SEMANTICS = (EXECUTION_CONFIRMED_SEMANTICS, *FROZEN_TERMINAL_SEMANTICS)


def _f(x: float, digits: int = 4) -> str:
    return f"{x:,.{digits}f}"


def _op(x: float, digits: int = 4) -> str:
    """ตัวถูกดำเนินการหลังเครื่องหมาย (+ − ×): ค่าติดลบใส่วงเล็บให้อ่านง่าย"""
    text = _f(x, digits)
    return f"({text})" if text.startswith("-") else text


def _meta_num(row, key: str) -> float | None:
    """ฟิลด์ meta ที่เป็นตัวเลข finite; ไม่มี/NaN/อ่านไม่ได้ -> None"""
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _meta_flag(row, key: str) -> bool:
    value = row.get(key)
    return isinstance(value, (bool, np.bool_)) and bool(value)


class _LedgerExplainer:
    def __init__(self, df: pd.DataFrame, p0: float | None):
        self.src = df.reset_index(drop=True)
        self.trace: list[dict] = []
        self.fixed = recompute_gated_ledger(
            self.src, p0=p0, trace=self.trace).reset_index(drop=True)
        self.traced = len(self.trace) == len(self.src)
        self.ctx = _reference_context(self.src, p0)       # (P₀, FIX_C) | None
        self.known = _known_cashflow_flags(self.src)
        self.semantics = _semantics(self.src).tolist()
        self.frozen_v2 = [s in FROZEN_TERMINAL_SEMANTICS for s in self.semantics]
        self.p0_from = ("state pointer (anchor ของ chain)" if p0 is not None
                        else "ราคา Pₙ ของแถว genesis")
        has_basis = {"มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)"}.issubset(self.src.columns)
        self.fix_c = (float(self.src["มูลค่าพอร์ต (USD)"].astype(float).iloc[0]
                            - self.src["ส่วนต่างเป้าหมาย (USD)"].astype(float).iloc[0])
                      if has_basis and len(self.src) else None)

    # ---- helpers -----------------------------------------------------------
    def _val(self, i: int, col: str) -> float:
        return float(self.fixed.at[i, col])

    def _where(self, idx: int | None) -> str:
        if idx is None or "DNA step" not in self.src.columns:
            return "—"
        return f"DNA step {int(self.src['DNA step'].iloc[idx])}"

    def _acted_origin(self, t: dict) -> str:
        """P_acted มาจากแถวไหน: seed ของ genesis (ยังไม่เคย act) หรือราคาแถว act ล่าสุด"""
        idx = t.get("acted_idx_before")
        kind = self.trace[idx]["kind"] if idx is not None else None
        if kind == "genesis":
            return "P₀ = quote ของแถว genesis (ยังไม่เคย act)"
        if kind == "window_start":
            return f"ราคาแถวแรกของช่วงที่โหลด · {self._where(idx)}"
        return f"ราคา act ล่าสุด · {self._where(idx)}"

    def _freeze_reason(self, i: int) -> str:
        row = self.src.iloc[i]
        status = str(row.get("สถานะ", ""))
        cashflow = row.get("cashflow_status")
        cashflow = "—" if pd.isna(cashflow) else str(cashflow)
        if status == PASS_DNA_ZERO:
            return "PASS_DNA_ZERO — DNA signal = 0 ไม่เทรด"
        if status == PASS_THRESHOLD:
            return "PASS_THRESHOLD — signal = 1 แต่ |ส่วนต่างเป้าหมาย| ≤ DIFF ไม่ยิง order"
        if status in (READY_BUY, READY_SELL):
            if self.semantics[i] in _EXECUTION_SEMANTICS:
                return (f"{status} เป็นแค่ intent — cashflow_status = {cashflow} "
                        "ยังไม่มี fill ที่ FINALIZED (qty/price > 0)")
            return f"{status} แต่ DNA signal ≠ 1 หรือจำนวนสั่ง = 0 จึงไม่นับเป็น act"
        return f"สถานะ {status} ไม่ใช่การเทรด"

    def _title(self, i: int, symbol: str, label: str) -> str:
        step = int(self.src["DNA step"].iloc[i]) if "DNA step" in self.src.columns else i
        return f"{symbol} · DNA step {step} · {label}"

    def _stored(self, i: int, symbol: str, col: str, why: str) -> str:
        return "\n".join([self._title(i, symbol, "ค่าที่บันทึกไว้"),
                          f"{symbol} = {_f(self._val(i, col))}", why])

    # ---- Rₙ ---------------------------------------------------------------
    def reference(self, i: int) -> str:
        col = "Rₙ อ้างอิง (USD)"
        if col not in self.fixed.columns:
            return ""
        R = self._val(i, col)
        if self.ctx is None or not self.known[i]:
            why = (f"semantics {self.semantics[i]!r} เก่า/ไม่รู้จัก — dashboard ไม่คำนวณทับ"
                   if not self.known[i] else
                   "ไม่ทราบ P₀ (ไม่มีแถว genesis และไม่มี state pointer)")
            return self._stored(i, "Rₙ", col, why)
        p0, fix_c = self.ctx
        P = float(self.src["ราคา Pₙ (USD)"].iloc[i])
        return "\n".join([
            self._title(i, "Rₙ", "เส้นอ้างอิง ไม่เคยแช่แข็ง"),
            "Rₙ = FIX_C × ln(Pₙ / P₀)",
            f"   = {_f(fix_c)} × ln({_f(P)} / {_f(p0)})",
            f"   = {_f(fix_c)} × {_op(math.log(P / p0), 6)}",
            f"   = {_f(R)}",
            f"FIX_C = มูลค่าพอร์ต − ส่วนต่างเป้าหมาย (แถวแรก) = {_f(fix_c)}",
            f"P₀ = {_f(p0)} จาก {self.p0_from}",
            "Pₙ คือ quote ของแถวนี้ — Rₙ ขยับทุกแถวแม้ไม่ได้เทรด",
        ])

    # ---- ΔAₙ / Aₙ / Eₙ ----------------------------------------------------
    def delta(self, i: int) -> str:
        col = "ΔAₙ ต่อสเต็ป (USD)"
        if col not in self.fixed.columns:
            return ""
        if not self.traced:
            return self._stored(i, "ΔAₙ", col, "คอลัมน์ไม่ครบ — dashboard ไม่คำนวณ ledger ใหม่")
        if self.frozen_v2[i]:
            return self._persisted(i, "delta")
        t = self.trace[i]
        kind, d = t["kind"], self._val(i, col)
        if kind in ("act", "genesis_fill"):
            fix_c, pf, pa = self.fix_c, t["p_fill"], t["acted_before"]
            fill = ("ราคา fill จริง (execution_price) — quote Pₙ = "
                    f"{_f(float(self.src['ราคา Pₙ (USD)'].iloc[i]))}"
                    if t.get("fill_from", "execution") == "execution" else
                    "Pₙ ของแถวนี้ (ราคาตัดสินใจ — gated_theoretical_v2)")
            return "\n".join([
                self._title(i, "ΔAₙ", "act — เทรดจริง"),
                "ΔAₙ = FIX_C × (P_fill / P_acted − 1)",
                f"    = {_f(fix_c)} × ({_f(pf)} / {_f(pa)} − 1)",
                f"    = {_f(d)}",
                f"P_fill = {_f(pf)} = {fill}",
                f"P_acted = {_f(pa)} = {self._acted_origin(t)}",
                "หลัง act นี้ P_acted เลื่อนมาเป็น P_fill",
            ])
        if kind == "frozen":
            return "\n".join([
                self._title(i, "ΔAₙ", "แช่แข็ง ไม่เทรดรอบนี้"),
                "ΔAₙ = 0",
                f"เหตุผล: {self._freeze_reason(i)}",
                "ไม่มี order → ไม่มีกำไร/ขาดทุนก้าวนี้",
                f"P_acted ค้างที่ {_f(t['acted_before'])} = {self._acted_origin(t)}",
            ])
        if kind == "genesis":
            return "\n".join([
                self._title(i, "ΔAₙ", "genesis — แถวแรกของ chain"),
                "ΔAₙ = 0 — ยังไม่เคยเทรด ไม่มีราคา act ให้เทียบ",
                f"P_acted ตั้งต้น = {_f(t['acted_before'])} = {self._acted_origin(t)}",
            ])
        if kind == "window_start":
            return self._stored(i, "ΔAₙ", col, "แถวแรกของช่วงที่โหลด (chain ถูกตัดหน้า) "
                                "ไม่มีแถวก่อนให้เทียบ — คงค่าที่บันทึก คำนวณใหม่จากแถวถัดไป")
        return self._stored(i, "ΔAₙ", col, f"semantics {self.semantics[i]!r} เก่า/ไม่รู้จัก "
                            "— dashboard ไม่เดาสูตรใหม่ให้")

    def actual(self, i: int) -> str:
        col = "Aₙ สะสม (USD)"
        if col not in self.fixed.columns:
            return ""
        if not self.traced:
            return self._stored(i, "Aₙ", col, "คอลัมน์ไม่ครบ — dashboard ไม่คำนวณ ledger ใหม่")
        if self.frozen_v2[i]:
            return self._persisted(i, "actual")
        t = self.trace[i]
        kind, A = t["kind"], self._val(i, col)
        d = self._val(i, "ΔAₙ ต่อสเต็ป (USD)")
        reset = ([f"Aₙ₋₁ ถูกรีเซ็ตเป็น 0 เพราะ semantics เปลี่ยน "
                  f"({t.get('prev_semantics')} → {self.semantics[i]})"]
                 if t.get("reset") else [])
        if kind in ("act", "genesis_fill"):
            return "\n".join([
                self._title(i, "Aₙ", "สะสมต่อจากก้าวก่อน"),
                "Aₙ = Aₙ₋₁ + ΔAₙ",
                f"   = {_f(t['A_prev'])} + {_op(d)}",
                f"   = {_f(A)}", *reset])
        if kind == "frozen":
            return "\n".join([
                self._title(i, "Aₙ", "ค้างไว้ (แช่แข็ง)"),
                "Aₙ = Aₙ₋₁   (ΔAₙ = 0 จึงไม่เพิ่ม)",
                f"   = {_f(t['A_prev'])}", *reset])
        if kind == "genesis":
            return "\n".join([self._title(i, "Aₙ", "genesis — แถวแรกของ chain"),
                              "Aₙ = 0 — เริ่มนับสะสมจากศูนย์"])
        if kind == "window_start":
            return self._stored(i, "Aₙ", col, "baseline ที่บันทึกไว้ ณ แถวแรกของช่วงที่โหลด")
        return self._stored(i, "Aₙ", col, f"semantics {self.semantics[i]!r} เก่า/ไม่รู้จัก "
                            "— dashboard ไม่เดาสูตรใหม่ให้")

    def excess(self, i: int) -> str:
        col = "Eₙ ส่วนเกินสะสม (USD)"
        if col not in self.fixed.columns:
            return ""
        if not self.traced:
            return self._stored(i, "Eₙ", col, "คอลัมน์ไม่ครบ — dashboard ไม่คำนวณ ledger ใหม่")
        if self.frozen_v2[i]:
            return self._persisted(i, "excess")
        t = self.trace[i]
        kind, E = t["kind"], self._val(i, col)
        A = self._val(i, "Aₙ สะสม (USD)")
        if kind in ("act", "genesis_fill"):
            R = self._val(i, "Rₙ อ้างอิง (USD)")
            return "\n".join([
                self._title(i, "Eₙ", "act — เทียบกับเส้นอ้างอิงที่ราคาแถวนี้"),
                "Eₙ = Aₙ − Rₙ",
                f"   = {_f(A)} − {_op(R)}",
                f"   = {_f(E)}",
                "บวก = ที่ทำได้จริง (Aₙ) เกินเส้นอ้างอิง (Rₙ)"])
        if kind == "frozen" and self.ctx is not None:
            p0, fix_c = self.ctx
            pa = t["acted_before"]
            ref = fix_c * math.log(pa / p0)
            return "\n".join([
                self._title(i, "Eₙ", "แช่แข็ง (smooth)"),
                "Eₙ = Aₙ − FIX_C × ln(P_acted / P₀)",
                f"   = {_f(A)} − {_f(fix_c)} × ln({_f(pa)} / {_f(p0)})",
                f"   = {_f(A)} − {_op(ref)}",
                f"   = {_f(E)}",
                "ใช้ P_acted (ไม่ใช่ Pₙ) → ค่านิ่งช่วงไม่เทรด ไม่แกว่งตามราคาตลาด"])
        if kind == "genesis":
            return "\n".join([self._title(i, "Eₙ", "genesis — แถวแรกของ chain"),
                              "Eₙ = 0 — ยังไม่มีส่วนเกิน (Aₙ = 0)"])
        if kind in ("frozen", "window_start"):
            why = ("ไม่ทราบ P₀ (ไม่มีแถว genesis และไม่มี state pointer)" if kind == "frozen"
                   else "baseline ที่บันทึกไว้ ณ แถวแรกของช่วงที่โหลด")
            return self._stored(i, "Eₙ", col, why)
        return self._stored(i, "Eₙ", col, f"semantics {self.semantics[i]!r} เก่า/ไม่รู้จัก "
                            "— dashboard ไม่เดาสูตรใหม่ให้")

    # ---- แถว frozen terminal v2/v3: ค่า ledger เป็นของ worker (ห้ามคำนวณทับ) ----
    def _persisted(self, i: int, which: str) -> str:
        row = self.src.iloc[i]
        symbol, col = {"delta": ("ΔAₙ", "ΔAₙ ต่อสเต็ป (USD)"),
                       "actual": ("Aₙ", "Aₙ สะสม (USD)"),
                       "excess": ("Eₙ", "Eₙ ส่วนเกินสะสม (USD)")}[which]
        value = self._val(i, col)
        title = self._title(i, symbol, "ค่าที่ worker บันทึก (terminal fill)")
        keep = "dashboard แสดงค่าที่ freeze ไว้ ไม่คำนวณทับ"
        finalized = row.get("cashflow_status") == CASHFLOW_FINALIZED
        price = _meta_num(row, "execution_price")
        prev_price = _meta_num(row, "previous_action_price")
        prev_A = _meta_num(row, "previous_actual_cumulative")
        basis = _meta_num(row, "R_basis")
        if which == "delta":
            if finalized and _meta_flag(row, "initial_funding"):
                return "\n".join([title, "ΔAₙ = 0 — initial funding",
                                  f"policy {FUNDING_BASELINE_POLICY}: ซื้อครั้งแรกเป็นเงินตั้งต้น "
                                  "ไม่ใช่กำไร/ขาดทุน", keep])
            if finalized and price is not None and prev_price and self.fix_c is not None:
                return "\n".join([
                    title, "ΔAₙ = FIX_C × (execution_price / previous_action_price − 1)",
                    f"    = {_f(self.fix_c)} × ({_f(price)} / {_f(prev_price)} − 1)",
                    f"    = {_f(value)}", keep])
            if not finalized:
                status = row.get("cashflow_status")
                return "\n".join([title, "ΔAₙ = 0 — ยังไม่มี fill ที่ FINALIZED "
                                  f"(cashflow_status = {'—' if pd.isna(status) else status})",
                                  keep])
        elif which == "actual":
            d = self._val(i, "ΔAₙ ต่อสเต็ป (USD)")
            if finalized and prev_A is not None:
                return "\n".join([title, "Aₙ = previous_actual_cumulative + ΔAₙ",
                                  f"   = {_f(prev_A)} + {_op(d)}", f"   = {_f(value)}", keep])
            if not finalized:
                return "\n".join([title, f"Aₙ = {_f(value)} — ค้างที่ fill ล่าสุดที่มองเห็นตอนสังเกต",
                                  "แถวที่ไม่มี fill ไม่ขยับ Aₙ", keep])
        else:
            A = self._val(i, "Aₙ สะสม (USD)")
            if basis is not None:
                lines = [title, "Eₙ = Aₙ − R_basis", f"   = {_f(A)} − {_op(basis)}",
                         f"   = {_f(value)}",
                         "R_basis = ฐานอ้างอิง ณ fill ที่ worker บันทึก (Rₙ − funding_reference_offset)",
                         keep]
                mark = _meta_num(row, "E_mark_at_observation")
                if mark is not None:
                    lines.insert(-1, f"E_mark (ตามราคาตลาดตอนสังเกต) = {_f(mark)} — แยกต่างหาก "
                                     "ไม่ใช่ Eₙ")
                return "\n".join(lines)
        return "\n".join([title, f"{symbol} = {_f(value)}", keep])


def ledger_explanations(df: pd.DataFrame, p0: float | None = None) -> list[dict[str, str]]:
    """ที่มาของตัวเลข Rₙ / ΔAₙ / Aₙ / Eₙ ของทุกแถว (ลำดับตรงกับ ``recompute_gated_ledger``)

    คืน list ยาวเท่าจำนวนแถว แต่ละช่อง = ``{ชื่อคอลัมน์ ledger: ข้อความหลายบรรทัด}``
    บรรทัดแรกเป็นหัวข้อ ที่เหลือคือสูตรพร้อมแทนค่าจริงของแถวนั้น. read-only:
    เรียก ``recompute_gated_ledger`` ด้วย ``trace`` แล้วเล่าเฉพาะสิ่งที่มันทำ
    """
    if df.empty:
        return []
    ex = _LedgerExplainer(df, p0)
    return [{"Rₙ อ้างอิง (USD)": ex.reference(i),
             "ΔAₙ ต่อสเต็ป (USD)": ex.delta(i),
             "Aₙ สะสม (USD)": ex.actual(i),
             "Eₙ ส่วนเกินสะสม (USD)": ex.excess(i)}
            for i in range(len(ex.src))]


# ============================================================================
# Rebalancing 101 — gated demo (DNA gate + บัญชีแบบแช่แข็ง / frozen ledger)
# ----------------------------------------------------------------------------
# playground เชิงสอน: สุ่มราคาแล้วเดิน ledger historical gated_theoretical_v2 เพื่อเห็นว่า
# "รอบที่ DNA signal=1 แต่ตัดสินใจ PASS (จำนวนสั่ง = 0)" ถูกบันทึก
# ในบัญชีแบบแช่แข็งอย่างไร — holdings แช่ตั้งแต่ act ล่าสุด, ΔAₙ = 0, Eₙ ค้าง (smooth)
# สูตรตรงกับ gated_rebalancing_cashflow_from_prices ของ Webull_Dashboard/manual_tools.py
# และ compute_recurrence ของ lego-firebase/lego_one_row.py
# ============================================================================

MAX_REBALANCING_STEPS = 2000
DEFAULT_GATED_DNA = "26021034252903219354832053493"


def simulate_rebalancing_prices(p0: float, vol: float, drift: float,
                                steps: int, seed: int) -> list[float]:
    """เส้นราคาสุ่มของ Testing Lab (geometric Brownian ต่อรอบ) — deterministic

    ``Pᵢ = Pᵢ₋₁ × exp((drift − vol²/2) + vol × Z)`` โดย ``Z`` จาก default_rng(seed)
    floor ที่ ``P₀ × 1e-8`` เพื่อให้ ln(Pᵢ/P₀) นิยามได้เสมอ
    """
    if not all(math.isfinite(float(x)) for x in (p0, vol, drift)):
        raise ValueError("p0, vol, and drift must be finite")
    if p0 <= 0:
        raise ValueError("p0 must be greater than 0")
    if vol < 0:
        raise ValueError("vol cannot be negative")
    if steps < 2 or steps > MAX_REBALANCING_STEPS:
        raise ValueError(f"steps must be between 2 and {MAX_REBALANCING_STEPS}")

    rng = np.random.default_rng(int(seed))
    prices: list[float] = []
    price = float(p0)
    for _ in range(int(steps)):
        shock = (drift - 0.5 * vol * vol) + vol * float(rng.standard_normal())
        price = max(price * math.exp(shock), p0 * 1e-8)
        prices.append(price)
    return prices


def rebalancing_cashflow_from_prices(prices: Iterable[float], fix_c: float,
                                     p0: float) -> list[dict[str, float]]:
    """เส้น pass-all (act ทุกรอบ) — baseline เทียบกับ gated

    Step 0 anchor ที่ P₀ ; ทุกราคา ``Pᵢ``: ``ΔAᵢ = Fix_c × (Pᵢ/Pᵢ₋₁ − 1)``,
    ``Rₙ = Fix_c × ln(Pₙ/P₀)``, ``Eₙ = Aₙ − Rₙ``
    (เท่ากับ gated ที่ actions เป็น 1 ทุกตัว)
    """
    if not math.isfinite(float(fix_c)) or not math.isfinite(float(p0)):
        raise ValueError("fix_c and p0 must be finite")
    if fix_c <= 0 or p0 <= 0:
        raise ValueError("fix_c and p0 must be greater than 0")

    rows: list[dict[str, float]] = [{
        "step": 0, "price": float(p0), "delta_actual": 0.0,
        "actual_cumulative": 0.0, "ln_reference": 0.0, "excess": 0.0,
    }]
    previous = float(p0)
    actual = 0.0
    for step, raw_price in enumerate(prices, start=1):
        price = float(raw_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Every price must be finite and greater than 0")
        delta = fix_c * (price / previous - 1.0)
        actual += delta
        reference = fix_c * math.log(price / p0)
        rows.append({
            "step": step, "price": price, "delta_actual": float(delta),
            "actual_cumulative": float(actual), "ln_reference": float(reference),
            "excess": float(actual - reference),
        })
        previous = price
    return rows


def gated_rebalancing_cashflow_from_prices(prices: Iterable[float], fix_c: float,
                                           p0: float,
                                           actions: Iterable[int]
                                           ) -> list[dict[str, float]]:
    """Gated demo ledger (gated_theoretical_v2) — เหมือน lego-firebase engine

    ``actions[i] ∈ {0,1}`` คือ gate ของรอบ ``i+1`` (รอบ 0 = จุด anchor เสมอ):
      act (1):  ``ΔAᵢ = Fix_c × (Pᵢ/P_acted − 1)`` โดย ``P_acted`` = ราคารอบ act
                ล่าสุด แล้วเลื่อน ``P_acted = Pᵢ`` ; ``Eᵢ = Aᵢ − Rᵢ``
      pass (0): ``ΔAᵢ = 0``, ``Aᵢ`` ค้าง, ``P_acted`` แช่แข็ง และ
                ``Eᵢ = Aᵢ − Fix_c × ln(P_acted/P₀)`` (smooth — ค้างค่า act ล่าสุด)

    เหตุผลเศรษฐศาสตร์: ช่วง pass ไม่มีการ rebalance — holdings แช่แข็งตั้งแต่
    act ล่าสุด กำไรจริงจึงเป็นก้อนเดียวเทียบราคาแช่แข็งตอน act ใหม่
    คุณสมบัติ: ``Eₙ`` ไม่ลด และ ≥ 0 เสมอ (จาก ``x − 1 ≥ ln x`` ต่อ segment)
    """
    if not math.isfinite(float(fix_c)) or not math.isfinite(float(p0)):
        raise ValueError("fix_c and p0 must be finite")
    if fix_c <= 0 or p0 <= 0:
        raise ValueError("fix_c and p0 must be greater than 0")

    rows: list[dict[str, float]] = [{
        "step": 0, "action": 1, "price": float(p0), "acted_price": float(p0),
        "delta_actual": 0.0, "actual_cumulative": 0.0, "ln_reference": 0.0,
        "excess": 0.0,
    }]
    acted = float(p0)
    actual = 0.0
    for step, (raw_price, raw_action) in enumerate(zip(prices, actions), start=1):
        price = float(raw_price)
        action = int(raw_action)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Every price must be finite and greater than 0")
        if action not in (0, 1):
            raise ValueError("Every action must be 0 or 1")
        reference = fix_c * math.log(price / p0)
        if action == 1:
            delta = fix_c * (price / acted - 1.0)
            actual += delta
            acted = price
            excess = actual - reference
        else:
            delta = 0.0
            excess = actual - fix_c * math.log(acted / p0)
        rows.append({
            "step": step, "action": action, "price": price,
            "acted_price": float(acted), "delta_actual": float(delta),
            "actual_cumulative": float(actual), "ln_reference": float(reference),
            "excess": float(excess),
        })
    return rows


def build_gate_actions(dna_code: str, n_rounds: int) -> list[int]:
    """decode DNA -> gate array 0/1 แล้วตัด/วนให้ยาวเท่า n_rounds (รอบ act ต่อ demo)

    ยึด decode เดียวกับ engine (dna_engine.decode_dna) — gate เพี้ยนไม่ได้
    DNA สั้นกว่าจำนวนรอบ -> วนซ้ำ (เฉพาะ demo; engine จริง step +1 จน DNA หมดแล้ว fail)
    """
    if n_rounds < 0:
        raise ValueError("n_rounds ต้อง >= 0")
    gate = [int(x) for x in decode_dna(dna_code.strip())]
    if not gate:
        raise ValueError("decode_dna คืน array ว่าง")
    if len(gate) < n_rounds:
        reps = -(-n_rounds // len(gate))       # ceil division
        gate = gate * reps
    return gate[:n_rounds]



def system_health_rows(warnings) -> pd.DataFrame:
    """Bounded operator view; resolved incidents remain history, not live alarms."""
    if not isinstance(warnings, dict):
        return pd.DataFrame()
    records = []
    for key, value in warnings.items():
        if not isinstance(value, dict):
            continue
        active = value.get("active")
        records.append({"kind": value.get("kind", str(key)),
                        "state": "ACTIVE" if active is True else (
                            "RESOLVED" if active is False else "HISTORY"),
                        "count": value.get("count"),
                        "last_at": value.get("last_at") or value.get("at"),
                        "retry_after": value.get("retry_after"),
                        "resolved_at": value.get("resolved_at")})
    return pd.DataFrame(records)
