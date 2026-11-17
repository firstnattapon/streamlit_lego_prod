"""lego_payoff.py — เส้น payoff จาก position จริงใน Webull (pure: ไม่ import streamlit / webull SDK)

แท็บ 📐 Payoff มี 2 มุมมองที่ **ห้ามปนกัน** (model ledger ≠ กำไรจริง):

💵 บัญชีจริง (Webull) — แกน y = P/L เทียบทุนของ Webull, ต่อ 1 underlying (S₀ = ``last_price`` ของหุ้นใน
positions, U₀ = Σ ``unrealized_profit_loss`` ทุก leg ของตัวนั้น)
  ① T+0 วันนี้        U₀ + q·(S−S₀) + Σoption m·n·[BS(S) − BS(S₀)]                         ผ่าน (S₀, U₀)
  ② ณ หมดอายุ LEAPS  U₀ + q·(S−S₀) + Σoption m·n·[V_h(S) − mark]   h = วันหมดอายุใกล้สุด
                     option ที่หมดวัน h ใช้ intrinsic · ที่หมดทีหลังใช้ BS ด้วย T ที่เหลือ ณ วัน h
  ③ bot rebalance ต่อ U₀ + [F(S) − F(S₀)] + Σoption m·n·[BS(S) − BS(S₀)]                      ผ่าน (S₀, U₀)
     F(u) = FIX_C·ln(u') − L·u' โดย u' = min(u, u*) และ u* = FIX_C / L (L = leaps_shares ของ bot)
     = ΔRₙ ของ LEGO ถ้า LEAPS เป็นหุ้นจริง + ส่วนต่างระหว่าง LEAPS จริงกับหุ้น L ตัว:
       FIX_C·ln(S/S₀) + [m·n·ΔBS − L·(S − S₀)] เมื่อ S, S₀ ≤ u*
     หุ้นจริงเป้าหมาย = FIX_C/u − L และ SELL ได้ไม่เกินหุ้นจริง (PASS_SELL_EXCEEDS_REAL) → เกิน u* ส่วนหุ้นราบ
     rebalance ต่อเนื่อง ไม่จำลอง DNA gate/DIFF และไม่นับ E ที่ได้เพิ่มจากการแกว่ง จึงเป็นค่าขั้นต่ำ
  ④ ③ + Eₙ           เส้น ③ บวก Eₙ สะสมของ bot (``execution_cashflow.excess`` ใน state) — E ที่เก็บจากความแกว่งแล้ว
                     ไม่อยู่ใน unrealized ของ Webull (เป็นเงินสดที่ปิดไปแล้ว) · Eₙ มาจาก model ledger ของ bot
                     เส้นนี้จึงเป็นเส้นเดียวที่รวม 2 ledger และระบุชื่อไว้ชัด · ห่างจาก ③ เท่ากับ Eₙ ทุกราคา

🧬 LEGO model ledger — ตัวเลขชุดเดียวกับแท็บ Live (semantics execution_terminal_funding_v3: Eₙ = Aₙ − R_basis)
  R(S)       = R_basis + FIX_C·ln(S / P_acted)
  พื้น A(S)   = Aₙ + FIX_C·[ln(min(S,u*)) − ln(min(P_acted,u*))]     (เกิน u* = pass → A แช่แข็ง)
  ที่ S = P_acted ได้ R_basis และ Aₙ ตรงกับ state · ของจริง A ≥ พื้นนี้ เพราะทุก act ได้ FIX_C·[(x−1) − ln x] ≥ 0

Fail closed: ข้อมูลที่เส้นไหนต้องใช้ขาดหรือไม่ตรงกัน → เส้นนั้นเป็น NaN ทั้งเส้น (หน้าเว็บซ่อนและบอกเหตุผล) ไม่ประมาณ
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lego_charts import FILL_ALPHA, FONT_FAMILY, LINE, REFERENCE_DASH, theme_tokens

RISK_FREE = 0.04                 # r ของ Black-Scholes (แสดงใน caption) · ไม่มีปันผล
GRID_POINTS = 241
MIN_GRID_PCT = 1.0               # ln(0) = −∞ → ข้อมูลเริ่มที่ 1% ของ S₀ แม้ช่วงที่เลือกเริ่มที่ 0 (แกน x ยังเริ่มที่ 0)
NEW_YORK = ZoneInfo("America/New_York")
US_CLOSE = time(16, 0)           # option หมดอายุ 16:00 เวลา New York
YEAR_SECONDS = 365.0 * 86400.0
CONSISTENCY_TOL = 0.01           # หุ้น: Webull กับ state ของ bot ต่างได้ไม่เกินนี้
IV_BOUNDS = (1e-4, 5.0)

VALUE_COL, GAP_COL = "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)"   # FIX_C = Vₙ − gapₙ (E1)
LINE_KEYS = ("t0", "expiry", "rebalance", "rebalance_e")
LINE_NAMES = {"t0": "① T+0 วันนี้", "expiry": "② ณ หมดอายุ LEAPS", "rebalance": "③ bot rebalance ต่อ",
              "rebalance_e": "④ ③ + Eₙ"}
LINE_SHORT = {"t0": "①", "expiry": "②", "rebalance": "③", "rebalance_e": "④"}
LEGO_NAMES = {"R": "R(S) อ้างอิง", "A": "พื้น A(S)"}
PRICE_COL, PCT_COL = "price", "pct"
TABLE_PCTS = (50, 75, 90, 100, 110, 125, 150, 200)


# ---------------------------------------------------------------- อ่านค่า

def _num(value) -> float | None:
    """ตัวเลข finite หรือ None (รับ str แบบที่ Webull ส่งมา เช่น "60.00000")"""
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_token_txt(text) -> tuple[str, int, str]:
    """เนื้อหา token.txt ของ SDK: บรรทัด 1 token · 2 expires (ms) · 3 status (ตัดบรรทัดว่างหัวท้าย)"""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 3 or not lines[1].isdigit():
        raise ValueError("WEBULL_TOKEN_TXT ต้องมี 3 บรรทัด: token / expires / status")
    return lines[0], int(lines[1]), lines[2]


def occ_symbol(symbol: str, expiry: date, kind: str, strike: float) -> str:
    """สัญลักษณ์ option แบบ OCC เช่น ``AAA270521C00007000``"""
    return f"{str(symbol).upper()}{expiry:%y%m%d}{'C' if kind == 'call' else 'P'}{round(strike * 1000):08d}"


def _date(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Leg:
    """หนึ่งขาของ position — หุ้น หรือ option หนึ่งสัญญา (จำนวนติดเครื่องหมาย: SELL = ลบ)"""
    symbol: str                  # underlying
    kind: str                    # "stock" | "call" | "put"
    qty: float                   # หุ้น หรือ จำนวนสัญญา
    multiplier: float            # หุ้น = 1, option = option_contract_multiplier
    mark: float | None           # last_price จาก positions (option combo: mid ของ snapshot)
    unrealized: float            # unrealized_profit_loss ของ Webull (รวมค่าธรรมเนียมแล้ว)
    cost: float | None = None
    strike: float | None = None
    expiry: date | None = None
    iv: float | None = None
    delta: float | None = None
    iv_source: str | None = None  # "webull" | "mark"
    occ: str | None = None

    @property
    def is_option(self) -> bool:
        return self.kind in ("call", "put")


# OCC: ชื่อหุ้น + yymmdd + C/P + strike×1000 (8 หลัก) — แบบมาตรฐานมีช่องว่างคั่นหลังชื่อหุ้นได้
_OCC = re.compile(r"^([A-Z][A-Z0-9.]*?)(\d{6})([CP])(\d{8})$")
SKIP_PREFIX = "ข้าม position ที่อ่านไม่ได้: "


def parse_occ(symbol) -> dict | None:
    """``AAA270521C00007000`` → {symbol, expiry, kind, strike} · ไม่ใช่ OCC → None"""
    match = _OCC.match(str(symbol or "").upper().replace(" ", ""))
    if not match:
        return None
    root, ymd, cp, strike = match.groups()
    try:
        expiry = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError:
        return None
    return {"symbol": root, "expiry": expiry, "kind": "call" if cp == "C" else "put",
            "strike": int(strike) / 1000.0}


def _option_rows(position: Mapping) -> list[Mapping]:
    """ขาของ option: ``legs`` (Webull OpenAPI v3 ตามเอกสาร) หรือ ``items`` (รูปแบบของ Webull MCP)
    · ไม่มีทั้งคู่แต่ตัว position เป็น option เอง (มี option_type หรือ symbol เป็น OCC) → ใช้ position เป็นขาเดียว"""
    for key in ("legs", "items"):
        rows = position.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, Mapping)]
    if position.get("option_type") or parse_occ(position.get("symbol")):
        return [position]
    return []


def option_contracts(raw_positions: Iterable[Mapping]) -> list[str]:
    """สัญลักษณ์ OCC ของทุก option ใน positions (ไม่ซ้ำ เรียงตามลำดับที่พบ)"""
    out: list[str] = []
    for position in raw_positions or ():
        if not isinstance(position, Mapping):
            continue
        for row in _option_rows(position):
            spec = _option_spec(position, row)
            if spec and spec["occ"] not in out:
                out.append(spec["occ"])
    return out


def _option_spec(position: Mapping, row: Mapping) -> dict | None:
    """ชนิด / วันหมดอายุ / strike จาก field ของขา · ขาดตัวไหนใช้ค่าจากสัญลักษณ์ OCC (ของขาหรือของ position)
    · ชื่อหุ้น = รากของ OCC ถ้ามี ไม่งั้น symbol ของขาหรือของ position"""
    if not isinstance(row, Mapping):
        return None
    occ = parse_occ(row.get("symbol")) or parse_occ(position.get("symbol")) or {}
    kind = str(row.get("option_type") or occ.get("kind") or "").lower()
    expiry = _date(row.get("option_expire_date")) or occ.get("expiry")
    strike = _num(row.get("option_exercise_price"))
    strike = strike if strike is not None else occ.get("strike")
    symbol = str(occ.get("symbol") or row.get("symbol") or position.get("symbol") or "").upper()
    if kind not in ("call", "put") or expiry is None or strike is None or not symbol:
        return None
    return {"symbol": symbol, "kind": kind, "expiry": expiry, "strike": strike,
            "occ": occ_symbol(symbol, expiry, kind, strike)}


def _leg_sign(position: Mapping, row: Mapping, count: float | None) -> float:
    """ซื้อ (+) หรือขาย (−): ``side`` ถ้ามี (MCP) → v3 ไม่มี side ให้ดูเครื่องหมายของจำนวน → ไม่บอก = ซื้อ"""
    side = str(row.get("side") or position.get("side") or "").upper()
    if side in ("SELL", "SHORT", "SELL_SHORT"):
        return -1.0
    if side in ("BUY", "LONG"):
        return 1.0
    total = _num(position.get("quantity"))
    return -1.0 if (count is not None and count < 0) or (total is not None and total < 0) else 1.0


def _field_names(mapping: Mapping) -> str:
    """ชื่อ field เท่านั้น (ไม่มีค่า) — ใช้บอกว่า Webull ส่งอะไรมาเมื่อแปลงไม่ได้"""
    return ", ".join(sorted(str(key) for key in mapping.keys())) or "(ว่าง)"


def underlying_symbols(raw_positions) -> list[str]:
    """ชื่อหุ้นของทุก position (option ใช้ชื่อหุ้นอ้างอิง ไม่ใช่สัญลักษณ์ OCC) — ใช้ขอราคาหุ้น"""
    legs, _ = parse_positions(raw_positions)
    return sorted({leg.symbol for leg in legs})


def _quote_mid(quote: Mapping | None) -> float | None:
    if not quote:
        return None
    bid, ask = _num(quote.get("bid")), _num(quote.get("ask"))
    if bid is not None and ask is not None and ask >= bid > 0:
        return (bid + ask) / 2
    price = _num(quote.get("price"))
    return price if price is not None and price > 0 else None


def parse_positions(raw_positions, option_quotes: Mapping | None = None) -> tuple[list[Leg], list[str]]:
    """positions ของ Webull → Leg · ข้าม position ที่อ่านไม่ได้แล้วบอก **ชื่อ field** ที่ขาด (ไม่มีค่าบัญชี)"""
    legs: list[Leg] = []
    problems: list[str] = []
    quotes = option_quotes or {}
    for position in raw_positions or ():
        if not isinstance(position, Mapping):
            problems.append("position ไม่ใช่ object")
            continue
        kind = str(position.get("instrument_type") or "").upper()
        symbol = str(position.get("symbol") or "").upper()
        unrealized = _num(position.get("unrealized_profit_loss"))
        if kind == "EQUITY":
            qty, mark = _num(position.get("quantity")), _num(position.get("last_price"))
            missing = [name for name, value in (("symbol", symbol), ("quantity", qty),
                                                ("last_price", mark),
                                                ("unrealized_profit_loss", unrealized))
                       if value in (None, "")]
            if missing:
                problems.append(f"EQUITY: ไม่พบ field {', '.join(missing)}")
                continue
            legs.append(Leg(symbol, "stock", qty, 1.0, mark, unrealized,
                            cost=_num(position.get("cost_price"))))
        elif kind == "OPTION":
            rows = _option_rows(position)
            if not rows or unrealized is None:
                missing = [name for name, ok in (("legs/items", bool(rows)),
                                                 ("unrealized_profit_loss", unrealized is not None))
                           if not ok]
                problems.append(f"OPTION: ไม่พบ {', '.join(missing)} (มี field: {_field_names(position)})")
                continue
            single = len(rows) == 1
            for index, row in enumerate(rows):
                spec = _option_spec(position, row)
                count = _num(row.get("quantity"))
                if count is None and row is position:
                    count = _num(position.get("quantity"))
                if spec is None or count is None:
                    problems.append("OPTION: ขาอ่านไม่ได้ ต้องมี option_type / option_expire_date / "
                                    f"option_exercise_price (หรือ symbol แบบ OCC) และ quantity "
                                    f"(มี field: {_field_names(row)})")
                    continue
                sign = _leg_sign(position, row, count)
                quote = quotes.get(spec["occ"])
                mark = _num(position.get("last_price")) if single else _quote_mid(quote)
                legs.append(Leg(
                    spec["symbol"], spec["kind"], sign * abs(count),
                    _num(row.get("option_contract_multiplier")) or 100.0, mark,
                    unrealized if index == 0 else 0.0,     # combo: P/L ของทั้งก้อนอยู่ที่ขาแรก
                    cost=_num(row.get("cost_price")) or (_num(position.get("cost_price")) if single else None),
                    strike=spec["strike"], expiry=spec["expiry"], occ=spec["occ"],
                    iv=_positive(quote.get("imp_vol")) if quote else None,
                    delta=_num(quote.get("delta")) if quote else None,
                    iv_source="webull" if quote and _positive(quote.get("imp_vol")) else None))
        else:
            problems.append(f"instrument_type {kind or '(ว่าง)'} ยังไม่รองรับ")
    return legs, problems


def _positive(value) -> float | None:
    out = _num(value)
    return out if out is not None and out > 0 else None


# ---------------------------------------------------------------- Black-Scholes

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float | None, kind: str) -> float:
    """ราคา option แบบยุโรป (ไม่มีปันผล) · T ≤ 0 หรือ σ ≤ 0 → intrinsic"""
    intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    if T <= 0 or not sigma or sigma <= 0 or S <= 0 or K <= 0:
        return intrinsic
    root = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / root
    d2 = d1 - root
    discount = K * math.exp(-r * T)
    if kind == "call":
        return S * norm_cdf(d1) - discount * norm_cdf(d2)
    return discount * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float | None, kind: str) -> float | None:
    if T <= 0 or not sigma or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0


def implied_vol(price: float | None, S: float, K: float, T: float, r: float, kind: str,
                bounds: tuple[float, float] = IV_BOUNDS) -> float | None:
    """แก้ σ จากราคา (bisection) · ราคาอยู่นอกช่วงที่ σ ใน ``bounds`` ให้ได้ → None (ไม่เดา)"""
    if price is None or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = bounds
    p_lo, p_hi = bs_price(S, K, T, r, lo, kind), bs_price(S, K, T, r, hi, kind)
    if not p_lo - 1e-9 <= price <= p_hi + 1e-9:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, kind) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def _expiry_close(expiry: date) -> datetime:
    return datetime.combine(expiry, US_CLOSE, tzinfo=NEW_YORK)


def years_to_expiry(expiry: date, now: datetime) -> float:
    """เวลาเหลือถึง 16:00 New York ของวันหมดอายุ (ปี = 365 วัน) ไม่ต่ำกว่า 0"""
    return max(0.0, (_expiry_close(expiry) - now).total_seconds() / YEAR_SECONDS)


# ---------------------------------------------------------------- state ของ bot (PROD · v3)

@dataclass(frozen=True)
class LegoChain:
    """ค่าจาก ``webull_lego_state`` ของ chain ล่าสุดของ symbol (+ FIX_C จากแถวล่าสุดของ chain)"""
    chain_key: str
    symbol: str
    fix_c: float
    leaps_shares: float
    real_holdings: float
    p0: float | None
    p_acted: float
    actual: float
    r_basis: float
    excess: float
    updated_at: str

    @property
    def u_star(self) -> float:
        """ราคาที่หุ้นจริงเป้าหมายเหลือ 0 (เกินนี้ SELL เกินหุ้นจริง → PASS) · ไม่มี LEAPS = ไม่มีเพดาน"""
        return self.fix_c / self.leaps_shares if self.leaps_shares > 0 else math.inf


def _stamp(value) -> pd.Timestamp:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return pd.Timestamp.min.tz_localize("UTC") if pd.isna(stamp) else stamp


def lego_chains(state_raw, rows_by_chain: Mapping | None) -> tuple[dict[str, LegoChain], dict[str, str]]:
    """symbol → LegoChain ของ chain ที่ ``updated_at`` ล่าสุด · ค่าที่จำเป็นหาย/ไม่ finite → ไม่ใส่ (fail closed)

    ``rows_by_chain``: chain_key → แถวล่าสุดของ chain (``webull_lego_rows/<last_run_id>``)
    คืน (chains, เหตุผลของ symbol ที่ใช้ไม่ได้)
    """
    candidates: dict[str, list[tuple[pd.Timestamp, str, Mapping]]] = {}
    for key, state in (state_raw or {}).items() if isinstance(state_raw, Mapping) else ():
        if not isinstance(state, Mapping):
            continue
        symbol = str(state.get("symbol") or str(key).split("_")[0]).upper()
        candidates.setdefault(symbol, []).append((_stamp(state.get("updated_at")), str(key), state))
    rows = rows_by_chain or {}
    chains: dict[str, LegoChain] = {}
    issues: dict[str, str] = {}
    for symbol, found in candidates.items():
        stamp, key, state = max(found, key=lambda item: (item[0], item[1]))
        row = rows.get(key) if isinstance(rows.get(key), Mapping) else {}
        cash = state.get("execution_cashflow") if isinstance(state.get("execution_cashflow"), Mapping) else {}

        def pick(primary, fallback):
            value = _num(cash.get(primary))
            return value if value is not None else _num(state.get(fallback))

        value, gap = _num(row.get(VALUE_COL)), _num(row.get(GAP_COL))
        fix_c = None if value is None or gap is None else value - gap
        fields = {
            "FIX_C (แถวล่าสุด)": fix_c,
            "P_acted": pick("last_action_price", "prev_price"),
            "Aₙ": pick("actual_cumulative", "prev_actual"),
            "R_basis": pick("r_basis", "r_basis"),
            "Eₙ": pick("excess", "prev_excess"),
            "leaps_shares": _num(state.get("leaps_shares", 0.0)),
            "หุ้นจริง (prev_holdings)": _num(state.get("prev_holdings")),
        }
        missing = [name for name, v in fields.items() if v is None]
        if missing or fields["FIX_C (แถวล่าสุด)"] <= 0 or fields["P_acted"] <= 0 \
                or fields["leaps_shares"] < 0:
            issues[symbol] = (f"{symbol}: state ของ bot ไม่ครบ ({', '.join(missing) or 'ค่าไม่ถูกต้อง'}) "
                              "— ซ่อนเส้น ③ และมุมมอง LEGO")
            continue
        scale = 1e-6 * max(1.0, abs(fields["FIX_C (แถวล่าสุด)"]))
        if abs(fields["Eₙ"] - (fields["Aₙ"] - fields["R_basis"])) > scale:
            issues[symbol] = f"{symbol}: Eₙ ≠ Aₙ − R_basis ใน state (ไม่ใช่ v3) — ซ่อนเส้น ③ และมุมมอง LEGO"
            continue
        p0 = _num(state.get("p0"))
        chains[symbol] = LegoChain(
            chain_key=key, symbol=symbol, fix_c=fields["FIX_C (แถวล่าสุด)"],
            leaps_shares=fields["leaps_shares"], real_holdings=fields["หุ้นจริง (prev_holdings)"],
            p0=p0 if p0 and p0 > 0 else None, p_acted=fields["P_acted"], actual=fields["Aₙ"],
            r_basis=fields["R_basis"], excess=fields["Eₙ"], updated_at=str(state.get("updated_at") or ""))
    return chains, issues


def check_consistency(legs: Sequence[Leg], chain: LegoChain, tol: float = CONSISTENCY_TOL) -> str | None:
    """Webull กับ bot เห็นตรงกันไหม: หุ้นจริง และ LEAPS (long call × multiplier) · ไม่ตรง → เหตุผล"""
    real = sum(leg.qty for leg in legs if leg.kind == "stock")
    leaps = sum(leg.qty * leg.multiplier for leg in legs if leg.kind == "call" and leg.qty > 0)
    if abs(real - chain.real_holdings) > tol + 1e-9:
        return (f"{chain.symbol}: หุ้นจริงใน Webull {real:,.2f} ≠ bot {chain.real_holdings:,.2f} "
                "— ซ่อนเส้น ③ และมุมมอง LEGO จนกว่าจะตรงกัน")
    if abs(leaps - chain.leaps_shares) > tol + 1e-9:
        return (f"{chain.symbol}: LEAPS ใน Webull {leaps:,.0f} หุ้น ≠ bot {chain.leaps_shares:,.0f} "
                "— ซ่อนเส้น ③ และมุมมอง LEGO จนกว่าจะตรงกัน")
    return None


# ---------------------------------------------------------------- รวมข้อมูลต่อ symbol

@dataclass
class SymbolBook:
    symbol: str
    spot: float | None
    legs: list[Leg]
    chain: LegoChain | None = None
    chain_issue: str | None = None        # มี = ห้ามใช้ chain (เส้น ③ / มุมมอง LEGO)
    quote: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def unrealized(self) -> float:
        return float(sum(leg.unrealized for leg in self.legs))

    @property
    def options(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.is_option]

    @property
    def usable_chain(self) -> LegoChain | None:
        return None if self.chain_issue else self.chain

    @property
    def options_priced(self) -> bool:
        """option ทุกขามี IV (จาก Webull หรือแก้จาก mark) — ไม่มี = เส้น ① ③ ซ่อน"""
        return all(leg.iv for leg in self.options)


def build_books(snap: Mapping, chains: Mapping[str, LegoChain], chain_issues: Mapping[str, str],
                now: datetime, r: float = RISK_FREE) -> tuple[list[SymbolBook], list[str]]:
    """snapshot จาก Webull + state ของ bot → ข้อมูลต่อ symbol (เรียงตามชื่อ) และหมายเหตุรวม"""
    options = snap.get("options")
    legs, problems = parse_positions(snap.get("positions") or [], options or {})
    notes = [f"{SKIP_PREFIX}{p}" for p in problems]
    if options is None and any(leg.is_option for leg in legs):
        notes.append("ดึงราคา option จาก Webull ไม่ได้ — IV และ delta คำนวณจาก mark ของ positions")
    quotes = snap.get("quotes") or {}
    books: list[SymbolBook] = []
    for symbol in sorted({leg.symbol for leg in legs}):
        own = [leg for leg in legs if leg.symbol == symbol]
        quote = dict(quotes.get(symbol) or {})
        stock = [leg for leg in own if leg.kind == "stock"]
        spot = stock[0].mark if stock else _positive(quote.get("price"))
        book = SymbolBook(symbol, spot if spot and spot > 0 else None, [], quote=quote)
        if book.spot is None:
            book.notes.append(f"{symbol}: ไม่มีราคาหุ้น — ซ่อนกราฟของตัวนี้")
        for leg in own:
            if leg.is_option and book.spot is not None:
                T = years_to_expiry(leg.expiry, now)
                iv, source = leg.iv, leg.iv_source
                if not iv:
                    iv = implied_vol(leg.mark, book.spot, leg.strike, T, r, leg.kind)
                    source = "mark" if iv else None
                delta = leg.delta if leg.delta is not None else bs_delta(book.spot, leg.strike, T, r,
                                                                          iv, leg.kind)
                leg = replace(leg, iv=iv, iv_source=source, delta=delta)
                if not iv:
                    book.notes.append(f"{symbol}: ราคา option {leg.occ} ไม่พอคำนวณ IV — ซ่อนเส้น ① ③ ของตัวนี้")
            book.legs.append(leg)
        book.chain = chains.get(symbol)
        if book.chain is None:
            book.chain_issue = chain_issues.get(symbol) or (
                f"{symbol}: ไม่มี chain ของ bot ใน Firebase PROD — แสดงเฉพาะเส้น ① ②")
        else:
            book.chain_issue = check_consistency(book.legs, book.chain)
        books.append(book)
    return books, notes


def exposure_ratio(book: SymbolBook) -> float | None:
    """exposure จริง = (หุ้นจริง + Σ delta·m·n)·S₀ / FIX_C — bot นับ LEAPS เป็นหุ้นเต็ม (delta 1)"""
    chain = book.usable_chain
    if chain is None or book.spot is None or any(leg.delta is None for leg in book.options):
        return None
    shares = sum(leg.qty * (1.0 if leg.kind == "stock" else leg.delta * leg.multiplier)
                 for leg in book.legs)
    return shares * book.spot / chain.fix_c


# ---------------------------------------------------------------- เส้น

def price_grid(spot: float, lo_pct: float = 40, hi_pct: float = 200,
               extra: Iterable[float] = (), n: int = GRID_POINTS) -> np.ndarray:
    """ราคาห่างเท่ากันจาก lo% ถึง hi% ของ S₀ + จุดสำคัญ (S₀, strike, u*) ที่อยู่ในช่วง
    · lo ต่ำสุด MIN_GRID_PCT (ราคา 0 ทำให้เส้น ln เป็น −∞)"""
    lo, hi = spot * max(lo_pct, MIN_GRID_PCT) / 100.0, spot * hi_pct / 100.0
    points = [spot, *[float(x) for x in extra if x is not None and math.isfinite(float(x))]]
    grid = np.concatenate([np.linspace(lo, hi, n), [p for p in points if lo <= p <= hi]])
    return np.unique(grid)


def _option_change(legs: Sequence[Leg], spot: float, S: np.ndarray, now: datetime, r: float) -> np.ndarray:
    """Σ m·n·[BS(S) − BS(S₀)] ณ ตอนนี้ · option ขาไหนไม่มี IV → NaN ทั้งเส้น"""
    total = np.zeros_like(S, dtype=float)
    for leg in legs:
        if not leg.is_option:
            continue
        if not leg.iv:
            return np.full_like(S, np.nan, dtype=float)
        T = years_to_expiry(leg.expiry, now)
        base = bs_price(spot, leg.strike, T, r, leg.iv, leg.kind)
        total += leg.multiplier * leg.qty * (
            np.array([bs_price(s, leg.strike, T, r, leg.iv, leg.kind) for s in S]) - base)
    return total


def _expiry_change(legs: Sequence[Leg], S: np.ndarray, now: datetime, r: float) -> np.ndarray:
    """Σ m·n·[V_h(S) − mark] ณ วันหมดอายุใกล้สุด h · ไม่มี option → NaN (ไม่มีวันหมดอายุให้วาด)"""
    options = [leg for leg in legs if leg.is_option]
    if not options:
        return np.full_like(S, np.nan, dtype=float)
    horizon = min(leg.expiry for leg in options)
    at = _expiry_close(horizon)
    total = np.zeros_like(S, dtype=float)
    for leg in options:
        if leg.mark is None:
            return np.full_like(S, np.nan, dtype=float)
        if leg.expiry == horizon:
            value = np.array([bs_price(s, leg.strike, 0.0, r, None, leg.kind) for s in S])
        else:
            if not leg.iv:
                return np.full_like(S, np.nan, dtype=float)
            T = years_to_expiry(leg.expiry, at)
            value = np.array([bs_price(s, leg.strike, T, r, leg.iv, leg.kind) for s in S])
        total += leg.multiplier * leg.qty * (value - leg.mark)
    return total


def _log_floor(fix_c: float, leaps: float, u: np.ndarray | float) -> np.ndarray:
    """F(u) = FIX_C·ln(u') − L·u' โดย u' = min(u, u*)"""
    u_star = fix_c / leaps if leaps > 0 else math.inf
    clipped = np.minimum(np.asarray(u, dtype=float), u_star)
    return fix_c * np.log(clipped) - leaps * clipped


def symbol_curves(legs: Sequence[Leg], spot: float, now: datetime, chain: LegoChain | None,
                  grid: np.ndarray, r: float = RISK_FREE) -> pd.DataFrame:
    """price, t0, expiry, rebalance, rebalance_e (USD เทียบทุนของ Webull) · ข้อมูลไม่พอ → NaN ทั้งเส้น

    rebalance_e = rebalance + Eₙ สะสมของ bot (ต้องมี chain เหมือน ③)
    """
    S = np.asarray(grid, dtype=float)
    U0 = float(sum(leg.unrealized for leg in legs))
    stock = sum(leg.qty for leg in legs if leg.kind == "stock") * (S - spot)
    options_now = _option_change(legs, spot, S, now, r)
    t0 = U0 + stock + options_now
    expiry = U0 + stock + _expiry_change(legs, S, now, r)
    if chain is None:
        rebalance = np.full_like(S, np.nan, dtype=float)
        excess = np.nan
    else:
        F = _log_floor(chain.fix_c, chain.leaps_shares, S)
        rebalance = U0 + (F - _log_floor(chain.fix_c, chain.leaps_shares, spot)) + options_now
        excess = chain.excess
    return pd.DataFrame({PRICE_COL: S, "t0": t0, "expiry": expiry, "rebalance": rebalance,
                         "rebalance_e": rebalance + excess})


def book_curves(book: SymbolBook, now: datetime, grid: np.ndarray, r: float = RISK_FREE) -> pd.DataFrame:
    return symbol_curves(book.legs, book.spot, now, book.usable_chain, grid, r)


def lego_view_curves(chain: LegoChain, grid: np.ndarray) -> pd.DataFrame:
    """price, R, A_floor, gap = A − R(S) (model ledger: LEAPS นับเป็นหุ้นเต็ม · ห้ามปนกับ P/L ของ Webull)

    gap = Eₙ ปัจจุบันจนถึง u* · เกิน u* bot pass (Eₙ ของ LEGO แช่แข็ง) แต่ R(S) ยังวิ่ง gap จึงหด = แพ้ถือเฉยๆ
    """
    S = np.asarray(grid, dtype=float)
    R = chain.r_basis + chain.fix_c * np.log(S / chain.p_acted)
    A = chain.actual + (_log_floor(chain.fix_c, 0.0, np.minimum(S, chain.u_star))
                        - _log_floor(chain.fix_c, 0.0, min(chain.p_acted, chain.u_star)))
    return pd.DataFrame({PRICE_COL: S, "R": R, "A_floor": A, "gap": A - R})


def pct_grid(lo_pct: float = 40, hi_pct: float = 200, n: int = GRID_POINTS) -> np.ndarray:
    """% ของ S₀ (100 = ราคาตอนนี้) สำหรับโหมดรวมพอร์ต · เริ่มอย่างน้อย MIN_GRID_PCT"""
    lo_pct = max(lo_pct, MIN_GRID_PCT)
    points = [100.0] if lo_pct <= 100.0 <= hi_pct else []
    return np.unique(np.concatenate([np.linspace(lo_pct, hi_pct, n), points]))


def portfolio_curves(books: Sequence[SymbolBook], pcts: np.ndarray, now: datetime,
                     r: float = RISK_FREE) -> pd.DataFrame:
    """รวมทุกตัว ณ % ของ S₀ เดียวกัน (สมมติทุกตัวขยับ % เท่ากัน) · ตัวไหนเส้นหาย → เส้นรวมหาย (NaN)"""
    pcts = np.asarray(pcts, dtype=float)
    out = pd.DataFrame({PCT_COL: pcts})
    for key in LINE_KEYS:
        out[key] = 0.0
    priced = [book for book in books if book.spot is not None]
    if not priced:
        out[list(LINE_KEYS)] = np.nan
        return out
    for book in priced:
        curves = book_curves(book, now, book.spot * pcts / 100.0, r)
        for key in LINE_KEYS:
            out[key] = out[key] + curves[key].to_numpy()
    return out


def portfolio_lego(books: Sequence[SymbolBook], pcts: np.ndarray) -> pd.DataFrame | None:
    """มุมมอง LEGO รวมทุก chain ที่ใช้ได้ ณ % เดียวกัน (สอดคล้องกับ "Eₙ ผลรวมทุกตัว" ของแท็บ Live)"""
    usable = [book for book in books if book.usable_chain is not None and book.spot is not None]
    if not usable:
        return None
    pcts = np.asarray(pcts, dtype=float)
    out = pd.DataFrame({PCT_COL: pcts, "R": 0.0, "A_floor": 0.0})
    for book in usable:
        curves = lego_view_curves(book.usable_chain, book.spot * pcts / 100.0)
        out["R"] += curves["R"].to_numpy()
        out["A_floor"] += curves["A_floor"].to_numpy()
    out["gap"] = out["A_floor"] - out["R"]
    return out


# ---------------------------------------------------------------- มุมมองแบบ 11 สมการ (แอป React)
# R = สมการที่ 6 (Benchmark ไม่ถูกรวมใน Net) · deep call LEAPS = สมการที่ 5 (piecewise δ) · Payoff = สมการที่ 3 (Net = Σ Active)
#   y₆ = 0 + FIX_C·ln(S/x₀₁)·δ₂
#   y₅ = 0 + FIX_C·ln(S/x₀₁)·δ(S)   δ(S) = { S ≥ x₀₁ → δ₂ ; else → δ₁ }
#   y₃ = Σ(เส้นที่ Active)            (y₆ ไม่ถูกรวม)
# x₀₁ = P₀ ของ chain (R = 0 ตรงกับ Rₙ ของ LEGO) · δ₂ = 1.0 · bias = 0
# δ₁ = (มูลค่า option ตอนนี้ + หุ้นจริง × ราคา) ÷ FIX_C  — โหมด Dynamic ของแอป React ด้วยค่าจาก Webull
ELEVEN_NAMES = {"y6": "y₆ Benchmark (R)", "y5": "y₅ deep call LEAPS", "y3": "y₃ Net = Σ Active"}
ELEVEN_SHORT = {"y6": "y₆", "y5": "y₅", "y3": "y₃"}
ELEVEN_ACTIVE = ("y5",)          # เส้นที่รวมเป็น y₃ ได้ (y₆ เป็น Benchmark ไม่ถูกรวม)
DELTA2 = 1.0


@dataclass(frozen=True)
class ElevenParams:
    symbol: str
    fix_c: float
    x01: float                   # P₀ ของ chain = จุดสลับ δ และ anchor ของ y₆
    delta1: float
    delta2: float
    option_value: float          # Σ option ยาว m·n·mark ตอนนี้
    stock_value: float           # หุ้นจริง × S₀


def eleven_params(book: SymbolBook) -> tuple[ElevenParams | None, str | None]:
    """ค่าของสมการ 5/6 ต่อ symbol · ข้อมูลไม่พอ → (None, เหตุผล) (fail closed)"""
    chain = book.usable_chain
    if chain is None:
        return None, book.chain_issue or f"{book.symbol}: ไม่มี chain ของ bot"
    if not chain.p0:
        return None, f"{book.symbol}: state ของ bot ไม่มี P₀ — วาดแบบ 11 สมการไม่ได้"
    if book.spot is None:
        return None, f"{book.symbol}: ไม่มีราคาหุ้น"
    longs = [leg for leg in book.options if leg.qty > 0]
    if any(leg.mark is None for leg in longs):
        return None, f"{book.symbol}: ไม่มีราคา option สำหรับคิด δ₁"
    option_value = float(sum(leg.multiplier * leg.qty * leg.mark for leg in longs))
    stock_value = float(sum(leg.qty for leg in book.legs if leg.kind == "stock")) * book.spot
    return ElevenParams(book.symbol, chain.fix_c, chain.p0, (option_value + stock_value) / chain.fix_c,
                        DELTA2, option_value, stock_value), None


def eleven_curves(params: ElevenParams, grid: np.ndarray, active: Iterable[str] = ELEVEN_ACTIVE) -> pd.DataFrame:
    """price, y6, y5, y3 (= Σ Active · y₆ ไม่ถูกรวม)"""
    S = np.asarray(grid, dtype=float)
    base = params.fix_c * np.log(S / params.x01)
    out = pd.DataFrame({PRICE_COL: S, "y6": base * params.delta2,
                        "y5": base * np.where(S >= params.x01, params.delta2, params.delta1)})
    out["y3"] = eleven_net(out, active)
    return out


def eleven_net(df: pd.DataFrame, active: Iterable[str]) -> np.ndarray:
    """y₃ = Σ เส้นที่ Active (ไม่มี Active = 0) · Benchmark y₆ ไม่ถูกรวมเสมอ"""
    keys = [k for k in active if k in ELEVEN_ACTIVE]
    total = np.zeros(len(df))
    for key in keys:
        total = total + df[key].to_numpy(dtype=float)
    return total


def eleven_portfolio(books: Sequence[SymbolBook], pcts: np.ndarray,
                     active: Iterable[str] = ELEVEN_ACTIVE) -> tuple[pd.DataFrame | None, list[str]]:
    """รวมทุกตัวที่ใช้ได้ ณ % ของ S₀ เดียวกัน (แต่ละตัวใช้ P₀ / δ₁ ของตัวเอง) · คืน (df, เหตุผลของตัวที่ข้าม)"""
    pcts = np.asarray(pcts, dtype=float)
    out = pd.DataFrame({PCT_COL: pcts, "y6": 0.0, "y5": 0.0})
    skipped, used = [], 0
    for book in books:
        params, issue = eleven_params(book)
        if params is None:
            skipped.append(issue)
            continue
        curves = eleven_curves(params, book.spot * pcts / 100.0, active)
        out["y6"] += curves["y6"].to_numpy()
        out["y5"] += curves["y5"].to_numpy()
        used += 1
    if not used:
        return None, skipped
    out["y3"] = eleven_net(out, active)
    return out, skipped


def break_evens(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """จุดตัด 0 แบบ linear interpolate (ข้ามช่วงที่เป็น NaN)"""
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out: list[float] = []
    for i in range(1, len(xs)):
        a, b = ys[i - 1], ys[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0:
            point = xs[i - 1]
        elif (a < 0 < b) or (a > 0 > b):
            point = xs[i - 1] + (0 - a) * (xs[i] - xs[i - 1]) / (b - a)
        else:
            continue
        if not out or abs(point - out[-1]) > 1e-9:
            out.append(float(point))
    if len(ys) and ys[-1] == 0 and (not out or abs(xs[-1] - out[-1]) > 1e-9):
        out.append(float(xs[-1]))
    return out


def curve_table(df: pd.DataFrame, x_col: str, columns: Mapping[str, str], at: Sequence[float]) -> pd.DataFrame:
    """ตารางค่า ณ จุดที่เลือก (interpolate) — table view ของกราฟ"""
    xs = df[x_col].to_numpy(dtype=float)
    rows = []
    for x in at:
        if not xs.min() <= x <= xs.max():
            continue
        row = {x_col: x}
        for key, label in columns.items():
            y = df[key].to_numpy(dtype=float)
            row[label] = float(np.interp(x, xs, y)) if np.isfinite(y).all() else None
        rows.append(row)
    return pd.DataFrame(rows)


def legs_frame(legs: Sequence[Leg]) -> pd.DataFrame:
    kind_th = {"stock": "หุ้น", "call": "Call", "put": "Put"}
    return pd.DataFrame([{
        "ชนิด": kind_th.get(leg.kind, leg.kind),
        "สัญญา/หุ้น": leg.qty,
        "strike": leg.strike,
        "หมดอายุ": leg.expiry.isoformat() if leg.expiry else None,
        "ทุน/หน่วย": leg.cost,
        "mark": leg.mark,
        "IV": leg.iv,
        "ที่มา IV": {"webull": "Webull", "mark": "แก้จาก mark"}.get(leg.iv_source or "", None),
        "delta": leg.delta,
        "P/L (USD)": leg.unrealized,
    } for leg in legs])


# ---------------------------------------------------------------- กราฟ (Plotly)
# hover การ์ดเดียวทุกเส้น (ค่านำ ชื่อตาม) · ปุ่มซูม ±10/25/50% รอบราคาตอนนี้ (แกน y ปรับตามช่วง) · range slider
# · โซนกำไร/ขาดทุนจาง ๆ คู่ป้าย ▲/▼ (สี status ไม่ทำหน้าที่ลำพัง) · ป้ายจุดคุ้มทุน/P/L ตอนนี้ · title + subtitle
# ในภาพ (ดาวน์โหลด PNG แล้วอ่านรู้เรื่อง) · ทุกอย่างทำงานในเบราว์เซอร์ ไม่ rerun Streamlit

USD = "%{y:,.2f} USD"
# d3 format ใน hover ห้ามขึ้นต้นด้วย "+": Plotly adjustFormat เติม "~" หน้า แล้ว d3 อ่านไม่ได้ → ทศนิยมเต็ม
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}   # dataviz status palette (คู่กับ ▲/▼ + ข้อความเสมอ)
ZONE_ALPHA = 0.045
ZOOM_PRESETS = (10, 25, 50)                           # ±% รอบราคาตอนนี้
ALL_LABEL = "ทั้งหมด"
# Plotly ทาสีปุ่มที่เลือก/ชี้อยู่เป็น #F4FAFF ตายตัว → ปุ่มต้องเป็นพื้นอ่อน + ตัวหนังสือเข้มทุกธีม (ธีมมืดเคยอ่านไม่ออก)
MENU_COLORS = {"bg": "#e9e8e3", "text": "#0b0b0b", "border": "rgba(11,11,11,0.18)"}
Y_PAD = 0.08


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def chart_config(filename: str = "payoff") -> dict:
    """config ของ st.plotly_chart: ดาวน์โหลด PNG ชื่อไฟล์ตามตัว (2×) · ดับเบิลคลิก = กลับมุมมองเริ่มต้น"""
    return {"displaylogo": False, "scrollZoom": False, "doubleClick": "reset",
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": filename}}


def _y_range(x: np.ndarray, series: Sequence[np.ndarray], lo: float, hi: float) -> list[float]:
    """ช่วงแกน y ที่พอดีกับค่าของทุกเส้นใน [lo, hi] (รวม 0 เสมอ) + ขอบ 8%"""
    mask = (x >= lo) & (x <= hi)
    values = [0.0]
    for y in series:
        y = np.asarray(y, dtype=float)[mask]
        values += y[np.isfinite(y)].tolist()
    y_lo, y_hi = min(values), max(values)
    pad = (y_hi - y_lo) * Y_PAD or 1.0
    return [y_lo - pad, y_hi + pad]


def _zoom_menu(tok: dict, x: np.ndarray, series: Sequence[np.ndarray], center: float | None,
               pct_axis: bool, full: tuple[list[float], list[float]]) -> list[dict]:
    """ปุ่ม ±10/25/50% รอบราคาตอนนี้ + ทั้งหมด — แต่ละปุ่มตั้งทั้งแกน x และ y (y พอดีกับช่วงที่เห็น)"""
    buttons = []
    lo_x, hi_x = float(x.min()), float(x.max())
    if center is not None:
        for p in ZOOM_PRESETS:
            lo, hi = (center - p, center + p) if pct_axis else (center * (1 - p / 100), center * (1 + p / 100))
            lo, hi = max(lo, lo_x), min(hi, hi_x)
            if hi - lo <= 0:
                continue
            buttons.append(dict(label=f"±{p}%", method="relayout",
                                args=[{"xaxis.range": [lo, hi], "yaxis.range": _y_range(x, series, lo, hi)}]))
    buttons.append(dict(label=ALL_LABEL, method="relayout",
                        args=[{"xaxis.range": full[0], "yaxis.range": full[1]}]))
    return [dict(type="buttons", direction="right", showactive=True, active=len(buttons) - 1,
                 x=1.0, xanchor="right", y=1.02, yanchor="bottom",       # แถวเดียวกับ legend (legend ชิดซ้าย)
                 pad=dict(t=0, r=0), bgcolor=MENU_COLORS["bg"], bordercolor=MENU_COLORS["border"],
                 borderwidth=1, font=dict(family=FONT_FAMILY, size=12, color=MENU_COLORS["text"]),
                 buttons=buttons)]


def _layout(tok: dict, *, height: int, x_title: str, y_title: str, title: str | None,
            subtitle: str | None, hover_title: str, x_range: list[float], y_range: list[float]) -> dict:
    text = dict(color=tok["text_secondary"], family=FONT_FAMILY, size=12)
    layout = dict(
        height=height,
        margin=dict(l=8, r=96, t=118 if title else 72, b=8),
        paper_bgcolor=tok["surface"], plot_bgcolor=tok["surface"],
        font=dict(family=FONT_FAMILY, size=13, color=tok["text_secondary"]),
        hovermode="x unified", hoverdistance=-1,
        hoverlabel=dict(bgcolor=tok["tooltip"], bordercolor=tok["grid"], namelength=-1, align="left",
                        font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                    traceorder="normal",        # fill="tonexty" ทำให้ Plotly กลับลำดับ legend เอง
                    itemclick="toggle", itemdoubleclick="toggleothers", itemsizing="constant",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        xaxis=dict(title=dict(text=x_title, font=text), tickfont=text, automargin=True, range=x_range,
                   showgrid=False, zeroline=False, showline=True, linecolor=tok["axis"],
                   ticks="outside", tickcolor=tok["axis"], ticklen=4, hoverformat=",.2f",
                   showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                   spikedash="solid", spikecolor=tok["axis"],
                   unifiedhovertitle=dict(text=hover_title),
                   rangeslider=dict(visible=True, thickness=0.07, bgcolor=tok["surface"], range=x_range,
                                    bordercolor=tok["grid"], borderwidth=1)),
        yaxis=dict(title=dict(text=y_title, font=text), tickfont=text, automargin=True, range=y_range,
                   tickformat=",.0f", hoverformat=",.2f", showgrid=True, gridcolor=tok["grid"],
                   zeroline=True, zerolinecolor=tok["zero"], zerolinewidth=1, showline=False, ticks=""),
    )
    if title:
        layout["title"] = dict(
            text=f"<b>{title}</b>", x=0, xanchor="left", xref="paper", y=0.985, yanchor="top",
            yref="container", font=dict(family=FONT_FAMILY, size=17, color=tok["text_primary"]),
            subtitle=dict(text=subtitle or "", font=dict(family=FONT_FAMILY, size=12,
                                                         color=tok["text_secondary"])))
    return layout


def _spread(values: Sequence[float], gap: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed, previous = [0.0] * len(values), None
    for i in order:
        y = float(values[i])
        if previous is not None and y < previous + gap:
            y = previous + gap
        placed[i] = previous = y
    if order:
        overshoot = placed[order[-1]] - float(values[order[-1]])
        placed = [y - overshoot / 2 for y in placed]
    return placed


def _end_labels(fig: go.Figure, tok: dict, x_last: float, items: list[tuple[str, float]],
                span: tuple[float, float]) -> None:
    """label ค่าที่ปลายขวาของเส้น (≤ 4 เส้น · สี text token — ตัวตนมาจาก legend/สีเส้น)"""
    if not items or len(items) > 4:
        return
    gap = (span[1] - span[0]) * 0.07 or 1.0
    for (label, value), y in zip(items, _spread([v for _, v in items], gap)):
        fig.add_annotation(x=x_last, y=y, text=f"{label} <b>{value:+,.0f}</b>", showarrow=False,
                           xanchor="left", xshift=8,
                           font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))


def _vline(fig: go.Figure, x: float, tok: dict, text: str, *, dash: str, strong: bool = False) -> None:
    """เส้นตั้ง + ป้ายชื่อด้านในขอบบนของกราฟ (ไม่ทับ legend ที่อยู่เหนือกราฟ)"""
    fig.add_vline(x=x, line=dict(color=tok["axis"] if strong else tok["grid"], width=1, dash=dash),
                  annotation=dict(text=text, yanchor="top", font=dict(family=FONT_FAMILY, size=12,
                                                                     color=tok["text_secondary"])),
                  annotation_position="top right")


def _chip(fig: go.Figure, tok: dict, x: float, y: float, text: str, *, ax: int, ay: int) -> None:
    """ป้ายเล็กชี้จุด (P/L ตอนนี้ / คุ้มทุน) — สี text token บนพื้น tooltip"""
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor=tok["axis"], ax=ax, ay=ay, bgcolor=tok["tooltip"],
                       bordercolor=tok["grid"], borderwidth=1, borderpad=3,
                       font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))


def _zones(fig: go.Figure, tok: dict, y_range: list[float]) -> None:
    """โซนกำไร (y > 0) / ขาดทุน (y < 0) จาง ๆ + ป้าย ▲ กำไร / ▼ ขาดทุน ชิดเส้น 0 ด้านซ้าย"""
    lo, hi = y_range
    if hi > 0:
        fig.add_hrect(y0=0, y1=hi, fillcolor=_rgba(STATUS["good"], ZONE_ALPHA), line_width=0, layer="below")
        fig.add_annotation(xref="paper", x=0.005, y=0, yanchor="bottom", xanchor="left", yshift=2,
                           text="▲ กำไร", showarrow=False,
                           font=dict(family=FONT_FAMILY, size=11, color=tok["text_secondary"]))
    if lo < 0:
        fig.add_hrect(y0=lo, y1=0, fillcolor=_rgba(STATUS["critical"], ZONE_ALPHA), line_width=0, layer="below")
        fig.add_annotation(xref="paper", x=0.005, y=0, yanchor="top", xanchor="left", yshift=-2,
                           text="▼ ขาดทุน", showarrow=False,
                           font=dict(family=FONT_FAMILY, size=11, color=tok["text_secondary"]))


def _finite_span(df: pd.DataFrame, keys: Sequence[str]) -> tuple[float, float]:
    values = np.concatenate([df[k].to_numpy(dtype=float) for k in keys]) if keys else np.array([0.0])
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    return float(min(0.0, values.min())), float(max(0.0, values.max()))


def payoff_figure(df: pd.DataFrame, *, x_col: str = PRICE_COL, x_title: str = "ราคาหุ้น (USD)",
                  show: Sequence[str] = LINE_KEYS, spot: float | None = None,
                  now_pl: float | None = None, strikes: Sequence[float] = (),
                  u_star: float | None = None, theme: str | None = None,
                  title: str | None = None, subtitle: str | None = None,
                  x_start: float | None = None, height: int = 580) -> go.Figure:
    """มุมมองบัญชีจริง: เส้น ① ② ③ ④ (เฉพาะที่เลือกและมีข้อมูล) + เส้นตั้ง S₀ / strike / u* + จุดคุ้มทุนของ ①"""
    tok = theme_tokens(theme)
    fig = go.Figure()
    x = df[x_col].to_numpy(dtype=float)
    pct_axis = x_col == PCT_COL
    # ④ = ③ + Eₙ: สีเดียวกับ ③ (ตัวตนเดียวกัน — สีที่ 4 ของ palette ไม่ผ่าน validator แบบทุกคู่)
    # แยกด้วยเส้นจุด + label ปลายเส้น และระบายช่องระหว่าง ③ กับ ④ = Eₙ เมื่อวาด ③ ด้วย
    green = tok["series"][2]
    styles = {"t0": dict(color=tok["series"][0], width=LINE["total"]),
              "expiry": dict(color=tok["series"][1], width=LINE["actual"], dash=REFERENCE_DASH),
              "rebalance": dict(color=green, width=LINE["excess"]),
              "rebalance_e": dict(color=green, width=LINE["total"], dash="dot")}
    drawn = [k for k in LINE_KEYS if k in show and np.isfinite(df[k].to_numpy(dtype=float)).all()]
    x_fmt = "%{x:,.0f}%" if pct_axis else "%{x:,.2f}"
    if spot and drawn and not pct_axis:          # แถวแรกของ hover: ห่างจากราคาตอนนี้กี่ %
        fig.add_trace(go.Scatter(
            x=x, y=df[drawn[0]], name="Δ จาก S₀", mode="lines", showlegend=False,
            line=dict(color="rgba(0,0,0,0)", width=0), customdata=(x / spot - 1.0) * 100.0,
            hovertemplate="ห่างจากราคาตอนนี้ <b>%{customdata:.1f}%</b><extra></extra>"))
    for key in drawn:
        band = key == "rebalance_e" and "rebalance" in drawn       # trace ก่อนหน้าคือ ③ เสมอ
        fig.add_trace(go.Scatter(x=x, y=df[key], name=LINE_NAMES[key], mode="lines",
                                 line=styles[key], fill="tonexty" if band else None,
                                 fillcolor=_rgba(green, FILL_ALPHA["excess"]) if band else None,
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {LINE_NAMES[key]}<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    if x_start is not None:                    # เช่น ช่วง 0–200%: แกน x เริ่มที่ 0 แม้ข้อมูลเริ่มที่ 1%
        lo = min(lo, float(x_start))
    full_y = _y_range(x, [df[k].to_numpy(dtype=float) for k in drawn], lo, hi)
    _zones(fig, tok, full_y)
    for k in strikes:
        if lo <= k <= hi:
            _vline(fig, k, tok, f"K {k:g}", dash="dot")
    if u_star is not None and math.isfinite(u_star) and lo <= u_star <= hi:
        _vline(fig, u_star, tok, f"u* {u_star:,.2f}", dash="dash")
    if spot is not None:
        _vline(fig, spot, tok, "ตอนนี้" if pct_axis else f"ตอนนี้ {spot:,.2f}", dash="solid", strong=True)
        if now_pl is not None:
            fig.add_trace(go.Scatter(
                x=[spot], y=[now_pl], name="P/L ตอนนี้", mode="markers", showlegend=False,
                marker=dict(size=10, color=tok["text_primary"], line=dict(color=tok["surface"], width=2)),
                hovertemplate="<b>%{y:,.2f}</b> USD  P/L ตอนนี้<extra></extra>"))
            _chip(fig, tok, spot, now_pl, f"ตอนนี้ <b>{now_pl:+,.2f}</b>", ax=-56, ay=36)
    if "t0" in drawn:
        points = break_evens(x, df["t0"])
        if points:
            fig.add_trace(go.Scatter(
                x=points, y=[0.0] * len(points), name="คุ้มทุน ①", mode="markers", showlegend=False,
                marker=dict(size=10, color=tok["surface"], line=dict(color=tok["series"][0], width=2)),
                hovertemplate=f"คุ้มทุน ① ที่ <b>{x_fmt}</b><extra></extra>"))
            for point in points:
                label = f"{point:,.0f}%" if pct_axis else f"{point:,.2f}"
                _chip(fig, tok, point, 0.0, f"คุ้มทุน ① <b>{label}</b>", ax=40, ay=-34)
    hover_title = ("ทุกตัวที่ <b>%{x:.0f}%</b> ของราคาตอนนี้" if pct_axis
                   else "ราคา <b>%{x:,.2f}</b> USD")
    fig.update_layout(**_layout(tok, height=height, x_title=x_title, y_title="P/L เทียบทุน (USD)",
                                title=title, subtitle=subtitle, hover_title=hover_title,
                                x_range=[lo, hi], y_range=full_y))
    fig.update_layout(updatemenus=_zoom_menu(
        tok, x, [df[k].to_numpy(dtype=float) for k in drawn], 100.0 if pct_axis else spot, pct_axis,
        ([lo, hi], full_y)))
    _end_labels(fig, tok, hi, [(LINE_SHORT[k], float(df[k].iloc[-1])) for k in drawn],
                _finite_span(df, drawn))
    return fig


def lego_ledger_figure(df: pd.DataFrame, *, x_col: str = PRICE_COL, x_title: str = "ราคาหุ้น (USD)",
                       p_acted: float | None = None, u_star: float | None = None,
                       spot: float | None = None, theme: str | None = None,
                       title: str | None = None, subtitle: str | None = None,
                       x_start: float | None = None, height: int = 540) -> go.Figure:
    """มุมมอง LEGO model ledger: R(S) (เส้นประอ้างอิง สีเดียวกับ Rₙ ของแท็บ Live) + พื้น A(S) (สีเดียวกับ Aₙ)

    ไม่มีโซนกำไร/ขาดทุน เพราะ model ledger ไม่ใช่ P/L ของบัญชี
    """
    tok = theme_tokens(theme)
    fig = go.Figure()
    x = df[x_col].to_numpy(dtype=float)
    pct_axis = x_col == PCT_COL
    fig.add_trace(go.Scatter(
        x=x, y=df["R"], name=LEGO_NAMES["R"], mode="lines",
        line=dict(color=tok["text_secondary"], width=LINE["reference"], dash=REFERENCE_DASH),
        hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {LEGO_NAMES['R']}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=df["A_floor"], name=LEGO_NAMES["A"], mode="lines",
        line=dict(color=tok["series"][0], width=LINE["actual"]),
        customdata=df["gap"], hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {LEGO_NAMES['A']}"
                                            "<br><b>%{customdata:,.2f}</b> USD  A − R(S)<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    if x_start is not None:
        lo = min(lo, float(x_start))
    series = [df["R"].to_numpy(dtype=float), df["A_floor"].to_numpy(dtype=float)]
    full_y = _y_range(x, series, lo, hi)
    if u_star is not None and math.isfinite(u_star) and lo <= u_star <= hi:
        _vline(fig, u_star, tok, f"u* {u_star:,.2f}", dash="dash")
    if p_acted is not None and lo <= p_acted <= hi:
        _vline(fig, p_acted, tok, f"P_acted {p_acted:,.2f}", dash="solid", strong=True)
    hover_title = ("ทุกตัวที่ <b>%{x:.0f}%</b> ของราคาตอนนี้" if pct_axis
                   else "ราคา <b>%{x:,.2f}</b> USD")
    fig.update_layout(**_layout(tok, height=height, x_title=x_title, y_title="USD (model ledger)",
                                title=title, subtitle=subtitle, hover_title=hover_title,
                                x_range=[lo, hi], y_range=full_y))
    center = 100.0 if pct_axis else (spot if spot is not None else p_acted)
    fig.update_layout(updatemenus=_zoom_menu(tok, x, series, center, pct_axis, ([lo, hi], full_y)))
    _end_labels(fig, tok, hi, [("R", float(df["R"].iloc[-1])), ("A", float(df["A_floor"].iloc[-1]))],
                _finite_span(df, ["R", "A_floor"]))
    return fig


def eleven_figure(df: pd.DataFrame, *, active: Sequence[str] = ELEVEN_ACTIVE, show_benchmark: bool = True,
                  x01: float | None = None, spot: float | None = None, x_col: str = PRICE_COL,
                  x_title: str = "ราคาหุ้น (USD)", x_start: float | None = None, theme: str | None = None,
                  title: str | None = None, subtitle: str | None = None, height: int = 560) -> go.Figure:
    """มุมมองแบบ 11 สมการ: y₆ Benchmark (เส้นประ ไม่รวม) · y₅ deep call LEAPS · y₃ Net = Σ Active (หนา)

    ไม่มีโซนกำไร/ขาดทุน เพราะเป็น model ไม่ใช่ P/L ของบัญชี
    """
    tok = theme_tokens(theme)
    fig = go.Figure()
    x = df[x_col].to_numpy(dtype=float)
    pct_axis = x_col == PCT_COL
    keys = [k for k in active if k in ELEVEN_ACTIVE]
    styles = {"y6": dict(color=tok["text_secondary"], width=LINE["reference"], dash=REFERENCE_DASH),
              "y5": dict(color=tok["series"][2], width=LINE["excess"]),
              "y3": dict(color=tok["text_primary"], width=LINE["total"] + 2.0)}
    drawn = (["y6"] if show_benchmark else []) + keys + ["y3"]
    # วาด y₃ (หนา) ก่อนเส้นที่ Active เพื่อให้ y₅ บางทับอยู่ด้านบนและยังมองเห็นเมื่อค่าเท่ากัน
    # (Active มีแค่ y₅ → y₃ = y₅) · legend เรียง y₆ → y₅ → y₃ ด้วย legendrank
    order = (["y6"] if show_benchmark else []) + ["y3"] + keys
    rank = {key: i for i, key in enumerate(drawn)}
    for key in order:
        fig.add_trace(go.Scatter(x=x, y=df[key], name=ELEVEN_NAMES[key], mode="lines", line=styles[key],
                                 legendrank=1000 + rank[key],
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {ELEVEN_NAMES[key]}<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    if x_start is not None:
        lo = min(lo, float(x_start))
    series = [df[k].to_numpy(dtype=float) for k in drawn]
    full_y = _y_range(x, series, lo, hi)
    if x01 is not None and lo <= x01 <= hi:
        _vline(fig, x01, tok, f"x₀₁ = P₀ {x01:,.2f} · สลับ δ", dash="dash")
    if spot is not None:
        _vline(fig, spot, tok, "ตอนนี้" if pct_axis else f"ตอนนี้ {spot:,.2f}", dash="solid", strong=True)
    hover_title = ("ทุกตัวที่ <b>%{x:.0f}%</b> ของราคาตอนนี้" if pct_axis else "ราคา <b>%{x:,.2f}</b> USD")
    fig.update_layout(**_layout(tok, height=height, x_title=x_title, y_title="USD (model 11 สมการ)",
                                title=title, subtitle=subtitle, hover_title=hover_title,
                                x_range=[lo, hi], y_range=full_y))
    fig.update_layout(updatemenus=_zoom_menu(tok, x, series, 100.0 if pct_axis else spot, pct_axis,
                                             ([lo, hi], full_y)))
    _end_labels(fig, tok, hi, [(ELEVEN_SHORT[k], float(df[k].iloc[-1])) for k in drawn],
                _finite_span(df, drawn))
    return fig
