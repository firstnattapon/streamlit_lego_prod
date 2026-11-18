"""lego_positions.py — position จริงจาก Webull + chain ของ bot ต่อ symbol (pure: ไม่ import streamlit / webull SDK)

ชั้นข้อมูลของแท็บ 🎯 Goal (สูตรและกราฟอยู่ใน lego_goal.py):
  parse_positions  อ่าน positions ของ Webull (OpenAPI v3 ``legs`` และแบบ MCP ``items``) → Leg ต่อขา
  lego_chains      อ่าน ``webull_lego_state`` + แถวล่าสุดของ chain → FIX_C, P₀, Eₙ (fail closed)
  build_books      รวมเป็น SymbolBook ต่อ symbol (ราคา, ขา, chain) · IV / delta ของ option จาก Webull
                   หรือแก้จาก mark ด้วย Black-Scholes
ข้อมูลที่ขาดหรือไม่ตรงกัน → บอกเหตุผล ไม่ประมาณ
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

RISK_FREE = 0.04                 # r ของ Black-Scholes · ไม่มีปันผล
NEW_YORK = ZoneInfo("America/New_York")
US_CLOSE = time(16, 0)           # option หมดอายุ 16:00 เวลา New York
YEAR_SECONDS = 365.0 * 86400.0
CONSISTENCY_TOL = 0.01           # หุ้น: Webull กับ state ของ bot ต่างได้ไม่เกินนี้
IV_BOUNDS = (1e-4, 5.0)

VALUE_COL, GAP_COL = "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)"   # FIX_C = Vₙ − gapₙ (E1)

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
                              "— ไม่ใช้ chain นี้")
            continue
        scale = 1e-6 * max(1.0, abs(fields["FIX_C (แถวล่าสุด)"]))
        if abs(fields["Eₙ"] - (fields["Aₙ"] - fields["R_basis"])) > scale:
            issues[symbol] = f"{symbol}: Eₙ ≠ Aₙ − R_basis ใน state (ไม่ใช่ v3) — ไม่ใช้ chain นี้"
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
                "— ไม่ใช้ chain นี้จนกว่าจะตรงกัน")
    if abs(leaps - chain.leaps_shares) > tol + 1e-9:
        return (f"{chain.symbol}: LEAPS ใน Webull {leaps:,.0f} หุ้น ≠ bot {chain.leaps_shares:,.0f} "
                "— ไม่ใช้ chain นี้จนกว่าจะตรงกัน")
    return None


# ---------------------------------------------------------------- รวมข้อมูลต่อ symbol

@dataclass
class SymbolBook:
    symbol: str
    spot: float | None
    legs: list[Leg]
    chain: LegoChain | None = None
    chain_issue: str | None = None        # มี = ห้ามใช้ chain
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
        """option ทุกขามี IV (จาก Webull หรือแก้จาก mark)"""
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
            book.notes.append(f"{symbol}: ไม่มีราคาหุ้น")
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
                    book.notes.append(f"{symbol}: ราคา option {leg.occ} ไม่พอคำนวณ IV และ delta")
            book.legs.append(leg)
        book.chain = chains.get(symbol)
        if book.chain is None:
            book.chain_issue = chain_issues.get(symbol) or (
                f"{symbol}: ไม่มี chain ของ bot ใน Firebase PROD")
        else:
            book.chain_issue = check_consistency(book.legs, book.chain)
        books.append(book)
    return books, notes
