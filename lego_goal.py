"""แท็บ 🎯 Payoff — Payoff = ขาหุ้น + ขา LEAPS เทียบ "เป้า" FIX_C·ln จากสถานะจริง (ยังไม่คิด premium)

โมดูลนี้แยกจากแท็บอื่น (isolation): ไม่ import lego_positions / streamlit · รับ "book" ต่อ symbol แบบ duck typing
(symbol, spot, legs, usable_chain, chain) · วาดกราฟเองด้วย token สีของ lego_charts

ชื่อตัวแปร (แทนเลข y ของแอป React 11 สมการ):

    goal        (y₆) เป้า            = fix_c · ln(x / p0)                         ไม่รวมใน payoff
    stock_leg   (y₁) ขาหุ้น           = shares · spot · ln(x / p0)
    leaps_leg   (y₈) ขา LEAPS         = Σ leaps_shares · [max(0, x − K) − max(0, p0 − K)]
    payoff      (y₃) Payoff          = stock_leg + leaps_leg
    payoff_e         Payoff + Eₙ      = payoff + excess          excess = Eₙ สะสมของ bot ถึงวันนี้ (ค่าคงที่ทั้งเส้น)
    gap              ห่างจากเป้า      = payoff − goal
    gap_e            ห่างจากเป้า + Eₙ = payoff_e − goal

น้ำหนักจากสถานะจริง (วัดที่ราคาตอนนี้ spot):

    stock_weight = shares · spot / fix_c
    leaps_weight = Σ delta · leaps_shares · spot / fix_c
    coverage     = stock_weight + leaps_weight

x = ราคาบนแกน 0 ถึง 1.5 × p0 · fix_c, p0, excess = chain ของ bot · spot, shares, strike, delta = Webull ·
leaps_shares = 100 × สัญญา · ขา LEAPS เริ่มที่ 0 ที่ p0 (เทียบกับเป้าได้) และนับ LEAPS เป็นหุ้นเต็มเหนือ strike

รวมพอร์ต: แกน = % ของ P₀ (สมมติทุกตัวขยับ % ของ P₀ ของตัวเองเท่ากัน) · แต่ละเส้น = ผลรวมของทุกตัว
ที่ราคา p0ᵢ × % / 100 · ค่า "ตอนนี้" = ผลรวมของแต่ละตัวที่ราคาตอนนี้ของตัวเอง ·
% ตอนนี้ (เทียบเท่า) = 100 · exp(Σ fix_cᵢ · ln(pctᵢ / 100) / Σ fix_cᵢ) — ที่ % นี้เส้นเป้ารวม = ผลรวมเป้าตอนนี้พอดี

กราฟเวลา (Eₙ เทียบต้นทุน option และ Cash Burn) · แกน x = เวลา UTC · ตัวเดียวหรือรวมพอร์ต (ผลรวมทุกตัว) · 4 เส้น:

    trend       Eₙ เฉลี่ยต่อวัน     = Σ Ēᵢ · max(0, t − t0ᵢ)   Ēᵢ = Eᵢ(now) / วันตั้งแต่ t0ᵢ (วันปฏิทิน) · ต่อไปอนาคต (ประ)
                                   สีเขียวเข้ม · สวิตช์ to_expiry (หน้าเว็บเปิดเป็นค่าเริ่มต้น) = ต่อเส้นประจน LEAPS หมดอายุเร็วสุด
                                   Eᵢ = Eₙ ของ fill ล่าสุด (finalized_runs ของ bot) · ไม่วาดเส้น Eₙ เอง
    ev_cost     ต้นทุน option (ev) = Σ leaps_shares · (cost − max(0, p0 − K))   ไม่คิด intrinsic ที่ P₀ · เส้นแนวนอนฝั่งบวก
    cash_burn   Cash Burn สะสม    = min(0, min_{s ≤ t} Σ Aᵢ(s))   ติดลบหรือ 0 เท่านั้น ค้างค่าที่ลึกสุด (เส้นแดง)
    cash_burn_now Cash Burn ตอนนี้ = min(0, Σ Aᵢ(t))              ไม่ค้างค่าลึกสุด · ขั้นบันไดเทาบางมาก + จุดสีแดง · จุดสุดท้าย = ตอนนี้

t0ᵢ = เวลาแถวแรกของ chain (P₀) · ไม่มี → เวลา fill แรก · คืนทุน ev ที่ t* = now + (ev_cost − E(now)) / ΣĒᵢ วัน
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lego_charts import FONT_FAMILY, LINE, REFERENCE_DASH, theme_tokens

GOAL_SCHEMA = 10                    # 10 = กราฟเวลา: Cash Burn ตอนนี้ = ขั้นบันไดเทาบางมาก + จุดสีแดง (9 = จุดอย่างเดียว)
PRICE, PCT = "price", "pct"
ALL = "รวมพอร์ต"
LINE_KEYS = ("goal", "stock_leg", "leaps_leg", "payoff", "payoff_e")
VALUE_KEYS = (*LINE_KEYS, "gap", "gap_e")
NAMES = {"goal": "เป้า (FIX_C · ln)", "stock_leg": "ขาหุ้น", "leaps_leg": "ขา LEAPS",
         "payoff": "Payoff = ขาหุ้น + ขา LEAPS", "payoff_e": "Payoff + Eₙ (bot)",
         "gap": "ห่างจากเป้า", "gap_e": "ห่างจากเป้า + Eₙ"}
CHIP = {"payoff": "Payoff", "payoff_e": "Payoff + Eₙ"}
SHORT = {"goal": "เป้า", "stock_leg": "หุ้น", "leaps_leg": "LEAPS", "payoff": "Payoff", "payoff_e": "Payoff+Eₙ"}
SHOWN = ("goal", "payoff_e")        # ค่าเริ่มต้นเปิดเฉพาะเป้า + Payoff + Eₙ (ไม่มี Eₙ → เป้า + Payoff) · ที่เหลือคลิก legend
GAP_SHOWN = "gap_e"                 # กราฟห่างจากเป้า: ค่าเริ่มต้นเปิดเฉพาะ ห่างจากเป้า + Eₙ (ไม่มี Eₙ → ห่างจากเป้า)
RANGE_HI = 1.5                      # แกนราคา 0 ถึง 1.5 × P₀ (รวมพอร์ต: 0 ถึง 150% ของ P₀)
MIN_FRACTION = 0.01                 # ln(0) = −∞ → ข้อมูลเริ่มที่ 1% ของ P₀ (แกนยังเริ่มที่ 0)
Y_FIT_FROM = 0.25                   # แกน y พอดีกับค่าตั้งแต่ 25% ของ P₀ (ซ้ายกว่านั้น ln ลงลึกจนกราฟแบน)
GRID_POINTS = 301
BAND_ALPHA = 0.12
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}   # dataviz status palette (คู่กับ ▲/▼ + ข้อความเสมอ)
ZONE_ALPHA = 0.045
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False, "doubleClick": "reset",
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "goal"}}


@dataclass(frozen=True)
class LeapsCall:
    strike: float
    leaps_shares: float           # 100 × จำนวนสัญญา (multiplier × qty)
    delta: float | None
    expiry: date | None = None
    cost: float | None = None     # ราคาที่จ่ายต่อหุ้น (cost_price ของ Webull) · None = Webull ไม่ส่งมา


@dataclass(frozen=True)
class GoalInputs:
    symbol: str
    fix_c: float
    p0: float
    spot: float
    shares: float
    calls: tuple[LeapsCall, ...] = ()
    excess: float | None = None   # Eₙ สะสมของ bot (chain) · None = ไม่มีเส้น Payoff + Eₙ
    fills: tuple[tuple[pd.Timestamp, float, float], ...] = ()   # (เวลา, Aₙ, Eₙ) ของแต่ละ fill เรียงตาม seq
    t0: pd.Timestamp | None = None                              # เวลาเริ่ม chain (P₀)
    t0_source: str | None = None                                # "row" = แถวแรก · "fill" = fill แรก

    @property
    def stock_value(self) -> float:
        return self.shares * self.spot

    @property
    def stock_weight(self) -> float:
        return self.stock_value / self.fix_c

    @property
    def leaps_exposure(self) -> float | None:
        """Σ delta · leaps_shares · spot · None เมื่อ LEAPS ขาไหนไม่มี delta (ไม่เดาเป็น 1)"""
        if any(c.delta is None for c in self.calls):
            return None
        return sum(c.delta * c.leaps_shares for c in self.calls) * self.spot

    @property
    def leaps_weight(self) -> float | None:
        exposure = self.leaps_exposure
        return None if exposure is None else exposure / self.fix_c

    @property
    def coverage(self) -> float | None:
        leaps = self.leaps_weight
        return None if leaps is None else self.stock_weight + leaps

    @property
    def leaps_shares(self) -> float:
        return float(sum(c.leaps_shares for c in self.calls))

    @property
    def needed_leaps_shares(self) -> float:
        """จำนวนหุ้นเทียบเท่าที่เป้าต้องการจากขา LEAPS ที่ p0 = (fix_c − shares · spot) / p0"""
        return (self.fix_c - self.stock_value) / self.p0

    @property
    def pct_now(self) -> float:
        """ราคาตอนนี้เป็น % ของ P₀"""
        return self.spot / self.p0 * 100.0

    @property
    def option_ev(self) -> float | None:
        """ต้นทุน time value ของ LEAPS นับจาก P₀ = Σ leaps_shares · (cost − max(0, p0 − K))
        · ไม่มี LEAPS → 0 · ขาไหนไม่มี cost → None (ไม่เดา)"""
        if any(c.cost is None for c in self.calls):
            return None
        return float(sum(c.leaps_shares * (c.cost - max(0.0, self.p0 - c.strike)) for c in self.calls))

    def option_text(self) -> str:
        return " + ".join(f"{c.leaps_shares / 100:g}×C{c.strike:g}" for c in self.calls) or "—"


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive(value) -> float | None:
    out = _finite(value)
    return out if out is not None and out > 0 else None


def goal_inputs(book) -> tuple[GoalInputs | None, str | None]:
    """ค่าของ symbol จาก book · ขาดอะไร → (None, เหตุผล) ไม่เดาค่า (fail closed)"""
    symbol = book.symbol
    chain = getattr(book, "usable_chain", None)
    if chain is None:
        why = ("ไม่มี chain ของ bot ใน Firebase PROD" if getattr(book, "chain", None) is None
               else "จำนวนหุ้นหรือ LEAPS ใน Webull กับ bot ไม่ตรงกัน")
        return None, f"{symbol}: {why} (ต้องใช้ FIX_C และ P₀) — ไม่แสดงในแท็บนี้"
    fix_c, p0, spot = _positive(chain.fix_c), _positive(chain.p0), _positive(book.spot)
    if fix_c is None or p0 is None:
        return None, f"{symbol}: state ของ bot ไม่มี FIX_C หรือ P₀ — ไม่แสดงในแท็บนี้"
    if spot is None:
        return None, f"{symbol}: ไม่มีราคาหุ้นตอนนี้ — ไม่แสดงในแท็บนี้"
    shares, calls = 0.0, []
    for leg in book.legs:
        if leg.kind == "stock":
            if leg.qty < 0:
                return None, f"{symbol}: มีหุ้น short — แท็บนี้รองรับเฉพาะฝั่ง buy"
            shares += float(leg.qty)
        elif leg.kind == "call" and leg.qty > 0 and _positive(leg.strike) is not None:
            calls.append(LeapsCall(float(leg.strike), float(leg.qty) * float(leg.multiplier),
                                   None if leg.delta is None else float(leg.delta), leg.expiry,
                                   _positive(getattr(leg, "cost", None))))
        else:
            return None, f"{symbol}: มี option ที่ไม่ใช่ long call (put หรือขาย) — แท็บนี้รองรับเฉพาะฝั่ง buy"
    excess = _finite(getattr(chain, "excess", None))
    fills = tuple((pd.Timestamp(f.at), float(f.actual), float(f.excess)) for f in getattr(chain, "fills", ()) or ())
    t0 = getattr(chain, "t0", None)
    return GoalInputs(symbol, fix_c, p0, spot, shares, tuple(calls), excess, fills,
                      None if t0 is None else pd.Timestamp(t0), getattr(chain, "t0_source", None)), None


def collect(books: Sequence) -> tuple[list[GoalInputs], list[str]]:
    """ทุก symbol ที่คำนวณได้ + เหตุผลของตัวที่ไม่ได้"""
    items, issues = [], []
    for book in books:
        inputs, issue = goal_inputs(book)
        if inputs is None:
            issues.append(issue)
        else:
            items.append(inputs)
    return items, issues


# ---------------------------------------------------------------- เส้น

def goal_grid(inputs: GoalInputs, n: int = GRID_POINTS) -> np.ndarray:
    """ราคา 1% ถึง 150% ของ P₀ (แกนเริ่มที่ 0) + จุดสำคัญ (spot, p0, strike) ที่อยู่ในช่วง"""
    lo, hi = inputs.p0 * MIN_FRACTION, inputs.p0 * RANGE_HI
    extra = [inputs.spot, inputs.p0, *(c.strike for c in inputs.calls)]
    return np.unique(np.concatenate([np.linspace(lo, hi, n), [x for x in extra if lo <= x <= hi]]))


def portfolio_pct_now(items: Sequence[GoalInputs]) -> float:
    """% ของ P₀ "ตอนนี้" ของทั้งพอร์ตเป็นค่าเดียว = ค่าเฉลี่ยเรขาคณิตถ่วงด้วย FIX_C
    → Σ fix_cᵢ · ln(pct* / 100) = Σ fix_cᵢ · ln(pctᵢ / 100) คือเส้นเป้ารวมที่ pct* เท่ากับผลรวมเป้าตอนนี้พอดี"""
    total = sum(i.fix_c for i in items)
    return 100.0 * math.exp(sum(i.fix_c * math.log(i.pct_now / 100.0) for i in items) / total)


def pct_grid(items: Sequence[GoalInputs] = (), n: int = GRID_POINTS) -> np.ndarray:
    """% ของ P₀ จาก 1 ถึง 150 + % ตอนนี้ของแต่ละตัวและของทั้งพอร์ต"""
    lo, hi = MIN_FRACTION * 100.0, RANGE_HI * 100.0
    extra = [100.0, *(i.pct_now for i in items), *([portfolio_pct_now(items)] if items else [])]
    return np.unique(np.concatenate([np.linspace(lo, hi, n), [x for x in extra if lo <= x <= hi]]))


def goal_curves(inputs: GoalInputs, prices: Iterable[float]) -> pd.DataFrame:
    x = np.asarray(list(prices), dtype=float)
    log = np.log(x / inputs.p0)
    goal = inputs.fix_c * log
    stock_leg = inputs.stock_value * log
    leaps_leg = np.zeros_like(x)
    for c in inputs.calls:
        leaps_leg = leaps_leg + c.leaps_shares * (np.maximum(0.0, x - c.strike) - max(0.0, inputs.p0 - c.strike))
    payoff = stock_leg + leaps_leg
    payoff_e = payoff + (np.nan if inputs.excess is None else inputs.excess)
    return pd.DataFrame({PRICE: x, "goal": goal, "stock_leg": stock_leg, "leaps_leg": leaps_leg,
                         "payoff": payoff, "payoff_e": payoff_e, "gap": payoff - goal, "gap_e": payoff_e - goal})


def portfolio_curves(items: Sequence[GoalInputs], pcts: Iterable[float]) -> pd.DataFrame:
    """ผลรวมทุกตัวที่ราคา p0ᵢ × % / 100 · ตัวไหนไม่มี Eₙ → payoff_e / gap_e ทั้งพอร์ตเป็น NaN (ไม่รวมแค่บางตัว)"""
    pct = np.asarray(list(pcts), dtype=float)
    total = np.zeros((len(pct), len(VALUE_KEYS)))
    for inputs in items:
        total = total + goal_curves(inputs, inputs.p0 * pct / 100.0)[list(VALUE_KEYS)].to_numpy(dtype=float)
    out = pd.DataFrame(total, columns=list(VALUE_KEYS))
    out.insert(0, PCT, pct)
    return out


def goal_at(inputs: GoalInputs, price: float | None = None) -> dict[str, float | None]:
    """ค่าทุกเส้นที่ราคาเดียว · payoff_e / gap_e เป็น None เมื่อไม่มี Eₙ"""
    row = goal_curves(inputs, [inputs.spot if price is None else price]).iloc[0]
    return {key: _finite(row[key]) for key in VALUE_KEYS}


def summary(items: Sequence[GoalInputs]) -> dict[str, float | None]:
    """ค่าตอนนี้ (แต่ละตัวที่ราคาของตัวเอง) รวมกัน + น้ำหนักเทียบ Σ FIX_C · ขาดค่าตัวไหน → None"""
    fix_c = sum(i.fix_c for i in items)
    exposures = [i.leaps_exposure for i in items]
    leaps = None if any(e is None for e in exposures) else sum(exposures) / fix_c
    stock = sum(i.stock_value for i in items) / fix_c
    out: dict[str, float | None] = {"fix_c": fix_c, "stock_weight": stock, "leaps_weight": leaps,
                                    "coverage": None if leaps is None else stock + leaps}
    excess = [i.excess for i in items]
    out["excess"] = None if any(e is None for e in excess) else float(sum(excess))
    nows = [goal_at(i) for i in items]
    for key in VALUE_KEYS:
        values = [now[key] for now in nows]
        out[key] = None if any(v is None for v in values) else float(sum(values))
    return out


def overview_table(items: Sequence[GoalInputs]) -> pd.DataFrame:
    """ตารางสรุปทุกตัว (ค่าที่ราคาตอนนี้)"""
    rows = []
    for inputs in items:
        now = goal_at(inputs)
        leaps, cover = inputs.leaps_weight, inputs.coverage
        rows.append({
            "ตัว": inputs.symbol, "หุ้นจริง": inputs.shares, "LEAPS": inputs.option_text(),
            "ราคาตอนนี้": inputs.spot, "P₀": inputs.p0, "ตอนนี้ % ของ P₀": inputs.pct_now, "FIX_C": inputs.fix_c,
            "ขาหุ้น %": inputs.stock_weight * 100,
            "ขา LEAPS %": None if leaps is None else leaps * 100,
            "ครอบคลุม %": None if cover is None else cover * 100,
            "Payoff ตอนนี้": now["payoff"], "เป้าตอนนี้": now["goal"], "ห่างจากเป้า": now["gap"],
            "Eₙ (bot)": inputs.excess, "Payoff + Eₙ ตอนนี้": now["payoff_e"], "ห่างจากเป้า + Eₙ": now["gap_e"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- สมการ (ข้อความ)

def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:+,.2f}"


def equation_text(inputs: GoalInputs, price: float | None = None) -> str:
    """สมการพร้อมค่าจริงแทนลงไป (ที่ราคา x = ราคาตอนนี้ ถ้าไม่ระบุ)"""
    x = inputs.spot if price is None else float(price)
    v = goal_at(inputs, x)
    log = math.log(x / inputs.p0)
    calls = "\n".join(f"  LEAPS      = {c.leaps_shares:g} หุ้น (Call K {c.strike:g}"
                      f"{'' if c.expiry is None else f' · หมด {c.expiry.isoformat()}'}"
                      f" · delta {'—' if c.delta is None else f'{c.delta:.4f}'})" for c in inputs.calls)
    leaps_terms = " + ".join(f"{c.leaps_shares:g} · [{max(0.0, x - c.strike):,.4g} − {max(0.0, inputs.p0 - c.strike):,.4g}]"
                             for c in inputs.calls) or "0"
    leaps_w, cover = inputs.leaps_weight, inputs.coverage
    delta_terms = " + ".join(f"{c.delta:.4f} · {c.leaps_shares:g}" for c in inputs.calls
                             if c.delta is not None) or "0"
    return "\n".join([
        f"ตัวแปร ({inputs.symbol} · สถานะจริงล่าสุด)",
        f"  fix_c      = {inputs.fix_c:,.2f}   เป้า FIX_C ของ chain",
        f"  p0         = {inputs.p0:,.4g}   ราคาตั้งต้นของ chain",
        f"  spot       = {inputs.spot:,.4g}   ราคาตอนนี้",
        f"  shares     = {inputs.shares:,.2f}   หุ้นจริง",
        calls or "  LEAPS      = ไม่มี",
        "",
        "น้ำหนัก (วัดที่ spot)",
        f"  stock_weight = shares · spot / fix_c = {inputs.shares:,.2f} · {inputs.spot:,.4g} / {inputs.fix_c:,.0f}"
        f" = {inputs.stock_weight:.3f}",
        f"  leaps_weight = Σ delta · leaps_shares · spot / fix_c = ({delta_terms}) · {inputs.spot:,.4g} / "
        f"{inputs.fix_c:,.0f} = {'—' if leaps_w is None else f'{leaps_w:.3f}'}",
        f"  coverage     = stock_weight + leaps_weight = {'—' if cover is None else f'{cover:.1%}'}",
        "",
        f"เส้น (ที่ x = {x:,.4g} · ln(x / p0) = {log:+.6f})",
        f"  goal      = fix_c · ln(x / p0)  = {inputs.fix_c:,.0f} · {log:+.6f} = {v['goal']:+,.2f}",
        f"  stock_leg = shares · spot · ln(x / p0) = {inputs.stock_value:,.2f} · {log:+.6f} = {v['stock_leg']:+,.2f}",
        f"  leaps_leg = Σ leaps_shares · [max(0, x − K) − max(0, p0 − K)] = {leaps_terms} = {v['leaps_leg']:+,.2f}",
        f"  payoff    = stock_leg + leaps_leg = {v['payoff']:+,.2f}",
        f"  gap       = payoff − goal = {v['gap']:+,.2f}",
        f"  excess    = Eₙ สะสมของ bot (chain) = {_fmt(inputs.excess)}",
        f"  payoff_e  = payoff + excess = {_fmt(v['payoff_e'])}",
        f"  gap_e     = payoff_e − goal = {_fmt(v['gap_e'])}",
        "",
        "LEAPS ที่เป้าต้องการ ณ p0 = (fix_c − shares · spot) / p0 = "
        f"({inputs.fix_c:,.0f} − {inputs.stock_value:,.2f}) / {inputs.p0:,.4g} = {inputs.needed_leaps_shares:,.1f} หุ้น"
        f" · ถืออยู่ {inputs.leaps_shares:g} หุ้น",
    ])


def portfolio_equation_text(items: Sequence[GoalInputs]) -> str:
    """สมการรวมพอร์ต + ค่าตอนนี้ของแต่ละตัวและผลรวม"""
    total = summary(items)
    width = max([len(i.symbol) for i in items] + [3])
    rows = [f"  {i.symbol:<{width}}  payoff {v['payoff']:+,.2f} · goal {v['goal']:+,.2f} · gap {v['gap']:+,.2f}"
            f" · excess {_fmt(i.excess)} · payoff_e {_fmt(v['payoff_e'])} · gap_e {_fmt(v['gap_e'])}"
            f"   (ตอนนี้ {i.pct_now:.1f}% ของ P₀)"
            for i, v in ((i, goal_at(i)) for i in items)]
    return "\n".join([
        f"รวมพอร์ต ({len(items)} ตัว) — แต่ละตัวใช้ fix_c, p0, หุ้น, LEAPS และ excess ของตัวเอง",
        "แกน: x_i = p0_i · pct / 100  (สมมติทุกตัวขยับ % ของ P₀ เท่ากัน)",
        "ตอนนี้ (เทียบเท่า) pct* = 100 · exp(Σ fix_c_i · ln(pct_i / 100) / Σ fix_c_i) = "
        f"{portfolio_pct_now(items):.2f}%  (goal_รวม(pct*) = ผลรวม goal ตอนนี้)",
        f"  goal_รวม(pct)     = Σ fix_c_i · ln(pct / 100) = {total['fix_c']:,.0f} · ln(pct / 100)",
        "  payoff_รวม(pct)   = Σ [stock_leg_i + leaps_leg_i](x_i)",
        f"  payoff_e_รวม(pct) = payoff_รวม(pct) + Σ excess_i = payoff_รวม(pct) {_fmt(total['excess'])}",
        "  gap_รวม = payoff_รวม − goal_รวม · gap_e_รวม = payoff_e_รวม − goal_รวม",
        "",
        "ค่าตอนนี้ (แต่ละตัวที่ราคาตอนนี้ของตัวเอง)",
        *rows,
        f"  {'รวม':<{width}}  payoff {total['payoff']:+,.2f} · goal {total['goal']:+,.2f} · gap {total['gap']:+,.2f}"
        f" · excess {_fmt(total['excess'])} · payoff_e {_fmt(total['payoff_e'])} · gap_e {_fmt(total['gap_e'])}",
    ])


# ---------------------------------------------------------------- กราฟ (Plotly)

@dataclass(frozen=True)
class Axis:
    """แกน x ของกราฟ: คอลัมน์, ชื่อ, ปลายขวา, จุดเริ่มพอดีแกน y และเส้นตั้ง"""
    col: str
    title: str
    hover: str
    x_max: float
    y_fit_from: float
    marks: tuple[tuple[float, str, str, bool], ...] = ()       # (x, ป้าย, dash, เข้ม)
    suffix: str = ""


def symbol_axis(inputs: GoalInputs) -> Axis:
    marks = tuple((c.strike, f"K {c.strike:g}", "dot", False) for c in inputs.calls)
    marks += ((inputs.p0, f"P₀ {inputs.p0:,.4g}", "dash", False),
              (inputs.spot, f"ตอนนี้ {inputs.spot:,.4g}", "solid", True))
    return Axis(PRICE, "ราคาหุ้น x (USD)", "ราคา <b>%{x:,.2f}</b> USD", inputs.p0 * RANGE_HI,
                inputs.p0 * Y_FIT_FROM, marks)


def portfolio_axis(items: Sequence[GoalInputs]) -> Axis:
    now = portfolio_pct_now(items)
    return Axis(PCT, "% ของ P₀ (สมมติทุกตัวขยับ % ของ P₀ เท่ากัน)", "<b>%{x:.1f}%</b> ของ P₀ ทุกตัว",
                RANGE_HI * 100.0, Y_FIT_FROM * 100.0,
                ((100.0, "P₀ (100%)", "dash", False), (now, f"ตอนนี้ {now:.1f}%", "solid", True)), "%")


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _y_range(x: np.ndarray, series: Sequence[np.ndarray], x_from: float) -> list[float]:
    """พอดีกับค่าตั้งแต่ x_from (รวม 0 เสมอ) + ขอบ 8% · ซ้ายกว่านั้นเส้นออกนอกกรอบ (ปุ่ม Autoscale ดูทั้งหมด)"""
    mask = x >= x_from
    values = [0.0]
    for y in series:
        y = np.asarray(y, dtype=float)[mask]
        values += y[np.isfinite(y)].tolist()
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08 or 1.0
    return [lo - pad, hi + pad]


def _layout(tok: dict, axis: Axis, *, height: int, title: str | None, subtitle: str | None, y_title: str,
            y_range: list[float]) -> dict:
    text = dict(color=tok["text_secondary"], family=FONT_FAMILY, size=12)
    layout = dict(
        height=height, margin=dict(l=8, r=96, t=118 if title else 72, b=8),
        paper_bgcolor=tok["surface"], plot_bgcolor=tok["surface"],
        font=dict(family=FONT_FAMILY, size=13, color=tok["text_secondary"]),
        hovermode="x unified", hoverdistance=-1,
        hoverlabel=dict(bgcolor=tok["tooltip"], bordercolor=tok["grid"], namelength=-1, align="left",
                        font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", traceorder="normal",
                    itemclick="toggle", itemdoubleclick="toggleothers", bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        xaxis=dict(title=dict(text=axis.title, font=text), tickfont=text, automargin=True,
                   range=[0.0, axis.x_max], ticksuffix=axis.suffix, showgrid=False, zeroline=False,
                   showline=True, linecolor=tok["axis"], ticks="outside", tickcolor=tok["axis"], ticklen=4,
                   hoverformat=",.2f", showspikes=True, spikemode="across", spikesnap="cursor",
                   spikethickness=1, spikedash="solid", spikecolor=tok["axis"],
                   unifiedhovertitle=dict(text=axis.hover)),
        yaxis=dict(title=dict(text=y_title, font=text), tickfont=text, automargin=True, range=y_range,
                   tickformat=",.0f", hoverformat=",.2f", showgrid=True, gridcolor=tok["grid"],
                   zeroline=True, zerolinecolor=tok["zero"], zerolinewidth=1, showline=False, ticks=""),
    )
    if title:
        layout["title"] = dict(
            text=f"<b>{title}</b>", x=0, xanchor="left", xref="paper", y=0.985, yanchor="top",
            yref="container", font=dict(family=FONT_FAMILY, size=17, color=tok["text_primary"]),
            subtitle=dict(text=subtitle or "", font=dict(family=FONT_FAMILY, size=12, color=tok["text_secondary"])))
    return layout


def _marks(fig: go.Figure, tok: dict, axis: Axis) -> None:
    """เส้นตั้ง (strike จุด · P₀ ประ · ตอนนี้ ทึบ) + ป้ายด้านในขอบบน"""
    label = dict(yanchor="top", font=dict(family=FONT_FAMILY, size=12, color=tok["text_secondary"]))
    for x, text, dash, strong in axis.marks:
        if 0 <= x <= axis.x_max:
            fig.add_vline(x=x, line=dict(color=tok["axis"] if strong else tok["grid"], width=1, dash=dash),
                          annotation=dict(text=text, **label), annotation_position="top right")


def _spread(values: Sequence[float], gap: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed, previous = [0.0] * len(values), None
    for i in order:
        y = float(values[i])
        if previous is not None and y < previous + gap:
            y = previous + gap
        placed[i] = previous = y
    if order:
        shift = (placed[order[-1]] - float(values[order[-1]])) / 2
        placed = [y - shift for y in placed]
    return placed


def shown_keys(df: pd.DataFrame) -> tuple[str, ...]:
    """เส้นที่เปิดตอนเริ่ม: เป้า + Payoff + Eₙ · ไม่มี Eₙ → เป้า + Payoff"""
    return SHOWN if df["payoff_e"].notna().any() else ("goal", "payoff")


def goal_figure(df: pd.DataFrame, axis: Axis, *, now: tuple[float, dict] | None = None, theme: str | None = None,
                title: str | None = None, subtitle: str | None = None, height: int = 560) -> go.Figure:
    """เป้า (ประ) · ขาหุ้น · ขา LEAPS · Payoff (หนา) · Payoff + Eₙ · ตอนเริ่มเปิดเฉพาะเป้ากับ Payoff + Eₙ
    (ที่เหลือคลิก legend) · แถบจางระหว่าง Payoff กับ Payoff + Eₙ = Eₙ (ซ่อน/โชว์พร้อมเส้น Payoff)
    · now = (x, ค่าจาก goal_at) → จุดและป้ายค่าตอนนี้ (ตัวเดียว)"""
    tok = theme_tokens(theme)
    blue = tok["series"][0]
    styles = {"goal": dict(color=tok["text_secondary"], width=LINE["reference"] + 0.5, dash=REFERENCE_DASH),
              "stock_leg": dict(color=tok["series"][2], width=LINE["excess"] + 0.5),
              "leaps_leg": dict(color=tok["series"][1], width=LINE["excess"] + 0.5),
              "payoff": dict(color=tok["text_primary"], width=LINE["total"] + 2.0),
              "payoff_e": dict(color=blue, width=LINE["total"] + 1.0)}
    has_e = bool(df["payoff_e"].notna().any())
    keys = [k for k in LINE_KEYS if k != "payoff_e" or has_e]
    shown = shown_keys(df)
    x = df[axis.col].to_numpy(dtype=float)
    fig = go.Figure()
    for rank, key in enumerate(keys):
        fig.add_trace(go.Scatter(x=x, y=df[key], name=NAMES[key], mode="lines", line=styles[key],
                                 visible=True if key in shown else "legendonly", legendrank=1000 + rank,
                                 legendgroup="payoff" if key == "payoff" else key,
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {NAMES[key]}<extra></extra>"))
    if has_e:                                  # แถบ Eₙ อยู่กลุ่มเดียวกับเส้น Payoff (คลิก legend แล้วซ่อน/โชว์พร้อมกัน)
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                 y=np.concatenate([df["payoff_e"].to_numpy(dtype=float),
                                                   df["payoff"].to_numpy(dtype=float)[::-1]]),
                                 name="แถบ Eₙ", mode="lines", fill="toself", fillcolor=_rgba(blue, BAND_ALPHA),
                                 line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup="payoff",
                                 visible=True if "payoff" in shown else "legendonly"))
    y_range = _y_range(x, [df[k].to_numpy(dtype=float) for k in keys], axis.y_fit_from)
    _marks(fig, tok, axis)
    if now is not None:
        at, values = now
        key = "payoff_e" if has_e and values.get("payoff_e") is not None else "payoff"
        fig.add_trace(go.Scatter(x=[at], y=[values[key]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=10, color=tok["surface"], line=dict(color=styles[key]["color"],
                                                                                      width=2))))
        gap = values["gap_e"] if key == "payoff_e" else values["gap"]
        fig.add_annotation(x=at, y=values[key], text=(f"ตอนนี้: {CHIP[key]} <b>{values[key]:+,.0f}</b> · "
                                                      f"เป้า {values['goal']:+,.0f} · ห่าง {gap:+,.0f}"),
                           showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=tok["axis"], ax=120, ay=84,
                           bgcolor=tok["tooltip"], bordercolor=tok["grid"], borderwidth=1, borderpad=3,
                           font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))
    fig.update_layout(**_layout(tok, axis, height=height, title=title, subtitle=subtitle, y_title="USD (model)",
                                y_range=y_range))
    ends = [(SHORT[k], float(df[k].iloc[-1])) for k in shown]
    spacing = (y_range[1] - y_range[0]) * 0.06
    for (label, value), y in zip(ends, _spread([v for _, v in ends], spacing)):
        fig.add_annotation(x=axis.x_max, y=y, text=f"{label} <b>{value:+,.0f}</b>", showarrow=False,
                           xanchor="left", xshift=8, font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))
    return fig


def gap_figure(df: pd.DataFrame, axis: Axis, *, theme: str | None = None, height: int = 340) -> go.Figure:
    """ห่างจากเป้า = Payoff − เป้า และ Payoff + Eₙ − เป้า · เส้น 0 = เป้า · โซนจาง ▲ เหนือเป้า / ▼ ใต้เป้า
    · ตอนเริ่มเปิดเฉพาะ ห่างจากเป้า + Eₙ (ไม่มี Eₙ → ห่างจากเป้า) · อีกเส้นคลิก legend"""
    tok = theme_tokens(theme)
    x = df[axis.col].to_numpy(dtype=float)
    keys = ["gap"] + (["gap_e"] if df["gap_e"].notna().any() else [])
    shown = GAP_SHOWN if len(keys) > 1 else "gap"
    styles = {"gap": dict(color=tok["text_primary"], width=LINE["total"] + 0.5),
              "gap_e": dict(color=tok["series"][0], width=LINE["excess"] + 1.0)}
    fig = go.Figure()
    for key in keys:
        fig.add_trace(go.Scatter(x=x, y=df[key], name=NAMES[key], mode="lines", showlegend=len(keys) > 1,
                                 line=styles[key], visible=True if key == shown else "legendonly",
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {NAMES[key]}<extra></extra>"))
    y_range = _y_range(x, [df[k].to_numpy(dtype=float) for k in keys], axis.y_fit_from)
    _marks(fig, tok, axis)
    for y0, y1, color, text, anchor, shift in ((0, y_range[1], STATUS["good"], "▲ เหนือเป้า", "bottom", 2),
                                               (y_range[0], 0, STATUS["critical"], "▼ ใต้เป้า", "top", -2)):
        if y1 > y0:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=_rgba(color, ZONE_ALPHA), line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, y=0, yanchor=anchor, xanchor="left", yshift=shift,
                               text=text, showarrow=False,
                               font=dict(family=FONT_FAMILY, size=11, color=tok["text_secondary"]))
    layout = _layout(tok, axis, height=height, title=None, subtitle=None, y_title="ห่างจากเป้า (USD)",
                     y_range=y_range)
    layout["margin"] = dict(l=8, r=24, t=56 if len(keys) > 1 else 24, b=8)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------- กราฟเวลา: Eₙ เทียบต้นทุน option (ev) และ Cash Burn

TIME = "time"
DAY = pd.Timedelta(days=1)
MIN_SPAN = pd.Timedelta(hours=1)        # chain เพิ่งเริ่มไม่ถึง 1 ชม. → ยังไม่หาค่าเฉลี่ยต่อวัน
HORIZON = pd.Timedelta(days=365)         # ไม่มีวันหมดอายุ LEAPS → ต่อเส้นเฉลี่ยไม่เกิน 1 ปี
MAX_PROJECT_DAYS = 36500.0               # วันคืนทุนไกลกว่า 100 ปี = ไม่บอกวันที่
COVERED_LEAD = 0.25                      # ไม่มีจุดให้ชน → ต่อเส้นเฉลี่ยอีก 25% ของช่วงที่ผ่านมา
PROJECT_POINTS = 61
EXPIRY_ZONE, EXPIRY_HOUR = "America/New_York", 16
CASH_KEYS = ("trend", "ev_cost", "cash_burn", "burn_now")
CASH_SHORT = {"trend": "เฉลี่ย", "ev_cost": "ต้นทุน ev", "cash_burn": "Cash Burn", "burn_now": "Burn ตอนนี้"}
BURN_NOW_SIZE, BURN_NOW_WIDTH = 6, 0.75   # Cash Burn ตอนนี้: ขั้นบันไดเทาบางมาก + จุดสีแดงที่ทุกเวลาของกราฟ
MARK_CHAR, MARK_ROW = 0.008, 16      # ป้ายเส้นตั้ง: ความกว้างโดยประมาณต่อตัวอักษร (สัดส่วนแกน x) · ระยะเลื่อนลงต่อแถว (px)
TREND_GREEN = {"light": "#006400", "dark": "#1b7a34"}   # เส้น Eₙ เฉลี่ย = เขียวเข้ม · คอนทราสต์กับพื้น ≥ 3:1 ทั้งสองธีม
CASH_CONFIG = dict(CHART_CONFIG, toImageButtonOptions=dict(CHART_CONFIG["toImageButtonOptions"], filename="goal_cash"))


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _expiry_close(expiry: date) -> pd.Timestamp:
    """LEAPS หมดอายุ 16:00 เวลา New York ของวันหมดอายุ (เป็น UTC)"""
    return (pd.Timestamp(expiry).tz_localize(EXPIRY_ZONE) + pd.Timedelta(hours=EXPIRY_HOUR)).tz_convert("UTC")


def first_expiry(items: Sequence[GoalInputs]) -> pd.Timestamp | None:
    """เวลาหมดอายุ (UTC) ของ LEAPS ที่หมดอายุเร็วสุดในทุกตัว · ไม่มีวันหมดอายุเลย → None"""
    expiries = [c.expiry for i in items for c in i.calls if c.expiry is not None]
    return _expiry_close(min(expiries)) if expiries else None


def _chain_steps(inputs: GoalInputs) -> pd.DataFrame:
    """จุดขั้นบันไดของ chain: (t0, Aₙ 0, Eₙ 0) + ทุก fill · เวลาซ้ำใช้ค่าหลังสุด · index = เวลา UTC"""
    points = [(_utc(inputs.t0), 0.0, 0.0)] if inputs.t0 is not None else []
    points += [(_utc(at), float(actual), float(excess)) for at, actual, excess in inputs.fills]
    if not points:
        return pd.DataFrame(columns=["actual", "excess"], dtype=float)
    df = pd.DataFrame(points, columns=[TIME, "actual", "excess"]).sort_values(TIME, kind="stable")
    return df.groupby(TIME).last()


def cash_timeline(items: Sequence[GoalInputs], now) -> pd.DataFrame:
    """เวลารวมของทุกตัว (ค่าล่าสุดของแต่ละตัวที่ ≤ t · 0 ก่อน t0) → excess = Σ Eᵢ · actual = Σ Aᵢ ·
    cash_burn = min(0, ค่าต่ำสุดสะสมของ actual) + จุดสุดท้ายที่ now · ไม่มีข้อมูลเวลาเลย → DataFrame ว่าง"""
    steps = {i.symbol: _chain_steps(i) for i in items}
    steps = {symbol: df for symbol, df in steps.items() if not df.empty}
    columns = [TIME, "excess", "actual", "cash_burn", "cash_burn_now"]
    if not steps:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({key: pd.concat({s: df[key] for s, df in steps.items()}, axis=1)
                        .sort_index().ffill().fillna(0.0).sum(axis=1) for key in ("excess", "actual")})
    now = _utc(now)
    if out.index[-1] < now:
        out.loc[now] = out.iloc[-1]
    out["cash_burn"] = np.minimum(0.0, out["actual"].cummin())
    out["cash_burn_now"] = np.minimum(0.0, out["actual"])          # ไม่ค้างค่าลึกสุด · ตัวสุดท้าย = ตอนนี้
    out.index.name = TIME
    return out.reset_index()[columns]


@dataclass(frozen=True)
class ChainPace:
    """Eₙ เฉลี่ยต่อวันของ chain ตั้งแต่ t0 (P₀) ถึง now"""
    symbol: str
    t0: pd.Timestamp | None
    t0_source: str | None
    excess: float                  # Eₙ ของ fill ล่าสุด (ยังไม่มี fill = 0)
    days: float | None             # วันปฏิทินตั้งแต่ t0
    per_day: float | None          # excess / days · chain เพิ่งเริ่ม < 1 ชม. → None


def chain_pace(inputs: GoalInputs, now) -> ChainPace:
    steps = _chain_steps(inputs)
    excess = float(steps["excess"].iloc[-1]) if not steps.empty else 0.0
    if inputs.t0 is None:
        return ChainPace(inputs.symbol, None, None, excess, None, None)
    t0 = _utc(inputs.t0)
    span = _utc(now) - t0
    days = span / DAY
    return ChainPace(inputs.symbol, t0, inputs.t0_source, excess, days, excess / days if span >= MIN_SPAN else None)


def cash_summary(items: Sequence[GoalInputs], now, *, to_expiry: bool = False) -> dict:
    """ค่าของกราฟเวลา: Eₙ ตอนนี้ · Eₙ เฉลี่ยต่อวัน (Σ ของแต่ละตัว) · ต้นทุน ev · Cash Burn ลึกสุด · วันคืนทุน ev
    · status: covered (Eₙ ≥ ต้นทุนแล้ว) · projected · after_expiry / after_horizon (ไม่ทัน) · never · None (ไม่มีต้นทุน)
    · to_expiry (สวิตช์ในหน้าเว็บ เปิดเป็นค่าเริ่มต้น · False = ต่อถึงวันคืนทุน): ต่อเส้นเฉลี่ยจน LEAPS หมดอายุเร็วสุด (end = expiry)
      ไม่เปลี่ยน status / วันคืนทุน · ไม่มีวันหมดอายุ หรือหมดอายุแล้ว หรือไม่มีค่าเฉลี่ย → ต่อเส้นแบบเดิม"""
    now = _utc(now)
    paces = [chain_pace(i, now) for i in items]
    df = cash_timeline(items, now)
    notes: list[str] = []
    for inputs, pace in zip(items, paces):
        if pace.t0 is None:
            notes.append(f"{inputs.symbol}: ไม่มีเวลาเริ่ม chain (t0) และยังไม่มี fill — ไม่อยู่ในกราฟเวลา")
            continue
        if pace.per_day is None:
            notes.append(f"{inputs.symbol}: chain เพิ่งเริ่มไม่ถึง 1 ชั่วโมง — ยังไม่คิด Eₙ เฉลี่ยต่อวัน")
        if pace.t0_source == "fill":
            notes.append(f"{inputs.symbol}: อ่านเวลาแถวแรกของ chain ไม่ได้ — t0 = เวลา fill แรก")
        if inputs.excess is not None and inputs.fills and abs(pace.excess - inputs.excess) > 0.005:
            notes.append(f"{inputs.symbol}: Eₙ ของ fill ล่าสุด {pace.excess:+,.2f} ≠ Eₙ ใน state {inputs.excess:+,.2f}"
                         " — กราฟเวลาใช้ค่าของ fill")
    timed = [p for p in paces if p.t0 is not None]
    t0 = min((p.t0 for p in timed), default=None)
    per_day = (None if not timed or any(p.per_day is None for p in timed)
               else float(sum(p.per_day for p in timed)))
    excess = None if df.empty else float(df["excess"].iloc[-1])
    cash_burn = None if df.empty else float(df["cash_burn"].iloc[-1])      # ค่าต่ำสุดสะสม → ตัวสุดท้าย = ลึกสุด
    cash_burn_now = None if df.empty else float(df["cash_burn_now"].iloc[-1])      # min(0, Σ Aₙ ล่าสุด)
    calls = [c for i in items for c in i.calls]
    if not calls:
        ev_cost = None
        notes.append("ไม่มี LEAPS — ไม่มีเส้นต้นทุน option")
    elif any(i.option_ev is None for i in items):
        ev_cost = None
        missing = ", ".join(i.symbol for i in items if i.option_ev is None)
        notes.append(f"{missing}: Webull ไม่ส่ง cost_price ของ LEAPS — ไม่วาดเส้นต้นทุน option")
    else:
        ev_cost = float(sum(i.option_ev for i in items))
    expiry = first_expiry(items)
    status, breakeven, end = None, None, now
    if ev_cost is not None and excess is not None:
        if excess >= ev_cost:
            status = "covered"
        elif per_day is None or per_day <= 0:
            status = "never"
        else:
            need = (ev_cost - excess) / per_day
            breakeven = (now + pd.Timedelta(days=need)).round("s") if need <= MAX_PROJECT_DAYS else None
            cap = expiry if expiry is not None else now + HORIZON
            if breakeven is not None and breakeven <= cap:
                status, end = "projected", breakeven
            else:
                status, end = ("after_expiry" if expiry is not None else "after_horizon"), cap
    if end <= now and per_day is not None and per_day > 0 and t0 is not None:
        end = now + (now - t0) * COVERED_LEAD
    extended = False
    if to_expiry and per_day is not None:
        if expiry is None:
            notes.append("ไม่มีวันหมดอายุ LEAPS — ต่อเส้นเฉลี่ยแบบเดิม")
        elif expiry <= now:
            notes.append(f"LEAPS หมดอายุแล้ว ({expiry:%Y-%m-%d}) — ต่อเส้นเฉลี่ยแบบเดิม")
        else:
            end, extended = expiry, True           # ทั้งก่อนและหลังวันคืนทุน · Ē ≤ 0 ก็ต่อ (เห็นว่าไม่ทัน)
    return {"now": now, "t0": t0, "end": end, "days": None if t0 is None else (now - t0) / DAY,
            "excess": excess, "per_day": per_day, "ev_cost": ev_cost, "cash_burn": cash_burn,
            "cash_burn_now": cash_burn_now, "breakeven": breakeven, "status": status, "expiry": expiry,
            "to_expiry": extended, "paces": paces, "notes": notes}


def trend_value(summary: dict, at) -> float:
    """เส้นเฉลี่ย ณ เวลา at = Σ Ēᵢ · max(0, at − t0ᵢ) (วัน) · ที่ now = Eₙ ตอนนี้"""
    at = _utc(at)
    return float(sum(p.per_day * max(0.0, (at - p.t0) / DAY) for p in summary["paces"]
                     if p.t0 is not None and p.per_day is not None))


def trend_points(summary: dict, times: Iterable) -> tuple[list, list, list, list] | None:
    """(x, y) ของเส้นเฉลี่ยช่วงที่ผ่านมา (t0 → now) และช่วงคาด (now → end) · ไม่มีค่าเฉลี่ย → None"""
    if summary["per_day"] is None or summary["t0"] is None:
        return None
    now, end = summary["now"], summary["end"]
    starts = [p.t0 for p in summary["paces"] if p.t0 is not None]
    past = sorted({_utc(t) for t in times if summary["t0"] <= _utc(t) <= now} | set(starts) | {now})
    future = list(pd.date_range(now, end, periods=PROJECT_POINTS)) if end > now else []
    return (past, [trend_value(summary, t) for t in past], future, [trend_value(summary, t) for t in future])


def breakeven_text(summary: dict) -> str:
    """วันคืนทุน ev เป็นข้อความ (ใช้ทั้ง subtitle และสมการ)"""
    status, when, expiry = summary["status"], summary["breakeven"], summary["expiry"]
    day = (lambda t: t.strftime("%Y-%m-%d"))
    if status is None:
        return "— (ไม่มีต้นทุน ev)"
    if status == "covered":
        return "Eₙ ครอบคลุมต้นทุน ev แล้ว"
    if status == "never":
        return "ยังคาดไม่ได้ (ไม่มี Eₙ เฉลี่ยต่อวันที่เป็นบวก)"
    left = "" if when is None else f"คาด {day(when)} (อีก {(when - summary['now']) / DAY:,.0f} วัน)"
    if status == "projected":
        return left
    limit = (f"LEAPS หมดอายุ {day(expiry)}" if status == "after_expiry" else "1 ปี")
    return f"{left} — ไม่ทัน{limit}" if left else f"ไม่ทัน{limit}"


def cash_figure(df: pd.DataFrame, summary: dict, *, theme: str | None = None,
                title: str | None = None, subtitle: str | None = None, height: int = 520) -> go.Figure:
    """4 เส้นตามเวลา: Eₙ เฉลี่ยต่อวัน (เขียวเข้ม · ทึบถึงตอนนี้ แล้วประไปอนาคต) · ต้นทุน option ev (แนวนอนฝั่งบวก)
    · Cash Burn สะสม (แดง ขั้นบันได ≤ 0) · Cash Burn ตอนนี้ (ขั้นบันไดเทาบางมาก + จุดสีแดง = min(0, Σ Aₙ(t)))
    · เส้นตั้ง t0 / ตอนนี้ / คืนทุน ev / LEAPS หมดอายุ (เมื่อเปิดสวิตช์ to_expiry)"""
    tok = theme_tokens(theme)
    colors = {"trend": TREND_GREEN["dark" if theme == "dark" else "light"], "ev_cost": tok["text_secondary"],
              "cash_burn": STATUS["critical"], "burn_now": STATUS["critical"]}
    per_day = summary["per_day"]
    names = {"trend": ("Eₙ เฉลี่ยต่อวัน" if per_day is None else f"Eₙ เฉลี่ย {per_day:,.2f} USD/วัน") + " (ตั้งแต่ t0 P₀)",
             "ev_cost": "ต้นทุน option (ev ไม่คิด intrinsic)", "cash_burn": "Cash Burn สะสม (Aₙ ต่ำสุด, ≤ 0)",
             "burn_now": "Cash Burn ตอนนี้ (Aₙ ณ เวลานั้น, ≤ 0)"}
    x = pd.to_datetime(df[TIME], utc=True)
    trend = trend_points(summary, x)
    future = trend[2] if trend else []
    grid = sorted(set(x) | set(future))
    x_end = max([*grid, summary["end"]])
    fig = go.Figure()

    def line(key, xs, ys, rank, mode="lines", **style):
        fig.add_trace(go.Scatter(x=xs, y=ys, name=names[key], mode=mode, legendrank=1000 + rank,
                                 legendgroup=key, hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {names[key]}<extra></extra>",
                                 **style))

    if trend:
        line("trend", trend[0], trend[1], 1, line=dict(color=colors["trend"], width=LINE["excess"] + 0.5))
        if future:
            fig.add_trace(go.Scatter(x=future, y=trend[3], name=f"{names['trend']} · คาด", mode="lines",
                                     legendgroup="trend", showlegend=False,
                                     line=dict(color=colors["trend"], width=LINE["excess"] + 0.5, dash="dash"),
                                     hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {names['trend']} · คาด<extra></extra>"))
    ev = summary["ev_cost"]
    if ev is not None:
        line("ev_cost", grid, [ev] * len(grid), 2,
             line=dict(color=colors["ev_cost"], width=LINE["reference"] + 1.0, dash=REFERENCE_DASH))
    burn_now = summary.get("cash_burn_now")
    if "cash_burn_now" in df:                # วาดก่อนเส้นแดง → ช่วงที่ทับกัน เส้นแดงอยู่บน (จุดแดงยังเห็น)
        line("burn_now", x, df["cash_burn_now"], 4, mode="lines+markers",
             line=dict(color=tok["text_secondary"], width=BURN_NOW_WIDTH, shape="hv"),
             marker=dict(color=colors["burn_now"], size=BURN_NOW_SIZE))
    line("cash_burn", x, df["cash_burn"], 3, fill="tozeroy", fillcolor=_rgba(colors["cash_burn"], ZONE_ALPHA * 2),
         line=dict(color=colors["cash_burn"], width=LINE["total"], shape="hv"))

    values = [0.0, *df["cash_burn"], *(trend[1] + trend[3] if trend else []),
              *([ev] if ev is not None else [])]
    finite = [float(v) for v in values if np.isfinite(v)]
    lo, hi = min(finite), max(finite)
    pad = (hi - lo) * 0.08 or 1.0
    y_range = [lo - pad, hi + pad]
    label_font = dict(family=FONT_FAMILY, size=12, color=tok["text_secondary"])
    marks = [(summary["t0"], "t0 · P₀", "dash", False), (summary["now"], "ตอนนี้", "solid", True)]
    if summary["status"] == "projected":
        marks.append((summary["breakeven"], f"คืนทุน ev {summary['breakeven']:%Y-%m-%d}", "dot", True))
    if summary.get("to_expiry"):
        marks.append((summary["expiry"], f"LEAPS หมดอายุ {summary['expiry']:%Y-%m-%d}", "dashdot", True))
    left = summary["t0"] if summary["t0"] is not None else x.iloc[0]
    width = x_end - left
    rows: list[list[tuple[float, float]]] = []          # ป้ายที่ชนกัน (เช่น t0 กับตอนนี้ เมื่อต่อเส้นยาว) → แถวถัดลงไป
    for at, text, dash, strong in marks:
        if at is None:
            continue
        right = at >= x_end                  # เส้นที่ขอบขวา (คืนทุน = ปลายกราฟ) → ป้ายอยู่ซ้ายเส้น
        f = (at - left) / width if width > pd.Timedelta(0) else 0.0
        box = (f - MARK_CHAR * len(text), f) if right else (f, f + MARK_CHAR * len(text))
        row = next((n for n, used in enumerate(rows) if all(box[1] <= a or box[0] >= b for a, b in used)), len(rows))
        if row == len(rows):
            rows.append([])
        rows[row].append(box)
        fig.add_shape(type="line", x0=at, x1=at, xref="x", y0=0, y1=1, yref="paper",
                      line=dict(color=tok["axis"] if strong else tok["grid"], width=1, dash=dash))
        fig.add_annotation(x=at, y=1, xref="x", yref="paper", text=text, showarrow=False,
                           xanchor="right" if right else "left", yanchor="top", xshift=-4 if right else 4,
                           yshift=-MARK_ROW * row, font=label_font)
    if summary["status"] == "projected":
        fig.add_trace(go.Scatter(x=[summary["breakeven"]], y=[ev], mode="markers", showlegend=False,
                                 hoverinfo="skip", marker=dict(size=10, color=tok["surface"],
                                                               line=dict(color=colors["trend"], width=2))))
    axis = Axis(TIME, "เวลา (UTC)", "<b>%{x|%Y-%m-%d %H:%M}</b> UTC", 0.0, 0.0)
    layout = _layout(tok, axis, height=height, title=title, subtitle=subtitle, y_title="USD (model)",
                     y_range=y_range)
    span = x_end - (summary["t0"] or x.iloc[0])
    layout["xaxis"].update(type="date", range=[(summary["t0"] or x.iloc[0]) - span * 0.01, x_end + span * 0.01],
                           ticksuffix="", hoverformat="%Y-%m-%d %H:%M")
    layout["margin"]["r"] = 136                                  # ป้ายค่าปลายเส้น เช่น "Cash Burn -1,591"
    fig.update_layout(**layout)
    ends = []
    if trend:
        ends.append(("trend", trend_value(summary, x_end)))
    if ev is not None:
        ends.append(("ev_cost", ev))
    ends.append(("cash_burn", summary["cash_burn"]))
    ends.append(("burn_now", burn_now))
    ends =[(k, v) for k, v in ends if v is not None]
    for (key, value), y in zip(ends, _spread([v for _, v in ends], (y_range[1] - y_range[0]) * 0.06)):
        fig.add_annotation(x=x_end, y=y, text=f"{CASH_SHORT[key]} <b>{value:+,.0f}</b>", showarrow=False,
                           xanchor="left", xshift=8, font=dict(family=FONT_FAMILY, size=12, color=colors[key]))
    return fig


def cash_equation_text(items: Sequence[GoalInputs], summary: dict) -> str:
    """สมการของกราฟเวลาพร้อมค่าจริง (ต่อท้ายข้อความใน expander สมการ)"""
    width = max([len(i.symbol) for i in items] + [3])
    stamp = (lambda t: "—" if t is None else t.strftime("%Y-%m-%d %H:%M"))
    rows = []
    for inputs, pace in zip(items, summary["paces"]):
        source = {"row": "แถวแรก", "fill": "fill แรก"}.get(pace.t0_source, "—")
        avg = ("—" if pace.per_day is None else
               f"{pace.excess:+,.2f} / {pace.days:,.2f} วัน = {pace.per_day:+,.2f} USD/วัน")
        terms = " + ".join(f"{c.leaps_shares:g} · ({'—' if c.cost is None else f'{c.cost:,.4g}'} − "
                           f"max(0, {inputs.p0:,.4g} − {c.strike:g}))" for c in inputs.calls) or "0 (ไม่มี LEAPS)"
        rows += [f"  {inputs.symbol:<{width}}  t0 {stamp(pace.t0)} UTC ({source}) · Ē = {avg}",
                 f"  {'':<{width}}  ev = {terms} = {_fmt(inputs.option_ev)}"]
    return "\n".join([
        f"กราฟเวลา — Eₙ เทียบต้นทุน option (ev) และ Cash Burn (ณ {stamp(summary['now'])} UTC)",
        "  Eₙ(t)        = Σ Eₙ ของ fill ล่าสุดที่ ≤ t ของแต่ละตัว (finalized_runs ของ bot · 0 ก่อน t0)",
        "  Ē            = Σ Eₙ(now)ᵢ / วันตั้งแต่ t0ᵢ (วันปฏิทิน)     เส้นเฉลี่ย(t) = Σ Ēᵢ · max(0, t − t0ᵢ)",
        "  ev_cost      = Σ leaps_shares · (cost − max(0, p0 − K))   (ไม่คิด intrinsic ที่ P₀)",
        "  cash_burn(t) = min(0, min_{s ≤ t} Σ Aₙ(s))",
        "  cash_burn_now(t) = min(0, Σ Aₙ(t))   (ไม่ค้างค่าลึกสุด · ขั้นบันไดเทา + จุดแดง · ค่าตอนนี้ = ที่ now)",
        "",
        *rows,
        f"  {'รวม':<{width}}  Eₙ {_fmt(summary['excess'])} · Ē {_fmt(summary['per_day'])} USD/วัน · "
        f"ev_cost {_fmt(summary['ev_cost'])} · cash_burn {_fmt(summary['cash_burn'])} · "
        f"cash_burn_now {_fmt(summary.get('cash_burn_now'))}",
        f"  คืนทุน ev: {breakeven_text(summary)}",
        *([f"  ต่อเส้นเฉลี่ยจน LEAPS หมดอายุ {summary['expiry']:%Y-%m-%d %H:%M} UTC: "
           f"เส้นเฉลี่ย = {_fmt(trend_value(summary, summary['expiry']))}"] if summary.get("to_expiry") else []),
    ])
