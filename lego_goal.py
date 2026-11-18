"""แท็บ 🎯 Goal — Payoff = ขาหุ้น + ขา LEAPS เทียบ "เป้า" FIX_C·ln จากสถานะจริง (ยังไม่คิด premium)

โมดูลนี้แยกจากแท็บอื่น (isolation): ไม่ import lego_payoff · รับ "book" ต่อ symbol แบบ duck typing
(symbol, spot, legs, usable_chain, chain) จากข้อมูลชุดเดียวกับแท็บ 📐 Payoff · วาดกราฟเองด้วย token สีของ lego_charts

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

x = ราคาบนแกน · fix_c, p0, excess = chain ของ bot · spot, shares, strike, delta = Webull · leaps_shares = 100 × สัญญา
ขา LEAPS เริ่มที่ 0 ที่ p0 (เทียบกับเป้าได้) และนับ LEAPS เป็นหุ้นเต็มเหนือ strike ตามหลัก LEGO
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

GOAL_SCHEMA = 2                     # 2 = มี payoff_e (Payoff + Eₙ)
PRICE = "price"
LINE_KEYS = ("goal", "stock_leg", "leaps_leg", "payoff", "payoff_e")
VALUE_KEYS = (*LINE_KEYS, "gap", "gap_e")
NAMES = {"goal": "เป้า (FIX_C · ln)", "stock_leg": "ขาหุ้น", "leaps_leg": "ขา LEAPS",
         "payoff": "Payoff = ขาหุ้น + ขา LEAPS", "payoff_e": "Payoff + Eₙ (bot)",
         "gap": "ห่างจากเป้า", "gap_e": "ห่างจากเป้า + Eₙ"}
SHORT = {"goal": "เป้า", "stock_leg": "หุ้น", "leaps_leg": "LEAPS", "payoff": "Payoff", "payoff_e": "Payoff+Eₙ"}
END_LABELS = ("goal", "payoff", "payoff_e", "leaps_leg")      # ป้ายปลายเส้น ≤ 4 (ที่เหลือดูจาก legend)
BAND_ALPHA = 0.12
MIN_PCT = 1.0                       # ราคา 0 ทำให้ ln เป็น −∞ → ข้อมูลเริ่มที่ 1% ของราคาตอนนี้
GRID_POINTS = 241
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}   # dataviz status palette (คู่กับ ▲/▼ + ข้อความเสมอ)
ZONE_ALPHA = 0.045
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False, "doubleClick": "reset",
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "goal"}}


@dataclass(frozen=True)
class LeapsCall:
    strike: float
    leaps_shares: float           # 100 × จำนวนสัญญา (multiplier × qty)
    delta: float | None
    expiry: date | None = None


@dataclass(frozen=True)
class GoalInputs:
    symbol: str
    fix_c: float
    p0: float
    spot: float
    shares: float
    calls: tuple[LeapsCall, ...] = ()
    excess: float | None = None   # Eₙ สะสมของ bot (chain) · None = ไม่มีเส้น Payoff + Eₙ

    @property
    def stock_value(self) -> float:
        return self.shares * self.spot

    @property
    def stock_weight(self) -> float:
        return self.stock_value / self.fix_c

    @property
    def leaps_weight(self) -> float | None:
        """None เมื่อ LEAPS ขาไหนไม่มี delta (ไม่เดาเป็น 1)"""
        if any(c.delta is None for c in self.calls):
            return None
        return sum(c.delta * c.leaps_shares for c in self.calls) * self.spot / self.fix_c

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

    def option_text(self) -> str:
        return " + ".join(f"{c.leaps_shares / 100:g}×C{c.strike:g}" for c in self.calls) or "—"


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


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
                                   None if leg.delta is None else float(leg.delta), leg.expiry))
        else:
            return None, f"{symbol}: มี option ที่ไม่ใช่ long call (put หรือขาย) — แท็บนี้รองรับเฉพาะฝั่ง buy"
    excess = _finite(getattr(chain, "excess", None))
    return GoalInputs(symbol, fix_c, p0, spot, shares, tuple(calls), excess), None


def goal_grid(inputs: GoalInputs, lo_pct: float, hi_pct: float, n: int = GRID_POINTS) -> np.ndarray:
    """ราคาห่างเท่ากัน lo%..hi% ของราคาตอนนี้ + จุดสำคัญ (spot, p0, strike) ที่อยู่ในช่วง"""
    lo, hi = inputs.spot * max(lo_pct, MIN_PCT) / 100.0, inputs.spot * max(hi_pct, lo_pct + 1) / 100.0
    extra = [inputs.spot, inputs.p0, *(c.strike for c in inputs.calls)]
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


def goal_at(inputs: GoalInputs, price: float | None = None) -> dict[str, float | None]:
    """ค่าทุกเส้นที่ราคาเดียว · payoff_e / gap_e เป็น None เมื่อไม่มี Eₙ"""
    row = goal_curves(inputs, [inputs.spot if price is None else price]).iloc[0]
    return {key: _finite(row[key]) for key in VALUE_KEYS}


def goal_overview(books: Sequence) -> tuple[pd.DataFrame, list[str]]:
    """ตารางสรุปทุกตัว (ค่าที่ราคาตอนนี้) + เหตุผลของตัวที่แสดงไม่ได้"""
    rows, issues = [], []
    for book in books:
        inputs, issue = goal_inputs(book)
        if inputs is None:
            issues.append(issue)
            continue
        now = goal_at(inputs)
        leaps, cover = inputs.leaps_weight, inputs.coverage
        rows.append({
            "ตัว": inputs.symbol, "หุ้นจริง": inputs.shares, "LEAPS": inputs.option_text(),
            "ราคาตอนนี้": inputs.spot, "P₀": inputs.p0, "FIX_C": inputs.fix_c,
            "ขาหุ้น %": inputs.stock_weight * 100,
            "ขา LEAPS %": None if leaps is None else leaps * 100,
            "ครอบคลุม %": None if cover is None else cover * 100,
            "Payoff ตอนนี้": now["payoff"], "เป้าตอนนี้": now["goal"], "ห่างจากเป้า": now["gap"],
            "Eₙ (bot)": inputs.excess, "Payoff + Eₙ ตอนนี้": now["payoff_e"], "ห่างจากเป้า + Eₙ": now["gap_e"]})
    return pd.DataFrame(rows), issues


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


# ---------------------------------------------------------------- กราฟ (Plotly)

def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _y_range(series: Sequence[np.ndarray]) -> list[float]:
    values = [0.0]
    for y in series:
        y = np.asarray(y, dtype=float)
        values += y[np.isfinite(y)].tolist()
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08 or 1.0
    return [lo - pad, hi + pad]


def _layout(tok: dict, *, height: int, title: str | None, subtitle: str | None, y_title: str,
            x_range: list[float], y_range: list[float]) -> dict:
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
        xaxis=dict(title=dict(text="ราคาหุ้น x (USD)", font=text), tickfont=text, automargin=True,
                   range=x_range, showgrid=False, zeroline=False, showline=True, linecolor=tok["axis"],
                   ticks="outside", tickcolor=tok["axis"], ticklen=4, hoverformat=",.2f",
                   showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                   spikedash="solid", spikecolor=tok["axis"],
                   unifiedhovertitle=dict(text="ราคา <b>%{x:,.2f}</b> USD")),
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


def _markers(fig: go.Figure, tok: dict, inputs: GoalInputs, lo: float, hi: float) -> None:
    """เส้นตั้ง strike (จุด) · p0 (ประ) · ราคาตอนนี้ (ทึบ) + ป้ายด้านในขอบบน"""
    marks = [(c.strike, f"K {c.strike:g}", "dot", False) for c in inputs.calls]
    marks += [(inputs.p0, f"P₀ {inputs.p0:,.4g}", "dash", False),
              (inputs.spot, f"ตอนนี้ {inputs.spot:,.4g}", "solid", True)]
    for x, text, dash, strong in marks:
        if lo <= x <= hi:
            fig.add_vline(x=x, line=dict(color=tok["axis"] if strong else tok["grid"], width=1, dash=dash),
                          annotation=dict(text=text, yanchor="top",
                                          font=dict(family=FONT_FAMILY, size=12, color=tok["text_secondary"])),
                          annotation_position="top right")


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


def goal_figure(df: pd.DataFrame, inputs: GoalInputs, *, theme: str | None = None, title: str | None = None,
                subtitle: str | None = None, x_start: float | None = None, height: int = 560) -> go.Figure:
    """เป้า (ประ) · ขาหุ้น · ขา LEAPS · Payoff (หนา) · Payoff + Eₙ (แถบจางระหว่างสองเส้น = Eₙ)
    + เส้นตั้ง K / P₀ / ตอนนี้ + ป้ายค่าตอนนี้ · ไม่มี Eₙ → ไม่มีเส้น Payoff + Eₙ"""
    tok = theme_tokens(theme)
    blue = tok["series"][0]
    styles = {"goal": dict(color=tok["text_secondary"], width=LINE["reference"] + 0.5, dash=REFERENCE_DASH),
              "stock_leg": dict(color=tok["series"][2], width=LINE["excess"] + 0.5),
              "leaps_leg": dict(color=tok["series"][1], width=LINE["excess"] + 0.5),
              "payoff": dict(color=tok["text_primary"], width=LINE["total"] + 2.0),
              "payoff_e": dict(color=blue, width=LINE["excess"] + 1.0)}
    keys = [k for k in LINE_KEYS if k != "payoff_e" or inputs.excess is not None]
    x = df[PRICE].to_numpy(dtype=float)
    fig = go.Figure()
    for rank, key in enumerate(keys):         # payoff_e ต่อจาก payoff เสมอ → tonexty = แถบ Eₙ
        band = dict(fill="tonexty", fillcolor=_rgba(blue, BAND_ALPHA)) if key == "payoff_e" else {}
        fig.add_trace(go.Scatter(x=x, y=df[key], name=NAMES[key], mode="lines", line=styles[key],
                                 legendrank=1000 + rank, **band,
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {NAMES[key]}<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    if x_start is not None:
        lo = min(lo, float(x_start))
    y_range = _y_range([df[k].to_numpy(dtype=float) for k in keys])
    _markers(fig, tok, inputs, lo, hi)
    now = goal_at(inputs)
    fig.add_trace(go.Scatter(x=[inputs.spot], y=[now["payoff"]], mode="markers", showlegend=False,
                             hoverinfo="skip", marker=dict(size=10, color=tok["surface"],
                                                           line=dict(color=tok["text_primary"], width=2))))
    with_e = "" if now["payoff_e"] is None else f" · +Eₙ {now['payoff_e']:+,.0f}"
    fig.add_annotation(x=inputs.spot, y=now["payoff"],
                       text=(f"ตอนนี้: Payoff <b>{now['payoff']:+,.0f}</b>{with_e} · "
                             f"เป้า {now['goal']:+,.0f} · ห่าง {now['gap']:+,.0f}"),
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=tok["axis"], ax=120, ay=84,
                       bgcolor=tok["tooltip"], bordercolor=tok["grid"], borderwidth=1, borderpad=3,
                       font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))
    fig.update_layout(**_layout(tok, height=height, title=title, subtitle=subtitle, y_title="USD (model)",
                                x_range=[lo, hi], y_range=y_range))
    ends = [(SHORT[k], float(df[k].iloc[-1])) for k in END_LABELS if k in keys]
    gap = (y_range[1] - y_range[0]) * 0.06
    for (label, value), y in zip(ends, _spread([v for _, v in ends], gap)):
        fig.add_annotation(x=hi, y=y, text=f"{label} <b>{value:+,.0f}</b>", showarrow=False, xanchor="left",
                           xshift=8, font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]))
    return fig


def gap_figure(df: pd.DataFrame, inputs: GoalInputs, *, theme: str | None = None, x_start: float | None = None,
               height: int = 340) -> go.Figure:
    """ห่างจากเป้า = Payoff − เป้า (+ เส้นรวม Eₙ) · เส้น 0 = เป้า · โซนจาง ▲ เหนือเป้า / ▼ ใต้เป้า (สี status คู่ข้อความ)"""
    tok = theme_tokens(theme)
    x = df[PRICE].to_numpy(dtype=float)
    keys = ["gap"] + (["gap_e"] if inputs.excess is not None else [])
    styles = {"gap": dict(color=tok["text_primary"], width=LINE["total"] + 0.5),
              "gap_e": dict(color=tok["series"][0], width=LINE["excess"] + 1.0)}
    fig = go.Figure()
    for key in keys:
        fig.add_trace(go.Scatter(x=x, y=df[key], name=NAMES[key], mode="lines", showlegend=len(keys) > 1,
                                 line=styles[key],
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {NAMES[key]}<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    if x_start is not None:
        lo = min(lo, float(x_start))
    y_range = _y_range([df[k].to_numpy(dtype=float) for k in keys])
    _markers(fig, tok, inputs, lo, hi)
    for y0, y1, color, text, anchor, shift in ((0, y_range[1], STATUS["good"], "▲ เหนือเป้า", "bottom", 2),
                                               (y_range[0], 0, STATUS["critical"], "▼ ใต้เป้า", "top", -2)):
        if y1 > y0:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=_rgba(color, ZONE_ALPHA), line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, y=0, yanchor=anchor, xanchor="left", yshift=shift,
                               text=text, showarrow=False,
                               font=dict(family=FONT_FAMILY, size=11, color=tok["text_secondary"]))
    layout = _layout(tok, height=height, title=None, subtitle=None, y_title="ห่างจากเป้า (USD)",
                     x_range=[lo, hi], y_range=y_range)
    layout["margin"] = dict(l=8, r=24, t=56 if len(keys) > 1 else 24, b=8)
    fig.update_layout(**layout)
    return fig
