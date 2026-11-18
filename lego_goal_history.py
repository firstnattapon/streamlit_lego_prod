"""แท็บ 🧪 ประวัติเป้า (ทดสอบ) — ห่างจากเป้า + Eₙ ตามเวลา จากแถวเก่าของ bot (pure: ไม่ import streamlit)

แยกจากแท็บอื่น: ใช้แค่นิยามของ lego_goal (GoalInputs, ชื่อเส้น) · ไม่แก้แท็บ 🎯 Goal

แต่ละแถวของ chain (``webull_lego_rows`` ที่ ``chain_key`` ตรงกับ chain ปัจจุบันของ symbol) = ค่าที่แท็บ 🎯 Goal
จะแสดง "ณ เวลานั้น" ที่ราคา x = Pₜ ของแถว ด้วยจำนวนหุ้นจริง, LEAPS และ Eₙ ของแถวนั้น:

    goal_t      = fix_c · ln(Pₜ / p0)                                  (= Rₙ ของแถว)
    stock_leg_t = sharesₜ · Pₜ · ln(Pₜ / p0)
    leaps_leg_t = (leapsₜ / leaps_now) · Σ leaps_shares · [max(0, Pₜ − K) − max(0, p0 − K)]
    payoff_t    = stock_leg_t + leaps_leg_t · payoff_e_t = payoff_t + Eₜ
    gap_t       = payoff_t − goal_t          · gap_e_t = payoff_e_t − goal_t

fix_c, p0 = chain ปัจจุบันของ bot · strike K = LEAPS ใน Webull ตอนนี้ (แถวเก่าไม่เก็บ strike) ·
leapsₜ > 0 แต่ตอนนี้ไม่มี LEAPS → ไม่รู้ K → แถวนั้นไม่มีค่า (fail closed) · ยังไม่คิด premium
รวมพอร์ต: เรียงทุกแถวตามเวลา แต่ละตัวใช้ค่าล่าสุดของตัวเองจนกว่าจะมีแถวใหม่ แล้วรวมกัน (เริ่มเมื่อทุกตัวมีแถวแล้ว)
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lego_charts import FONT_FAMILY, LINE, theme_tokens
from lego_goal import NAMES, VALUE_KEYS, GoalInputs

HISTORY_SCHEMA = 1
TIME, PRICE, HOLDINGS, LEAPS, EXCESS, CHAIN = ("เวลา (UTC)", "ราคา Pₙ (USD)", "real_holdings", "leaps_shares",
                                               "Eₙ ส่วนเกินสะสม (USD)", "chain_key")
COLUMNS = ("time", "price", "shares", "leaps", "excess")
TREND_DAYS = 7
SHOWN = "gap_e"                     # ค่าเริ่มต้นเปิดเฉพาะ ห่างจากเป้า + Eₙ (ห่างจากเป้าคลิก legend)
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}   # dataviz status palette (คู่กับ ▲/▼ + ข้อความเสมอ)
ZONE_ALPHA = 0.045
# ตลาดสหรัฐฯ 13:30–21:00 UTC (EDT/EST) → ซ่อนเสาร์–อาทิตย์และ 21:00–13:00 UTC ไม่ให้เส้นลากผ่านช่วงปิดตลาด
RANGE_BREAKS = (dict(bounds=["sat", "mon"]), dict(bounds=[21, 13], pattern="hour"))
CHART_CONFIG = {"displaylogo": False, "scrollZoom": False, "doubleClick": "reset",
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "goal_history"}}


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def chain_rows(rows_raw, chain_key: str) -> tuple[pd.DataFrame, list[str]]:
    """แถวของ chain เรียงตามเวลา (time, price, shares, leaps, excess) · แถวที่ยังไม่ commit ไม่นับ ·
    ค่าไม่ครบ → ข้ามและบอกจำนวน (ไม่เดาค่า)"""
    records, bad = [], 0
    for row in (rows_raw.values() if isinstance(rows_raw, Mapping) else ()):
        if not isinstance(row, Mapping) or row.get(CHAIN) != chain_key or row.get("committed") is False:
            continue
        time = pd.to_datetime(row.get(TIME), utc=True, errors="coerce")
        price, shares = _finite(row.get(PRICE)), _finite(row.get(HOLDINGS))
        leaps, excess = _finite(row.get(LEAPS)), _finite(row.get(EXCESS))
        if pd.isna(time) or price is None or price <= 0 or shares is None or shares < 0 \
                or leaps is None or leaps < 0 or excess is None:
            bad += 1
            continue
        records.append((time, price, shares, leaps, excess))
    df = pd.DataFrame(records, columns=list(COLUMNS))
    df = df.sort_values("time", kind="stable").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df, ([f"{chain_key}: ข้าม {bad} แถวที่ค่าไม่ครบ"] if bad else [])


def history(inputs: GoalInputs, rows: pd.DataFrame) -> pd.DataFrame:
    """ค่าของแท็บ 🎯 Goal ณ แต่ละแถว (ราคา, หุ้นจริง, LEAPS และ Eₙ ของแถวนั้น)"""
    x = rows["price"].to_numpy(dtype=float)
    log = np.log(x / inputs.p0)
    goal = inputs.fix_c * log
    stock_leg = rows["shares"].to_numpy(dtype=float) * x * log
    structure = np.zeros_like(x)
    for c in inputs.calls:
        structure = structure + c.leaps_shares * (np.maximum(0.0, x - c.strike) - max(0.0, inputs.p0 - c.strike))
    leaps = rows["leaps"].to_numpy(dtype=float)
    if inputs.leaps_shares > 0:
        leaps_leg = structure * leaps / inputs.leaps_shares
    else:                                    # ไม่มี LEAPS ตอนนี้ → ไม่รู้ strike ของแถวที่เคยมี
        leaps_leg = np.where(leaps > 0, np.nan, 0.0)
    payoff = stock_leg + leaps_leg
    payoff_e = payoff + rows["excess"].to_numpy(dtype=float)
    return pd.DataFrame({"time": rows["time"].to_numpy(), "price": x, "excess": rows["excess"].to_numpy(dtype=float),
                         "goal": goal, "stock_leg": stock_leg, "leaps_leg": leaps_leg, "payoff": payoff,
                         "payoff_e": payoff_e, "gap": payoff - goal, "gap_e": payoff_e - goal})


def portfolio_history(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """รวมทุกตัวตามเวลา: แต่ละตัวใช้ค่าล่าสุดของตัวเองจนกว่าจะมีแถวใหม่ · เริ่มเมื่อทุกตัวมีแถวแล้ว"""
    cols = ["excess", *VALUE_KEYS]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["time", *cols])
    index = pd.DatetimeIndex(sorted(set().union(*(f["time"] for f in frames))))
    parts = [f.set_index("time")[cols].reindex(index).ffill() for f in frames]
    started = np.logical_and.reduce([p["excess"].notna().to_numpy() for p in parts])
    total = sum(parts)[started]
    return total.rename_axis("time").reset_index()


def trend(df: pd.DataFrame, col: str, days: float = TREND_DAYS) -> dict | None:
    """ค่าล่าสุด เทียบกับแถวล่าสุดที่เก่ากว่า ``days`` วัน (ไม่มี → แถวแรก, partial) + อัตราต่อวัน"""
    data = df[["time", col]].dropna()
    if len(data) < 2:
        return None
    last_time, last = data["time"].iloc[-1], float(data[col].iloc[-1])
    before = data[data["time"] <= last_time - pd.Timedelta(days=days)]
    partial = before.empty
    ref = data.iloc[0] if partial else before.iloc[-1]
    elapsed = (last_time - ref["time"]) / pd.Timedelta(days=1)
    change = last - float(ref[col])
    return {"last": last, "ref": float(ref[col]), "change": change, "days": elapsed,
            "rate": change / elapsed if elapsed > 0 else None, "partial": partial}


def gap_change(df: pd.DataFrame, days: float = TREND_DAYS) -> dict | None:
    """ช่องว่างกับเป้า (|ห่างจากเป้า + Eₙ|) แคบลงหรือกว้างขึ้นเทียบ ``days`` วันก่อน"""
    t = trend(df, "gap_e", days)
    if t is None:
        return None
    width_change = abs(t["last"]) - abs(t["ref"])
    return {**t, "width_change": width_change,
            "word": "แคบลง" if width_change < 0 else ("กว้างขึ้น" if width_change > 0 else "เท่าเดิม"),
            "side": "เหนือเป้า" if t["last"] > 0 else ("ใต้เป้า" if t["last"] < 0 else "ตรงเป้า")}


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """แถวสุดท้ายของแต่ละวัน (UTC) + Eₙ ที่เพิ่มในวันนั้น"""
    if df.empty:
        return pd.DataFrame()
    day = df.assign(วัน=df["time"].dt.strftime("%Y-%m-%d")).groupby("วัน", sort=True).tail(1)
    out = pd.DataFrame({
        "วัน (UTC)": day["วัน"].to_numpy(),
        "ห่างจากเป้า + Eₙ": day["gap_e"].to_numpy(), "ห่างจากเป้า": day["gap"].to_numpy(),
        "Eₙ สะสม": day["excess"].to_numpy(), "Payoff + Eₙ": day["payoff_e"].to_numpy(), "เป้า": day["goal"].to_numpy()})
    out["Eₙ เพิ่มในวัน"] = out["Eₙ สะสม"].diff()
    if "price" in day.columns:
        out.insert(1, "ราคา", day["price"].to_numpy())
    return out


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
            y_range: list[float], legend: bool) -> dict:
    text = dict(color=tok["text_secondary"], family=FONT_FAMILY, size=12)
    layout = dict(
        height=height, margin=dict(l=8, r=24, t=(110 if title else 24) + (24 if legend else 0), b=8),
        paper_bgcolor=tok["surface"], plot_bgcolor=tok["surface"],
        font=dict(family=FONT_FAMILY, size=13, color=tok["text_secondary"]),
        hovermode="x unified", hoverdistance=-1, showlegend=legend,
        hoverlabel=dict(bgcolor=tok["tooltip"], bordercolor=tok["grid"], namelength=-1, align="left",
                        font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", traceorder="normal",
                    itemclick="toggle", itemdoubleclick="toggleothers", bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        xaxis=dict(title=dict(text="เวลา (UTC · ซ่อนช่วงตลาดปิด)", font=text), tickfont=text, automargin=True,
                   type="date", rangebreaks=list(RANGE_BREAKS), showgrid=False, zeroline=False, showline=True,
                   linecolor=tok["axis"], ticks="outside", tickcolor=tok["axis"], ticklen=4,
                   hoverformat="%Y-%m-%d %H:%M", showspikes=True, spikemode="across", spikesnap="cursor",
                   spikethickness=1, spikedash="solid", spikecolor=tok["axis"],
                   unifiedhovertitle=dict(text="<b>%{x|%Y-%m-%d %H:%M}</b> UTC")),
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


def gap_history_figure(df: pd.DataFrame, *, theme: str | None = None, title: str | None = None,
                       subtitle: str | None = None, height: int = 440) -> go.Figure:
    """ห่างจากเป้า + Eₙ (เปิด) และ ห่างจากเป้า (คลิก legend) ตามเวลา · เส้น 0 = เป้า · โซน ▲ เหนือ / ▼ ใต้"""
    tok = theme_tokens(theme)
    styles = {"gap": dict(color=tok["text_primary"], width=LINE["total"] + 0.5),
              "gap_e": dict(color=tok["series"][0], width=LINE["total"] + 0.5)}
    fig = go.Figure()
    for key in ("gap", "gap_e"):
        fig.add_trace(go.Scatter(x=df["time"], y=df[key], name=NAMES[key], mode="lines", line=styles[key],
                                 visible=True if key == SHOWN else "legendonly", connectgaps=False,
                                 hovertemplate=f"<b>%{{y:,.2f}}</b> USD  {NAMES[key]}<extra></extra>"))
    y_range = _y_range([df[k].to_numpy(dtype=float) for k in ("gap", "gap_e")])
    for y0, y1, color, text, anchor, shift in ((0, y_range[1], STATUS["good"], "▲ เหนือเป้า", "bottom", 2),
                                               (y_range[0], 0, STATUS["critical"], "▼ ใต้เป้า", "top", -2)):
        if y1 > y0:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=_rgba(color, ZONE_ALPHA), line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, y=0, yanchor=anchor, xanchor="left", yshift=shift,
                               text=text, showarrow=False,
                               font=dict(family=FONT_FAMILY, size=11, color=tok["text_secondary"]))
    fig.update_layout(**_layout(tok, height=height, title=title, subtitle=subtitle, y_title="ห่างจากเป้า (USD)",
                                y_range=y_range, legend=True))
    return fig


def excess_history_figure(df: pd.DataFrame, *, theme: str | None = None, title: str | None = None,
                          subtitle: str | None = None, height: int = 340) -> go.Figure:
    """Eₙ สะสมของ bot ตามเวลา (เส้นเดียว → ไม่มี legend · ชื่ออยู่ในหัวกราฟ)"""
    tok = theme_tokens(theme)
    fig = go.Figure(go.Scatter(x=df["time"], y=df["excess"], name="Eₙ สะสม (bot)", mode="lines",
                               line=dict(color=tok["series"][2], width=LINE["total"] + 0.5),
                               hovertemplate="<b>%{y:,.2f}</b> USD  Eₙ สะสม (bot)<extra></extra>"))
    fig.update_layout(**_layout(tok, height=height, title=title, subtitle=subtitle, y_title="Eₙ สะสม (USD)",
                                y_range=_y_range([df["excess"].to_numpy(dtype=float)]), legend=False))
    return fig
