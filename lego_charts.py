"""lego_charts.py — กราฟ Plotly ของ dashboard (pure: ไม่ import streamlit/firebase)

แยกจาก streamlit_app.py เพื่อให้ test figure ได้ตรง ๆ:
  - ledger_figure        : Rₙ (อ้างอิง) vs Aₙ vs Eₙ ของ chain เดียว (Live + Rebalancing 101)
  - total_excess_figure  : Σ Eₙ รวมทุกหุ้น + เส้นรายหุ้น ตามเวลา (UTC)

กติกาการออกแบบ (dataviz): แกน y เดียว, เส้นบาง 1–2px (ค่าใน ``LINE``), จุด act 5px มีวงขอบสีพื้น, ตัวหนังสือใช้สี
text token ไม่ใช้สีเส้น, grid จาง, hover แบบ crosshair, legend เมื่อมี ≥ 2 series และ
label ค่าล่าสุดที่ปลายเส้นเมื่อมีไม่เกิน 4 series. สี categorical ผ่าน validate_palette.js
ทั้งโหมด light และ dark (dark เป็น step ที่เลือกสำหรับพื้นมืด ไม่ใช่การกลับสีอัตโนมัติ)
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

THEME_TOKENS = {
    "light": {
        "surface": "#ffffff",            # พื้นหลัง Streamlit — ใช้เป็นวงขอบ marker
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "axis": "rgba(11,11,11,0.35)",
        "grid": "rgba(11,11,11,0.08)",
        "zero": "rgba(11,11,11,0.30)",
        "tooltip": "#ffffff",
        "button": "rgba(11,11,11,0.06)",
        "button_active": "rgba(42,120,214,0.18)",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#0e1117",
        "text_primary": "#f0efec",
        "text_secondary": "#c3c2b7",
        "axis": "rgba(240,239,236,0.35)",
        "grid": "rgba(240,239,236,0.10)",
        "zero": "rgba(240,239,236,0.35)",
        "tooltip": "#1a1a19",
        "button": "rgba(240,239,236,0.08)",
        "button_active": "rgba(57,135,229,0.30)",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}
FONT_FAMILY = '"Source Sans", "Source Sans Pro", "Noto Sans Thai", sans-serif'
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
LEDGER_NAMES = ("Rₙ อ้างอิง", "Aₙ สะสม", "Eₙ ส่วนเกินสะสม")
ACT_NAME = "act (เทรดจริง)"
TOTAL_NAME = "Σ Eₙ รวม"
OTHER_NAME = "อื่น ๆ"
TOTAL_COL = "รวม"                 # ต้องตรงกับ lego_dash_core.TOTAL_EXCESS_COL
MAX_ACT_MARKERS = 60              # มากกว่านี้จุดจะทับกันจนอ่านไม่ออก -> แสดงเฉพาะเส้น
# ความหนาเส้น (px) / ขนาดจุด / ความทึบพื้นที่ใต้เส้น — ปรับที่นี่ที่เดียว
# เส้นบางแบบกราฟการเงิน: เส้นอ้างอิงบางสุด, เส้นผลรวมหนาสุดเพื่อให้ยังเด่นกว่าเส้นรายหุ้น
LINE = {
    "reference": 1.0,       # Rₙ อ้างอิง (เส้นประ)
    "actual": 1.25,         # Aₙ สะสม
    "excess": 1.5,          # Eₙ ส่วนเกินสะสม
    "symbol": 1.25,         # Eₙ รายหุ้น ในกราฟผลรวม
    "total": 2.0,           # Σ Eₙ ผลรวม
}
REFERENCE_DASH = "4px,3px"
ACT_MARKER = {"size": 5, "ring": 1}
FILL_ALPHA = {"excess": 0.06, "total": 0.05}
MAX_DIRECT_LABELS = 4
# ห้ามขึ้นต้นด้วย "+": Plotly adjustFormat เติม "~" หน้า format ที่ไม่ขึ้นต้นด้วย ~ , . 0 $
# "+,.2f" จึงกลายเป็น "~+,.2f" ที่ d3 อ่านไม่ได้ แล้ว hover โชว์ทศนิยมเต็ม
USD = "%{y:,.2f} USD"


def theme_tokens(theme: str | None) -> dict:
    """token สีของธีม — อะไรที่ไม่ใช่ "dark" ใช้ light"""
    return THEME_TOKENS["dark" if theme == "dark" else "light"]


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(tok: dict, *, height: int, x_title: str) -> dict:
    text = dict(color=tok["text_secondary"], family=FONT_FAMILY, size=12)
    return dict(
        height=height,
        margin=dict(l=8, r=128, t=56, b=8),
        # ต้องทึบ: Plotly วาดเส้นรองใต้ spike ด้วย combine(plot_bgcolor, paper_bgcolor)
        # ถ้าโปร่งใสจะได้สีขาว -> แถบขาวหนา 3px บนธีมมืด
        paper_bgcolor=tok["surface"],
        plot_bgcolor=tok["surface"],
        font=dict(family=FONT_FAMILY, size=13, color=tok["text_secondary"]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=tok["tooltip"], bordercolor=tok["axis"],
                        font=dict(family=FONT_FAMILY, size=13,
                                  color=tok["text_primary"])),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT_FAMILY, size=13, color=tok["text_primary"])),
        # automargin ต้องตั้งเอง: st.plotly_chart ไม่ใช้ template default ของ Plotly
        # (ไม่ตั้ง = เลขแกน y ถูกตัด และเลขแกน x หาย)
        xaxis=dict(title=dict(text=x_title, font=text), tickfont=text, automargin=True,
                   showgrid=False, zeroline=False, showline=True,
                   linecolor=tok["axis"], ticks="outside", tickcolor=tok["axis"],
                   ticklen=4, showspikes=True, spikemode="across", spikesnap="cursor",
                   spikethickness=1, spikedash="solid", spikecolor=tok["axis"]),
        yaxis=dict(title=dict(text="USD", font=text), tickfont=text, automargin=True,
                   tickformat=",.2~f", showgrid=True, gridcolor=tok["grid"],
                   zeroline=True, zerolinecolor=tok["zero"], zerolinewidth=1,
                   showline=False, ticks=""),
    )


def _spread(values: Sequence[float], gap: float) -> list[float]:
    """เลื่อนตำแหน่ง label ในแนวตั้งให้ห่างกันอย่างน้อย ``gap`` (หน่วยข้อมูล) คงลำดับเดิม"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous = None
    for i in order:
        y = float(values[i])
        if previous is not None and y < previous + gap:
            y = previous + gap
        placed[i] = previous = y
    if order:                               # ดันกลับลงครึ่งหนึ่งของส่วนที่เกิน ให้กลุ่มอยู่กลางค่าจริง
        overshoot = placed[order[-1]] - float(values[order[-1]])
        placed = [y - overshoot / 2 for y in placed]
    return placed


def _direct_labels(fig: go.Figure, tok: dict, x_last, items: list[tuple[str, float]],
                   all_values: Sequence[float]) -> None:
    """label ค่าล่าสุดที่ปลายเส้น (สี text token — ตัวตนของเส้นมาจาก legend/สีเส้น)"""
    if not items:
        return
    lo, hi = min([0.0, *all_values]), max([0.0, *all_values])
    gap = (hi - lo) * 0.085 or 1.0
    ys = _spread([value for _, value in items], gap)
    for (label, value), y in zip(items, ys):
        fig.add_annotation(x=x_last, y=y, text=f"{label} <b>{value:+,.2f}</b>",
                           showarrow=False, xanchor="left", xshift=10,
                           font=dict(family=FONT_FAMILY, size=12,
                                     color=tok["text_primary"]))


def ledger_figure(x: Sequence, r: Sequence[float], a: Sequence[float],
                  e: Sequence[float], *, acted: Sequence[bool] | None = None,
                  hover_text: Sequence[str] | None = None, x_title: str = "DNA step",
                  theme: str | None = None, height: int = 420) -> go.Figure:
    """Rₙ (เส้นประอ้างอิง) vs Aₙ (+ จุด act) vs Eₙ (เส้นหลักพร้อมพื้นที่ใต้เส้น)

    ``acted``: True = แถวที่ act (เทรดจริง) — แสดงเป็นจุดบนเส้น Aₙ (ไม่เกิน
    ``MAX_ACT_MARKERS`` จุด); ``hover_text``: ข้อความเสริมต่อแถว เช่น เวลา + สถานะ
    """
    tok = theme_tokens(theme)
    xs, rs, as_, es = list(x), [float(v) for v in r], [float(v) for v in a], [float(v) for v in e]
    r_name, a_name, e_name = LEDGER_NAMES
    blue, orange = tok["series"][0], tok["series"][1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=rs, name=r_name, mode="lines",
        line=dict(color=tok["text_secondary"], width=LINE["reference"], dash=REFERENCE_DASH),
        hovertemplate=f"{r_name}: {USD}<extra></extra>"))
    a_template = f"{a_name}: {USD}"
    if hover_text is not None:
        a_template += "<br><span style='font-size:12px'>%{customdata}</span>"
    fig.add_trace(go.Scatter(
        x=xs, y=as_, name=a_name, mode="lines",
        line=dict(color=blue, width=LINE["actual"]),
        customdata=list(hover_text) if hover_text is not None else None,
        hovertemplate=a_template + "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xs, y=es, name=e_name, mode="lines",
        line=dict(color=orange, width=LINE["excess"]),
        fill="tozeroy", fillcolor=_rgba(orange, FILL_ALPHA["excess"]),
        hovertemplate=f"{e_name}: {USD}<extra></extra>"))
    if acted is not None:
        flags = [bool(v) for v in acted]
        if 0 < sum(flags) <= MAX_ACT_MARKERS:
            fig.add_trace(go.Scatter(
                x=[v for v, f in zip(xs, flags) if f],
                y=[v for v, f in zip(as_, flags) if f],
                name=ACT_NAME, mode="markers", hoverinfo="skip",
                marker=dict(size=ACT_MARKER["size"], color=blue,
                            line=dict(color=tok["surface"], width=ACT_MARKER["ring"]))))
    fig.update_layout(**_base_layout(tok, height=height, x_title=x_title))
    if xs:
        _direct_labels(fig, tok, xs[-1],
                       [("Rₙ", rs[-1]), ("Aₙ", as_[-1]), ("Eₙ", es[-1])],
                       rs + as_ + es)
    return fig


def _fold_symbols(timeline: pd.DataFrame, limit: int) -> pd.DataFrame:
    """มีหุ้นเกิน ``limit`` -> เก็บ ``limit − 1`` ตัวที่ |Eₙ ล่าสุด| มากสุด ที่เหลือรวมเป็น "อื่น ๆ"
    (ไม่สร้างสีที่ 9 — dataviz: series เกินจำนวนสีให้พับเป็น Other)"""
    symbols = [c for c in timeline.columns if c != TOTAL_COL]
    if len(symbols) <= limit:
        return timeline[symbols]
    latest = timeline[symbols].iloc[-1].abs().sort_values(ascending=False, kind="stable")
    keep = sorted(latest.index[:limit - 1])
    out = timeline[keep].copy()
    out[OTHER_NAME] = timeline[[s for s in symbols if s not in keep]].sum(axis=1)
    return out


def total_excess_figure(timeline: pd.DataFrame, *, theme: str | None = None,
                        height: int = 440) -> go.Figure:
    """Σ Eₙ รวมทุกหุ้น (เส้นหนาสีหมึก) + Eₙ รายหุ้น (เส้นบาง) แบบขั้นบันได ตามเวลา (UTC)

    ``timeline`` มาจาก ``lego_dash_core.total_excess_timeline`` — index = เวลา,
    คอลัมน์ = หุ้น + "รวม". เส้นเป็นขั้นบันได (hv) เพราะ Eₙ ค้างค่าจนกว่าจะมีแถวใหม่.
    สีผูกกับหุ้นตามลำดับชื่อ (ไม่ผูกกับอันดับมูลค่า)
    """
    tok = theme_tokens(theme)
    fig = go.Figure()
    x = list(timeline.index)
    per_symbol = _fold_symbols(timeline, len(tok["series"]))
    labels: list[tuple[str, float]] = []
    values: list[float] = []
    for slot, symbol in enumerate(per_symbol.columns):
        y = per_symbol[symbol].astype(float).tolist()
        values += y
        other = symbol == OTHER_NAME
        fig.add_trace(go.Scatter(
            x=x, y=y, name=str(symbol), mode="lines", line_shape="hv",
            line=dict(width=LINE["symbol"], dash="dot" if other else "solid",
                      color=tok["text_secondary"] if other else tok["series"][slot]),
            hovertemplate=f"{symbol}: {USD}<extra></extra>"))
        if y:
            labels.append((str(symbol), y[-1]))
    if TOTAL_COL in timeline.columns:
        total = timeline[TOTAL_COL].astype(float).tolist()
        values += total
        fig.add_trace(go.Scatter(
            x=x, y=total, name=TOTAL_NAME, mode="lines", line_shape="hv",
            line=dict(width=LINE["total"], color=tok["text_primary"]),
            fill="tozeroy", fillcolor=_rgba(tok["text_primary"][:7], FILL_ALPHA["total"]),
            hovertemplate=f"<b>{TOTAL_NAME}: {USD}</b><extra></extra>"))
    layout = _base_layout(tok, height=height, x_title="")
    layout["xaxis"].update(
        title=None,                     # วันที่อธิบายตัวเอง; UTC อยู่ใน hover — ที่ใต้แกนเป็นของ legend
        type="date", hoverformat="%Y-%m-%d %H:%M UTC",
        rangeselector=dict(
            x=1, xanchor="right", y=1.02, yanchor="bottom",
            bgcolor=tok["button"], activecolor=tok["button_active"],
            font=dict(family=FONT_FAMILY, size=12, color=tok["text_primary"]),
            buttons=[dict(count=1, label="1D", step="day", stepmode="backward"),
                     dict(count=7, label="1W", step="day", stepmode="backward"),
                     dict(count=1, label="1M", step="month", stepmode="backward"),
                     dict(label="ทั้งหมด", step="all")]))
    # legend ใต้กราฟ: แถวปุ่มช่วงเวลาอยู่ด้านบน หุ้นหลายตัวจะไม่ทับปุ่ม
    layout["legend"].update(y=-0.16, yanchor="top")
    layout["margin"].update(b=24)
    fig.update_layout(**layout)
    if x:
        items = labels if len(labels) <= MAX_DIRECT_LABELS else []
        if TOTAL_COL in timeline.columns:
            items = items + [("Σ", float(timeline[TOTAL_COL].iloc[-1]))]
        _direct_labels(fig, tok, x[-1], items, values)
    return fig
