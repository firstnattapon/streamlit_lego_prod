"""test_charts.py — กราฟ Plotly (lego_charts) + "Eₙ ผลรวมทุกตัว ตามเวลา" (excess_total_timeline)

ค่าที่คาดหวังของ Eₙ มาจาก STORED_E (test_execution_confirmed_contract คำนวณแยกอิสระ) ไม่ได้ย้อนจากโค้ด
dashboard · ค่าปลายกราฟต้องเท่ากับ metric "Eₙ สะสม (USD) ผลรวมทุกตัว" (latest_excess_total)
"""
from __future__ import annotations

import pandas as pd
import pytest

from lego_charts import (ACT_NAME, LEDGER_NAMES, MAX_ACT_MARKERS, OTHER_NAME,
                         THEME_TOKENS, TOTAL_NAME, _spread, ledger_figure,
                         total_excess_figure)
from lego_dash_core import (TOTAL_EXCESS_COL, excess_total_timeline, latest_excess_total,
                            rows_to_df)
from test_day_filter_total import DAY1, DAY2, P0, STORED_E, chain, prod_db, rtdb_rows
from test_paper_view import PAPER


# --------------------------------------------------------------------------- timeline

def test_timeline_ends_at_the_total_metric_and_counts_the_same_chains():
    db = prod_db()            # AAPL_old + AAPL_new, QBTS_x และ chain PAPER ที่หลงอยู่ใน DB ของ PROD
    rows, state = rows_to_df(db["webull_lego_rows"]), db["webull_lego_state"]
    timeline = excess_total_timeline(rows, state)

    assert list(timeline.columns) == ["AAPL", "QBTS", TOTAL_EXCESS_COL]
    total = latest_excess_total(rows, state)["total"]
    assert timeline[TOTAL_EXCESS_COL].iloc[-1] == pytest.approx(total, abs=1e-12)
    assert total == pytest.approx(STORED_E[5] + STORED_E[4])
    # AAPL_old (30 ก.ค.) เป็น chain เก่า และ chain PAPER ไม่ถูกนับ — ไม่มีเวลาของทั้งคู่ในกราฟ
    assert timeline.index.min() == pd.Timestamp(f"{DAY1}T14:00:00Z")
    assert str(timeline.index.tz) == "UTC" and timeline.index.is_monotonic_increasing


def test_timeline_follows_each_chain_and_counts_zero_before_its_first_row():
    df = rows_to_df(rtdb_rows(chain("AAPL_new", "AAPL"),
                              chain("QBTS_x", "QBTS", n=5, days=(DAY2,) * 5)))
    state = {"AAPL_new": {"p0": P0}, "QBTS_x": {"p0": P0}}
    timeline = excess_total_timeline(df, state)

    assert len(timeline) == 11
    assert timeline["AAPL"].tolist() == pytest.approx(STORED_E[:6] + [STORED_E[5]] * 5)
    assert timeline["QBTS"].tolist() == pytest.approx([0.0] * 6 + STORED_E[:5])
    assert (timeline["AAPL"] + timeline["QBTS"] - timeline[TOTAL_EXCESS_COL]).abs().max() <= 1e-12
    assert timeline[TOTAL_EXCESS_COL].iloc[-1] == pytest.approx(
        latest_excess_total(df, state)["total"], abs=1e-12)


def test_the_paper_page_draws_its_paper_chain_and_prod_does_not():
    rows, state = rows_to_df(PAPER["webull_lego_rows"]), PAPER["webull_lego_state"]
    on_paper = excess_total_timeline(rows, state, paper_page=True)
    assert on_paper[TOTAL_EXCESS_COL].iloc[-1] == pytest.approx(
        latest_excess_total(rows, state, paper_page=True)["total"], abs=1e-12)
    assert excess_total_timeline(rows, state).empty       # หน้า PROD ไม่นับ chain PAPER


def test_rows_without_a_readable_time_leave_the_timeline_only():
    rows = chain("QBTS_x", "QBTS")
    rows.loc[2, "เวลา (UTC)"] = "not-a-time"
    df = rows_to_df(rtdb_rows(rows))
    state = {"QBTS_x": {"p0": P0}}
    timeline = excess_total_timeline(df, state)
    assert timeline["QBTS"].tolist() == pytest.approx(STORED_E[:2] + STORED_E[3:6])
    assert latest_excess_total(df, state)["total"] == pytest.approx(STORED_E[5])
    assert excess_total_timeline(pd.DataFrame(), {}).empty


# --------------------------------------------------------------------------- figures

def _ledger(theme=None, acted=None, n=5):
    xs = list(range(n))
    return ledger_figure(xs, [0.5 * i for i in xs], [1.0 * i for i in xs],
                         [0.25 * i for i in xs], acted=acted, theme=theme,
                         hover_text=[f"t{i}" for i in xs])


def test_ledger_figure_traces_single_axis_and_usd_hover():
    fig = _ledger(acted=[False, True, False, True, True])
    names = [trace.name for trace in fig.data]
    assert names == [*LEDGER_NAMES, ACT_NAME]
    assert "yaxis2" not in fig.layout.to_plotly_json()          # แกน y เดียว
    for trace in fig.data[:3]:
        # ",.2f" ขึ้นต้นด้วย "," — Plotly ไม่เติม "~" (format ที่ขึ้นต้นด้วย "+" จะเสีย)
        assert "%{y:,.2f} USD" in trace.hovertemplate
    assert "%{customdata}" in fig.data[1].hovertemplate
    assert list(fig.data[3].x) == [1, 3, 4]                       # จุด act เฉพาะแถว act
    assert fig.data[3].marker.size == 8
    assert fig.data[0].line.dash == "dash"                        # Rₙ = เส้นอ้างอิง
    assert len(fig.layout.annotations) == 3                       # label ค่าล่าสุด Rₙ/Aₙ/Eₙ


def test_ledger_figure_hides_act_markers_when_too_dense_or_absent():
    n = MAX_ACT_MARKERS + 1
    assert len(_ledger(acted=[True] * n, n=n).data) == 3
    assert len(_ledger(acted=[False] * 5).data) == 3
    assert len(_ledger(acted=None).data) == 3


def test_figures_use_selected_theme_tokens_with_opaque_surface():
    for theme, key in (("dark", "dark"), ("light", "light"), (None, "light")):
        tok = THEME_TOKENS[key]
        fig = _ledger(theme=theme, acted=[True] * 5)
        assert fig.data[1].line.color == tok["series"][0]
        assert fig.data[2].line.color == tok["series"][1]
        assert fig.data[3].marker.line.color == tok["surface"]
        # พื้นต้องทึบ: Plotly วาดเส้นรองใต้ spike ด้วยสีพื้น (โปร่งใส = แถบขาวบนธีมมืด)
        assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == tok["surface"]
    assert THEME_TOKENS["dark"]["series"] != THEME_TOKENS["light"]["series"]


def _timeline(symbols: list[str], rows: int = 4) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame({s: [float(k + i) for i in range(rows)]
                          for k, s in enumerate(symbols, start=1)}, index=index)
    frame[TOTAL_EXCESS_COL] = frame.sum(axis=1)
    return frame


def test_total_excess_figure_lines_per_stock_plus_bold_total():
    fig = total_excess_figure(_timeline(["AAPL", "NVDA", "TSLA"]), theme="light")
    tok = THEME_TOKENS["light"]
    assert [t.name for t in fig.data] == ["AAPL", "NVDA", "TSLA", TOTAL_NAME]
    assert [t.line.color for t in fig.data[:3]] == tok["series"][:3]
    assert all(t.line.shape == "hv" for t in fig.data)
    assert fig.data[-1].line.width == 3 and fig.data[-1].line.color == tok["text_primary"]
    assert fig.layout.xaxis.type == "date"
    assert "yaxis2" not in fig.layout.to_plotly_json()
    texts = [a.text for a in fig.layout.annotations]
    assert len(texts) == 4 and texts[-1].startswith("Σ")       # ≤ 4 หุ้น: label ทุกเส้น


def test_total_excess_figure_folds_beyond_palette_into_other_and_keeps_sum():
    symbols = [f"S{i:02d}" for i in range(10)]
    timeline = _timeline(symbols)
    fig = total_excess_figure(timeline, theme="dark")
    names = [t.name for t in fig.data]
    assert len(names) == 9 and names[-2:] == [OTHER_NAME, TOTAL_NAME]
    plotted = sum(pd.Series(t.y) for t in fig.data[:-1])
    assert (plotted - timeline[TOTAL_EXCESS_COL].to_numpy()).abs().max() <= 1e-9
    kept = names[:-2]                                  # 7 ตัวที่ |Eₙ ล่าสุด| มากสุด
    assert kept == sorted(symbols[-7:])
    texts = [a.text for a in fig.layout.annotations]
    assert len(texts) == 1 and texts[0].startswith("Σ")   # > 4 หุ้น: label เฉพาะยอดรวม


def test_spread_keeps_order_and_min_gap():
    placed = _spread([5.0, 5.1, -3.0, 5.05], gap=1.0)
    order = sorted(range(4), key=lambda i: [5.0, 5.1, -3.0, 5.05][i])
    ys = [placed[i] for i in order]
    assert all(b - a >= 1.0 - 1e-12 for a, b in zip(ys, ys[1:]))
