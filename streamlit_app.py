"""streamlit_app.py — Dashboard (read-only) อ่านจาก Firebase RTDB + Rebalancing 101

4 แท็บ:
  📊 Live Dashboard   — อ่าน Firebase ครั้งเดียวต่อการโหลดหน้า (lego_live_data.Snapshot) แล้วแสดงใน
                        st.fragment: ตัวเลือก Chain (จำไว้ใน URL ?chain=) → หัวข้อ "หุ้น · Live" พร้อมสถานะ
                        สด/ค้าง/ผิดพลาด เวลาอัปเดต และปุ่มรีเฟรช → ตัวกรองวัน (UTC) → แท็บย่อย
                        ข้อมูล (การ์ด metric รวม Eₙ ผลรวมทุกตัว, ตาราง 17 คอลัมน์, Order audit,
                        Integrity E1–E8 ที่คำนวณเมื่อเปิดเท่านั้น, State pointer) และ กราฟ (Rₙ/Aₙ/Eₙ ตามวัน
                        + Eₙ ผลรวมทุกตัวตามเวลา) — สลับ Chain/แท็บ/วันไม่อ่าน Firebase ซ้ำ ไม่คำนวณซ้ำ
                        (กราฟทั้งหมดเป็น Plotly จาก lego_charts.py)
  🎓 Rebalancing 101  — playground เชิงสอน "gated demo (DNA gate + บัญชีแบบแช่แข็ง)"
                        ไม่ต้องมี Firebase: สุ่มราคา -> เดิน ledger gated_theoretical_v2
                        เห็นว่ารอบ "จำนวนสั่ง = 0" (pass/แช่แข็ง) ถูกบันทึกอย่างไร
  🌐 Web Apps         — คู่มือ/playground แบบโต้ตอบ (HTML ล้วนใน web_apps/ แสดงใน iframe)
                        คัดลอกจากแท็บ Web Apps ของ pages/Manual.py (leg-code):
                        Rebalancing 101 + Rebalancing Playground — ไม่เรียก Firebase
  ⚡ Cheat Sheet       — แท็บย่อย 📝 Summary: สรุป 10 ข้อ + ประโยคเดียว (cheat_sheet/summary.md)
                        และ 🔀 Diagrams: แผนภาพ Mermaid (.mmd ใน cheat_sheet/): main rebalancing,
                        rollover (beta += (Fix_c − Fix_c_new) * ln(P_acted / P₀)) และ Option Scale
                        (จ่าย K ก่อน → เก็บ E → ขยาย Fix_c ด้วยกำไรที่ล็อกแล้ว) แสดงเป็นภาพ PNG ที่ render ไว้ล่วงหน้า

deploy บน streamlit.app: ใส่ service account JSON + DB URL ใน st.secrets (เฉพาะแท็บ Live)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from contextlib import contextmanager
from pathlib import Path

import firebase_admin
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from firebase_admin import credentials, db

from dna_engine import DNAError
import lego_dash_core as _lego_dash_core
from lego_hover_table import build_hover_table_html, hover_table_height, style_status_rows

# Streamlit reruns the entry script in a long-lived process. During a hot deploy,
# sys.modules can still hold the previous lego_dash_core while this file is newer.
# Reload once when any public API used by this entry point is missing, then import
# the public API. This prevents a mixed-version process (new streamlit_app.py +
# stale lego_dash_core.py) from crashing with AttributeError after deployment.
_REQUIRED_DASH_CORE_API = (
    "FROZEN_TERMINAL_SEMANTICS",
    "system_health_rows",
    "LEGACY_DROPPED_COLS",
    "ledger_explanations",
    "audit_with_ledger_time",
    "HOLDINGS_COL",
    "table_tooltips",
    "leaps_summary",
    # หน้าไม่เรียกฟังก์ชันนี้แล้ว (ปิดป้ายแจ้ง) แต่ยังเป็นตัวบอกว่า core รองรับ genesis_baseline=True
    # ที่ recompute_gated_ledger/table_tooltips ถูกเรียกใน lego_live_data — core เก่ากว่านั้นต้องถูกโหลดใหม่
    "genesis_baseline_info",
    "ALL_DAYS",
    "day_options",
    "day_positions",
    "chain_p0_hint",
    "latest_excess_total",
    "excess_total_timeline",
)
if any(not hasattr(_lego_dash_core, name) for name in _REQUIRED_DASH_CORE_API):
    _lego_dash_core = importlib.reload(_lego_dash_core)

from lego_dash_core import (ALL_DAYS, DEFAULT_GATED_DNA, HOLDINGS_COL,
                            MAX_REBALANCING_STEPS, MONEY_COLS,
                            audit_with_ledger_time, build_gate_actions,
                            day_options, day_positions, excess_total_timeline,
                            filter_audit_rows,
                            gated_rebalancing_cashflow_from_prices,
                            latest_excess_total, order_columns, pending_broker_fee_count,
                            rebalancing_cashflow_from_prices,
                            simulate_rebalancing_prices)

# เหตุผลเดียวกับด้านบน: lego_live_data ที่แคชไว้จากรุ่นก่อนต้องไม่ทำให้หน้าใหม่ล่ม
import lego_live_data as _lego_live_data

_REQUIRED_LIVE_DATA_API = ("fetch_snapshot", "Snapshot", "Freshness", "describe",
                           "fmt_num", "fmt_text", "state_table")
if any(not hasattr(_lego_live_data, name) for name in _REQUIRED_LIVE_DATA_API):
    _lego_live_data = importlib.reload(_lego_live_data)

from lego_live_data import (ERROR, LIVE, STALE, ChainView, Freshness, Snapshot, SnapshotError,
                            describe, fetch_snapshot, fmt_num, fmt_text, state_table)
from lego_charts import (MAX_ACT_MARKERS, PLOTLY_CONFIG, ledger_figure,
                         total_excess_figure)

# Web Apps: คู่มือ/playground แบบ HTML ล้วน (self-contained) แสดงใน iframe ผ่าน
# components.html — ชุดนี้คัดลอกมาจากแท็บ "🌐 Web Apps" ของ pages/Manual.py (leg-code)
WEB_APPS_DIR = Path(__file__).resolve().parent / "web_apps"
WEB_APPS = {
    "Rebalancing 101": "rebalancing101.html",
    "Rebalancing Playground": "rebalancing_playground.html",
}

# Cheat Sheet: แผนภาพ Mermaid (.mmd) — แท็บแสดงภาพ PNG ที่ render ไว้ล่วงหน้าด้วย
# tools/render_cheat_sheet.py ชื่อภาพมี hash ของ .mmd (<ชื่อ>.<sha8>.png) จึงรู้ทันทีเมื่อ
# แก้ .mmd แล้วยังไม่ render ใหม่ (แทนที่จะโชว์ภาพเก่าเงียบ ๆ)
CHEAT_SHEET_DIR = Path(__file__).resolve().parent / "cheat_sheet"
CHEAT_SHEETS = {               # หัวข้อ -> ชื่อไฟล์ .mmd (ไม่รวมนามสกุล)
    "1. Main rebalancing": "main_rebalancing",
    "2. Rollover": "rollover",
    "3. Option Scale": "option_scale",
}
# แท็บย่อย Summary แสดงไฟล์ Markdown นี้ (สรุป 10 ข้อ + ประโยคเดียว) — แก้ข้อความที่ไฟล์ได้เลย
CHEAT_SUMMARY = "summary.md"


def _has_secret(key: str) -> bool:
    try:                      # ไม่มีไฟล์ secrets เลย -> st.secrets จะ raise ไม่ใช่คืน False
        return key in st.secrets
    except Exception:
        return False


# LEGO_MODE=paper เก็บหุ้นและ fill ปลอมใน database แยก (lego-firebase-paper) ของ project เดียวกัน
PROD_DB_SECRET, PAPER_DB_SECRET = "FIREBASE_DB_URL", "FIREBASE_DB_URL_PAPER"
DATABASE_PROD, DATABASE_PAPER = "PROD", "PAPER (จำลอง)"


def _secret_url(key: str) -> str | None:
    if not _has_secret(key):
        return None
    return str(st.secrets[key]).strip() or None


def _prod_url() -> str | None:
    """database ของ PROD — ไม่ตั้ง = หน้านี้อ่าน paper อย่างเดียว"""
    return _secret_url(PROD_DB_SECRET)


def _paper_url() -> str | None:
    """database ของ paper เมื่อ secret FIREBASE_DB_URL_PAPER ระบุไว้"""
    return _secret_url(PAPER_DB_SECRET)


def _secrets_ready() -> bool:
    """ต้องมี service account และ database อย่างน้อยหนึ่งตัว (PROD หรือ paper)"""
    return _has_secret("FIREBASE_SA_JSON") and bool(_prod_url() or _paper_url())


def _ref(path: str, paper: bool = False):
    """อ่านอย่างเดียว: PROD ตามเดิม หรือ database ของ paper (app และ service account เดียวกัน)"""
    return db.reference(path, url=_paper_url()) if paper else db.reference(path)


@st.cache_resource
def _init():
    if not firebase_admin._apps:
        cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_SA_JSON"]))
        # ไม่มี PROD ก็ตั้ง app บน paper — หน้า paper ระบุ url เองทุกครั้งอยู่แล้ว
        firebase_admin.initialize_app(cred, {"databaseURL": _prod_url() or _paper_url()})
    return True


def _db_name(paper: bool) -> str:
    return "PAPER" if paper else "PROD"


@contextmanager
def safe_section(action: str):
    """ส่วนเสริมล้มได้โดยไม่พาหน้าล้ม: คำเตือน "<action>ไม่ได้ (…)" แทนเนื้อหา (st.rerun/st.stop ผ่านไปปกติ)"""
    try:
        yield
    except Exception as exc:
        st.warning(f"{action}ไม่ได้ ({type(exc).__name__}: {exc})")


TOTAL_EXCESS_LABEL = "Eₙ สะสม (USD) ผลรวมทุกตัว"
DAY_FILTER_LABEL = "วัน (UTC)"
STATUS_COLORS = {LIVE: "green", STALE: "orange", ERROR: "red"}


def render_ledger_table(view: ChainView, rows: list[int]) -> None:
    """ตาราง 17 คอลัมน์ของแถว ``rows`` (ตัวกรองวัน): โหมดดูที่มา (default) หรือ st.dataframe เรียงได้

    ที่มาคำนวณจากค่าที่ DB บันทึกทั้ง chain แล้วเลือกเฉพาะแถว — P_acted/Aₙ₋₁ อาจมาจากแถวของวันก่อน
    """
    show = order_columns(view.ledger.iloc[rows].copy())
    for col in MONEY_COLS:
        if col in show:
            show[col] = show[col].astype(float).round(2)
    if st.toggle("ดูที่มา", value=True, key="live_hover_table",
                 help="ชี้เมาส์ (หรือกด Tab) ที่ตัวเลข Rₙ / ΔAₙ / Aₙ / Eₙ เพื่อดูสูตรพร้อมแทนค่าจริง — "
                      "ปิดเพื่อใช้ตารางแบบเรียง/กรองได้ (st.dataframe) แต่ไม่มี tooltip ที่มา"):
        try:   # LEAPS แทนหุ้น: แถวที่ bot ใช้ LEAPS ได้ tooltip ที่คอลัมน์ 7 (หุ้นจริง + LEAPS) ด้วย
            tips, tip_cols = view.tooltips()
            components.html(build_hover_table_html(show, [tips[i] for i in rows], tip_cols, MONEY_COLS,
                                                   theme=_streamlit_theme()),
                            height=hover_table_height(len(show)))
            st.caption("ชี้เมาส์ (หรือกด Tab) ที่ Rₙ · ΔAₙ · Aₙ · Eₙ"
                       + (" · จำนวนถือครอง (หุ้นจริง + LEAPS)" if HOLDINGS_COL in tip_cols else "")
                       + " เพื่อดูสูตรที่แทนค่าจริงและแถวที่ P_acted/Aₙ₋₁ มาจาก (tooltip แสดง 4 ตำแหน่ง)")
            return
        except Exception as exc:   # tooltip เป็นส่วนเสริม — ห้ามทำให้ตารางหลักล่ม
            st.warning(f"สร้างตารางแบบชี้เมาส์ไม่ได้ ({type(exc).__name__}: {exc}) "
                       "— แสดงตารางปกติแทน")
    st.dataframe(style_status_rows(show, MONEY_COLS), width="stretch")


def excess_total_metric(rows: pd.DataFrame, state, paper: bool) -> tuple[str, str]:
    """(ค่า, help) ของ metric "Eₙ สะสม (USD) ผลรวมทุกตัว" — ส่วนเสริม ห้ามทำให้หน้าล่ม

    ``rows`` = ทุกแถวของฐานข้อมูลนี้ (ก่อนเลือก chain); หน้า PROD ไม่นับ chain PAPER
    """
    try:
        summary = latest_excess_total(rows, state, paper_page=paper)
    except Exception as exc:
        return "—", f"คำนวณผลรวมไม่ได้ ({type(exc).__name__}: {exc})"
    total = summary["total"]
    # round + 0.0: ผลรวมที่ปัดแล้วเป็นศูนย์ติดลบ (เช่น -1e-16) แสดง "0.00" ไม่ใช่ "-0.00"
    value = "—" if total is None else f"{round(total, 2) + 0.0:.2f}"
    return value, summary["text"]


def render_total_excess(snap: Snapshot, paper: bool) -> None:
    """กราฟ "Eₙ ผลรวมทุกตัว ตามเวลา" — chain ชุดเดียวกับ metric ผลรวม จึงจบที่ค่าเดียวกับ metric"""
    st.subheader("Eₙ ผลรวมทุกตัว ตามเวลา", anchor=False)
    with safe_section("สร้างกราฟ Eₙ ผลรวมทุกตัว"):
        timeline = snap.memo("total_timeline", lambda: excess_total_timeline(
            snap.rows, snap.state, paper_page=paper))
        if timeline.empty:
            st.caption("ยังไม่มี chain ที่นับได้ (หรือไม่มีแถวที่มีเวลา UTC และ Eₙ) จึงยังไม่มีกราฟ")
            return
        st.plotly_chart(total_excess_figure(timeline, theme=_streamlit_theme()),
                        theme=None, config=PLOTLY_CONFIG, key=f"live_total_excess:{_db_name(paper)}")
        st.caption(
            "เส้นหนา = ผลรวม · เส้นบาง = Eₙ ของแต่ละหุ้น · นับ chain ชุดเดียวกับ metric "
            f"“{TOTAL_EXCESS_LABEL}” (chain ล่าสุดของแต่ละหุ้น"
            + ("" if paper else ", ไม่นับ chain PAPER")
            + ") จึงจบที่ค่าเดียวกัน · Eₙ เป็นค่าสะสมอยู่แล้ว: ผลรวม ณ เวลา t = ผลบวก Eₙ ล่าสุดของแต่ละ "
            "chain (ไม่ cumsum ซ้ำ), ก่อนแถวแรกของ chain นับเป็น 0 · กราฟนี้ไม่ขึ้นกับตัวกรองวัน "
            "(ใช้ปุ่ม 1D / 1W / 1M หรือลากเพื่อซูม)")
        with st.expander("ตาราง Eₙ ล่าสุดรายหุ้น", expanded=False):
            table = timeline.iloc[-1].rename("Eₙ ล่าสุด (USD)").astype(float).round(2).to_frame()
            table.index.name = "หุ้น"
            st.dataframe(table, width="stretch")


def live_ledger_figure(view: pd.DataFrame):
    """กราฟ Rₙ / Aₙ / Eₙ ของแถวที่ตัวกรองวันแสดง — จุด act = แถวที่ ΔAₙ ≠ 0"""
    delta = "ΔAₙ ต่อสเต็ป (USD)"
    acted = (view[delta].astype(float).abs() > 1e-12).tolist() if delta in view.columns else None
    parts = [view[col].astype(str) for col in ("เวลา (UTC)", "สถานะ") if col in view.columns]
    hover = parts[0].str.cat(parts[1:], sep=" · ").tolist() if parts else None
    return ledger_figure(
        view["DNA step"].tolist(),
        view["Rₙ อ้างอิง (USD)"], view["Aₙ สะสม (USD)"], view["Eₙ ส่วนเกินสะสม (USD)"],
        acted=acted, hover_text=hover, x_title="DNA step", theme=_streamlit_theme())


def render_day_filter(df: pd.DataFrame, chain, paper: bool) -> list[int]:
    """selectbox "วัน (UTC)" (ใช้ร่วมทั้งแท็บ ข้อมูล/กราฟ) — คืนตำแหน่งแถวที่ตาราง กราฟ และ Order audit แสดง

    ``df`` คือ ledger ที่คำนวณจากทั้ง chain แล้ว ตัวกรองแค่เลือกแถว ไม่คำนวณอะไรใหม่.
    ค่าเริ่มต้น = วันล่าสุดที่ chain มีแถว (ตัวแรกของรายการ); key ผูกกับฐานข้อมูล, chain และวันล่าสุด
    จึงกลับไปวันล่าสุดเองเมื่อสลับ PROD/PAPER เปลี่ยน chain หรือมีวันใหม่ ส่วนวันที่เลือกเองยังค้างอยู่
    ระหว่างกด widget อื่น
    """
    options = day_options(df)
    if not options:                       # ไม่มีเวลาที่อ่านได้ -> แสดงทุกแถวเหมือนเดิม
        return list(range(len(df)))
    pick, note = st.columns([1, 3], vertical_alignment="bottom")
    day = pick.selectbox(
        DAY_FILTER_LABEL, options, index=0, key=f"live_day:{_db_name(paper)}:{chain}:{options[0]}",
        help="เลือกวัน (UTC) ที่จะแสดงในตาราง 17 คอลัมน์ กราฟ และ Order audit — ค่าเริ่มต้นคือวันล่าสุด "
             "ที่ chain มีแถว · ตัวเลขทุกตัวยังคำนวณจากทั้ง chain ตัวกรองแค่เลือกแถวที่แสดง "
             "(metric และ Integrity check ใช้ทั้ง chain เสมอ)")
    rows = day_positions(df, day)
    note.caption(f"แสดง {len(rows):,} จาก {len(df):,} แถวของ chain นี้ · "
                 "Rₙ / ΔAₙ / Aₙ / Eₙ และ tooltip คำนวณจากทั้ง chain")
    return rows


def render_header(title: str, fresh: Freshness, fetched_at: pd.Timestamp, chain,
                  now: pd.Timestamp) -> None:
    """หัวข้อเดียว + ป้าย สด/ค้าง/ผิดพลาด (badge ไม่ใช่ st.warning) + เวลาอัปเดต + ปุ่มรีเฟรชมุมขวา"""
    head, status, action = st.columns([4, 6, 1.4], vertical_alignment="center")
    head.subheader(title, anchor=False)
    where = "" if chain is None else " · chain `" + str(chain).replace("`", "'") + "`"
    status.markdown(f":{STATUS_COLORS[fresh.status]}-badge[{fresh.label}] {describe(fresh, now)}{where}")
    if action.button("รีเฟรช", key="live_refresh", icon=":material/refresh:", width="stretch",
                     help=f"ดึงข้อมูลล่าสุดเมื่อ {fetched_at:%Y-%m-%d %H:%M:%S} UTC — กดเพื่อดึงใหม่ "
                          "(หน้าไม่รีเฟรชเอง)"):
        st.rerun(scope="app")      # รันเต็มสคริปต์ = อ่าน Firebase ใหม่


def render_health(snap: Snapshot) -> None:
    """node เสริมที่อ่านไม่ได้ + System health — ต้องเห็นได้ก่อนมีแถวแรก (ระบบพักเพราะ Authentication)"""
    for note in snap.notes:
        st.warning(note)
    if snap.health.empty:
        return
    if snap.alarm:
        st.error("ระบบพักการทำงานจากปัญหา Authentication — ตรวจ credentials/token และเวลาลองใหม่")
    with st.expander("System health — สถานะปัจจุบันและประวัติ", expanded=snap.alarm):
        st.dataframe(snap.health, width="stretch")
        st.caption("RESOLVED/HISTORY เป็นประวัติ ไม่ใช่หลักฐานว่าบัญชีพร้อมเทรดเงินจริง")


def render_chain_picker(snap: Snapshot):
    """Chain แบบ 1 คลิก ผูกกับ ?chain= (ซ่อนเมื่อมีตัวเดียว) — ค่าที่ไม่มีใน options กลับไป chain ล่าสุด"""
    chains = snap.chains()
    if len(chains) <= 1:
        return chains[0] if chains else None
    return st.radio("Chain", chains, index=chains.index(snap.default_chain()), horizontal=True,
                    key="chain", bind="query-params", label_visibility="collapsed")


def render_alerts(view: ChainView, rows: list[int]) -> None:
    """คำเตือนที่ต้องเห็นทั้งแท็บ ข้อมูล และ กราฟ (เรื่องเงิน/ความถูกต้องห้ามซ่อนอยู่หลังแท็บ)"""
    if view.paper_marked:
        st.markdown(":orange-background[**PAPER**] chain นี้: จำนวนหุ้นและ fill เป็นการจำลอง "
                    "(Preview จริง ไม่เคย Place) ไม่ใช่เงินจริง")
    if view.n_corrections:
        st.warning("⚠️ engine เขียน ledger " + (
            f"ไม่ตรง execution-confirmed contract {view.n_corrections} แถว — ตาราง/กราฟคำนวณใหม่จาก fill "
            "ที่ FINALIZED เท่านั้น; pending/rejected/PASS แช่แข็ง ควรตรวจ engine และ order audit ต้นทาง"
            if view.execution_mode else
            f"ผิด {view.n_corrections} แถว (แถว PASS ที่ยัง act — ΔAₙ ≠ 0) — ตาราง/กราฟถูก **คำนวณใหม่ตาม"
            "หลักการแช่แข็ง** (ทุก PASS ทั้ง PASS_DNA_ZERO และ PASS_THRESHOLD → ΔAₙ = 0, Aₙ ค้าง) "
            "ควรแก้ engine ต้นทางด้วย"))
    # นับจาก audit ทั้ง chain: คำเตือน money fence ต้องไม่หายเพราะตัวกรองวัน
    if pending := pending_broker_fee_count(view.audit_rows()):
        st.warning(f"มี {pending} fill ที่ broker fee ยังเป็น PENDING — "
                   "ระบบไม่ถือเป็นศูนย์และยังไม่ปลด money fence จนกว่าจะกระทบยอด"
                   + (" (นับทั้ง chain ไม่ขึ้นกับวันที่เลือก)" if len(rows) < len(view.ledger) else ""))
    for note in view.notes:
        st.warning(note)


def render_audit(view: ChainView, rows: list[int]) -> None:
    """Order audit (redacted) ของแถวที่ตัวกรองวันแสดง (วันของ order = วันของแถวที่ order ผูกอยู่)"""
    chain_adf = view.audit_rows()
    if chain_adf.empty:
        return
    st.subheader("Order audit", anchor=False)
    shown = view.ledger.iloc[rows]
    adf = (filter_audit_rows(view.audit, shown["run_id"] if "run_id" in shown.columns else [])
           if len(rows) < len(view.ledger) else chain_adf)
    if adf.empty:
        st.caption(f"ไม่มี order ในวันที่เลือก — เลือก “{ALL_DAYS}” เพื่อดู order ทั้ง chain")
        return
    adf = audit_with_ledger_time(adf, shown)
    first = [name for name in ("เวลา (UTC)", "run_id", "status", "reject_reason",
                               "terminal_reason", "filled_quantity", "broker_fee_status")
             if name in adf]
    st.dataframe(adf[first + [name for name in adf if name not in first]], width="stretch")


def render_data_tab(snap: Snapshot, view: ChainView, rows: list[int], paper: bool) -> None:
    """การ์ด 5 metric → คำอธิบาย semantics → ตาราง 17 คอลัมน์ → Order audit → Integrity → State"""
    latest = view.ledger.iloc[-1]          # fmt_num/fmt_text: null -> "—" ไม่ล่ม
    total_value, total_help = snap.memo("total_metric", lambda: excess_total_metric(
        snap.rows, snap.state, paper))
    cards = st.columns(5)
    cards[0].metric("สถานะตัดสินใจล่าสุด" if view.execution_mode else "สถานะล่าสุด",
                    fmt_text(latest.get("สถานะ")))
    cards[1].metric("DNA step", fmt_num(latest.get("DNA step"), 0))
    cards[2].metric("Pₙ (USD)", fmt_num(latest.get("ราคา Pₙ (USD)")))
    cards[3].metric("Eₙ สะสม (USD)", fmt_num(latest.get("Eₙ ส่วนเกินสะสม (USD)")))
    cards[4].metric(TOTAL_EXCESS_LABEL, total_value, help=total_help)
    if view.frozen_v2_mode and (mark := fmt_num(latest.get("E_mark_at_observation"))) != "—":
        st.caption(f"E_mark ล่าสุด: {mark} USD · "
                   "Eₙ เป็นค่าที่ freeze จาก terminal fill และ dashboard ไม่คำนวณทับ")
    if view.execution_mode:
        st.caption(f"Execution cashflow ล่าสุด: {fmt_text(latest.get('cashflow_status'))} · "
                   "READY_BUY/READY_SELL คือ decision intent ไม่ใช่หลักฐานว่า fill แล้ว")
        st.caption("execution_confirmed_v1: act เฉพาะ cashflow_status=FINALIZED ที่มี "
                   "execution_quantity และ execution_price > 0 · ΔAₙ ใช้ราคาที่ fill จริง; "
                   "pending/rejected/PASS แช่แข็ง Aₙ/P_acted · Rₙ ยังวิ่งตาม quote Pₙ ทุกแถว")
    else:
        st.caption("gated_theoretical_v2: แถว act (READY_BUY/SELL) ΔAₙ = FIX_C×(Pₙ/P_acted−1) "
                   "เทียบราคาแถว act ล่าสุด · แถว pass (ไม่เทรด — PASS_DNA_ZERO หรือ PASS_THRESHOLD, "
                   "จำนวนสั่ง = 0) ΔAₙ = 0, Aₙ ค้าง, Eₙ ค้างค่า act ล่าสุด (smooth) — Eₙ ไม่ลดและ ≥ 0 เสมอ")
    if view.leaps:
        st.caption("LEAPS แทนหุ้น: จำนวนถือครอง = หุ้นจริง + LEAPS (หุ้นจำลอง, 1 สัญญา = 100) — "
                   f"แถวล่าสุด {view.leaps['text']} · SELL ได้ไม่เกินหุ้นจริง · LEAPS เปลี่ยนกลาง chain "
                   f"{view.leaps['changes']} ครั้ง (บันทึกที่แถวนั้นใน leaps_shares_changed_from)")
    with safe_section("แสดงตาราง 17 คอลัมน์"):
        render_ledger_table(view, rows)
    with safe_section("แสดง Order audit"):
        render_audit(view, rows)

    # Integrity ตรวจค่าที่ DB บันทึก (ก่อน recompute เพื่อแสดง) — O(n²) จึงคำนวณเมื่อเปิดเท่านั้น
    with st.expander("🔎 Integrity E1–E8", key="live_integrity", on_change="rerun") as box:
        if box.open:
            with safe_section("ตรวจ Integrity"), st.spinner("กำลังตรวจสมการ"):
                report, ok = view.integrity()
                st.dataframe(report, width="stretch")
                if ok:
                    st.success("ข้อมูลที่บันทึกผ่านการตรวจสมการในช่วงที่โหลด — "
                               "ไม่ใช่การรับรองความพร้อมเทรดเงินจริง")
                else:
                    st.error("พบแถวที่ไม่สอดคล้องสมการ — ตรวจ chain/engine ก่อนเชื่อกราฟ")

    if entry := snap.entry(view.key):
        with st.expander("State pointer"):
            st.dataframe(state_table(entry), width="stretch")


def render_chart_tab(snap: Snapshot, view: ChainView, rows: list[int], paper: bool) -> None:
    """กราฟ Rₙ / Aₙ / Eₙ ของ chain ตามตัวกรองวัน → กราฟ Eₙ ผลรวมทุกตัวตามเวลา"""
    st.subheader("Rₙ / Aₙ / Eₙ", anchor=False)
    with safe_section("สร้างกราฟ Rₙ / Aₙ / Eₙ"):
        st.plotly_chart(live_ledger_figure(view.ledger.iloc[rows]), theme=None,
                        config=PLOTLY_CONFIG, key="live_ledger_chart")
        st.caption("จุดบนเส้น Aₙ = แถว act (ΔAₙ ≠ 0) · ชี้เมาส์ดูค่าทุกเส้น ณ DNA step นั้น "
                   "พร้อมเวลา (UTC) และสถานะ · ลากเพื่อซูม ดับเบิลคลิกเพื่อรีเซ็ต")
    render_total_excess(snap, paper)


def _render_live(snap: Snapshot, paper: bool, now: pd.Timestamp) -> None:
    render_health(snap)
    if snap.rows.empty:
        render_header("Live", snap.freshness(None, now), snap.fetched_at, None, now)
        st.info("ยังไม่มีแถว committed — รอ Cloud Function รอบแรก")
        return
    chain = render_chain_picker(snap)       # อยู่เหนือทุกอย่าง: chain ที่เสียยังสลับไปตัวอื่นได้
    render_header(" · ".join(part for part in (snap.symbol(chain), "Live") if part),
                  snap.freshness(chain, now), snap.fetched_at, chain, now)
    view = snap.chain_view(chain)
    rows = render_day_filter(view.ledger, chain, paper)
    render_alerts(view, rows)
    # on_change="rerun" -> .open บอกแท็บที่เปิดอยู่ จึงรันเฉพาะแท็บนั้น (สลับแท็บ = rerun แค่ fragment นี้)
    tab_data, tab_chart = st.tabs(["ข้อมูล", "กราฟ"], key="live_view", on_change="rerun")
    if tab_data.open:
        with tab_data:
            render_data_tab(snap, view, rows, paper)
    if tab_chart.open:
        with tab_chart:
            render_chart_tab(snap, view, rows, paper)


@st.fragment
def live_panel(snap: Snapshot, paper: bool) -> None:
    """เนื้อหาแท็บ Live — widget ข้างในทำให้ rerun แค่ส่วนนี้บน ``snap`` เดิม (ไม่อ่าน Firebase ซ้ำ)"""
    try:
        _render_live(snap, paper, pd.Timestamp.now(tz="UTC"))
    except Exception as exc:       # ไม่ขึ้น traceback และไม่พาอีก 3 แท็บหาย (Chain ยังสลับได้จากด้านบน)
        st.error(f"แสดงหน้า Live ไม่ได้ ({type(exc).__name__}: {exc})")


def render_live_dashboard(paper: bool = False) -> None:
    # แท็บนี้ต้องมี Firebase — ไม่มี secrets ก็แค่แจ้ง แล้วไม่ล้มทั้งหน้า (แท็บ 101 ยังใช้ได้)
    if not _secrets_ready():
        st.info("ตั้ง st.secrets: FIREBASE_SA_JSON และ FIREBASE_DB_URL หรือ FIREBASE_DB_URL_PAPER ก่อน จึงจะเห็น "
                "ข้อมูลสด — ระหว่างนี้เปิดแท็บ 🎓 Rebalancing 101 เพื่อลอง gated demo ได้เลย")
        return
    try:
        _init()
        with st.spinner("กำลังโหลดข้อมูล"):
            # อ่านที่นี่ที่เดียว (รันเต็มสคริปต์) — การกด widget ในแท็บนี้ rerun แค่ live_panel
            snap = fetch_snapshot(lambda path: _ref(path, paper).get())
    except Exception as exc:
        now = pd.Timestamp.now(tz="UTC")
        render_header("Live", Freshness(ERROR, None, None, "โหลดข้อมูลไม่ได้"), now, None, now)
        detail = str(exc) if isinstance(exc, SnapshotError) else f"{type(exc).__name__}: {exc}"
        st.error(f"โหลดข้อมูลไม่ได้ — {detail} · กด รีเฟรช เพื่อลองใหม่")
        return
    live_panel(snap, paper)


def render_rebalancing_101() -> None:
    st.subheader("🧬 Gated demo — DNA gate + บัญชีแบบแช่แข็ง (gated_theoretical_v2)")
    st.caption(
        "แบบจำลองเชิงสอนของ historical semantics `gated_theoretical_v2`: ใน Testing Lab "
        "ที่ไม่มี threshold นี้ gate 1 ถือเป็น act และ gate 0 แช่แข็ง · Live Dashboard ปัจจุบัน "
        "รองรับ `execution_confirmed_v1` ซึ่ง "
        "ขยับ Aₙ เฉพาะ broker-confirmed fill; Testing Lab นี้ยังคง input/output เดิม"
    )
    st.markdown(
        "- **act (READY_BUY/READY_SELL, จำนวนสั่ง > 0 — เทรดจริง):** ΔAₙ = Fix_c × "
        "(Pₙ/P_acted − 1), เลื่อน P_acted = Pₙ, Eₙ = Aₙ − Rₙ\n"
        "- **pass (ไม่เทรด — จำนวนสั่ง = 0):** ΔAₙ = 0, holdings แช่แข็งตั้งแต่ act ล่าสุด, "
        "Aₙ ค้าง, P_acted แช่แข็ง, Eₙ smooth (ค้างค่า act ล่าสุด) — **รวมทั้ง PASS_DNA_ZERO "
        "(signal = 0) และ PASS_THRESHOLD (signal = 1 แต่ |gap| ≤ DIFF)**: ทั้งสองไม่ยิง order "
        "จึงแช่แข็ง ledger เหมือนกัน (ΔAₙ = 0) — Eₙ ไม่ลดและ ≥ 0 เสมอ (x − 1 ≥ ln x)"
    )

    pc = st.columns(2)
    with pc[0]:
        st.markdown("#### 1) เส้นอ้างอิงทางทฤษฎี")
        st.code("Rₙ = Fix_c × ln(Pₙ / P₀)", language=None)
        st.caption("กระแสเงินสดอ้างอิงของการรักษามูลค่าคงที่แบบต่อเนื่อง (act ทุกรอบ)")
    with pc[1]:
        st.markdown("#### 2) เส้น gated จริง (แช่แข็งช่วง pass)")
        st.code("act:  ΔAₙ = Fix_c × (Pₙ/P_acted − 1),  Eₙ = Aₙ − Rₙ\n"
                "pass: ΔAₙ = 0 (แช่แข็ง),  Eₙ = Aₙ − Fix_c × ln(P_acted/P₀)",
                language=None)
        st.caption("P_acted = ราคารอบ act ล่าสุด — ค้างไว้ตลอดช่วง pass")

    st.markdown("#### Testing Lab — สุ่มราคา แล้วเดิน ledger ตาม DNA gate")
    if "dash_gated_seed" not in st.session_state:
        st.session_state.dash_gated_seed = 101

    r1 = st.columns(3)
    fix_c = r1[0].number_input("Fix_c (มูลค่าเป้าหมาย)", min_value=0.01,
                               value=1500.0, step=100.0, format="%.2f")
    p0 = r1[1].number_input("ราคาเริ่มต้น P₀", min_value=0.01, value=100.0,
                            format="%.5f")
    vol = r1[2].number_input("ความผันผวน/รอบ (%)", min_value=0.0, max_value=40.0,
                             value=4.0, step=0.1)
    r2 = st.columns(3)
    drift = r2[0].number_input("แนวโน้ม/รอบ (%)", min_value=-10.0, max_value=10.0,
                               value=0.0, step=0.01)
    steps = r2[1].number_input("จำนวนรอบ", min_value=2,
                               max_value=MAX_REBALANCING_STEPS, value=100, step=1)
    seed = r2[2].number_input("Seed", min_value=0, step=1, key="dash_gated_seed")
    dna_code = st.text_input(
        "DNA code (gate 0/1 ต่อรอบ — decode เดียวกับ engine: bypass:N / [1,N] / stream)",
        value=DEFAULT_GATED_DNA, key="dash_gated_dna")

    try:
        prices = simulate_rebalancing_prices(float(p0), float(vol) / 100.0,
                                             float(drift) / 100.0, int(steps),
                                             int(seed))
        actions = build_gate_actions(dna_code, len(prices))
        gated_rows = gated_rebalancing_cashflow_from_prices(
            prices, float(fix_c), float(p0), actions)
        pass_rows = rebalancing_cashflow_from_prices(prices, float(fix_c), float(p0))
        g_final, p_final = gated_rows[-1], pass_rows[-1]

        n_pass = len(actions) - sum(actions)
        m = st.columns(4)
        # เดโมนี้ไม่มีชั้น threshold — DNA gate จึงเท่ากับการตัดสินใจเทรด (act=เทรด, pass=แช่แข็ง)
        m[0].metric("act (เทรด) / pass (แช่แข็ง)", f"{sum(actions)} / {n_pass}")
        m[1].metric("Gated Aₙ", f"{g_final['actual_cumulative']:+,.2f}")
        m[2].metric("Gated Eₙ (smooth)", f"{g_final['excess']:+,.2f}")
        m[3].metric("Δ เทียบ pass-all Aₙ",
                    f"{g_final['actual_cumulative'] - p_final['actual_cumulative']:+,.2f}")

        st.markdown("#### กราฟ — Rₙ (อ้างอิง) vs Aₙ (gated) vs Eₙ (smooth)")
        st.plotly_chart(
            ledger_figure([row["step"] for row in gated_rows],
                          [row["ln_reference"] for row in gated_rows],
                          [row["actual_cumulative"] for row in gated_rows],
                          [row["excess"] for row in gated_rows],
                          acted=[row["action"] == 1 for row in gated_rows],
                          x_title="รอบ", theme=_streamlit_theme()),
            theme=None, config=PLOTLY_CONFIG, key="demo_ledger_chart")
        st.caption("Aₙ = gated · Eₙ = smooth · จุดบนเส้น Aₙ = รอบ act (gate 1) — "
                   f"แสดงจุดเมื่อมี act ไม่เกิน {MAX_ACT_MARKERS} รอบ")

        st.markdown("#### ตาราง ledger (gated)")
        gated_frame = pd.DataFrame(gated_rows)
        st.dataframe(
            gated_frame.rename(columns={
                "step": "รอบ", "action": "gate", "price": "ราคา Pᵢ",
                "acted_price": "P_acted (แช่แข็ง)", "delta_actual": "ΔA รอบนี้",
                "actual_cumulative": "Aₙ", "ln_reference": "Rₙ",
                "excess": "Eₙ (smooth)",
            }),
            width="stretch", hide_index=True,
        )
        st.download_button("ดาวน์โหลด CSV (gated)",
                           data=gated_frame.to_csv(index=False),
                           file_name="rebalancing_gated_demo.csv",
                           mime="text/csv")
        st.info(
            "ข้อสังเกต: gated ไม่ได้ดีกว่า pass-all เสมอ — ผลต่างมาจาก cross-terms "
            "ของช่วงที่ข้าม (ตลาด trend การข้ามชนะ, ตลาด mean-revert การ act ทุกรอบชนะ) "
            "DNA gate จึงต้องมาจากการคัดเลือกด้วย backtest/GA ไม่ใช่ข้ามมั่ว"
        )
        st.caption("แบบจำลองเพื่อการเรียนรู้ · ยังไม่รวม spread, slippage และภาษี")
    except (DNAError, ValueError) as exc:
        st.error(f"คำนวณไม่ได้: {exc}")


_HTML_OPEN_TAG = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_DATA_THEME_ATTR = re.compile(r"""\sdata-theme\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""",
                              re.IGNORECASE)


def apply_iframe_theme(app_html: str, theme: str | None) -> str:
    """บังคับสีของ web app ให้ตรงกับธีม Streamlit — หน้า HTML รองรับ ``:root[data-theme]``

    ``theme`` ไม่ใช่ "light"/"dark" (ไม่รู้ธีม) -> คืนเดิม ให้ตาม prefers-color-scheme
    ของเบราว์เซอร์. ไฟล์ที่เป็น fragment (ไม่มี ``<html>``) จะถูกครอบด้วย
    ``<html data-theme=...>`` อย่างเดียว ไม่เติม doctype: ต้นฉบับไม่มี และ iframe
    srcdoc ไม่เข้าโหมด quirks เพราะไม่มี doctype อยู่แล้ว เนื้อหาจึงเหมือนเดิมทุกไบต์
    """
    if theme not in ("light", "dark"):
        return app_html
    attr = f' data-theme="{theme}"'
    match = _HTML_OPEN_TAG.search(app_html)
    if match is None:
        return f"<html{attr}>{app_html}"
    tag = _DATA_THEME_ATTR.sub("", match.group(0))
    return app_html[:match.start()] + tag[:-1] + attr + ">" + app_html[match.end():]


def _streamlit_theme() -> str | None:
    try:
        return getattr(st.context.theme, "type", None)
    except Exception:        # ธีมเป็นส่วนเสริม — ห้ามทำให้หน้าล่ม
        return None


def render_web_app(filename: str, *, height: int = 900) -> None:
    """Render a bundled, self-contained HTML app in an isolated iframe."""
    try:
        app_html = (WEB_APPS_DIR / filename).read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"{exc.__class__.__name__}: {exc}")
        return
    components.html(apply_iframe_theme(app_html, _streamlit_theme()),
                    height=height, scrolling=True)


def render_web_apps() -> None:
    st.subheader("Interactive Rebalancing Web Apps")
    st.caption(
        "เปิดคู่มือและ playground แบบโต้ตอบได้จากแดชบอร์ดนี้ "
        "โดยทำงานอยู่ใน iframe แยก ไม่เรียก Firebase และไม่อ่าน secrets"
    )
    selected_web_app = st.radio(
        "เลือก Web App",
        options=tuple(WEB_APPS),
        horizontal=True,
        label_visibility="collapsed",
        key="web_app_choice",
    )
    render_web_app(WEB_APPS[selected_web_app])


def mmd_digest(source: str) -> str:
    """8 hex ตัวแรกของ SHA-256 ของข้อความ .mmd — อ่านแบบ text จึงไม่ขึ้นกับ CRLF/LF
    (tools/render_cheat_sheet.py ใช้กติกาเดียวกัน)"""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]


def render_cheat_sheet(stem: str) -> None:
    """แสดงแผนภาพ ``cheat_sheet/<stem>.mmd`` (ภาพ PNG ที่ render ไว้ + ซอร์ส + ปุ่มดาวน์โหลด)"""
    try:
        source = (CHEAT_SHEET_DIR / f"{stem}.mmd").read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"{exc.__class__.__name__}: {exc}")
        return
    image = CHEAT_SHEET_DIR / f"{stem}.{mmd_digest(source)}.png"
    if image.is_file():
        # output_format="PNG": ค่าเริ่มต้น "auto" ส่งภาพที่ไม่มี alpha เป็น JPEG แบบสูญเสีย ตัวอักษรในแผนภาพจะมัว
        st.image(str(image), output_format="PNG")   # render ที่ scale 3 — เบราว์เซอร์ย่อให้พอดีคอลัมน์ จึงคมบนจอ retina
    else:
        st.warning("ภาพของแผนภาพนี้ยังไม่ตรงกับไฟล์ .mmd ล่าสุด — "
                   "รัน `python tools/render_cheat_sheet.py` แล้ว commit ภาพใหม่")
    with st.expander("ซอร์ส .mmd — คัดลอกไปวางใน Mermaid ได้"):
        st.code(source, language=None)
    st.download_button("ดาวน์โหลด .mmd", data=source, file_name=f"{stem}.mmd",
                       mime="text/plain", key=f"cheat_sheet_{stem}")


def render_cheat_summary() -> None:
    """แสดง ``cheat_sheet/summary.md`` (สรุป 10 ข้อ + ประโยคเดียว) — ข้อความอยู่ในไฟล์ ไม่ได้อยู่ในโค้ด"""
    try:
        text = (CHEAT_SHEET_DIR / CHEAT_SUMMARY).read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"{exc.__class__.__name__}: {exc}")
        return
    st.markdown(text)


def render_cheat_sheets() -> None:
    st.subheader("⚡ Cheat Sheet — รู้ทันขั้นตอนในหน้าเดียว")
    tab_summary, tab_diagrams = st.tabs(["📝 Summary", "🔀 Diagrams"])
    with tab_summary:
        render_cheat_summary()
    with tab_diagrams:
        st.caption(
            "แผนภาพ Mermaid (.mmd) ของ main rebalancing, rollover และ Option Scale — แสดงเป็นภาพที่ render ไว้ล่วงหน้า "
            "จึงเปิดได้ทันทีและใช้ออฟไลน์ได้ · ซอร์ส .mmd เป็น Mermaid มาตรฐาน คัดลอกไปใช้ต่อได้"
        )
        for column, (title, stem) in zip(st.columns(len(CHEAT_SHEETS)), CHEAT_SHEETS.items()):
            with column:
                st.markdown(f"#### {title}")
                render_cheat_sheet(stem)


PROD_DB_INSTANCE = "lego-firebase-prod"


def render_prod_banner(paper: bool = False) -> None:
    """ป้ายบอกว่าหน้านี้คือ PROD และเตือนถ้า secrets ชี้ไป DB อื่น (เตือนเท่านั้น ไม่บล็อก)"""
    # markdown ไม่ใช่ st.warning: เทสต์เดิมใช้ "ไม่มี warning" เป็นสัญญาณว่าหน้าปกติ
    if paper:
        st.markdown(":orange-background[**PAPER**] จำนวนหุ้นและ fill ในหน้านี้เป็นการจำลอง "
                    "ไม่ใช่เงินจริง: Preview จริง แต่ไม่เคย Place (อ่านจาก FIREBASE_DB_URL_PAPER)")
        if "paper" not in str(_paper_url()):
            st.error("FIREBASE_DB_URL_PAPER ไม่ได้ชี้ไป database ของ paper — ตรวจ st.secrets")
        return
    st.markdown(f":red-background[**PROD**] อ่านข้อมูลบัญชีจริงของ lego-tick-prod "
                f"จาก {PROD_DB_INSTANCE} (read-only)")
    if _secrets_ready() and PROD_DB_INSTANCE not in str(_prod_url()):
        st.error(f"FIREBASE_DB_URL ไม่ได้ชี้ไป {PROD_DB_INSTANCE} อาจเป็น DB ของ UAT "
                 "ตรวจ st.secrets ก่อนเชื่อตัวเลขในหน้านี้")


def main():
    st.set_page_config(page_title="LEGO PROD", layout="wide")
    st.title("🧬 LEGO Shannon Demon: PROD")
    # ตั้งทั้งคู่ = มีตัวเลือก; ตั้ง FIREBASE_DB_URL_PAPER อย่างเดียว = paper อย่างเดียว
    # ไม่ตั้ง FIREBASE_DB_URL_PAPER = PROD อย่างเดียวเหมือนเดิม
    if _paper_url() is None:
        paper = False
    elif _prod_url() is None:
        paper = True
    else:
        paper = st.sidebar.radio("ฐานข้อมูล", [DATABASE_PROD, DATABASE_PAPER],
                                 key="database") == DATABASE_PAPER
    render_prod_banner(paper)

    tab_live, tab_101, tab_web, tab_cheat = st.tabs(
        ["📊 Live Dashboard", "🎓 Rebalancing 101", "🌐 Web Apps", "⚡ Cheat Sheet"])
    with tab_live:
        render_live_dashboard(paper)
    with tab_101:
        render_rebalancing_101()
    with tab_web:
        render_web_apps()
    with tab_cheat:
        render_cheat_sheets()


if __name__ == "__main__":
    main()
