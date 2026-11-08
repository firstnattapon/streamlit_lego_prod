"""lego_live_data.py — ชั้นข้อมูลของแท็บ Live Dashboard (pure: ไม่ import streamlit / firebase_admin)

หลักการ: อ่าน Firebase 4 node **ครั้งเดียวต่อการรันเต็มสคริปต์** เป็น ``Snapshot`` แล้วหน้าเว็บส่ง snapshot
เข้า ``st.fragment`` — สลับ Chain / แท็บ / วัน / สวิตช์ = rerun เฉพาะ fragment บน snapshot เดิม
ไม่อ่าน Firebase ซ้ำและไม่คำนวณซ้ำ (ผลคำนวณถูก memo ไว้บนตัว snapshot และหายไปพร้อมกัน ไม่มีแคช global)

สูตรทั้งหมดอยู่ใน lego_dash_core และถูกเรียกแบบ ``core.fn(...)`` ตอนใช้งาน (ไม่ผูกชื่อไว้ตอน import)
reload guard ของ streamlit_app.py และ monkeypatch ในเทสต์จึงยังมีผล — โมดูลนี้แค่จัดลำดับการเรียก จำผล
และบอกว่าข้อมูล "สด" หรือไม่
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

import lego_dash_core as core

ROWS_PATH = "webull_lego_rows"
STATE_PATH = "webull_lego_state"
AUDIT_PATH = "webull_lego_order_audit"
WARNINGS_PATH = "webull_lego_warnings"
# อ่านไม่ได้ = ไม่มีอะไรให้แสดง (SnapshotError) · node อื่นอ่านไม่ได้ = แสดงที่เหลือพร้อมคำเตือน
REQUIRED_PATHS = (ROWS_PATH, STATE_PATH)
READ_ORDER = (WARNINGS_PATH, STATE_PATH, ROWS_PATH, AUDIT_PATH)

TIME_COL = core.COLUMN_ORDER[0]                  # "เวลา (UTC)"
EXCESS_COL = core.LEDGER_COLS[3]                 # "Eₙ ส่วนเกินสะสม (USD)"

LIVE, STALE, ERROR = "live", "stale", "error"
STATUS_LABEL = {LIVE: "สด", STALE: "ค้าง", ERROR: "ผิดพลาด"}
SLOT_FALLBACK_S = 900.0        # 15 นาที: slot ของ fixture (26 slot/session 13:30–19:45 UTC)
SLOT_BOUNDS_S = (60.0, 3600.0)
STALE_AFTER_SLOTS = 2          # ไม่มีแถวใหม่เกิน 2 slot = ค้าง
US_MARKET = ZoneInfo("America/New_York")
US_SESSION = (time(9, 30), time(16, 0))


class SnapshotError(RuntimeError):
    """อ่าน node ที่จำเป็น (rows / state) ไม่ได้ — หน้าแสดงข้อผิดพลาดและปุ่มลองใหม่แทน"""


def _utc(now: pd.Timestamp | None) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")


def _as_dict(value) -> dict:
    """state / audit ที่ไม่ใช่ dict (None, list, ข้อความ) -> {} — กัน ``.values()`` / ``.get`` ล่ม"""
    return value if isinstance(value, dict) else {}


def _stamp(value) -> pd.Timestamp | None:
    """เวลา ISO (มี/ไม่มี offset) -> Timestamp UTC; อ่านไม่ได้ -> None (กติกาเดียวกับ core.utc_days)"""
    if not isinstance(value, str):
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce", format="mixed")
    return None if pd.isna(stamp) else stamp


def fetch_snapshot(read: Callable[[str], Any], now: pd.Timestamp | None = None) -> Snapshot:
    """อ่าน 4 node ผ่าน ``read(path)`` (หน้าเว็บส่ง ``lambda p: _ref(p, paper).get()``) เป็น Snapshot"""
    raw: dict[str, Any] = {}
    notes: list[str] = []
    for path in READ_ORDER:
        try:
            raw[path] = read(path)
        except Exception as exc:
            reason = f"อ่าน {path} ไม่ได้ ({type(exc).__name__}: {exc})"
            if path in REQUIRED_PATHS:
                raise SnapshotError(reason) from exc
            raw[path] = None
            notes.append(reason + " — ส่วนที่ใช้ข้อมูลนี้อาจไม่ครบ")
    try:
        rows = core.rows_to_df(raw[ROWS_PATH])
    except Exception as exc:
        raise SnapshotError(f"แปลง {ROWS_PATH} ไม่ได้ ({type(exc).__name__}: {exc})") from exc
    return Snapshot(rows=rows, state=_as_dict(raw[STATE_PATH]), audit=_as_dict(raw[AUDIT_PATH]),
                    health=core.system_health_rows(raw[WARNINGS_PATH]), fetched_at=_utc(now),
                    notes=notes)


class _Memo:
    """memo ต่ออ็อบเจกต์ — script run ของ session เดียวไม่ซ้อนกัน จึงไม่ต้องมีล็อก"""
    _memo: dict

    def memo(self, key, compute: Callable[[], Any]):
        """คำนวณครั้งแรกแล้วจำไว้ — exception ไม่ถูกจำ (ครั้งหน้าลองใหม่)"""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]


@dataclass(eq=False)
class Snapshot(_Memo):
    """ข้อมูลหนึ่งรอบการอ่าน + ผลคำนวณที่ memo ไว้ (ตายพร้อม snapshot)"""
    rows: pd.DataFrame
    state: dict
    audit: dict
    health: pd.DataFrame
    fetched_at: pd.Timestamp
    notes: list[str] = field(default_factory=list)
    _memo: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def alarm(self) -> bool:
        """System health มีรายการ ACTIVE (ระบบพักเพราะ Authentication)"""
        return "state" in self.health.columns and bool(self.health["state"].eq("ACTIVE").any())

    def chains(self) -> list:
        """chain ทั้งหมด เรียงตามชื่อ (ตำแหน่งปุ่มไม่เลื่อนเมื่อข้อมูลอัปเดต)"""
        if "chain_key" not in self.rows.columns:
            return []
        return self.memo("chains", lambda: sorted(self.rows["chain_key"].dropna().unique()))

    def default_chain(self):
        """chain ที่ active ล่าสุดตาม state.updated_at (กติกาเดิมของ core.default_chain_index)"""
        chains = self.chains()
        return chains[core.default_chain_index(chains, self.state)] if chains else None

    def entry(self, chain) -> dict:
        """state pointer ของ chain — ไม่มี/ผิดรูป -> {}"""
        return _as_dict(self.state.get(chain)) if chain is not None else {}

    def symbol(self, chain) -> str:
        """หุ้นของ chain: "สินทรัพย์" แถวล่าสุด -> state.symbol -> ชื่อ chain"""
        rows = self.chain_rows(chain)
        last = rows["สินทรัพย์"].iloc[-1] if "สินทรัพย์" in rows.columns and len(rows) else None
        for value in (last, self.entry(chain).get("symbol")):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(chain) if chain is not None else ""

    def chain_rows(self, chain) -> pd.DataFrame:
        """แถวของ chain แบบที่หน้าเดิมแบ่ง: หลาย chain กรองด้วย chain_key, chain เดียว/ไม่มี key ใช้ทั้ง frame"""
        if len(self.chains()) > 1:
            return self.rows[self.rows["chain_key"] == chain]
        return self.rows

    def chain_view(self, chain) -> ChainView:
        return self.memo(("chain", chain), lambda: ChainView.build(self, chain))

    def freshness(self, chain, now: pd.Timestamp | None = None) -> Freshness:
        return freshness(self.chain_rows(chain), self.entry(chain), _utc(now), alarm=self.alarm)


@dataclass(eq=False)
class ChainView(_Memo):
    """ledger ของ chain ที่เลือก — คำนวณตอนสร้างเฉพาะส่วนที่ทุกมุมมองใช้; ส่วนหนักคำนวณเมื่อถูกเรียก"""
    key: Any
    persisted: pd.DataFrame          # ค่าที่ DB บันทึก (tooltip + Integrity ตรวจจากชุดนี้)
    ledger: pd.DataFrame             # recompute(..., genesis_baseline=True) — ตัวเลขที่หน้าแสดง
    p0_hint: float | None
    n_corrections: int
    execution_mode: bool
    frozen_v2_mode: bool
    paper_marked: bool
    leaps: dict | None
    notes: list[str]
    audit: dict
    _memo: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def build(cls, snap: Snapshot, chain) -> ChainView:
        rows = snap.chain_rows(chain)
        notes: list[str] = []
        try:
            p0 = core.chain_p0_hint(snap.state, chain)
        except (TypeError, ValueError) as exc:
            p0 = None
            notes.append(f"อ่าน p0 ของ chain นี้ใน state pointer ไม่ได้ ({exc}) — ใช้ราคาแถว genesis แทน")
        semantics = (rows["semantics"].astype(str) if "semantics" in rows.columns
                     else pd.Series(dtype=str))
        frozen = tuple(getattr(core, "FROZEN_TERMINAL_SEMANTICS",
                               (core.EXECUTION_TERMINAL_FROZEN_V2,)))
        persisted = rows.copy(deep=True)
        # dashboard เป็น read-only: derive recurrence ใหม่ตาม semantics ก่อนแสดงเสมอ
        # (v2 ใช้ decision, v3 ใช้เฉพาะ broker fill ที่ FINALIZED) แล้วนับแถวที่ engine เขียนไม่ตรง
        fixed = core.recompute_gated_ledger(rows, p0=p0)
        return cls(
            key=chain, persisted=persisted,
            # n = 0 เป็นจุดตั้งต้น (R₀ = ΔA₀ = A₀ = E₀ = 0) — "แสดง" เท่านั้น ไม่แก้ DB
            ledger=core.recompute_gated_ledger(persisted, p0=p0, genesis_baseline=True),
            p0_hint=p0, n_corrections=core.count_ledger_corrections(rows, fixed),
            execution_mode=bool(semantics.isin([core.EXECUTION_CONFIRMED_SEMANTICS,
                                                *frozen]).any()),
            frozen_v2_mode=bool(semantics.isin(frozen).any()),
            # ตีตราจาก backend (LEGO_MODE=paper) — ขึ้นแม้ chain paper จะไปอยู่ผิด database
            paper_marked=bool(chain) and bool(snap.entry(chain).get("paper")),
            leaps=core.leaps_summary(persisted), notes=notes, audit=snap.audit)

    def tooltips(self) -> tuple[list[dict], list[str]]:
        """ที่มาของ Rₙ/ΔAₙ/Aₙ/Eₙ ทุกแถวของ chain (≈0.5 วิ/1,000 แถว) — genesis_baseline ตรงกับ ledger"""
        return self.memo("tips", lambda: core.table_tooltips(
            self.persisted, p0=self.p0_hint, genesis_baseline=True))

    def integrity(self) -> tuple[pd.DataFrame, bool]:
        """Integrity E1–E8 บนค่าที่ DB บันทึก — งานหนัก (O(n²)) หน้าเรียกเมื่อเปิด expander เท่านั้น"""
        return self.memo("integrity", lambda: core.integrity_report(
            self.persisted, p0_hint=self.p0_hint))

    def audit_rows(self) -> pd.DataFrame:
        """order audit ของทั้ง chain — ผูกด้วย run_id กัน order ข้าม chain ปนตาราง"""
        return self.memo("audit", lambda: core.filter_audit_rows(
            self.audit, self.ledger["run_id"] if "run_id" in self.ledger.columns else []))


# ---- สด / ค้าง / ผิดพลาด ------------------------------------------------------

@dataclass(frozen=True)
class Freshness:
    status: str                          # LIVE / STALE / ERROR
    last_update: pd.Timestamp | None     # เวลาที่ chain ขยับล่าสุด (UTC)
    age_s: float | None
    reason: str = ""

    @property
    def label(self) -> str:
        return STATUS_LABEL[self.status]


def _row_times(rows: pd.DataFrame, tail: int = 32) -> list[pd.Timestamp]:
    if TIME_COL not in rows.columns:
        return []
    return [s for s in (_stamp(v) for v in rows[TIME_COL].iloc[-tail:]) if s is not None]


def slot_seconds(rows: pd.DataFrame) -> float:
    """ความยาว slot ของ chain จากข้อมูลเอง: median ของระยะห่างแถวท้าย ๆ (ข้ามคืนเป็น outlier ที่ median ตัดทิ้ง)"""
    times = _row_times(rows)
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return SLOT_FALLBACK_S
    lo, hi = SLOT_BOUNDS_S
    return min(max(float(pd.Series(gaps).median()), lo), hi)


def market_open(now: pd.Timestamp) -> bool:
    """ตลาด US เปิดอยู่ไหม (จ–ศ 09:30–16:00 America/New_York) — ไม่มีปฏิทินวันหยุด จึงเป็นการประมาณ"""
    local = _utc(now).tz_convert(US_MARKET)
    return local.weekday() < 5 and US_SESSION[0] <= local.time() < US_SESSION[1]


def freshness(rows: pd.DataFrame, entry: dict, now: pd.Timestamp, *,
              alarm: bool = False) -> Freshness:
    """สถานะของ chain: สด = มีแถว/updated_at ภายใน 2 slot · ค้าง = นานกว่านั้น · ผิดพลาด = ระบบพัก"""
    now = _utc(now)
    times = _row_times(rows, tail=1)
    stamps = [s for s in (_stamp(_as_dict(entry).get("updated_at")), *times) if s is not None]
    last = max(stamps) if stamps else None
    age = None if last is None else max(0.0, (now - last).total_seconds())
    if alarm:
        return Freshness(ERROR, last, age, "ระบบพัก (Authentication)")
    if last is None:
        return Freshness(STALE, None, None, "ไม่มีเวลาอัปเดต")
    if age <= STALE_AFTER_SLOTS * slot_seconds(rows):
        return Freshness(LIVE, last, age)
    return Freshness(STALE, last, age, "ตลาดปิด" if not market_open(now) else "ไม่มีแถวใหม่")


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return "เมื่อครู่"
    for unit, size in (("วัน", 86400), ("ชม.", 3600), ("นาที", 60)):
        if seconds >= size:
            return f"{int(seconds // size):,} {unit}ที่แล้ว"
    return ""


def describe(fresh: Freshness, now: pd.Timestamp) -> str:
    """"อัปเดต 14:05 UTC · 3 นาทีที่แล้ว" (วันอื่นแสดงวันที่ด้วย) + เหตุผลถ้ามี"""
    parts = []
    if fresh.last_update is not None:
        same_day = fresh.last_update.date() == _utc(now).date()
        parts.append("อัปเดต " + fresh.last_update.strftime("%H:%M" if same_day else "%Y-%m-%d %H:%M")
                     + " UTC")
        parts.append(fmt_age(fresh.age_s))
    if fresh.reason:
        parts.append(fresh.reason)
    return " · ".join(p for p in parts if p)


# ---- ตัวเลขบนการ์ด ------------------------------------------------------------

def fmt_num(value, nd: int = 2) -> str:
    """ตัวเลข -> ข้อความ nd ตำแหน่ง; None / NaN / inf / อ่านไม่ได้ -> "—" (ไม่มี "-0.00")"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{round(number, nd) + 0.0:.{nd}f}"


def fmt_text(value) -> str:
    """ข้อความบนการ์ด; None / NaN / ว่าง -> "—" """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return str(value).strip() or "—"


def state_table(entry: dict) -> pd.DataFrame:
    """state pointer ของ chain เป็นตาราง ฟิลด์ -> ค่า (ค่าซ้อนแสดงเป็น JSON; ทุกค่าเป็นข้อความ)"""
    def cell(value) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return fmt_text(value)

    table = pd.DataFrame({"ค่า": {str(k): cell(v) for k, v in sorted(_as_dict(entry).items(),
                                                                       key=lambda kv: str(kv[0]))}})
    table.index.name = "ฟิลด์"
    return table
