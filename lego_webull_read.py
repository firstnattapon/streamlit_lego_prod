"""lego_webull_read.py — อ่าน position + quote จาก Webull OpenAPI แบบอ่านอย่างเดียว (ไม่ import streamlit)

คำสัญญา (ตรวจจากซอร์ส webull-openapi-python-sdk 3.0.4 และพิสูจน์ใน test_positions.py):
  R1 อ่านอย่างเดียว   ทุก request ผ่าน ``ReadOnlyGuard`` ที่ให้ผ่านเฉพาะ GET 4 path ใน ``READ_ONLY_ALLOWLIST``
                      endpoint ที่เปลี่ยนสถานะของ Webull (order place/preview/replace/cancel, token create/
                      refresh/check) เป็น POST ทั้งหมด จึงผ่านด่านนี้ไม่ได้ และถูกปฏิเสธก่อนออก network
  R2 ไม่ส่ง order     ไม่ import TradeClient หรือโมดูล order ใด ๆ — ใช้ AccountV2 / MarketData / OptionMarketData
                      ตรง ๆ (constructor แค่เก็บ client)
  R3 ไม่สร้าง token    ไม่สร้าง TradeClient / DataClient / QuotesClient เพราะ constructor ทั้งสามเรียก
                      ClientInitializer → TokenManager.init_token → create_token และเขียน conf/token.txt
                      (อาจต้องยืนยัน 2FA และกระทบ token ที่ bot PROD ใช้) — ที่นี่แค่ ``set_token`` เป็น header
  R4 ไม่เขียนไฟล์     ApiClient ไม่เขียนไฟล์ (มีแค่ set_file_logger ที่ไม่เรียก) และไม่ผ่าน TokenStorage
  R5 ไม่หลุดลง log    SDK log ``vars(request)`` (มี x-access-token / x-app-key / x-signature) ตอน Webull ตอบ
                      non-2xx → ปิด logger "webull" ไม่ให้ไหลขึ้น root · ข้อความ error ทุกอันผ่าน ``redact``
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import lego_positions

READ_ONLY_ALLOWLIST = frozenset({
    ("GET", "/trading/accounts/list"),
    ("GET", "/trading/assets/positions/list"),
    ("GET", "/market-data/stocks/snapshots/list"),
    ("GET", "/market-data/options/snapshots/list"),
})
STOCK_BATCH, OPTION_BATCH = 100, 20             # ข้อจำกัดจำนวน symbol ต่อ request ของ Webull
# 2 = positions แบบ Webull OpenAPI v3 (option อยู่ใน legs ไม่มี side) ขอราคาหุ้นด้วยชื่อหุ้นอ้างอิง
# streamlit_app ใช้ค่านี้ตรวจว่าโมดูลในหน่วยความจำเป็นรุ่นเก่าหรือไม่ (hot deploy)
SNAPSHOT_SCHEMA = 2
CONNECT_TIMEOUT_S, READ_TIMEOUT_S = 5, 10
DEBUG_ENV = "WEBULL_API_DEBUG"                  # SDK พิมพ์ request/response ทั้งก้อน (มี token) เมื่อเปิด
PRIVATE_KEYS = ("account_id", "account_number", "user_id")
_SECRET_FIELD = re.compile(
    r"(?i)(x-access-token|x-app-key|x-signature|app_secret|app_key|access_token|token)"
    r"(['\"]?\s*[:=]\s*['\"]?)([^'\"\s,}\]]+)")


class WebullReadError(RuntimeError):
    """อ่าน Webull ไม่ได้ — ข้อความภาษาไทยพร้อมวิธีแก้ และผ่าน ``redact`` แล้ว"""


class ReadOnlyGuard:
    """ครอบ ApiClient — มีแค่ ``get_response`` (ไม่มี ``__getattr__`` เพื่อไม่ให้มีทางอ้อม)"""
    __slots__ = ("_client",)

    def __init__(self, client) -> None:
        self._client = client

    def get_response(self, request):
        key = (str(request.get_method()).upper(), str(request.get_action_name()))
        if key not in READ_ONLY_ALLOWLIST:
            raise WebullReadError(f"blocked: {key[0]} {key[1]} (แท็บนี้อ่านอย่างเดียว) — เป็นบั๊ก ต้องแก้โค้ด")
        return self._client.get_response(request)


def redact(text, secrets: Iterable[str] = ()) -> str:
    """แทนค่าลับ (key / secret / token ที่ยาว ≥ 6) และค่าหลังชื่อ field ลับ ด้วย ``***``"""
    out = str(text)
    for secret in sorted({str(s) for s in secrets if s and len(str(s)) >= 6}, key=len, reverse=True):
        out = out.replace(secret, "***")
    return _SECRET_FIELD.sub(lambda m: m.group(1) + m.group(2) + "***", out)


def silence_sdk_logs() -> None:
    """R5: logger ของ SDK ไม่ไหลขึ้น root (Streamlit Cloud เก็บ stderr เป็น log)"""
    log = logging.getLogger("webull")
    log.handlers[:] = [logging.NullHandler()]
    log.propagate = False
    log.setLevel(logging.CRITICAL)


def _load_sdk() -> SimpleNamespace:
    """import เฉพาะส่วนอ่าน — ห้าม import trade_client / data_client / quotes_client / token_* / order_*"""
    try:
        from webull.core.client import ApiClient
        from webull.core.exception.exceptions import ClientException, ServerException
        from webull.data.quotes.market_data import MarketData
        from webull.data.quotes.option_market_data import OptionMarketData
        from webull.trade.trade.v2.account_info_v2 import AccountV2
    except ImportError as exc:
        raise WebullReadError("ยังไม่ได้ติดตั้ง webull-openapi-python-sdk (ตรวจ requirements.txt)") from exc
    return SimpleNamespace(ApiClient=ApiClient, AccountV2=AccountV2, MarketData=MarketData,
                           OptionMarketData=OptionMarketData, ServerException=ServerException,
                           ClientException=ClientException)


def _records(data) -> list:
    """list ตรง ๆ หรือ list ที่ซ้อนใน object (data / items / list / positions)"""
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in ("data", "items", "list", "positions", "accounts"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _json(response, what: str) -> list:
    status = getattr(response, "status_code", None)
    if status != 200:
        raise WebullReadError(f"Webull ตอบ {status} ตอนอ่าน{what}")
    try:
        return _records(response.json())
    except ValueError as exc:
        raise WebullReadError(f"Webull ตอบข้อมูลที่ไม่ใช่ JSON ตอนอ่าน{what}") from exc


def _server_error(sdk, exc, what: str) -> WebullReadError:
    if isinstance(exc, sdk.ServerException):
        return WebullReadError(f"Webull ตอบ HTTP {exc.get_http_status()} ({exc.get_error_code()}) ตอนอ่าน{what}")
    return WebullReadError(f"เชื่อมต่อ Webull ไม่ได้ตอนอ่าน{what} ({exc.get_error_code()})")


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _strip_private(position: Mapping) -> dict:
    return {k: v for k, v in position.items() if k not in PRIVATE_KEYS}


def fetch_account_snapshot(app_key: str, app_secret: str, region_id: str, token_txt: str, *,
                           environment: str = "prod", now: datetime | None = None,
                           sdk: SimpleNamespace | None = None) -> dict:
    """positions + ราคาหุ้น + ราคา option ของทุกบัญชี (อ่านอย่างเดียว) · ล้ม → ``WebullReadError`` (redact แล้ว)"""
    secrets = [str(app_key or ""), str(app_secret or "")]
    try:
        if str(environment).strip().lower() != "prod":
            raise WebullReadError("แท็บนี้รองรับเฉพาะ WEBULL_ENVIRONMENT = prod")
        if os.environ.get(DEBUG_ENV):
            raise WebullReadError(f"ปิด {DEBUG_ENV} ก่อน (debug ของ SDK พิมพ์ header ที่มี token)")
        try:
            token, _expires_ms, _status = lego_positions.parse_token_txt(token_txt)
        except ValueError as exc:
            raise WebullReadError(str(exc)) from None
        secrets.append(token)
        return _fetch(app_key, app_secret, region_id, token, now or datetime.now(timezone.utc),
                      sdk or _load_sdk())
    except WebullReadError as exc:
        raise WebullReadError(redact(str(exc), secrets)) from None
    except Exception as exc:                     # อะไรก็ตามที่หลุดมา: ชนิด + ข้อความที่ redact แล้ว
        raise WebullReadError(redact(f"{type(exc).__name__}: {exc}", secrets)) from None


def _fetch(app_key, app_secret, region_id, token, now: datetime, sdk: SimpleNamespace) -> dict:
    silence_sdk_logs()
    raw = sdk.ApiClient(app_key, app_secret, region_id,       # ไม่ยิง network ไม่เขียนไฟล์
                        connect_timeout=CONNECT_TIMEOUT_S, timeout=READ_TIMEOUT_S)
    raw.set_token(token)                                     # ไปเป็น header x-access-token เท่านั้น
    client = ReadOnlyGuard(raw)
    account = sdk.AccountV2(client)
    try:
        accounts = _json(account.get_account_list(), "รายการบัญชี")
    except (sdk.ServerException, sdk.ClientException) as exc:
        if isinstance(exc, sdk.ServerException):
            raise WebullReadError(
                f"token ใช้ไม่ได้ (HTTP {exc.get_http_status()} {exc.get_error_code()}) — "
                "แท็บนี้ไม่สร้าง token ใหม่ (กัน bot PROD สะดุด) ให้นำ token.txt ล่าสุดของ bot "
                "มาใส่ WEBULL_TOKEN_TXT") from None
        raise _server_error(sdk, exc, "รายการบัญชี") from None
    ids = [str(a.get("account_id")) for a in accounts if isinstance(a, Mapping) and a.get("account_id")]
    if not ids:
        raise WebullReadError("Webull ไม่ส่งบัญชีมา (ไม่พบ field account_id)")
    positions: list[dict] = []
    for account_id in ids:
        try:
            records = _json(account.get_account_position(account_id), "positions")
        except (sdk.ServerException, sdk.ClientException) as exc:
            raise _server_error(sdk, exc, "positions") from None
        positions += [_strip_private(p) for p in records if isinstance(p, Mapping)]

    legs, problems = lego_positions.parse_positions(positions)
    if positions and not legs:
        raise WebullReadError("รูปแบบข้อมูล Webull เปลี่ยน: " + " · ".join(sorted(set(problems))))
    notes: list[str] = []
    symbols = sorted({leg.symbol for leg in legs})       # ชื่อหุ้น (option → หุ้นอ้างอิง ไม่ใช่ OCC)
    quotes: dict[str, dict] = {}
    try:                                         # สถานะตลาด/เวลา quote และ S₀ ของตัวที่มีแต่ option
        for batch in _chunks(symbols, STOCK_BATCH):
            for q in _json(sdk.MarketData(client).get_snapshot(",".join(batch), "US_STOCK"), "ราคาหุ้น"):
                if isinstance(q, Mapping) and q.get("symbol"):
                    quotes[str(q["symbol"]).upper()] = {
                        "price": q.get("price"), "trade_status": q.get("trade_status"),
                        "quote_time": q.get("quote_time")}
    except (sdk.ServerException, sdk.ClientException, WebullReadError) as exc:
        notes.append(f"ดึงราคาหุ้นไม่ได้ ({type(exc).__name__}) — ไม่มีสถานะตลาด")
    occs = lego_positions.option_contracts(positions)
    options: dict[str, dict] | None = {}
    try:                                         # ไม่บังคับ: ล้มแล้วแก้ IV จาก mark ของ positions แทน
        for batch in _chunks(occs, OPTION_BATCH):
            for q in _json(sdk.OptionMarketData(client).get_option_snapshot(",".join(batch), "US_OPTION"),
                           "ราคา option"):
                if isinstance(q, Mapping) and q.get("symbol"):
                    options[str(q["symbol"]).upper()] = {
                        k: q.get(k) for k in ("price", "bid", "ask", "imp_vol", "delta", "quote_time")}
    except (sdk.ServerException, sdk.ClientException, WebullReadError):
        options = None
    return {"as_of": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "positions": positions, "quotes": quotes, "options": options,
            "account_count": len(ids), "notes": notes}
