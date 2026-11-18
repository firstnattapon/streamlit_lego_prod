"""ชั้นข้อมูลของแท็บ 🎯 Payoff — position จริง + chain ของ bot (lego_positions) · Webull อ่านอย่างเดียว
(lego_webull_read) · หน้าเว็บ (AppTest: Webull / Firebase ปลอม)

ข้อมูลทั้งหมดเป็นของสมมติ (symbol AAA/BBB, ค่าลับขึ้นต้นด้วย FAKE_TEST_) — repo เป็น public ห้ามมีตัวเลขบัญชีจริง
HTTP ปลอม: patch ``Response.get_response_object`` (จุดเดียวที่ SDK ยิง requests) จึงใช้ request + signing ของ SDK จริง
"""
from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import io
import json
import logging
import math
import os
import pkgutil
from datetime import date, datetime, timezone
from pathlib import Path

import firebase_admin
import firebase_admin.db  # noqa: F401  (goal_app patch firebase_admin.db.reference — ต้องโหลดก่อน)
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import lego_positions as positions
import lego_webull_read as webull_read
from lego_webull_read import READ_ONLY_ALLOWLIST, ReadOnlyGuard, WebullReadError, redact

ROOT = Path(__file__).parent
APP_PATH = str(ROOT / "streamlit_app.py")
NOW = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
FAKE_KEY, FAKE_SECRET, FAKE_TOKEN = "FAKE_TEST_KEY_0001", "FAKE_TEST_SECRET_0001", "FAKE_TEST_TOKEN_000"
TOKEN_TXT = f"\n{FAKE_TOKEN}\n1790000000000\nNORMAL\n"
OCC = "AAA270521C00007000"

STOCK = {"currency": "USD", "quantity": "60.00000", "position_id": "P-1", "symbol": "AAA",
         "instrument_type": "EQUITY", "cost_price": "10.00", "last_price": "9.00",
         "unrealized_profit_loss": "-60.00"}
CALL = {"currency": "USD", "quantity": "1.0000000000", "position_id": "P-2", "symbol": "AAA",
        "option_strategy": "SINGLE", "instrument_type": "OPTION", "cost_price": "4.90",
        "last_price": "3.30", "unrealized_profit_loss": "-160.00",
        "items": [{"side": "BUY", "quantity": "1.0000000000", "symbol": "AAA", "cost_price": "4.90",
                   "option_type": "CALL", "option_expire_date": "2027-05-21",
                   "option_exercise_price": "7.0000000000",
                   "option_contract_multiplier": "100.0000000000"}]}
POSITIONS = [STOCK, CALL]
OPTION_QUOTES = {OCC: {"imp_vol": "0.80", "delta": "0.80", "bid": "3.10", "ask": "3.50"}}
STATE = {
    "AAA_fake01": {"symbol": "AAA", "updated_at": "2026-11-16T20:45:04Z", "p0": 10.0,
                   "prev_holdings": 60.0, "leaps_shares": 100.0, "last_run_id": "run_fake_1",
                   "execution_cashflow": {"actual_cumulative": -100.0, "excess": 20.0,
                                          "r_basis": -120.0, "last_action_price": 9.0}},
    "AAA_older": {"symbol": "AAA", "updated_at": "2026-10-01T15:00:00Z", "p0": 12.0,
                  "prev_holdings": 1.0, "leaps_shares": 0.0, "last_run_id": "run_fake_0",
                  "execution_cashflow": {"actual_cumulative": 1.0, "excess": 1.0,
                                         "r_basis": 0.0, "last_action_price": 12.0}},
}
ROWS = {"AAA_fake01": {positions.VALUE_COL: 1468.27, positions.GAP_COL: -31.73},
        "AAA_older": {positions.VALUE_COL: 900.0, positions.GAP_COL: -100.0}}


def legs() -> list[positions.Leg]:
    out, problems = positions.parse_positions(POSITIONS, OPTION_QUOTES)
    assert not problems
    return out


def chain() -> positions.LegoChain:
    chains, issues = positions.lego_chains(STATE, ROWS)
    assert not issues
    return chains["AAA"]


# ---------------------------------------------------------------- Black-Scholes

def test_put_call_parity_and_the_example_time():
    T = positions.years_to_expiry(date(2027, 5, 21), NOW)
    assert T == pytest.approx(0.50742, abs=1e-5)          # 16:00 New York (EDT) = 20:00 UTC
    c = positions.bs_price(9, 7, T, 0.04, 0.8, "call")
    p = positions.bs_price(9, 7, T, 0.04, 0.8, "put")
    assert c - p == pytest.approx(9 - 7 * math.exp(-0.04 * T), abs=1e-9)
    assert c - p == pytest.approx(2.140645, abs=1e-6)
    assert positions.bs_price(9, 7, T, 0.04, 0.8, "call") == pytest.approx(3.0405, abs=1e-4)


def test_no_time_or_no_volatility_means_intrinsic():
    for S in (5.0, 7.0, 9.5):
        assert positions.bs_price(S, 7, 0.0, 0.04, 0.8, "call") == max(0.0, S - 7)
        assert positions.bs_price(S, 7, 0.0, 0.04, 0.8, "put") == max(0.0, 7 - S)
        assert positions.bs_price(S, 7, 0.5, 0.04, 0.0, "call") == max(0.0, S - 7)
        assert positions.bs_price(S, 7, 0.5, 0.04, None, "put") == max(0.0, 7 - S)


def test_implied_vol_round_trips_and_refuses_impossible_prices():
    T = 0.5
    for kind in ("call", "put"):
        price = positions.bs_price(9, 7, T, 0.04, 0.8, kind)
        assert positions.implied_vol(price, 9, 7, T, 0.04, kind) == pytest.approx(0.8, abs=1e-4)
    assert positions.implied_vol(1.0, 9, 7, T, 0.04, "call") is None      # ต่ำกว่า intrinsic: ไม่เดา
    assert positions.implied_vol(None, 9, 7, T, 0.04, "call") is None
    assert positions.implied_vol(3.0, 9, 7, 0.0, 0.04, "call") is None


# ---------------------------------------------------------------- อ่าน positions / token

def test_positions_are_parsed_like_the_webull_payload():
    out = legs()
    stock, call = out
    assert (stock.symbol, stock.kind, stock.qty, stock.multiplier, stock.mark, stock.unrealized) == \
        ("AAA", "stock", 60.0, 1.0, 9.0, -60.0)
    assert (call.kind, call.qty, call.multiplier, call.strike, call.expiry, call.mark) == \
        ("call", 1.0, 100.0, 7.0, date(2027, 5, 21), 3.30)
    assert call.occ == OCC == positions.occ_symbol("AAA", date(2027, 5, 21), "call", 7.0)
    assert (call.iv, call.delta, call.iv_source) == (0.8, 0.8, "webull")
    assert positions.option_contracts(POSITIONS) == [OCC]


def test_bad_positions_are_skipped_with_field_names_only():
    broken = [{"instrument_type": "EQUITY", "symbol": "AAA", "cost_price": "123.45"},
              {"instrument_type": "OPTION", "symbol": "AAA"},
              {"instrument_type": "CRYPTO", "symbol": "BTC"}]
    out, problems = positions.parse_positions(broken)
    assert out == []
    assert problems[0] == "EQUITY: ไม่พบ field quantity, last_price, unrealized_profit_loss"
    assert problems[1] == "OPTION: ไม่พบ legs/items, unrealized_profit_loss (มี field: instrument_type, symbol)"
    assert "CRYPTO" in problems[2]
    assert not any("123.45" in p for p in problems)                  # ไม่มีค่าบัญชีในข้อความ


def test_sold_options_are_negative_and_a_combo_keeps_its_pl_on_the_first_leg():
    combo = dict(CALL, unrealized_profit_loss="50.00", option_strategy="VERTICAL", items=[
        CALL["items"][0],
        dict(CALL["items"][0], side="SELL", option_exercise_price="9.0000000000")])
    quotes = {OCC: {"bid": "3.0", "ask": "3.4"}, "AAA270521C00009000": {"bid": "1.8", "ask": "2.2"}}
    long_leg, short_leg = positions.parse_positions([combo], quotes)[0]
    assert (long_leg.qty, short_leg.qty) == (1.0, -1.0)
    assert (long_leg.unrealized, short_leg.unrealized) == (50.0, 0.0)
    assert (long_leg.mark, short_leg.mark) == (pytest.approx(3.2), pytest.approx(2.0))


# รูปแบบทางการของ Webull OpenAPI v3 (GET /trading/assets/positions/list): option อยู่ใน "legs" และไม่มี "side"
# (เอกสาร developer.webull.co.th — Account Positions) · รูปแบบของ MCP ที่ใช้ตอนพัฒนาคือ "items" + "side"
V3_CALL = {"position_id": "P-2", "currency": "USD", "quantity": "1", "symbol": "AAA", "option_strategy": "SINGLE",
           "instrument_type": "OPTION", "cost_price": "4.90", "last_price": "3.30",
           "unrealized_profit_loss": "-160.00",
           "legs": [{"leg_id": "L-1", "symbol": "AAA", "quantity": "1", "option_type": "CALL",
                     "option_expire_date": "2027-05-21", "option_exercise_price": "7",
                     "option_contract_multiplier": "100", "option_contract_deliverable": "100",
                     "expiration_type": "AM"}]}
V3_POSITIONS = [STOCK, V3_CALL]


def _shape(leg: positions.Leg) -> tuple:
    return (leg.symbol, leg.kind, leg.qty, leg.multiplier, leg.mark, leg.unrealized, leg.strike, leg.expiry,
            leg.occ, leg.iv, leg.delta)


def test_v3_legs_parse_like_the_mcp_items():
    v3, problems = positions.parse_positions(V3_POSITIONS, OPTION_QUOTES)
    assert not problems
    assert [_shape(leg) for leg in v3] == [_shape(leg) for leg in legs()]
    assert v3[1].cost == 4.90                                         # legs ไม่มีทุน → ใช้ของ position (ขาเดียว)
    assert positions.option_contracts(V3_POSITIONS) == [OCC]
    assert positions.underlying_symbols(V3_POSITIONS) == ["AAA"]


def test_occ_symbols_on_the_leg_or_position_give_the_underlying_and_the_contract():
    assert positions.parse_occ(OCC) == {"symbol": "AAA", "expiry": date(2027, 5, 21), "kind": "call", "strike": 7.0}
    assert positions.parse_occ("BRK.B 270115P00450000")["symbol"] == "BRK.B"
    assert positions.parse_occ("AAA") is None and positions.parse_occ("AAA271399C00007000") is None
    leg_occ = dict(V3_CALL, legs=[{"leg_id": "L-1", "symbol": OCC, "quantity": "1"}])     # ไม่มี field option อื่น
    position_occ = {k: v for k, v in V3_CALL.items() if k != "legs"} | {"symbol": OCC}     # ไม่มี legs เลย
    for shape in (leg_occ, position_occ):
        out, problems = positions.parse_positions([STOCK, shape], OPTION_QUOTES)
        assert not problems and _shape(out[1]) == _shape(legs()[1])
        assert positions.underlying_symbols([STOCK, shape]) == ["AAA"]
        assert positions.option_contracts([shape]) == [OCC]


def test_v3_short_legs_use_the_sign_of_the_quantity():
    short = dict(V3_CALL, legs=[dict(V3_CALL["legs"][0], quantity="-2")])
    assert positions.parse_positions([short])[0][0].qty == -2.0
    short_total = dict(V3_CALL, quantity="-1")
    assert positions.parse_positions([short_total])[0][0].qty == -1.0
    assert positions.parse_positions([V3_CALL])[0][0].qty == 1.0                    # ไม่บอกอะไร = ซื้อ


def test_an_unreadable_leg_reports_field_names_only():
    broken = dict(V3_CALL, legs=[{"leg_id": "L-9", "symbol": "AAA", "quantity": "1", "cost": "777.77"}])
    out, problems = positions.parse_positions([broken])
    assert out == [] and problems == [
        "OPTION: ขาอ่านไม่ได้ ต้องมี option_type / option_expire_date / option_exercise_price "
        "(หรือ symbol แบบ OCC) และ quantity (มี field: cost, leg_id, quantity, symbol)"]
    assert "777.77" not in problems[0]


def test_v3_positions_build_full_books_with_the_bot_chain():
    chains, issues = positions.lego_chains(STATE, ROWS)
    books, notes = positions.build_books({"positions": V3_POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    book = books[0]
    assert not notes and book.chain_issue is None                     # LEAPS 1 × 100 = bot 100
    assert book.unrealized == pytest.approx(-220.0)                   # หุ้น + option
    assert book.spot == 9.0 and book.options[0].delta == 0.8 and book.chain.excess == 20.0


def test_token_txt_needs_three_lines():
    assert positions.parse_token_txt(TOKEN_TXT) == (FAKE_TOKEN, 1790000000000, "NORMAL")
    for bad in ("", FAKE_TOKEN, f"{FAKE_TOKEN}\nNORMAL", f"{FAKE_TOKEN}\nsoon\nNORMAL"):
        with pytest.raises(ValueError, match="3 บรรทัด"):
            positions.parse_token_txt(bad)


# ---------------------------------------------------------------- state ของ bot (chain)

def test_lego_chains_pick_the_latest_chain_and_read_v3_fields():
    c = chain()
    assert (c.chain_key, c.fix_c, c.leaps_shares, c.real_holdings) == ("AAA_fake01", pytest.approx(1500.0),
                                                                     100.0, 60.0)
    assert (c.p_acted, c.actual, c.r_basis, c.excess, c.p0) == (9.0, -100.0, -120.0, 20.0, 10.0)


def test_lego_chains_fail_closed():
    state = {"AAA_x": {**STATE["AAA_fake01"], "execution_cashflow": {}, "prev_price": None}}
    chains, issues = positions.lego_chains(state, {"AAA_x": ROWS["AAA_fake01"]})
    assert chains == {} and "P_acted" in issues["AAA"]
    chains, issues = positions.lego_chains({"AAA_x": STATE["AAA_fake01"]}, {})          # ไม่มีแถว → ไม่มี FIX_C
    assert chains == {} and "FIX_C" in issues["AAA"]
    v2 = {"AAA_x": {**STATE["AAA_fake01"],
                    "execution_cashflow": {**STATE["AAA_fake01"]["execution_cashflow"], "excess": 5.0}}}
    chains, issues = positions.lego_chains(v2, {"AAA_x": ROWS["AAA_fake01"]})
    assert chains == {} and "R_basis" in issues["AAA"]


def test_lego_chains_fall_back_to_the_state_pointer_fields():
    pointer = {k: v for k, v in STATE["AAA_fake01"].items() if k != "execution_cashflow"}
    pointer.update(prev_price=9.0, prev_actual=-100.0, r_basis=-120.0, prev_excess=20.0)
    chains, issues = positions.lego_chains({"AAA_x": pointer}, {"AAA_x": ROWS["AAA_fake01"]})
    assert not issues and chains["AAA"].p_acted == 9.0 and chains["AAA"].excess == 20.0


def test_consistency_between_webull_and_the_bot():
    c = chain()
    assert positions.check_consistency(legs(), c) is None
    near = [positions.Leg(**{**legs()[0].__dict__, "qty": 60.005}), legs()[1]]
    assert positions.check_consistency(near, c) is None
    off = [positions.Leg(**{**legs()[0].__dict__, "qty": 59.5}), legs()[1]]
    assert "หุ้นจริงใน Webull 59.50 ≠ bot 60.00" in positions.check_consistency(off, c)
    two = [legs()[0], positions.Leg(**{**legs()[1].__dict__, "qty": 2.0})]
    assert "LEAPS ใน Webull 200 หุ้น ≠ bot 100" in positions.check_consistency(two, c)


def test_build_books_marks_missing_chain_and_mismatch_and_solves_iv_from_mark():
    books, notes = positions.build_books({"positions": POSITIONS, "options": None}, {}, {}, NOW)
    book = books[0]
    assert book.usable_chain is None and "ไม่มี chain ของ bot" in book.chain_issue
    assert any("คำนวณจาก mark" in n for n in notes)
    call = book.options[0]
    assert call.iv_source == "mark" and positions.bs_price(
        9.0, 7.0, positions.years_to_expiry(call.expiry, NOW), positions.RISK_FREE, call.iv, "call") \
        == pytest.approx(3.30, abs=1e-6)
    cheap = [STOCK, dict(CALL, last_price="0.50")]                      # ต่ำกว่า intrinsic: แก้ IV ไม่ได้
    books, _ = positions.build_books({"positions": cheap, "options": None}, {}, {}, NOW)
    assert not books[0].options_priced and any("ไม่พอคำนวณ IV" in n for n in books[0].notes)
    mismatch = {"AAA_fake01": {**STATE["AAA_fake01"], "prev_holdings": 61.0}}
    chains, issues = positions.lego_chains(mismatch, ROWS)
    books, _ = positions.build_books({"positions": POSITIONS, "options": OPTION_QUOTES}, chains, issues, NOW)
    assert books[0].usable_chain is None and "≠ bot 61.00" in books[0].chain_issue


# ---------------------------------------------------------------- Webull อ่านอย่างเดียว (HTTP ปลอม)

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeHttp:
    """แทน Response.get_response_object: ตอบตาม (method, path) และจดทุก request ที่ SDK ส่ง"""

    def __init__(self, routes=None, error: Exception | None = None):
        self.routes = {
            ("GET", "/trading/accounts/list"): (200, [{"account_id": "ACC-FAKE-1", "account_number": "N-1"}]),
            ("GET", "/trading/assets/positions/list"): (200, [dict(p, account_id="ACC-FAKE-1") for p in POSITIONS]),
            ("GET", "/market-data/stocks/snapshots/list"): (
                200, [{"symbol": "AAA", "price": "9.05", "trade_status": "TRADING", "quote_time": 1}]),
            ("GET", "/market-data/options/snapshots/list"): (200, [dict(OPTION_QUOTES[OCC], symbol=OCC)]),
            **(routes or {})}
        self.error = error
        self.calls: list[tuple[str, str, dict, str]] = []

    def __call__(self, response):
        url = response.get_url()
        method, path = response.get_method(), url.split("?")[0]
        self.calls.append((method, path, dict(response.get_headers()), url))
        if self.error is not None:
            raise self.error
        status, body = self.routes.get((method, path), (404, {"error_code": "NOT_FOUND"}))
        return status, {"x-request-id": "req-fake"}, json.dumps(body).encode(), _Resp(status, body)


@pytest.fixture
def http(monkeypatch):
    from webull.core.http.response import Response
    fake = FakeHttp()
    monkeypatch.setattr(Response, "get_response_object", lambda self: fake(self))
    monkeypatch.delenv("WEBULL_API_DEBUG", raising=False)
    return fake


def fetch(**kwargs):
    return webull_read.fetch_account_snapshot(FAKE_KEY, FAKE_SECRET, "th", TOKEN_TXT, now=NOW, **kwargs)


def test_fetch_reads_four_get_endpoints_with_the_token_header(http):
    snap = fetch()
    assert [(m, p) for m, p, _, _ in http.calls] == [
        ("GET", "/trading/accounts/list"), ("GET", "/trading/assets/positions/list"),
        ("GET", "/market-data/stocks/snapshots/list"), ("GET", "/market-data/options/snapshots/list")]
    assert {(m, p) for m, p, _, _ in http.calls} <= READ_ONLY_ALLOWLIST
    for _, _, headers, _ in http.calls:
        assert headers["x-access-token"] == FAKE_TOKEN and headers["x-app-key"] == FAKE_KEY
    assert snap["account_count"] == 1 and snap["as_of"] == "2026-11-17T15:00:00+00:00"
    assert all("account_id" not in p and "account_number" not in p for p in snap["positions"])
    assert snap["quotes"]["AAA"]["trade_status"] == "TRADING"
    assert snap["options"][OCC]["imp_vol"] == "0.80"


def test_fetch_with_v3_positions_quotes_the_underlying_and_the_option(http):
    http.routes[("GET", "/trading/assets/positions/list")] = (
        200, [dict(STOCK), dict(V3_CALL, symbol=OCC, legs=[dict(V3_CALL["legs"][0], symbol=OCC)])])
    snap = fetch()
    urls = {p: url for _, p, _, url in http.calls}
    assert "symbols=AAA&" in urls["/market-data/stocks/snapshots/list"]          # ไม่ส่ง OCC ไปขอราคาหุ้น
    assert f"symbols={OCC}&" in urls["/market-data/options/snapshots/list"]
    books, notes = positions.build_books(snap, {}, {}, NOW)
    assert [leg.kind for leg in books[0].legs] == ["stock", "call"] and books[0].options[0].iv == 0.8


def _sdk_requests():
    """request ทุกคลาสของ SDK ที่สร้างได้โดยไม่ต้องมี argument (order v1–v3, account, data, token)"""
    from webull.core.request import ApiRequest
    found = {}
    for package in ("webull.trade.request", "webull.data.request", "webull.core.http.initializer"):
        module = importlib.import_module(package)
        for info in pkgutil.walk_packages(module.__path__, package + "."):
            mod = importlib.import_module(info.name)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, ApiRequest) and cls is not ApiRequest and cls.__module__ == mod.__name__:
                    try:
                        request = cls()
                    except TypeError:
                        continue
                    found[(str(request.get_method()).upper(), request.get_action_name())] = request
    return found


def test_r1_every_post_and_every_non_allowlisted_request_is_blocked_before_the_network(http):
    from webull.core.client import ApiClient
    guard = ReadOnlyGuard(ApiClient(FAKE_KEY, FAKE_SECRET, "th"))
    requests = _sdk_requests()
    posts = {key for key in requests if key[0] == "POST"}
    for must in [("POST", "/trading/orders/place"), ("POST", "/trading/orders/preview"),
                 ("POST", "/trading/orders/replace"), ("POST", "/trading/orders/cancel"),
                 ("POST", "/trading/orders/batch-place"), ("POST", "/openapi/trade/stock/order/place"),
                 ("POST", "/openapi/trade/option/order/place"), ("POST", "/trade/order/place"),
                 ("POST", "/auth/tokens/create"), ("POST", "/auth/tokens/check"),
                 ("POST", "/openapi/auth/token/refresh")]:
        assert must in posts, must
    blocked = 0
    for key, request in requests.items():
        if key in READ_ONLY_ALLOWLIST:
            continue
        with pytest.raises(WebullReadError, match="blocked"):
            guard.get_response(request)
        blocked += 1
    assert blocked >= len(posts) > 20
    assert http.calls == []                                            # ไม่มีอะไรออก network เลย
    assert ("GET", "/trading/orders/open-orders/list") in requests     # GET อื่นก็โดนกันด้วย
    assert not hasattr(guard, "_implementation_of_do_action") and not hasattr(guard, "__dict__")


def test_r2_the_wrapper_imports_no_order_or_token_module():
    source = (ROOT / "lego_webull_read.py").read_text(encoding="utf-8")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module}.{alias.name}" for alias in node.names]
    forbidden = ("trade_client", "data_client", "quotes_client", "token_manager", "token_operation",
                 "token_storage", "client_initializer", "order")
    assert not [name for name in imported if any(word in name.lower() for word in forbidden)], imported
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert '"POST"' not in code and "'POST'" not in code
    assert all(method == "GET" for method, _ in READ_ONLY_ALLOWLIST) and len(READ_ONLY_ALLOWLIST) == 4


def test_r3_no_token_is_created_or_checked(http, monkeypatch):
    from webull.core.http.initializer.client_initializer import ClientInitializer
    from webull.core.http.initializer.token.token_manager import TokenManager
    from webull.core.http.initializer.token.token_operation import TokenOperation
    from webull.core.http.initializer.token.token_storage import TokenStorage

    def forbidden(*args, **kwargs):
        raise AssertionError("token path must never run")

    for target, name in [(ClientInitializer, "initializer"), (ClientInitializer, "init_token"),
                         (TokenManager, "__init__"), (TokenManager, "init_token"),
                         (TokenOperation, "create_token"), (TokenOperation, "refresh_token"),
                         (TokenOperation, "check_token"), (TokenStorage, "__init__")]:
        monkeypatch.setattr(target, name, forbidden)
    assert fetch()["positions"]
    assert not any("/auth/" in path for _, path, _, _ in http.calls)


def test_r4_nothing_is_written_to_disk(http, monkeypatch, tmp_path):
    work, token_dir = tmp_path / "cwd", tmp_path / "token-dir"
    work.mkdir()
    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"write attempt: {file}")
        return real_open(file, mode, *args, **kwargs)

    def no_mkdir(*args, **kwargs):
        raise AssertionError("mkdir attempt")

    with monkeypatch.context() as m:
        m.chdir(work)
        m.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(token_dir))
        m.setattr(builtins, "open", read_only_open)
        m.setattr(io, "open", read_only_open)
        m.setattr(Path, "mkdir", no_mkdir)
        m.setattr(os, "mkdir", no_mkdir)
        m.setattr(os, "makedirs", no_mkdir)
        assert fetch()["positions"]
    assert list(work.iterdir()) == [] and not token_dir.exists()


def test_r5_a_rejected_token_stops_without_leaking_secrets(http, caplog):
    http.routes[("GET", "/trading/accounts/list")] = (401, {"error_code": "TOKEN_EXPIRED",
                                                            "message": f"bad token {FAKE_TOKEN}"})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(WebullReadError) as info:
        fetch()
    message = str(info.value)
    assert "token ใช้ไม่ได้ (HTTP 401 TOKEN_EXPIRED)" in message and "ไม่สร้าง token ใหม่" in message
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (FAKE_TOKEN, FAKE_KEY, FAKE_SECRET):
        assert secret not in message and secret not in logged
    assert [p for _, p, _, _ in http.calls] == ["/trading/accounts/list"]   # หยุดทันที ไม่ลองอย่างอื่น


def test_unexpected_errors_are_redacted(http):
    http.error = RuntimeError(f"boom {FAKE_TOKEN} x-app-key: {FAKE_KEY} app_secret={FAKE_SECRET}")
    with pytest.raises(WebullReadError) as info:
        fetch()
    message = str(info.value)
    assert "boom" in message and all(s not in message for s in (FAKE_TOKEN, FAKE_KEY, FAKE_SECRET))


def test_positions_error_and_schema_change_are_reported(http):
    http.routes[("GET", "/trading/assets/positions/list")] = (403, {"error_code": "FORBIDDEN"})
    with pytest.raises(WebullReadError, match="HTTP 403 \\(FORBIDDEN\\) ตอนอ่านpositions"):
        fetch()
    http.routes[("GET", "/trading/assets/positions/list")] = (
        200, [{"instrument_type": "EQUITY", "symbol": "AAA", "cost_price": "321.09"}])
    with pytest.raises(WebullReadError) as info:
        fetch()
    assert "รูปแบบข้อมูล Webull เปลี่ยน" in str(info.value) and "321.09" not in str(info.value)


def test_option_and_stock_snapshots_are_optional(http):
    http.routes[("GET", "/market-data/options/snapshots/list")] = (403, {"error_code": "NO_PERMISSION"})
    http.routes[("GET", "/market-data/stocks/snapshots/list")] = (500, {"error_code": "DOWN"})
    snap = fetch()
    assert snap["options"] is None and snap["quotes"] == {} and snap["positions"]
    assert any("ไม่มีสถานะตลาด" in n for n in snap["notes"])
    books, notes = positions.build_books(snap, {}, {}, NOW)
    assert books[0].options[0].iv_source == "mark" and any("คำนวณจาก mark" in n for n in notes)


def test_guards_before_any_client_is_built(http, monkeypatch):
    with pytest.raises(WebullReadError, match="WEBULL_ENVIRONMENT = prod"):
        fetch(environment="uat")
    with pytest.raises(WebullReadError, match="3 บรรทัด"):
        webull_read.fetch_account_snapshot(FAKE_KEY, FAKE_SECRET, "th", FAKE_TOKEN)
    monkeypatch.setenv("WEBULL_API_DEBUG", "true")
    with pytest.raises(WebullReadError, match="WEBULL_API_DEBUG"):
        fetch()
    assert http.calls == []


def test_the_signer_uses_the_given_key_not_the_environment(http, monkeypatch):
    monkeypatch.setenv("WEBULL_APP_KEY_ID", "FAKE_TEST_OTHER_KEY")
    monkeypatch.setenv("WEBULL_APP_KEY_SECRET", "FAKE_TEST_OTHER_SECRET")
    fetch()
    assert {headers["x-app-key"] for _, _, headers, _ in http.calls} == {FAKE_KEY}


def test_sdk_logs_do_not_reach_the_root_logger(http):
    fetch()
    log = logging.getLogger("webull")
    assert log.propagate is False and log.level == logging.CRITICAL


def test_redact_hides_values_and_named_fields():
    text = (f"key {FAKE_KEY} secret {FAKE_SECRET} {{'x-access-token': 'abc123', 'x-signature': 'sig=='}} "
            "token=zzz999 แท็บนี้ไม่สร้าง token ใหม่")
    out = redact(text, [FAKE_KEY, FAKE_SECRET, "", "abc"])
    assert FAKE_KEY not in out and FAKE_SECRET not in out
    assert "abc123" not in out and "sig==" not in out and "zzz999" not in out
    assert "ไม่สร้าง token ใหม่" in out


# ---------------------------------------------------------------- หน้าเว็บ (AppTest)

WEBULL_SECRETS = {"WEBULL_APP_KEY": FAKE_KEY, "WEBULL_APP_SECRET": FAKE_SECRET,
                  "WEBULL_REGION_ID": "th", "WEBULL_ENVIRONMENT": "prod", "WEBULL_TOKEN_TXT": TOKEN_TXT}


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def snapshot(**extra) -> dict:
    return {"as_of": "2026-11-17T15:00:00+00:00", "positions": POSITIONS, "options": OPTION_QUOTES,
            "quotes": {"AAA": {"price": "9.05", "trade_status": "TRADING"}}, "account_count": 1,
            "notes": [], **extra}


def goal_app(monkeypatch, *, snap=None, error=None, firebase=True, state=None, rows=None) -> AppTest:
    """แอปจริงกับ Webull / Firebase ปลอม · state/rows: chain_key → state / แถวล่าสุด (ค่าเริ่มต้น = fixture AAA)"""
    def fake_fetch(*args, **kwargs):
        if error is not None:
            raise error
        return snap if snap is not None else snapshot()

    monkeypatch.setattr(webull_read, "fetch_account_snapshot", fake_fetch)
    secrets = dict(WEBULL_SECRETS)
    if firebase:
        state, rows = state or STATE, rows or ROWS
        payloads = {"webull_lego_state": state, "webull_lego_rows": {}, "webull_lego_order_audit": {}}
        payloads.update({f"webull_lego_rows/{pointer['last_run_id']}": rows[key] for key, pointer in state.items()})

        class _Reference:
            def __init__(self, path, url=None):
                self.path = path

            def get(self):
                return payloads.get(self.path)

        monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
        monkeypatch.setattr(firebase_admin.db, "reference", _Reference)
        secrets.update({"FIREBASE_SA_JSON": "{}",
                        "FIREBASE_DB_URL": "https://lego-firebase-prod.mock.test"})
    app = AppTest.from_file(APP_PATH)
    app.secrets.update(secrets)
    return app


def test_without_webull_secrets_the_tab_only_explains():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)
    assert not app.exception and not app.error and not app.warning
    assert any("WEBULL_APP_KEY" in info.value for info in app.info)
    assert len(app.get("plotly_chart")) == 1                         # มีแค่กราฟของแท็บ 101


def test_a_skipped_position_is_a_warning_at_the_top(monkeypatch):
    broken = {k: v for k, v in V3_CALL.items() if k != "legs"}
    app = goal_app(monkeypatch, snap=snapshot(positions=[STOCK, broken]))
    app.run(timeout=30)
    assert not app.exception
    assert len(app.warning) == 1 and app.warning[0].value.startswith("ข้อมูลไม่ครบ: ข้าม position ที่อ่านไม่ได้")
    assert "มี field: cost_price, currency" in app.warning[0].value and "-160.00" not in app.warning[0].value


def test_the_reload_guards_and_the_removed_payoff_tab():
    import streamlit_app
    assert webull_read.SNAPSHOT_SCHEMA >= 2 and streamlit_app.goal.GOAL_SCHEMA >= 3
    assert streamlit_app.positions.SKIP_PREFIX and callable(streamlit_app.positions.parse_occ)
    assert not hasattr(streamlit_app, "render_history_tab")             # แท็บ 🧪 ประวัติเป้า ถูกลบแล้ว


def test_a_webull_error_stays_inside_the_tab(monkeypatch):
    app = goal_app(monkeypatch, error=WebullReadError("token ใช้ไม่ได้ (HTTP 401 X) — แท็บนี้ไม่สร้าง token ใหม่"))
    app.run(timeout=30)
    assert not app.exception
    assert [e.value for e in app.error] == ["ดึงข้อมูล Webull ไม่ได้ — token ใช้ไม่ได้ (HTTP 401 X) — "
                                            "แท็บนี้ไม่สร้าง token ใหม่"]
    assert len(app.tabs) >= 7 and len(app.number_input) == 6          # แท็บอื่นยังอยู่ครบ


def test_an_unexpected_error_shows_only_its_type(monkeypatch):
    app = goal_app(monkeypatch, error=RuntimeError(f"leak {FAKE_TOKEN}"))
    app.run(timeout=30)
    assert [e.value for e in app.error] == ["ดึงข้อมูล Webull ไม่ได้ (RuntimeError)"]


