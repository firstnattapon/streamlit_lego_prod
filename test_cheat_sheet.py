"""⚡ Cheat Sheet tab — the Mermaid sources, their rendered images and the rollover formula.

* ``cheat_sheet/*.mmd`` are the source of truth (plain Mermaid, usable anywhere Mermaid is).
* The tab shows PNGs rendered ahead of time by ``tools/render_cheat_sheet.py``. Their names
  carry a hash of the source, so a .mmd edited without re-rendering fails here and shows a
  warning in the app instead of a silently stale picture.
* The rollover diagram's formula, ``beta += (Fix_c − Fix_c_new) * ln(P_acted / P₀)``, is
  checked against the repo's own gated ledger — and so is where beta goes (on R, never on A).
* The Option Scale diagram is the app-v4 cycle with one box per step. Its order is pinned, its
  update is the rollover diagram's own line, and app-v4's worked example (K, S, Fix_c_new,
  ΣE / ΣK) is rebuilt on the same ledger, with the 12.4% rebuilt as a Black-Scholes premium.
"""
from __future__ import annotations

import importlib.util
import math
import random
import re
import struct
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lego_dash_core as core
import streamlit_app

ROOT = Path(__file__).parent
SHEET_DIR = ROOT / "cheat_sheet"
APP_PATH = str(ROOT / "streamlit_app.py")
STEMS = list(streamlit_app.CHEAT_SHEETS.values())

# Just enough Mermaid to catch typos without a renderer: declared nodes, edges, class lines.
_NODE = re.compile(r'\b([A-Za-z_]\w*)\s*(?:\[|\{)"')
_EDGE = re.compile(r'\b([A-Za-z_]\w*)(?:\[[^\]]*\]|\{[^}]*\})?\s*-->\s*(?:\|"[^"]*"\|\s*)?([A-Za-z_]\w*)')
_CLASSDEF = re.compile(r"^\s*classDef\s+(\w+)\s", re.MULTILINE)
_CLASS = re.compile(r"^\s*class\s+([\w,]+)\s+(\w+)\s*;", re.MULTILINE)
_LABEL = re.compile(r'\b([A-Za-z_]\w*)\s*[\[{]"([^"]*)"[\]}]')                      # node -> its text
_ARROW_LABEL = re.compile(r'\b([A-Za-z_]\w*)\s*-->\|"([^"]*)"\|\s*([A-Za-z_]\w*)')   # (from, text on the arrow, to)


def _source(stem: str) -> str:
    return (SHEET_DIR / f"{stem}.mmd").read_text(encoding="utf-8")


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return struct.unpack(">II", header[16:24])


def _graph(text: str):
    nodes = set(_NODE.findall(text))
    edges = _EDGE.findall(text)
    return nodes, edges


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_catalog_lists_the_three_diagrams():
    assert streamlit_app.CHEAT_SHEET_DIR == SHEET_DIR.resolve()
    assert streamlit_app.CHEAT_SHEETS == {"1. Main rebalancing": "main_rebalancing",
                                          "2. Rollover": "rollover",
                                          "3. Option Scale": "option_scale"}
    assert {path.stem for path in SHEET_DIR.glob("*.mmd")} == set(STEMS)    # no unlisted diagram


@pytest.mark.parametrize("stem", STEMS)
def test_mmd_source_is_a_well_formed_flowchart(stem):
    text = _source(stem)
    nodes, edges = _graph(text)

    assert text.startswith("flowchart TD\n")
    assert all(line.count('"') % 2 == 0 for line in text.splitlines())      # labels are closed
    assert nodes and "end" not in nodes                                      # `end` is a Mermaid keyword
    assert edges and all({a, b} <= nodes for a, b in edges)                  # no edge to an undeclared node
    assert nodes == {i for pair in edges for i in pair}                      # every node is wired in
    styles = set(_CLASSDEF.findall(text))
    for ids, name in _CLASS.findall(text):
        assert name in styles and set(ids.split(",")) <= nodes


def test_main_rebalancing_keeps_the_agreed_equations():
    text = _source("main_rebalancing")
    for needle in ("Gap = Q × P − Fix_c", "R = Fix_c × ln(P / P₀)", "ABS(Gap) ≤ DIFF",
                   "Qty = ABS(Gap) / P", "Σ(Qty × Price) / ΣQty",
                   "ΔA = Fix_c × (VWAP / P_acted − 1)", "A_new = A_old + ΔA",
                   "E_new = A_new − R", "P_acted = VWAP"):
        assert needle in text, needle


def test_rollover_diagram_is_the_simple_one():
    text = _source("rollover")
    nodes, edges = _graph(text)

    assert "beta += (Fix_c − Fix_c_new) * ln(P_acted / P₀)" in text   # the one formula
    assert "จุดหมุนคือ P_acted" in text                      # pivot = last acted price, not the live tick
    assert "R = beta + Fix_c * ln(P / P₀)" in text           # how main step 1 reads after a rollover
    for confusing in ("c′", "P′", "p₀", "·"):                # one name per thing (main's names), * for multiply
        assert confusing not in text, confusing
    assert "beta บวกที่ R เท่านั้น" in text                  # beta lives on R ...
    assert "A ไม่บวก beta" in text                           # ... never on A (see test_beta_goes_on_r_and_never_on_a)
    assert text.count("ln(") == 2                            # one ln for the update, one for R
    assert len(nodes) <= 6                                   # small enough to read at a glance
    # straight line with one skip: no loop back
    graph = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    def has_cycle(node, path=()):
        return node in path or any(has_cycle(nxt, (*path, node)) for nxt in graph.get(node, []))
    assert not any(has_cycle(start) for start in graph)


def test_option_scale_diagram_is_the_simple_one():
    text = _source("option_scale")
    nodes, edges = _graph(text)
    labels = dict(_LABEL.findall(text))
    step = {label.split()[0]: node for node, label in labels.items()}     # "1." -> its box ... "วัดผล" -> the last box

    assert set(labels) == nodes and len(nodes) <= 8                       # one box per app-v4 step, no more
    assert set(step) == {"1.", "2.", "3.", "3.1", "4.", "5.1", "5.2", "วัดผล"}   # numbered as in app-v4
    assert all(label.count("<br/>") <= 2 for label in labels.values())    # at most three short lines each
    for confusing in ("c′", "P′", "p₀", "β", "·", "×", "÷"):              # main's and rollover's names; * multiplies, / divides
        assert confusing not in text, confusing
    for needle in ("K = 12.4% * Fix_c * Hedge", "หุ้น 20% + LEAPS 80%", "E_cyc = E_new − E ตอนจ่าย", "S = E_cyc − K",
                   "ไม่จ่าย K ซ้ำ", "Zero-Cost ≠ ไม่เสี่ยง", "Fix_c_new = Fix_c + S", "ΣE / ΣK ≥ 1.0x"):
        assert needle in text, needle
    for line in ("beta += (Fix_c − Fix_c_new) * ln(P_acted / P₀)", "Fix_c = Fix_c_new"):
        assert line in text and line in _source("rollover"), line         # the very update the rollover diagram draws

    # The order is the rule. Nothing leads into step 1 (the next round is the last box's words, not an arrow),
    # "ไม่" goes back to 2 so K is never paid twice, and Scale Up is reachable only through Shield, after "ใช่".
    pay, collect, gate, trend, shield, scale, roll, measure = (
        step[s] for s in ("1.", "2.", "3.", "3.1", "4.", "5.1", "5.2", "วัดผล"))
    assert sorted(edges) == sorted([(pay, collect), (collect, gate), (gate, shield), (gate, trend), (trend, collect),
                                    (shield, scale), (scale, roll), (roll, measure)])
    assert sorted(_ARROW_LABEL.findall(text)) == sorted([(gate, "ใช่", shield), (gate, "ไม่", trend),
                                                         (trend, "กลับ 2", collect)])


# --------------------------------------------------------------------------- #
# rendered images
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem", STEMS)
def test_rendered_png_matches_the_current_source(stem):
    expected = SHEET_DIR / f"{stem}.{streamlit_app.mmd_digest(_source(stem))}.png"

    assert expected.is_file(), "the .mmd changed: run python tools/render_cheat_sheet.py"
    assert [p.name for p in SHEET_DIR.glob(f"{stem}.*.png")] == [expected.name]   # no stale leftovers
    width, height = _png_size(expected)
    assert width >= 900 and height > width        # rendered at scale 3, so it stays sharp on retina screens


def test_render_script_and_app_agree():
    spec = importlib.util.spec_from_file_location("render_cheat_sheet", ROOT / "tools" / "render_cheat_sheet.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    assert script.CHEAT_SHEET_DIR == SHEET_DIR.resolve()
    for text in ("flowchart TD\n", _source("rollover"), "ไทย ≤ 1"):
        assert script.source_digest(text) == streamlit_app.mmd_digest(text)


def test_digest_ignores_crlf_checkouts(tmp_path):
    lf, crlf = tmp_path / "lf.mmd", tmp_path / "crlf.mmd"
    lf.write_bytes(b'flowchart TD\n    A["x"] --> B["y"]\n')
    crlf.write_bytes(b'flowchart TD\r\n    A["x"] --> B["y"]\r\n')

    assert (streamlit_app.mmd_digest(lf.read_text(encoding="utf-8"))
            == streamlit_app.mmd_digest(crlf.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# the tab
# --------------------------------------------------------------------------- #
def test_cheat_sheet_tab_shows_all_diagrams():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    assert not app.exception
    # Cheat Sheet is still the last tab; its own two tabs follow it (Summary is in test_cheat_summary.py)
    assert [tab.label for tab in app.tabs][-3:] == ["⚡ Cheat Sheet", "📝 Summary", "🔀 Diagrams"]
    assert len(app.get("imgs")) == len(STEMS)                  # one picture per diagram
    urls = [img.url for element in app.get("imgs") for img in element.proto.imgs]
    assert len(urls) == len(STEMS) and all(url.endswith(".png") for url in urls)   # lossless: "auto" would send JPEG
    shown = [block.value for block in app.code]
    for stem in STEMS:
        assert _source(stem).rstrip("\n") in shown             # the .mmd text, ready to copy (st.code trims the last newline)
    assert not app.warning                                     # every picture is current
    assert len(app.number_input) == 6 and len(app.text_input) == 1   # no new inputs for the smoke tests


def test_a_stale_image_is_flagged_but_the_source_stays(monkeypatch, tmp_path):
    source = 'flowchart TD\n    A["a"] --> B["b"]\n'
    (tmp_path / "x.mmd").write_text(source, encoding="utf-8")
    (tmp_path / "x.00000000.png").write_bytes(b"an image rendered from an older source")
    monkeypatch.setattr(streamlit_app, "CHEAT_SHEET_DIR", tmp_path)

    def script():
        import streamlit_app
        streamlit_app.render_cheat_sheet("x")

    app = AppTest.from_function(script)
    app.run(timeout=20)

    assert not app.exception
    assert len(app.warning) == 1 and "render_cheat_sheet.py" in app.warning[0].value
    assert not app.get("imgs")
    assert [block.value for block in app.code] == [source.rstrip("\n")]


def test_a_missing_mmd_is_reported_instead_of_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(streamlit_app, "CHEAT_SHEET_DIR", tmp_path)

    def script():
        import streamlit_app
        streamlit_app.render_cheat_sheet("gone")

    app = AppTest.from_function(script)
    app.run(timeout=20)

    assert not app.exception
    assert len(app.error) == 1 and "FileNotFoundError" in app.error[0].value


# --------------------------------------------------------------------------- #
# the rollover formula, on the repo's own ledger
# --------------------------------------------------------------------------- #
def _beta_run(prices, p0, c0, actions, rollovers, live_pivot=False):
    """cheat_sheet/main_rebalancing.mmd + rollover.mmd as code (A and E stay frozen between acts)."""
    c, beta, acted, a = c0, 0.0, p0, 0.0
    e_jumps = []
    for i, (price, act) in enumerate(zip(prices, actions), start=1):
        if act:
            a += c * (price / acted - 1.0)                       # step 6, with the capital in force
            acted = price
        if i in rollovers:
            before = a - (beta + c * math.log(acted / p0))
            pivot = price if live_pivot else acted               # P_acted (the live tick is the wrong choice)
            beta += (c - rollovers[i]) * math.log(pivot / p0)    # the formula
            c = rollovers[i]
            e_jumps.append(a - (beta + c * math.log(acted / p0)) - before)
    return a, a - (beta + c * math.log(acted / p0)), e_jumps


def _fresh_chains(prices, p0, c0, actions, rollovers):
    """The same run as separate chains (genesis at P_acted when the capital changes): A and E add up."""
    total_a = total_e = 0.0
    start, c, acted, rows, flags = p0, c0, p0, [], []
    for i, (price, act) in enumerate(zip(prices, actions), start=1):
        rows.append(price)
        flags.append(act)
        acted = price if act else acted
        if i in rollovers:
            last = core.gated_rebalancing_cashflow_from_prices(rows, c, start, flags)[-1]
            total_a += last["actual_cumulative"]
            total_e += last["excess"]
            start, c, rows, flags = acted, rollovers[i], [], []
    if rows:
        last = core.gated_rebalancing_cashflow_from_prices(rows, c, start, flags)[-1]
        total_a += last["actual_cumulative"]
        total_e += last["excess"]
    return total_a, total_e


def test_beta_rollover_equals_opening_fresh_chains():
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randint(12, 120)
        p0 = rng.uniform(20, 300)
        prices, price = [], p0
        for _ in range(n):
            price *= math.exp(rng.gauss(-0.0008, 0.03))
            prices.append(price)
        actions = [1 if rng.random() < rng.choice([0.3, 0.7, 1.0]) else 0 for _ in range(n)]
        rollovers = {rng.randint(1, n - 1): rng.choice([250.0, 700.0, 1500.0, 2200.0, 5000.0])
                     for _ in range(rng.randint(0, 4))}
        c0 = rng.choice([500.0, 1500.0, 3000.0])

        a, e, jumps = _beta_run(prices, p0, c0, actions, rollovers)
        chained_a, chained_e = _fresh_chains(prices, p0, c0, actions, rollovers)

        assert a == pytest.approx(chained_a, abs=1e-8)
        assert e == pytest.approx(chained_e, abs=1e-8)
        assert all(abs(jump) < 1e-9 for jump in jumps)           # E does not move at the rollover


def test_beta_goes_on_r_and_never_on_a():
    # the Rebalancing Playground example (capital 100 → 200 at P_acted = 120), then three more rows
    prices, actions, rollovers = [110.0, 120.0, 135.0, 150.0, 165.0], [1, 1, 1, 0, 1], {2: 200.0}
    beta = (100.0 - 200.0) * math.log(120.0 / 100.0)                        # ≈ −18.23
    ref_a, ref_e = _fresh_chains(prices, 100.0, 100.0, actions, rollovers)  # the repo's ledger, one chain per capital
    a, e, _ = _beta_run(prices, 100.0, 100.0, actions, rollovers)

    assert abs(beta) > 1                                                    # the example must tell the placements apart
    assert (a, e) == (pytest.approx(ref_a, abs=1e-8), pytest.approx(ref_e, abs=1e-8))   # 88.5354 and 6.6125

    # Try every place beta could be added in the table. Only one keeps A equal to the cash that was
    # made (ref_a) and E equal to the fresh chains (ref_e).
    r_without_beta = 200.0 * math.log(165.0 / 100.0)                        # Fix_c_new * ln(P_acted / P₀)
    fits = []
    for on_r in (False, True):
        for on_a in (False, True):
            shown_a = ref_a + (beta if on_a else 0.0)
            shown_e = shown_a - ((beta if on_r else 0.0) + r_without_beta)
            if shown_a == pytest.approx(ref_a, abs=1e-8) and shown_e == pytest.approx(ref_e, abs=1e-8):
                fits.append((on_r, on_a))
    assert fits == [(True, False)]


def test_pivot_must_be_the_last_acted_price_not_the_live_tick():
    # act at 110, price drifts to 121 without acting, then the capital doubles
    prices, actions, rollovers = [110.0, 121.0], [1, 0], {2: 3000.0}

    *_, jump_acted = _beta_run(prices, 100.0, 1500.0, actions, rollovers)
    *_, jump_live = _beta_run(prices, 100.0, 1500.0, actions, rollovers, live_pivot=True)

    assert jump_acted == [pytest.approx(0.0, abs=1e-9)]
    assert jump_live == [pytest.approx((1500.0 - 3000.0) * math.log(110.0 / 121.0))]   # ≈ +142.97 of fake profit


def test_successive_rollovers_just_add_up():
    beta, c, p0 = 0.0, 1500.0, 100.0
    steps = [(2200.0, 112.0), (900.0, 95.0), (1500.0, 130.0)]
    for new_c, pivot in steps:
        beta += (c - new_c) * math.log(pivot / p0)
        c = new_c

    closed_form = ((1500.0 - 2200.0) * math.log(112 / 100) + (2200.0 - 900.0) * math.log(95 / 100)
                   + (900.0 - 1500.0) * math.log(130 / 100))
    assert beta == pytest.approx(closed_form, abs=1e-12)


# --------------------------------------------------------------------------- #
# Option Scale: app-v4's worked example, on the repo's own ledger
# --------------------------------------------------------------------------- #
def _premium_rate(strike=0.9, years=1.0, sigma=0.5, rate=0.04):
    """Black-Scholes put on 1 of stock, 10% out of the money: the 12.4% in K = 12.4% * Fix_c * Hedge."""
    def cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    d1 = (math.log(1.0 / strike) + (rate + sigma ** 2 / 2.0) * years) / (sigma * math.sqrt(years))
    d2 = d1 - sigma * math.sqrt(years)
    return strike * math.exp(-rate * years) * cdf(-d2) - cdf(-d1)


def test_option_scale_gives_the_app_v4_numbers():
    # Fix_c 1,500 and P₀ 10, price 15 then back to 10, Hedge 1: the cycle of app-v4's system map, step by step
    fix_c, p0, prices = 1500.0, 10.0, [15.0, 10.0]
    assert f"{_premium_rate():.1%}" in _source("option_scale")           # the 12.4% in the box is the Black-Scholes put

    k = _premium_rate() * fix_c                                          # 1. pay K first
    rows = core.gated_rebalancing_cashflow_from_prices(prices, fix_c, p0, [1, 1])   # 2. E from main rebalancing
    e_at_pay, e_up, e_down = (row["excess"] for row in rows)
    s_up, s_down = e_up - e_at_pay - k, e_down - e_at_pay - k            # 3. S = E_cyc − K
    assert (e_at_pay, round(k, 2)) == (0.0, 185.94)

    assert (round(e_up, 2), round(s_up, 2)) == (141.80, -44.14)          # S < 0 at 15: step 3.1, E stays ...
    assert round(rows[1]["ln_reference"], 2) == 608.20                   # ... and the trend R is what goes to Pool CF

    assert (round(e_down, 2), round(s_down, 2)) == (250.00, 64.06)       # S > 0 at 10: steps 4, 5.1, 5.2
    new_c = fix_c + s_down                                               # 5.1 Scale Up
    assert round(new_c, 2) == 1564.06
    a, e, jumps = _beta_run(prices, p0, fix_c, [1, 1], {2: new_c})       # 5.2 Rollover, the rollover diagram's own model
    assert (round(a, 2), round(e, 2)) == (250.00, 250.00) and abs(jumps[0]) < 1e-9   # E does not move
    assert round(_premium_rate() * new_c, 2) == 193.88                   # step 1 again, on the new Fix_c
    assert f"{e_down / k:.2f}x" == "1.34x"                               # the measure: ΣE / ΣK
