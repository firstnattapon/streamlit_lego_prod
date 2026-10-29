"""⚡ Cheat Sheet → 📝 Summary tab — the 10 key messages and the one sentence.

* ``cheat_sheet/summary.md`` is the source of truth; the tab shows it as Markdown, so the wording can be
  changed without touching code.
* Its claims are rebuilt on the repo's own ledger (``gated_rebalancing_cashflow_from_prices``) and on the
  Option Scale model of test_cheat_sheet.py, so the sentences cannot drift away from the numbers.
"""
from __future__ import annotations

import itertools
import math
import random
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lego_dash_core as core
import streamlit_app
from test_cheat_sheet import _premium_rate

ROOT = Path(__file__).parent
APP_PATH = str(ROOT / "streamlit_app.py")
SUMMARY = ROOT / "cheat_sheet" / streamlit_app.CHEAT_SUMMARY
FIX_C = 1500.0
BARS_PER_YEAR = 252 * 78            # 5-minute bars


def _text() -> str:
    return SUMMARY.read_text(encoding="utf-8")


def _sections() -> dict[str, list[str]]:
    """'#### heading' -> the non-empty lines under it, in file order."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in _text().splitlines():
        if line.startswith("#### "):
            current = sections.setdefault(line[5:].strip(), [])
        elif line.strip():
            assert current is not None, "the file must start with a '#### ' heading"
            current.append(line.strip())
    return sections


def _ledger(prices, p0=100.0):
    """(A, E, buy & hold) after the last price, acting at every price, Fix_c = 1,500."""
    rows = core.gated_rebalancing_cashflow_from_prices(prices, FIX_C, p0, [1] * len(prices))
    return rows[-1]["actual_cumulative"], rows[-1]["excess"], FIX_C * (prices[-1] / p0 - 1.0)


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
def test_summary_is_ten_numbered_messages_in_three_groups_and_one_sentence():
    sections = _sections()

    assert list(sections) == ["Main rebalancing", "Rollover", "Option Scale", "ประโยคเดียว"]
    groups = [sections[name] for name in list(sections)[:3]]
    assert [len(group) for group in groups] == [4, 3, 3]
    numbers = []
    for line in itertools.chain.from_iterable(groups):
        match = re.match(r"(\d+)\. \S", line)
        assert match, f"not a numbered message: {line!r}"
        numbers.append(int(match.group(1)))
    assert numbers == list(range(1, 11))             # no gaps; the numbering carries on across groups
    sentence = sections["ประโยคเดียว"]
    assert len(sentence) == 1 and sentence[0].startswith("**") and sentence[0].endswith("**")


def test_the_sentence_follows_the_real_cycle_and_ends_with_the_catch():
    sentence = _sections()["ประโยคเดียว"][0]

    # Main rebalancing collects E, Option Scale decides, Rollover is its last step (5.2)
    assert sentence.index("Main") < sentence.index("Option Scale") < sentence.index("Rollover")
    assert sentence.endswith("แต่ E บวกไม่ได้แปลว่าชนะการถือเฉยๆ**")


# --------------------------------------------------------------------------- #
# messages 3-4: E is never negative, yet a one-way market loses to holding
# --------------------------------------------------------------------------- #
def test_e_is_positive_yet_a_one_way_market_is_behind_holding_and_a_round_trip_is_ahead():
    for prices in ([110.0, 121.0, 133.1], [80.0, 64.0, 51.2]):       # three steps up, three steps down
        a, e, hold = _ledger(prices)
        assert e > 0 and a < hold
    a, e, hold = _ledger([110.0, 100.0])                             # the price comes back
    assert e > 0 and a > hold


@pytest.mark.parametrize("up, down, expected", [
    (1.10, 0.90, "tie"),         # E[x] = 1.00: no direction
    (1.12, 0.92, "behind"),      # E[x] = 1.02: drifting up
    (1.08, 0.88, "behind"),      # E[x] = 0.98: drifting down
])
def test_a_market_without_direction_ties_holding_on_average_and_drift_costs(up, down, expected):
    steps, mean_gap = 10, 0.0
    for path in itertools.product((up, down), repeat=steps):         # all 2**10 paths, equally likely
        prices, price = [], 100.0
        for x in path:
            price *= x
            prices.append(price)
        a, _, hold = _ledger(prices)
        mean_gap += (a - hold) / 2 ** steps

    if expected == "tie":
        assert mean_gap == pytest.approx(0.0, abs=1e-8)
    else:
        assert mean_gap < -1.0


# --------------------------------------------------------------------------- #
# message 7: beta hides the price where the reference line is zero
# --------------------------------------------------------------------------- #
def test_p_avg_is_the_price_where_the_reference_line_is_zero_and_is_not_a_cost_basis():
    # the Rebalancing Playground example: capital 100 -> 200 at P_acted = 120, P0 = 100
    p0, c_old, c_new, pivot = 100.0, 100.0, 200.0, 120.0
    beta = (c_old - c_new) * math.log(pivot / p0)
    p_avg = p0 * math.exp(-beta / c_new)

    assert p_avg == pytest.approx(math.sqrt(p0 * pivot))                        # 109.54, the geometric mean
    assert beta + c_new * math.log(p_avg / p0) == pytest.approx(0.0, abs=1e-12)  # R(P_avg) = 0
    # scaling down 1,500 -> 500 at 130 puts P_avg below every price the capital entered at (100 and 130)
    beta = (1500.0 - 500.0) * math.log(130.0 / 100.0)
    assert p0 * math.exp(-beta / 500.0) == pytest.approx(59.17, abs=0.01)


# --------------------------------------------------------------------------- #
# messages 9-10: E passes K only while the price keeps moving, and how long that takes
# --------------------------------------------------------------------------- #
def _years_until_e_passes_k(sigma, years, seed=2026):
    """First time E reaches K on a driftless random walk of 5-minute bars (the repo's ledger)."""
    rng = random.Random(seed)
    dt = 1.0 / BARS_PER_YEAR
    prices, price = [], 100.0
    for _ in range(int(years * BARS_PER_YEAR)):
        price *= math.exp(rng.gauss(-0.5 * sigma ** 2 * dt, sigma * math.sqrt(dt)))
        prices.append(price)
    rows = core.gated_rebalancing_cashflow_from_prices(prices, FIX_C, 100.0, [1] * len(prices))
    excess = [row["excess"] for row in rows]
    assert all(b >= a - 1e-9 for a, b in zip(excess, excess[1:]))     # E never goes down
    k = _premium_rate() * FIX_C
    return next((i / BARS_PER_YEAR for i, e in enumerate(excess) if e >= k), None)


@pytest.mark.parametrize("sigma, expected", [(0.5, 1.0), (0.3, 2.8)])
def test_how_long_e_takes_to_pass_k(sigma, expected):
    assert _years_until_e_passes_k(sigma, years=3.5) == pytest.approx(expected, abs=0.1)


def test_a_flat_price_never_passes_k():
    rows = core.gated_rebalancing_cashflow_from_prices([100.0] * 500, FIX_C, 100.0, [1] * 500)

    assert max(row["excess"] for row in rows) == 0.0                  # no wiggle, no E


def test_message_10_numbers_come_from_the_option_scale_model():
    k = _premium_rate()                                  # K / Fix_c, the 12.4% in the Option Scale box
    text = _text()

    assert f"{k:.1%}" == "12.4%" and "`K` = 12.4% ของทุน" in text
    assert round(math.sqrt(2 * k), 2) == 0.50 and "≥ ~50%/ปี" in text         # sigma >= sqrt(2k) pays K back in a year
    assert round(2 * k / 0.30 ** 2, 1) == 2.8 and "แกว่ง 30% ใช้ ~2.8 ปี" in text   # 2k / sigma^2 years


# --------------------------------------------------------------------------- #
# the tab
# --------------------------------------------------------------------------- #
def test_cheat_sheet_summary_tab_shows_the_file_first():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs][-3:] == ["⚡ Cheat Sheet", "📝 Summary", "🔀 Diagrams"]
    assert _text().strip() in [block.value for block in app.markdown]


def test_a_missing_summary_is_reported_instead_of_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(streamlit_app, "CHEAT_SHEET_DIR", tmp_path)

    def script():
        import streamlit_app
        streamlit_app.render_cheat_summary()

    app = AppTest.from_function(script)
    app.run(timeout=20)

    assert not app.exception
    assert len(app.error) == 1 and "FileNotFoundError" in app.error[0].value
