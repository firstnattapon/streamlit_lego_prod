"""UAT smoke test for the deployed Streamlit entry point."""
from pathlib import Path

import firebase_admin
import lego_dash_core
from streamlit.testing.v1 import AppTest

from test_execution_confirmed_contract import _execution_fixture


def test_app_starts_without_firebase_secrets_and_keeps_tab_ui():
    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    app.run(timeout=20)

    assert not app.exception
    assert len(app.title) == 1
    assert len(app.tabs) == 6       # 4 tabs + Summary and Diagrams inside the Cheat Sheet tab
    # The educational tab remains usable while the live tab explains that
    # Firebase secrets are absent. These counts protect its existing inputs;
    # the Web Apps and Cheat Sheet tabs add a radio, images and buttons, never a
    # number/text input.
    assert len(app.info) >= 1
    assert len(app.number_input) == 6
    assert len(app.text_input) == 1
    # Only the 101 ledger chart renders without Firebase; it is Plotly, not st.line_chart.
    assert len(app.get("plotly_chart")) == 1
    assert len(app.get("vega_lite_chart")) == 0        # st.line_chart renders as this


def test_live_execution_dashboard_renders_with_mocked_read_only_firebase(
        monkeypatch):
    rows = _execution_fixture()
    chain = str(rows.iloc[0]["chain_key"])
    payloads = {
        "webull_lego_rows": {
            str(row["run_id"]): row.to_dict() for _, row in rows.iterrows()
        },
        "webull_lego_state": {
            chain: {"p0": 100.0, "updated_at": "2026-08-01T15:00:00Z"}
        },
        "webull_lego_order_audit": {},
    }

    class _Reference:
        def __init__(self, path):
            self.path = path

        def get(self):
            return payloads.get(self.path)

    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", _Reference)
    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    app.secrets.update({
        "FIREBASE_SA_JSON": "{}",
        "FIREBASE_DB_URL": "https://mock.firebaseio.test",
    })

    app.run(timeout=20)

    assert not app.exception
    # 4 tabs + ข้อมูล/กราฟ inside Live + Summary and Diagrams inside the Cheat Sheet tab
    assert len(app.tabs) == 8
    metric_labels = {item.label for item in app.metric}
    assert {"สถานะตัดสินใจล่าสุด", "DNA step", "Pₙ (USD)",
            "Eₙ สะสม (USD)"}.issubset(metric_labels)
    # The ข้อมูล view opens first and only the open view runs: the 101 ledger is the only chart.
    assert len(app.get("plotly_chart")) == 1
    assert len(app.get("vega_lite_chart")) == 0
    assert any("Execution cashflow ล่าสุด" in item.value for item in app.caption)
    assert any("execution_confirmed_v1" in item.value for item in app.caption)
    # State pointer (this chain) and the 101 tab's ledger render as dataframes. The
    # 17-column table is the hover iframe by default (tooltips and the dataframe
    # fallback: test_ledger_hover.py); Integrity runs only while its expander is open.
    assert len(app.dataframe) >= 2
    assert not app.success

    app.session_state["live_integrity"] = True
    app.run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) >= 3 and app.success

    app.session_state["live_view"] = "กราฟ"
    app.run(timeout=20)
    assert not app.exception
    # Eₙ total over time + the selected chain's ledger + the 101 ledger, all Plotly.
    assert len(app.get("plotly_chart")) == 3
    assert len(app.get("vega_lite_chart")) == 0
    assert not any("ผลรวมทุกตัว" in item.value for item in app.warning)
    assert "DNA step" not in {item.label for item in app.metric}   # the cards belong to ข้อมูล



def test_auth_pause_visible_before_first_row_and_resolved_is_history(monkeypatch):
    warnings = {"auth_failure_example": {"kind": "AUTH_BACKOFF", "active": True,
                                           "count": 3, "retry_after": 9999999999}}
    class Reference:
        def __init__(self, path): self.path = path
        def get(self): return warnings if self.path == "webull_lego_warnings" else {}
    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", Reference)
    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    app.secrets.update({"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL": "https://mock.firebaseio.test"})
    app.run(timeout=20)
    assert not app.exception
    assert any("Authentication" in item.value for item in app.error)
    warnings["auth_failure_example"]["active"] = False
    app.run(timeout=20)
    assert not app.exception
    assert not any("Authentication" in item.value for item in app.error)


def test_hot_deploy_reloads_core_when_system_health_api_is_missing(monkeypatch):
    """A stale sys.modules core must not crash a newer Streamlit entry point."""
    assert hasattr(lego_dash_core, "FROZEN_TERMINAL_SEMANTICS")
    assert hasattr(lego_dash_core, "system_health_rows")

    # Reproduce the mixed-version state seen on Streamlit Cloud: the process has
    # an older core shape cached even though streamlit_app.py is already newer.
    monkeypatch.delattr(lego_dash_core, "system_health_rows")

    app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
    app.run(timeout=20)

    assert not app.exception
    assert callable(getattr(lego_dash_core, "system_health_rows", None))


def test_hot_deploy_reloads_core_that_predates_the_retired_cash_column():
    """A core cached from before cbf5da5 still lists the cash column in COLUMN_ORDER.

    It has every API the older guard checked, so without LEGACY_DROPPED_COLS in the
    guard the stale module is kept and the retired column keeps rendering.
    """
    retired = "ΔAₙ เงินจริง (USD)"
    original_order = list(lego_dash_core.COLUMN_ORDER)
    original_drop = lego_dash_core.LEGACY_DROPPED_COLS
    try:
        # Rebuild the pre-cbf5da5 shape: 18 columns, no legacy-drop API.
        stale_order = original_order[:]
        stale_order.insert(stale_order.index("Aₙ สะสม (USD)"), retired)
        lego_dash_core.COLUMN_ORDER[:] = stale_order
        del lego_dash_core.LEGACY_DROPPED_COLS

        app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
        app.run(timeout=20)

        assert not app.exception
        assert lego_dash_core.LEGACY_DROPPED_COLS == (retired,)
        assert retired not in lego_dash_core.COLUMN_ORDER
    finally:
        lego_dash_core.COLUMN_ORDER[:] = original_order
        lego_dash_core.LEGACY_DROPPED_COLS = original_drop


def test_prod_banner_and_db_url_warning(monkeypatch):
    class Reference:
        def __init__(self, path): self.path = path
        def get(self): return {}
    monkeypatch.setattr(firebase_admin, "_apps", {"test": object()})
    monkeypatch.setattr(firebase_admin.db, "reference", Reference)

    def run(db_url):
        app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
        app.secrets.update({"FIREBASE_SA_JSON": "{}", "FIREBASE_DB_URL": db_url})
        app.run(timeout=20)
        assert not app.exception
        assert "PROD" in app.title[0].value
        assert any("lego-tick-prod" in item.value for item in app.markdown)
        return any("FIREBASE_DB_URL" in item.value for item in app.error)

    assert run("https://lego-firebase-default-rtdb.asia-southeast1.firebasedatabase.app")
    assert not run("https://lego-firebase-prod.asia-southeast1.firebasedatabase.app")
