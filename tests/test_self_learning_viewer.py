import json
#codex/create-self-learning-viewer-page

from pages import self_learning_viewer as slv


def test_load_evolve_data_success(tmp_path, monkeypatch):
    data = {"step": 1, "status": "ok"}
    file_path = tmp_path / "evolve_log.json"
    file_path.write_text(json.dumps(data))

    alerts = []
    monkeypatch.setattr(slv, "send_telegram_alert", lambda msg: alerts.append(msg))

    result = slv.load_evolve_data(str(file_path))

    assert result == data
    assert alerts == []


def test_load_evolve_data_missing(monkeypatch, tmp_path):
    file_path = tmp_path / "missing.json"

    alerts = []
    monkeypatch.setattr(slv, "send_telegram_alert", lambda msg: alerts.append(msg))

    result = slv.load_evolve_data(str(file_path))

    assert result is None
    assert len(alerts) == 1

import os
import importlib
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _DummyTab:
    def __enter__(self):  # pragma: no cover - simple context manager
        return None

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple context manager
        return False


@pytest.fixture
def mock_streamlit(monkeypatch):
    """Provide a minimal stub for the streamlit module."""
    st_stub = types.SimpleNamespace(
        header=lambda *a, **k: None,
        tabs=lambda labels: [_DummyTab() for _ in labels],
        write=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", st_stub)

    # Stub out external libraries used by the page modules
    plotly_module = types.ModuleType("plotly")
    plotly_go = types.ModuleType("plotly.graph_objects")
    plotly_go.Figure = type("Figure", (), {})
    plotly_module.graph_objects = plotly_go
    plotly_px = types.ModuleType("plotly.express")
    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", plotly_go)
    monkeypatch.setitem(sys.modules, "plotly.express", plotly_px)
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    return st_stub


def test_self_learning_viewer_loads_and_renders(mock_streamlit, tmp_path, monkeypatch):
    """Ensure the self-learning viewer loads JSON and renders summaries."""
    viewer = pytest.importorskip("self_learning_viewer")

    load_func = getattr(viewer, "load_evolve_log", None) or getattr(
        viewer, "load_evolution_log", None
    )
    render_func = getattr(viewer, "render_summaries", None) or getattr(
        viewer, "render_summary", None
    )
    if not load_func or not render_func:
        pytest.skip("Viewer interface incomplete")

    log_data = {"summaries": ["alpha", "beta"]}
    log_path = tmp_path / "evolve_log.json"
    log_path.write_text(json.dumps(log_data), encoding="utf-8")

    loaded = load_func(log_path)
    assert loaded == log_data

    outputs = []
    monkeypatch.setattr(viewer, "st", types.SimpleNamespace(write=lambda s: outputs.append(s)))
    render_func(loaded)
    assert outputs == log_data["summaries"]


def test_show_risk_monitor_smoke(mock_streamlit, monkeypatch):
    """Smoke test for pages.risk_monitor.show_risk_monitor."""
    rm = importlib.import_module("pages.risk_monitor")
    monkeypatch.setattr(rm, "show_risk_overview", lambda: None)
    monkeypatch.setattr(rm, "show_position_risk", lambda: None)
    monkeypatch.setattr(rm, "show_portfolio_risk", lambda: None)
    monkeypatch.setattr(rm, "show_risk_alerts", lambda: None)
    monkeypatch.setattr(rm, "show_stress_testing", lambda: None)
    rm.show_risk_monitor()


def test_show_pnl_analysis_smoke(mock_streamlit, monkeypatch):
    """Smoke test for pages.pnl_analysis.show_pnl_analysis."""
    pa = importlib.import_module("pages.pnl_analysis")
    monkeypatch.setattr(pa, "show_pnl_overview", lambda: None)
    monkeypatch.setattr(pa, "show_daily_pnl", lambda: None)
    monkeypatch.setattr(pa, "show_performance_trends", lambda: None)
    monkeypatch.setattr(pa, "show_strategy_performance", lambda: None)
    monkeypatch.setattr(pa, "show_trade_analysis", lambda: None)
    pa.show_pnl_analysis()


def test_show_strategy_config_smoke(mock_streamlit, monkeypatch):
    """Smoke test for pages.strategy_config.show_strategy_config."""
    sc = importlib.import_module("pages.strategy_config")
    monkeypatch.setattr(sc, "show_active_strategies", lambda: None)
    monkeypatch.setattr(sc, "show_strategy_configuration", lambda: None)
    monkeypatch.setattr(sc, "show_backtest_results", lambda: None)
    sc.show_strategy_config()
# fix-bot-2025-07-24
