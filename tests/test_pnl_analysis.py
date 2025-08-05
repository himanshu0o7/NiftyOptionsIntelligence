import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import Database
from utils.success_rate_tracker import SuccessRateTracker
from pages import pnl_analysis


def setup_test_db(tmp_path):
    db_path = tmp_path / "test_trading_data.db"
    db = Database(str(db_path))
    SuccessRateTracker(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO daily_summary
        (date, total_trades, winning_trades, losing_trades, win_rate,
         total_pnl, avg_win, avg_loss, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("2025-07-01", 2, 1, 1, 50, 1000, 1500, -500, -500),
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO daily_summary
        (date, total_trades, winning_trades, losing_trades, win_rate,
         total_pnl, avg_win, avg_loss, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("2025-07-02", 1, 1, 0, 100, 2000, 2000, 0, -200),
    )

    cursor.execute(
        """
        INSERT INTO positions
        (symbol, token, product_type, exchange, quantity, avg_price, current_price,
         pnl, unrealized_pnl, realized_pnl, strategy_name, opened_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "NIFTY",
            "123",
            "MIS",
            "NSE",
            1,
            100,
            100,
            300,
            300,
            0,
            "strat",
            "2025-07-02",
            "OPEN",
        ),
    )

    conn.commit()
    conn.close()
    return db


def test_get_pnl_metrics(tmp_path, monkeypatch):
    db = setup_test_db(tmp_path)
    monkeypatch.setattr(pnl_analysis, "Database", lambda: db)
    monkeypatch.setattr(pnl_analysis, "send_telegram_alert", lambda msg: None)

    metrics = pnl_analysis.get_pnl_metrics("Last 3 Months")

    assert metrics["total_pnl"] == 3300.0
    assert metrics["realized_pnl"] == 3000.0
    assert metrics["unrealized_pnl"] == 300.0
    assert metrics["avg_trade_pnl"] == 1000.0
    assert metrics["win_rate"] == pytest.approx(66.67, rel=1e-2)


def test_get_daily_pnl_data(tmp_path, monkeypatch):
    db = setup_test_db(tmp_path)
    monkeypatch.setattr(pnl_analysis, "Database", lambda: db)
    monkeypatch.setattr(pnl_analysis, "send_telegram_alert", lambda msg: None)

    df = pnl_analysis.get_daily_pnl_data("2025-07-01", "2025-07-02")

    assert not df.empty
    assert list(df["daily_pnl"]) == [1000, 2000]
    assert list(df["trades_count"]) == [2, 1]


def test_get_performance_trends_data(tmp_path, monkeypatch):
    db = setup_test_db(tmp_path)
    monkeypatch.setattr(pnl_analysis, "Database", lambda: db)
    monkeypatch.setattr(pnl_analysis, "send_telegram_alert", lambda msg: None)

    df = pnl_analysis.get_performance_trends_data("Last 3 Months")

    assert "cumulative_pnl" in df.columns
    assert df.iloc[-1]["cumulative_pnl"] == 3000

