"""Tests for the RiskGuard module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import risk_guard


def _setup_guard(monkeypatch):
    monkeypatch.setattr(risk_guard.RiskGuard, "_get_current_commit", lambda self: "good")
    guard = risk_guard.RiskGuard()
    monkeypatch.setattr(guard, "_check_file_churn", lambda: None)
    monkeypatch.setattr(guard, "_check_log_size", lambda: None)
    return guard


def test_cpu_breach_triggers_alert_and_rollback(monkeypatch):
    alerts = []
    monkeypatch.setattr(risk_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(risk_guard.psutil, "cpu_percent", lambda interval=None: 100)
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)

    monkeypatch.setattr(risk_guard.subprocess, "run", fake_run)
    guard = _setup_guard(monkeypatch)
    guard.cpu_threshold = 10
    with pytest.raises(RuntimeError):
        guard.monitor()
    assert alerts
    assert calls and calls[0][:3] == ["git", "reset", "--hard"]


def test_log_size_breach(monkeypatch, tmp_path):
    alerts = []
    monkeypatch.setattr(risk_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(risk_guard.psutil, "cpu_percent", lambda interval=None: 0)
    monkeypatch.setattr(risk_guard.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(risk_guard.RiskGuard, "_get_current_commit", lambda self: "good")

    log_file = tmp_path / "test.log"
    log_file.write_bytes(b"x" * 20)

    guard = risk_guard.RiskGuard(log_size_threshold=10, log_path=str(log_file))
    monkeypatch.setattr(guard, "_check_file_churn", lambda: None)
    with pytest.raises(RuntimeError):
        guard.monitor()
    assert alerts


def test_custom_check_breach(monkeypatch):
    alerts = []
    monkeypatch.setattr(risk_guard, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(risk_guard.psutil, "cpu_percent", lambda interval=None: 0)
    monkeypatch.setattr(risk_guard.subprocess, "run", lambda *args, **kwargs: None)
    guard = _setup_guard(monkeypatch)

    def failing_check():
        return False, "failure"

    guard.register_check("fail", failing_check)
    with pytest.raises(RuntimeError):
        guard.monitor()
    assert alerts and "failure" in alerts[0]

