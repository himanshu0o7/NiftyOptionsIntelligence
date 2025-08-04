"""Tests for self_audit module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import self_audit


class DummyAlert:
    """Collect messages instead of sending Telegram alerts."""

    def __init__(self):
        self.messages = []

    def __call__(self, message: str):
        self.messages.append(message)


def test_reports_missing_module(tmp_path, monkeypatch):
    target = tmp_path / "mod.py"
    target.write_text("import nonexistent_module\n")
    dummy = DummyAlert()
    monkeypatch.setattr(self_audit, "send_telegram_alert", dummy)

    result = self_audit.scan_repository(base_path=str(tmp_path))

    assert any(entry["module"] == "nonexistent_module" for entry in result["missing_modules"])
    assert dummy.messages  # alert triggered


def test_handles_invalid_syntax(tmp_path, monkeypatch):
    target = tmp_path / "bad.py"
    target.write_text("def broken(:\n    pass")
    dummy = DummyAlert()
    monkeypatch.setattr(self_audit, "send_telegram_alert", dummy)

    result = self_audit.scan_repository(base_path=str(tmp_path))

    assert result["errors"]
    assert dummy.messages  # alert triggered
