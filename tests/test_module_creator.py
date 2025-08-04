import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import module_creator


def test_create_modules_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    alerts = []
    monkeypatch.setattr(module_creator, "send_telegram_alert", lambda msg: alerts.append(msg))

    module_creator.create_modules(["foo.py"])

    assert os.path.exists("foo.py")
    assert alerts == []


def test_create_modules_write_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    alerts = []

    def fake_alert(message):
        alerts.append(message)

    def fail_open(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(module_creator, "send_telegram_alert", fake_alert)
    monkeypatch.setattr("builtins.open", fail_open)

    module_creator.create_modules(["bad.py"])

    assert alerts and "bad.py" in alerts[0]
    assert not os.path.exists("bad.py")
