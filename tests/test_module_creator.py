# codex/wrap-file-writing-logic-in-try/except
import builtins
import os
import sys
from pathlib import Path


import module_creator


def test_create_modules_success(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module_creator, "send_telegram_alert", lambda msg: calls.append(msg))
    target_file = tmp_path / "subdir" / "test_mod.py"
    summary = module_creator.create_modules([str(target_file)])
    assert target_file.exists()
    assert summary["created"] == [str(target_file)]
    assert summary["failed"] == []
    assert not calls


def test_create_modules_dir_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module_creator, "send_telegram_alert", lambda msg: calls.append(msg))

    def fake_makedirs(path):
        raise OSError("dir error")

    monkeypatch.setattr(os, "makedirs", fake_makedirs)
    target_file = tmp_path / "subdir" / "test_mod.py"
    summary = module_creator.create_modules([str(target_file)])
    assert not target_file.exists()
    assert summary["created"] == []
    assert summary["failed"][0]["file"] == str(target_file)
    assert summary["failed"][0]["stage"] == "directory"
    assert calls


def test_create_modules_file_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module_creator, "send_telegram_alert", lambda msg: calls.append(msg))

    def fake_open(*args, **kwargs):
        raise OSError("write error")

    monkeypatch.setattr(builtins, "open", fake_open)
    target_file = tmp_path / "test_mod.py"
    summary = module_creator.create_modules([str(target_file)])
    assert not target_file.exists()
    assert summary["created"] == []
    assert summary["failed"][0]["file"] == str(target_file)
    assert summary["failed"][0]["stage"] == "file"
    assert calls

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
# fix-bot-2025-07-24
