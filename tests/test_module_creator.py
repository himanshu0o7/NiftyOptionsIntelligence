import builtins
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
