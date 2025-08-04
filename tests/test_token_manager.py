# codex/wrap-requests.get-in-try/except
from pathlib import Path
import sys

import pytest
from requests.exceptions import RequestException

sys.path.append(str(Path(__file__).resolve().parents[1]))
import token_manager


class DummyResponse:
    status_code = 200

    def json(self):
        return {"data": 1}

    def raise_for_status(self):
        pass


def test_download_scrip_master_success(tmp_path, monkeypatch):
    temp_file = tmp_path / "scrip.json"
    monkeypatch.setattr(token_manager, "LOCAL_SCRIP_FILE", str(temp_file))
    monkeypatch.setattr(token_manager.requests, "get", lambda *a, **k: DummyResponse())
    messages = []
    monkeypatch.setattr(token_manager, "send_telegram_alert", lambda m: messages.append(m))

    assert token_manager.download_scrip_master()
    assert temp_file.exists()
    assert messages == []


def test_download_scrip_master_failure(tmp_path, monkeypatch):
    temp_file = tmp_path / "scrip.json"
    monkeypatch.setattr(token_manager, "LOCAL_SCRIP_FILE", str(temp_file))

    def fake_get(*args, **kwargs):
        raise RequestException("offline")

    monkeypatch.setattr(token_manager.requests, "get", fake_get)
    messages = []
    monkeypatch.setattr(token_manager, "send_telegram_alert", lambda m: messages.append(m))

    assert not token_manager.download_scrip_master(retries=1)
    assert messages and "Failed to download scrip master" in messages[0]
    assert not temp_file.exists()
 
import os
import json
import builtins

import pytest
import requests

import token_manager


def _make_dummy_response(data):
    class DummyResponse:
        status_code = 200

        def json(self):
            return data

        def raise_for_status(self):
            pass

    return DummyResponse()


def test_download_scrip_master_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_get(url, timeout=10):
        return _make_dummy_response({"hello": "world"})

    alerts = []

    def fake_alert(message):
        alerts.append(message)

    monkeypatch.setattr(token_manager.requests, "get", fake_get)
    monkeypatch.setattr(token_manager, "send_telegram_alert", fake_alert)

    assert token_manager.download_scrip_master()
    assert os.path.exists(token_manager.LOCAL_SCRIP_FILE)
    assert alerts == []



def test_download_scrip_master_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_get(url, timeout=10):
        raise requests.exceptions.RequestException("boom")

    alerts = []

    def fake_alert(message):
        alerts.append(message)

    monkeypatch.setattr(token_manager.requests, "get", fake_get)
    monkeypatch.setattr(token_manager, "send_telegram_alert", fake_alert)

    result = token_manager.download_scrip_master(retries=2)
    assert result is False
    assert alerts  # ensure alert was triggered
    assert not os.path.exists(token_manager.LOCAL_SCRIP_FILE)


def test_load_scrip_data_uses_cache(monkeypatch, tmp_path):
    token_manager.clear_cache()
    data = [
        {"symbol": "ABC", "exchange": "NFO", "instrumenttype": "OPTIDX"}
    ]
    path = tmp_path / "scrip_master.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(token_manager, "LOCAL_SCRIP_FILE", str(path))

    download_calls = {"count": 0}

    def fake_download():
        download_calls["count"] += 1
        return True

    monkeypatch.setattr(token_manager, "download_scrip_master", fake_download)

    original_open = builtins.open
    open_calls = {"count": 0}

    def fake_open(*args, **kwargs):
        open_calls["count"] += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    df1 = token_manager.load_scrip_data()
    df2 = token_manager.load_scrip_data()

    assert download_calls["count"] == 1
    assert open_calls["count"] == 1
    assert df1 is df2


def test_clear_cache_resets(monkeypatch, tmp_path):
    token_manager.clear_cache()
    path = tmp_path / "scrip_master.json"
    path.write_text(json.dumps([]))
    monkeypatch.setattr(token_manager, "LOCAL_SCRIP_FILE", str(path))
    monkeypatch.setattr(token_manager, "download_scrip_master", lambda: True)

    token_manager.load_scrip_data()
    assert token_manager._scrip_data_cache is not None
    token_manager.clear_cache()
    assert token_manager._scrip_data_cache is None
# fix-bot-2025-07-24
