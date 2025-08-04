import os

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
