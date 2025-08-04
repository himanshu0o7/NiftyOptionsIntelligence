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
