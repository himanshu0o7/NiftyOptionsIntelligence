import json

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
