import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evolution_viewer import load_evolution_records


def test_load_evolution_records_valid(tmp_path):
    data = [{"test": 1}]
    path = tmp_path / "evolve_log.json"
    path.write_text(json.dumps(data))
    assert load_evolution_records(path) == data


def test_load_evolution_records_missing(monkeypatch, tmp_path):
    messages = []

    def fake_alert(msg):
        messages.append(msg)

    monkeypatch.setattr(
        "evolution_viewer.send_telegram_alert", fake_alert
    )
    result = load_evolution_records(tmp_path / "missing.json")
    assert result == []
    assert any("not found" in m for m in messages)


def test_load_evolution_records_empty_file(tmp_path):
    path = tmp_path / "evolve_log.json"
    path.write_text("")
    assert load_evolution_records(path) == []
