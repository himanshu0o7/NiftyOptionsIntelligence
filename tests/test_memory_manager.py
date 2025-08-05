import json
import os
import sqlite3
from datetime import datetime, timedelta

import memory_manager


def test_append_to_json_log_and_rotation(monkeypatch, tmp_path):
    alerts = []
    monkeypatch.setattr(memory_manager, "send_telegram_alert", lambda msg: alerts.append(msg))

    log_file = tmp_path / "test.json"
    memory_manager.append_to_json_log({"a": 1}, log_file=str(log_file), max_bytes=100)
    with open(log_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == [{"a": 1}]
    assert alerts == []

    # force rotation with large entry
    memory_manager.append_to_json_log({"b": "x" * 200}, log_file=str(log_file), max_bytes=100)
    files = list(tmp_path.iterdir())
    assert len(files) == 2  # rotated file + new log
    assert alerts == []


def test_append_to_sqlite_log_and_rotation(monkeypatch, tmp_path):
    alerts = []
    monkeypatch.setattr(memory_manager, "send_telegram_alert", lambda msg: alerts.append(msg))

    db_file = tmp_path / "test.db"
    table = "logs"
    for i in range(4):
        memory_manager.append_to_sqlite_log(table, {"i": i}, db_file=str(db_file), max_rows=3)

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    assert count == 3
    cur.execute(f"SELECT data FROM {table} ORDER BY id ASC")
    rows = [json.loads(r[0]) for r in cur.fetchall()]
    assert rows == [{"i": 1}, {"i": 2}, {"i": 3}]
    conn.close()
    assert alerts == []


def test_cleanup_old_logs(monkeypatch, tmp_path):
    alerts = []
    monkeypatch.setattr(memory_manager, "send_telegram_alert", lambda msg: alerts.append(msg))

    old_file = tmp_path / "old.log"
    new_file = tmp_path / "new.log"
    old_file.write_text("old")
    new_file.write_text("new")
    old_time = datetime.now() - timedelta(days=8)
    os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))

    memory_manager.cleanup_old_logs(log_dir=str(tmp_path), retention_days=7)
    assert not old_file.exists()
    assert new_file.exists()
    assert alerts == []
