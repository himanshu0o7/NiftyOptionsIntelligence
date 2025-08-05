import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any

from telegram_alerts import send_telegram_alert

MODULE_NAME = "memory_manager"


def append_to_json_log(log_data: Dict[str, Any], log_file: str = "logs/strategy_log.json", max_bytes: int = 5_000_000) -> None:
    """Append a dictionary to a JSON log file with simple rotation."""
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        entries = []
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        entries = loaded
                except json.JSONDecodeError:
                    entries = []

        entries.append(log_data)
        data_str = json.dumps(entries)
        if len(data_str.encode("utf-8")) > max_bytes:
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            rotated = f"{os.path.splitext(log_file)[0]}_{timestamp}.json"
            if os.path.exists(log_file):
                os.rename(log_file, rotated)
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump([log_data], f)
        else:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(data_str)
    except Exception as exc:  # pragma: no cover - alert on failure
        send_telegram_alert(f"{MODULE_NAME} append_to_json_log error: {exc}")


def append_to_sqlite_log(table: str, log_data: Dict[str, Any], db_file: str = "logs/memory_log.db", max_rows: int = 10000) -> None:
    """Insert log data into SQLite table with rotation on row count."""
    try:
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        cur.execute(
            f"INSERT INTO {table} (timestamp, data) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), json.dumps(log_data)),
        )
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        if count > max_rows:
            to_delete = count - max_rows
            cur.execute(
                f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY id ASC LIMIT ?)",
                (to_delete,),
            )
            conn.commit()
    except Exception as exc:  # pragma: no cover - alert on failure
        send_telegram_alert(f"{MODULE_NAME} append_to_sqlite_log error: {exc}")
    finally:
        if 'conn' in locals():
            conn.close()


def cleanup_old_logs(log_dir: str = "logs", retention_days: int = 7) -> None:
    """Remove log files older than ``retention_days`` days."""
    try:
        if not os.path.isdir(log_dir):
            return
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        for name in os.listdir(log_dir):
            path = os.path.join(log_dir, name)
            try:
                mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                if mtime < cutoff:
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        for root, _dirs, files in os.walk(path, topdown=False):
                            for f in files:
                                os.remove(os.path.join(root, f))
                            os.rmdir(root)
            except Exception as inner_exc:
                send_telegram_alert(f"{MODULE_NAME} cleanup file error: {inner_exc}")
    except Exception as exc:  # pragma: no cover - alert on failure
        send_telegram_alert(f"{MODULE_NAME} cleanup_old_logs error: {exc}")
