"""Self-learning viewer for evolution log."""

import json
from typing import List

from telegram_alerts import send_telegram_alert

EVOLVE_LOG_PATH = "evolve_log.json"


def load_evolution_records(path: str = EVOLVE_LOG_PATH) -> List[dict]:
    """Load evolution records from ``path``.

    Returns an empty list if the file is missing, empty, or contains
    invalid JSON.  Telegram alerts are sent on load errors.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            send_telegram_alert(
                "self_learning_viewer: evolve log format invalid, expected list"
            )
            return []
    except FileNotFoundError:
        send_telegram_alert("self_learning_viewer: evolve log not found")
        return []
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        send_telegram_alert(
            f"self_learning_viewer: JSON parse error in evolve log - {exc}"
        )
        return []
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(
            f"self_learning_viewer: unexpected error loading evolve log - {exc}"
        )
        return []


def main() -> None:
    records = load_evolution_records()
    if records:
        print(f"Loaded {len(records)} evolution record(s)")
    else:
        print("No evolution records found.")


if __name__ == "__main__":
    main()
