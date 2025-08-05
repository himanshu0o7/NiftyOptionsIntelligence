"""Thin wrapper to expose the Streamlit strategy configuration page.

This wrapper exists to maintain backward compatibility. It delegates to
``pages.strategy_config.show_strategy_config`` and reports any runtime errors
through Telegram alerts.
"""

from pages.strategy_config import show_strategy_config
from telegram_alerts import send_telegram_alert


def main() -> None:
    """Run the strategy configuration page with basic error handling."""
    try:
        show_strategy_config()
    except Exception as exc:  # noqa: BLE001
        send_telegram_alert(f"[strategy_config] {exc}")
        raise


if __name__ == "__main__":
    main()
