"""Thin wrapper to expose the Streamlit strategy configuration page.

This wrapper exists to maintain backward compatibility. It delegates to
``pages.strategy_config.show_strategy_config`` and reports any runtime errors
through Telegram alerts.
"""

# Dependency: This module requires `pages.strategy_config` and its `show_strategy_config` function.
from telegram_alerts import send_telegram_alert
try:
    from pages.strategy_config import show_strategy_config
except (ImportError, ModuleNotFoundError) as exc:
    send_telegram_alert(f"[strategy_config] Import error: {exc}")
    raise
def main() -> None:
    """Run the strategy configuration page with basic error handling."""
    try:
        show_strategy_config()
    except Exception as exc:  # noqa: BLE001
        send_telegram_alert(f"[strategy_config] {exc}")
        raise


if __name__ == "__main__":
    main()
