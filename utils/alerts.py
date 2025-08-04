import os
import logging
import requests
from datetime import datetime
from config.settings import Settings

# Load settings from env
_settings = Settings()

TELEGRAM_BOT_TOKEN = _settings.telegram_bot_token
TELEGRAM_CHAT_ID = _settings.telegram_chat_id

_alert_logger = logging.getLogger("alerts")


def send_alert(message: str) -> None:
    """
    Sends a message to the configured Telegram channel.

    Args:
        message (str): The message to send.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        _alert_logger.warning("Telegram credentials not set. Alert not sent.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code != 200:
            _alert_logger.error(
                f"Telegram API error {response.status_code}: {response.text}"
            )
        else:
            _alert_logger.info("Alert sent to Telegram.")
    except Exception as e:
        _alert_logger.exception(f"Error sending alert: {e}")


