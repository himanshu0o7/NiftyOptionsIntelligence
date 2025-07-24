# telegram_alerts.py

import os
import requests

# Make sure these are set in your environment or .env file
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # e.g., '123456789:ABCdefGHIjklMNOpqrSTUvwxYZ'
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # e.g., '@yourchannel' or user ID

def send_telegram_alert(message: str):
    """Send alert message via Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not set in environment.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📩 Telegram alert sent.")
        else:
            print(f"⚠️ Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending Telegram alert: {e}")

# Example test
if __name__ == "__main__":
    send_telegram_alert("🚨 *Test Alert* from KP5Bot.")

