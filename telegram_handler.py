import requests
import os
from dotenv import load_dotenv

load_dotenv()  # ✅ This loads the .env variables

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_alert(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram credentials not set")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        print("✅ Telegram alert sent")
    except Exception as e:
        print(f"❌ Telegram alert failed: {e}")
