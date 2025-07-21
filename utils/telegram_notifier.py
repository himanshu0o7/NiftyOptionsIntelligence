# utils/telegram_notifier.py

import os
import sys
import requests
from dotenv import load_dotenv

# Ensure current working directory is in sys.path for consistent module resolution
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    from utils.trend_detector import detect_trend
except ModuleNotFoundError:
    print("❌ Failed to import detect_trend from utils.trend_detector. Ensure trend_detector.py exists in utils/ directory.")
    raise

# Load environment variables from .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_trend_alert(symbol="NIFTY", expiry="25JUL2025"):
    result = detect_trend(symbol, expiry)

    message = (
        f"\U0001F4C8 *Trend Alert for {symbol} ({expiry})*\n"
        f"\U0001F50D *Trend:* {result['trend']}\n"
        f"\U0001F4DA *Reason:* {result['reason']}\n"
        f"\U0001F4CA *CE ∆:* {result['supporting_data'].get('ce_delta')}, "
        f"CE OI: {result['supporting_data'].get('ce_oi_change')}\n"
        f"\U0001F4C9 *PE ∆:* {result['supporting_data'].get('pe_delta')}, "
        f"PE OI: {result['supporting_data'].get('pe_oi_change')}"
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ Telegram alert sent.")
        else:
            print(f"❌ Failed to send alert: {response.text}")
    except Exception as e:
        print(f"❌ Error sending alert: {e}")

