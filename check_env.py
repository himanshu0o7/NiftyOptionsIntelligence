import os
from dotenv import load_dotenv

load_dotenv()  # Load .env

required_vars = [
    "ANGEL_API_KEY", "CLIENT_ID", "MPIN", "TOTP_SECRET",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"
]

print("🔍 Checking environment variables...\n")
for var in required_vars:
    value = os.getenv(var)
    if not value:
        print(f"❌ {var} is MISSING!")
    else:
        print(f"✅ {var} is set.")
