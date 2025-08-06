# daily_runner.py

from angel_utils import login, load_nfo_scrip_master
from strategy import check_market_signal

# Login
print("🔐 Logging in...")
client = login()
print("✅ Logged in.")

# Load or refresh scrip master
load_nfo_scrip_master(client, force_refresh=True)

# Strategy check
signal = check_market_signal(client, "NIFTY", 25000, "25JUL2024", "CE")

if signal:
    print("✅ Signal Found:", signal)
else:
    print("🚫 No trade signal today.")

