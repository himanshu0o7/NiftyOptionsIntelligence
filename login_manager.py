# login_manager.py
# Updated with token refresh using generateTokens, TOTP handling, and basic rate limit delay.

import os
import pyotp
from SmartApi import SmartConnect
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PIN = os.getenv("ANGEL_PIN")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

class AngelLoginManager:
    def __init__(self):
        self._validate_env()
        self.smartconnect = SmartConnect(api_key=API_KEY)
        self.session_data = None
        self.last_login = 0
        self.TOKEN_LIFE = 14 * 60  # Initial check; actual session ~28 hours, but refresh proactively

    def _validate_env(self):
        missing_vars = [var for var in ["ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET"] if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"❌ Missing environment variables: {', '.join(missing_vars)}")

    def generate_totp(self):
        try:
            totp = pyotp.TOTP(TOTP_SECRET)
            return totp.now()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to generate TOTP: {e}")

    def fresh_login(self):
        try:
            time.sleep(1)  # Rate limit delay
            totp = self.generate_totp()
            self.session_data = self.smartconnect.generateSession(
                clientCode=CLIENT_ID,  # Correct arg from SDK
                password=PIN,
                totp=totp
            )
            if not self.session_data['status']:
                raise ValueError(f"❌ Login failed: {self.session_data['message']}")
            self.last_login = time.time()
            return self.session_data
        except Exception as e:
            raise ConnectionError(f"❌ Login attempt failed: {e}")

    def ensure_fresh(self):
        if not self.session_data or (time.time() - self.last_login > self.TOKEN_LIFE):
            if self.session_data and 'refreshToken' in self.session_data['data']:
                try:
                    time.sleep(1)  # Delay for rate limit
                    refresh_resp = self.smartconnect.generateTokens(self.session_data['data']['refreshToken'])
                    if refresh_resp['status']:
                        self.session_data = refresh_resp
                        self.last_login = time.time()
                        return self.session_data
                except Exception as e:
                    print(f"Token refresh failed: {e}. Falling back to fresh login.")
            return self.fresh_login()
        return self.session_data

# ✅ Example:
# login = AngelLoginManager()
# session = login.ensure_fresh()
# smart = login.smartconnect

