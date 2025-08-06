# session_manager.py
# Updated expiry to 28 hours from docs, cache includes refreshToken, added error handling and delays.

import os
import json
import time
from login_manager import AngelLoginManager
from telegram_alerts import send_telegram_alert

SESSION_CACHE_FILE = "session_cache.json"

class SessionManager:
    def __init__(self):
        self.login = AngelLoginManager()
        self.smartconnect = self.login.smartconnect
        self.session_data = None
        self.last_login_time = 0
        self.expiry_duration = 28 * 3600  # 28 hours from SmartAPI docs

    def _save_to_cache(self):
        if self.session_data is None:
            return
        cache_data = {
            "session_data": self.session_data,
            "last_login_time": self.last_login_time
        }
        try:
            with open(SESSION_CACHE_FILE, "w") as f:
                json.dump(cache_data, f)
            os.chmod(SESSION_CACHE_FILE, 0o600)  # Secure permissions
# codex/replace-json.jsonencodeerror-exceptions
        except (IOError, TypeError, OverflowError, ValueError) as e:
            msg = f"[SessionManager] Error saving cache: {e}"
            print(msg)
            send_telegram_alert(msg)

        except (IOError, TypeError, OverflowError) as e:
            error_msg = f"session_manager: Error saving cache: {e}"
            print(error_msg)
            send_telegram_alert(f"⚠️ {error_msg}")
# fix-bot-2025-07-24

    def _load_from_cache(self):
        if not os.path.exists(SESSION_CACHE_FILE):
            return None
        try:
            with open(SESSION_CACHE_FILE, "r") as f:
                cache_data = json.load(f)
            if "session_data" in cache_data and "last_login_time" in cache_data:
                return cache_data
            else:
                msg = "[SessionManager] Invalid cache format"
                print(msg)
                send_telegram_alert(msg)
                return None
        except (IOError, json.JSONDecodeError) as e:
            msg = f"[SessionManager] Error loading cache: {e}"
            print(msg)
            send_telegram_alert(msg)
            return None

    def is_expired(self):
        return time.time() - self.last_login_time > self.expiry_duration

    def get_session(self):
        if self.session_data and not self.is_expired():
            return self.session_data

        cached = self._load_from_cache()
        if cached:
            self.session_data = cached["session_data"]
            self.last_login_time = cached["last_login_time"]
            if not self.is_expired():
                return self.session_data

        try:
            time.sleep(1)  # Rate limit delay
            self.session_data = self.login.fresh_login()
            self.last_login_time = time.time()
            self._save_to_cache()
            return self.session_data
        except Exception as e:
            raise RuntimeError(f"Failed to get session: {e}")

# ✅ Example Usage:
# sm = SessionManager()
# session = sm.get_session()
# smart = sm.smartconnect

