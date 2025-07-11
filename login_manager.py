# login_manager.py

import pyotp
from SmartApi import SmartConnect
import time

class AngelOneLogin:
    def __init__(self, api_key, client_id, mpin, totp_secret):
        self.api_key = api_key
        self.client_id = client_id
        self.mpin = mpin
        self.totp_secret = totp_secret
        self.connection = SmartConnect(api_key=self.api_key)
        self.session_data = None
        self.login_time = None
        self.token_expiry = 15 * 60  # 15 minutes validity (SmartAPI docs)

    def _generate_totp(self):
        return pyotp.TOTP(self.totp_secret).now()

    def _is_token_expired(self):
        return not self.login_time or (time.time() - self.login_time > self.token_expiry)

    def login(self, force_refresh=False):
        if self.session_data and not force_refresh and not self._is_token_expired():
            return self._get_token_payload()
        try:
            totp = self._generate_totp()
            self.session_data = self.connection.generateSession(self.client_id, self.mpin, totp)
            if not self.session_data.get('status', False):
                raise Exception(f"Login failed: {self.session_data.get('message', 'Unknown error')}")
            self.login_time = time.time()
            return self._get_token_payload()
        except Exception as e:
            raise Exception(f"Login Error: {e}")

    def _get_token_payload(self):
        return {
            "clientcode": self.session_data["data"]["clientcode"],
            "jwtToken": self.connection.access_token,
            "refreshToken": self.session_data["data"]["refreshToken"],
            "feedToken": self.connection.getfeedToken(),
            "api_key": self.api_key
        }

