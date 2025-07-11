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
        self.smartconnect = SmartConnect(api_key=api_key)
        self.session_data = None
        self.feed_token = None
        self.jwt_token = None
        self.last_login = 0
        self.TOKEN_LIFE = 14 * 60   # 14 minutes

    def fresh_login(self):
        totp = pyotp.TOTP(self.totp_secret).now()
        self.session_data = self.smartconnect.generateSession(self.client_id, self.mpin, totp)
        if not self.session_data or "data" not in self.session_data:
            raise Exception("Login failed! Check credentials or TOTP.")
        self.jwt_token = self.session_data["data"]["jwtToken"]
        self.feed_token = self.smartconnect.getfeedToken()
        self.last_login = time.time()
        return {
            "jwtToken": self.jwt_token,
            "feedToken": self.feed_token,
            "api_key": self.api_key,
            "clientcode": self.client_id
        }

    def ensure_fresh(self):
        if not self.jwt_token or (time.time() - self.last_login) > self.TOKEN_LIFE:
            return self.fresh_login()
        else:
            return {
                "jwtToken": self.jwt_token,
                "feedToken": self.feed_token,
                "api_key": self.api_key,
                "clientcode": self.client_id
            }
