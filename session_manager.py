import os
import json

SESSION_FILE = "session_tokens.json"

def load_session():
    if not os.path.exists(SESSION_FILE):
        print("❌ Session file not found.")
        return None

    with open(SESSION_FILE) as f:
        session = json.load(f)

    required_keys = ["jwtToken", "feedToken"]
    if all(k in session for k in required_keys):
        print("✅ Session loaded.")
        return session
    else:
        print("❌ Incomplete session data.")
        return None

