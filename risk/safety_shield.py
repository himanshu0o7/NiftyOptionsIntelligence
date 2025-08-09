# risk/safety_shield.py
from dataclasses import dataclass
import os, json, time, hashlib, threading
from pathlib import Path

def _env(key, default): 
    v=os.getenv(key, str(default)); 
    try: return int(v) if str(default).isdigit() else float(v)
    except: return v

@dataclass
class GuardConfig:
    run_mode: str = os.getenv("RUN_MODE","ALERT").upper()
    hard_kill: str = os.getenv("HARD_KILL_SWITCH","ON").upper()
    require_tg_confirm: str = os.getenv("REQUIRE_TG_CONFIRM","ON").upper()
    max_loss_day: float = _env("MAX_LOSS_DAY", 1200)
    max_loss_trade: float = _env("MAX_LOSS_TRADE", 350)
    max_positions: int = _env("MAX_POSITIONS", 3)
    dup_window_sec: int = _env("DUPLICATE_WINDOW_SEC", 30)
    hb_timeout_sec: int = _env("HEARTBEAT_TIMEOUT_SEC", 25)

class SafetyShield:
    def __init__(self, state_path=".runtime/state.json", cfg: GuardConfig|None=None):
        self.cfg = cfg or GuardConfig()
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.state = {"orders":[],"pnl_day":0.0,"last_heartbeat":0, "disabled": False}
        self._load()

    def _load(self):
        if self.state_path.exists():
            try: self.state |= json.loads(self.state_path.read_text())
            except: pass

    def _save(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def heartbeat(self):
        with self._lock:
            self.state["last_heartbeat"] = time.time()
            self._save()

    def is_stale(self) -> bool:
        return (time.time() - self.state.get("last_heartbeat",0)) > self.cfg.hb_timeout_sec

    def disable(self, reason:str):
        with self._lock:
            self.state["disabled"]=True
            self.state["disable_reason"]=reason
            self._save()

    def set_pnl_day(self, pnl: float):
        with self._lock:
            self.state["pnl_day"]=float(pnl); self._save()

    def _fingerprint(self, order: dict) -> str:
        # stable hash of essential order fields
        keys = ("symbol","strike","otype","qty","side","price")
        s = "|".join(str(order.get(k,"")) for k in keys)
        return hashlib.sha256(s.encode()).hexdigest()

    def _recent_duplicate(self, fp: str) -> bool:
        now = time.time()
        for o in reversed(self.state["orders"][-20:]):
            if o.get("fp")==fp and (now - o.get("ts",0)) < self.cfg.dup_window_sec:
                return True
        return False

    def _record(self, order: dict, status: str):
        self.state["orders"].append({"fp":self._fingerprint(order), "ts":time.time(), "status":status})
        self.state["orders"]=self.state["orders"][-200:]
        self._save()

    def can_send(self, order: dict, open_positions: int, est_risk_trade: float, require_confirm_cb=None) -> tuple[bool,str]:
        # fail-closed switches
        if self.cfg.hard_kill=="ON" or self.state.get("disabled", False):
            return False, "HARD_KILL active"
        if self.cfg.run_mode=="ALERT":
            return False, "ALERT mode (signals only)"
        if self.is_stale():
            return False, "No heartbeat (WS stale) – disarmed"

        # risk limits
        if self.state.get("pnl_day",0.0) <= -abs(self.cfg.max_loss_day):
            self.disable("Daily loss limit hit"); 
            return False, "Daily circuit breaker"
        if est_risk_trade > self.cfg.max_loss_trade:
            return False, f"Trade risk {est_risk_trade} > MAX_LOSS_TRADE {self.cfg.max_loss_trade}"
        if open_positions >= self.cfg.max_positions:
            return False, "Max positions reached"

        # duplicate suppression
        fp = self._fingerprint(order)
        if self._recent_duplicate(fp):
            return False, "Duplicate order suppressed"

        # human-in-the-loop for LIVE if required
        if self.cfg.require_tg_confirm == "ON" and self.cfg.run_mode=="LIVE" and callable(require_confirm_cb):
            ok = require_confirm_cb(order)
            if not ok: return False, "Telegram confirmation denied/timeout"

        self._record(order, "APPROVED")
        return True, "Approved"

    def after_send(self, order: dict, ok: bool, reason: str):
        self._record(order, "SENT_OK" if ok else f"SENT_FAIL:{reason}")
  
