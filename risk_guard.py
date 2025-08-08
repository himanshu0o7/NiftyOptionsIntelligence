# risk_guard.py (relevant parts only)
from __future__ import annotations
import datetime as dt
from typing import Dict, Any
from logging_setup import configure_logger

class RiskGuard:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log = configure_logger("kp5.risk")
        self._losses = 0
        self._halted = False

    def is_hard_stopped(self) -> bool:
        return self._halted

    def is_entry_cutoff(self) -> bool:
        cutoff = self.cfg["risk_management"]["stop_loss"]["time_based_sl"]["intraday_last_entry"]
        now = dt.datetime.now().astimezone().strftime("%H:%M")
        return now >= cutoff

    def can_enter(self, order: Dict[str, Any]) -> bool:
        if self._halted: return False
        rm = self.cfg["risk_management"]
        if self._losses >= rm.get("session_guards", {}).get("max_losses_halt", 3):
            self._halted = True
            self.log.error("halted_by_max_losses")
            return False
        # daily loss limit check (needs MTM feed)
        return True

    def register_entry(self, result: Dict[str, Any]):
        # update exposure counters etc.
        ...

    def register_sl(self, pnl_abs: float):
        if pnl_abs < 0:
            self._losses += 1
            guards = self.cfg["risk_management"].get("session_guards", {})
            if self._losses >= guards.get("max_losses_halt", 3):
                self._halted = True
                self.log.error("hard_stop_triggered")
