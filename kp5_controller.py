# kp5_controller.py
from __future__ import annotations
import time, signal, threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

from logging_setup import configure_logger
from risk_guard import RiskGuard
from order_executor import OrderExecutor
from live_stream_handler import LiveStreamHandler
from scrip_master_utils import resolve_atm_tokens
from utils.config import load_config

@dataclass
class ControllerState:
    running: bool = False
    last_heartbeat_ts: float = 0.0
    open_positions: int = 0
    losses_today: int = 0

class KP5Controller:
    """
    High-level orchestration for: login -> stream -> filter -> route orders -> monitor risk.
    All actions are fail-closed: on any critical error, stop new orders and square-off if needed.
    """
    def __init__(self, cfg: Dict[str, Any], heartbeat_sec: int = 5):
        self.log = configure_logger("kp5.controller")
        self.cfg = cfg
        self.state = ControllerState(running=False)
        self.heartbeat_sec = heartbeat_sec

        self.risk = RiskGuard(cfg)
        self.router = OrderExecutor(cfg)
        self.stream = LiveStreamHandler(cfg, on_snapshot=self.on_snapshot, on_disconnect=self.on_disconnect)

        self._hb_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # === Lifecycle ===
    def start(self):
        self.log.info("controller_start", extra={"capital": self.cfg["risk_management"]["total_capital"]})
        self.state.running = True
        self._wire_signals()

        # login + master contract preload + tokens
        self._bootstrap()

        # start streams
        self.stream.connect()
        self._start_heartbeat()

    def stop(self, reason: str = "manual"):
        if not self.state.running:
            return
        self.log.warning("controller_stop", extra={"reason": reason})
        self._stop_event.set()
        try:
            self.stream.disconnect()
        finally:
            self.router.graceful_square_off_all(reason="controller_stop")
            self.state.running = False

    def _wire_signals(self):
        def _graceful(*_):
            self.stop(reason="signal")
        signal.signal(signal.SIGINT, _graceful)
        signal.signal(signal.SIGTERM, _graceful)

    def _bootstrap(self):
        try:
            # Preload/validate instruments, resolve ATM CE/PE tokens
            atm = resolve_atm_tokens(self.cfg)
            self.log.info("bootstrap_resolved_tokens", extra={"atm": atm})
            self.router.set_active_symbols(atm)
        except Exception as e:
            self.log.exception("bootstrap_failed", extra={"error": str(e)})
            raise

    def _start_heartbeat(self):
        def _run():
            while not self._stop_event.wait(self.heartbeat_sec):
                self.state.last_heartbeat_ts = time.time()
                # Ops health + tunnel status if configured
                self.log.info("heartbeat", extra={
                    "open_positions": self.router.count_open_positions(),
                    "risk_hard_stop": self.risk.is_hard_stopped(),
                    "stream_alive": self.stream.is_alive(),
                })
                # Fail-closed on feed gap
                if not self.stream.is_alive():
                    self.log.error("feed_gap_detected")
                    self.router.freeze_new_orders()
        self._hb_thread = threading.Thread(target=_run, name="kp5-heartbeat", daemon=True)
        self._hb_thread.start()

    # === Stream callback ===
    def on_snapshot(self, snapshot: Dict[str, Any]):
        """
        snapshot contains: ltp, bid/ask, oi, oi_delta%, vol, iv, greeks, ts, symbol, side
        """
        if self.risk.is_hard_stopped():
            return

        # Pre-trade filters
        passed, reason = self._filters(snapshot)
        if not passed:
            if reason:
                self.log.debug("filter_reject", extra={"reason": reason, "sym": snapshot.get("symbol")})
            return

        # Size & risk calc (buyers: min(50% premium, ₹cap 2%))
        try:
            order = self.router.build_order(snapshot)  # MIS+LIMIT + slippage guard
        except Exception as e:
            self.log.exception("build_order_failed", extra={"error": str(e)})
            return

        if not self.risk.can_enter(order):
            self.log.info("risk_reject_entry", extra={"why": "limits_or_cooldown"})
            return

        # Route
        result = self.router.place(order)
        self.log.info("order_placed", extra={"result": result})
        self.risk.register_entry(result)

    def on_disconnect(self, why: str):
        self.log.error("stream_disconnected", extra={"why": why})
        self.router.freeze_new_orders()

    # === Filters (delta/IV/liquidity/TS) ===
    def _filters(self, s: Dict[str, Any]) -> tuple[bool, str]:
        g = s.get("greeks", {})
        delta = g.get("delta")
        iv = s.get("iv")
        if delta is None or iv is None:
            return False, "missing_greeks"

        gt = self.cfg["trading_rules"]["greeks_thresholds"]
        if not (gt["delta_min"] <= delta <= gt["delta_max"]):
            return False, "delta_band_fail"
        if not (gt["iv_min"] <= iv <= gt["iv_max"]):
            return False, "iv_band_fail"
        if (pct := s.get("iv_percentile")) is not None:
            if pct > gt.get("iv_percentile_max", 100):
                return False, "iv_percentile_high"

        # Liquidity guards
        lq = self.cfg["trading_rules"]["liquidity_requirements"]
        spread = s.get("spread_abs", 999)
        spread_pct = s.get("spread_pct", 999)
        if spread > lq["bid_ask_spread_max_abs"] and spread_pct > lq["bid_ask_spread_max_pct"]:
            return False, "wide_spread"

        if not s.get("rel_volume_ok", False):
            return False, "rel_volume_low"

        if s.get("oi_change_pct", 0) < lq["min_oi_change_percent"]:
            return False, "oi_change_low"
        if s.get("opp_oi_change_pct", 0) > lq["opposite_side_oi_unwind_percent"]:
            return False, "opp_side_not_unwinding"

        # Session guards (block new entries after cutoff)
        if self.risk.is_entry_cutoff():
            return False, "entry_cutoff"
        return True, ""
