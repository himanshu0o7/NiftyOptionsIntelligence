# order_executor.py (only the relevant bits)
from __future__ import annotations
import time
from typing import Dict, Any
from logging_setup import configure_logger
from utils.config import load_config

class OrderExecutor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.log = configure_logger("kp5.order")
        self._freeze = False
        self.active_symbols = {}

    def freeze_new_orders(self): self._freeze = True
    def unfreeze(self): self._freeze = False
    def set_active_symbols(self, sym_map: Dict[str, Any]): self.active_symbols = sym_map
    def count_open_positions(self) -> int: ...  # your existing impl

    def build_order(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        or_cfg = self.cfg.get("order_routing", {})
        if self._freeze:
            raise RuntimeError("frozen_by_controller")
        ltp = float(snapshot["ltp"])
        bps = float(or_cfg.get("slippage_bps", 15))
        price = round(ltp * (1 + bps/10000), 2)
        return {
            "symbol": snapshot["symbol"],
            "producttype": or_cfg.get("product_type", "MIS"),
            "ordertype": or_cfg.get("order_type", "LIMIT"),
            "price": price,
            "quantity": snapshot.get("qty", self._qty_from_risk(snapshot)),
            "validity": "DAY",
            "meta": {"ltp": ltp, "slip_bps": bps}
        }

    def _qty_from_risk(self, snapshot: Dict[str, Any]) -> int:
        lot = snapshot.get("lot_size", 75)
        # buyers: cap loss by min(50% premium, risk_per_trade)
        premium = float(snapshot["ltp"])
        risk = float(self.cfg["risk_management"]["risk_per_trade"])
        max_loss_per_lot = min(0.5 * premium * lot, risk)
        lots = max(1, int(risk // max(1, max_loss_per_lot)))  # >=1, conservative
        return lots * lot

    def place(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # Call your AngelOne placeOrder() and implement cancel/replace on timeout
        timeout = int(self.cfg["order_routing"].get("timeout_seconds", 5))
        t0 = time.time()
        # resp = self._broker.place(order)  # <-- your integration
        resp = {"status": "sent", "order": order, "t0": t0}
        # poll for fill within timeout; else cancel
        # ...
        return resp

    def graceful_square_off_all(self, reason: str):
        self.log.warning("square_off_all", extra={"reason": reason})
        # iterate open positions; send market exits (or LIMIT with protective price)
