from risk.safety_shield import SafetyShield, GuardConfig
shield = SafetyShield()

def place_order_guarded(broker, order: dict, open_positions: int, ltp: float):
    """
    order = {"symbol":"NIFTY","strike":24400,"otype":"PE","qty":75,"side":"BUY","price":ltp}
    est_risk_trade = order["qty"] * order["price"]   # option buying risk ~ premium paid
    """
    est_risk = order["qty"] * (order.get("price") or ltp)
    def tg_confirm(o): 
        # integrate with your send_telegram_message and wait for inline keyboard response
        # return True on 'CONFIRM', False otherwise
        return False  # default deny until wired
    ok, why = shield.can_send(order, open_positions, est_risk, require_confirm_cb=tg_confirm)
    if not ok:
        logger.warning("Order blocked: %s | %s", order, why)
        return {"ok": False, "reason": why}

    if os.getenv("RUN_MODE","ALERT").upper()=="SIM":
        # simulate fill; don't hit broker
        logger.info("SIM FILL: %s", order)
        return {"ok": True, "sim": True}

    # LIVE path
    try:
        resp = broker.place_order(order)  # your existing call
        shield.after_send(order, True, "OK")
        return {"ok": True, "resp": resp}
    except Exception as e:
        shield.after_send(order, False, str(e))
        logger.exception("Order failed")
        return {"ok": False, "reason": str(e)}
      
