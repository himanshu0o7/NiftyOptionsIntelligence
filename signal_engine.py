# signal_engine.py
import requests

def scan_and_alert(tokens, symbol, atm_strike):
    alerts = []
    ticks = []
    # Example logic for ATM ±2 CE/PE scanning (Dummy data)
    for offset in [-2, -1, 0, 1, 2]:
        for opt in ["CE", "PE"]:
            strike = atm_strike + offset * 100
            delta, iv = 0.6, 20  # Example values, replace with live API call
            ticks.append((symbol, strike, opt, delta, iv))

            # Alert criteria
            if delta > 0.5 and iv < 25:
                alerts.append((symbol, strike, opt, delta, iv))
                msg = f"📢 Signal: BUY {symbol} {strike}{opt} | Delta: {delta:.2f} | IV: {iv}%"
                url = f"https://api.telegram.org/bot{tokens['TELEGRAM_BOT_TOKEN']}/sendMessage"
                requests.post(url, data={"chat_id": tokens['TELEGRAM_CHAT_ID'], "text": msg})

    return {"ticks": ticks, "alerts": alerts}

