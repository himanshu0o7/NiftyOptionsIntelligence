# fix-bot-2025-07-24
"""
Utility functions for detecting market trends.

The core function exposed by this module, :func:`detect_trend`,
analyses option Greek data and open–interest changes to classify the
market as Bullish, Bearish or Sideways.  A small wrapper is used to
retrieve delta values either via a live call into
``greeks_handler.fetch_option_greeks`` (if available) or, if that
dependency is absent, by falling back to a sensible default.  The
open–interest change is fetched from ``utils.oi_data.get_oi_change``.

In a real‑time deployment you should replace the dummy delta values
returned by :func:`get_option_greek_data` with real data from your
broker’s API.
"""

import logging
from utils.oi_data import get_oi_change

try:
    # Attempt to import the live Greek fetcher.  This will fail in
    # environments where ``greeks_handler`` is unavailable (e.g. when
    # running locally without Angel One connectivity).
    from greeks_handler import fetch_option_greeks  # type: ignore
except ImportError:
    fetch_option_greeks = None


def get_option_greek_data(symbol: str, expiry: str, option_type: str, strike: int | None = None, tokens: dict | None = None) -> dict:
    """Fetch option Greek data for a given contract.

    Parameters
    ----------
    symbol: str
        Underlying index or stock symbol (e.g. ``"NIFTY"``).
    expiry: str
        Expiry date in Angel One format (e.g. ``"25JUL2025"``).  This
        parameter is currently unused by the fallback implementation.
    option_type: str
        ``"CE"`` for call options or ``"PE"`` for put options.
    strike: int | None
        Strike price of the option.  If ``None`` and the live Greek
        fetcher is available, the function may choose a reasonable
        default or return dummy data.
    tokens: dict | None
        Dictionary of authentication tokens required by the live API.

    Returns
    -------
    dict
        A mapping containing at least the key ``"delta"``.

    Notes
    -----
    This wrapper attempts to call ``fetch_option_greeks`` from
    :mod:`greeks_handler` if it exists.  If either the function is
    unavailable or an error is raised during the call, a deterministic
    dummy delta is returned: positive for calls and negative for puts.
    """
    # Use live greeks if available
    if fetch_option_greeks and strike is not None and tokens is not None:
        try:
            data = fetch_option_greeks(symbol, strike, option_type, tokens)
            return {"delta": data.get("delta")}
        except Exception as exc:
            logging.error(f"Greek fetch error: {exc}")
    # Fallback: return a representative delta
    return {"delta": 0.6 if option_type.upper() == "CE" else -0.6}


def detect_trend(symbol: str, expiry: str) -> dict:
    """Determine the trend of the market for the given symbol and expiry.

    A bullish trend is signalled when the call option delta exceeds 0.6
    **and** call open–interest change is positive.  A bearish trend is
    signalled when the put option delta is less than -0.6 and the put
    open–interest change is positive.  Otherwise the market is
    considered sideways.

    Parameters
    ----------
    symbol: str
        Underlying index (e.g. ``"NIFTY"`` or ``"BANKNIFTY"``).
    expiry: str
        Expiry date in Angel One format.

    Returns
    -------
    dict
        A dictionary with ``"trend"`` (``"Bullish"``, ``"Bearish"`` or
        ``"Sideways"``), ``"reason"`` (explanatory string) and
        ``"supporting_data"`` (raw values used in the decision).
    """
    try:
        # Greek deltas
        ce_greeks = get_option_greek_data(symbol, expiry, option_type="CE")
        pe_greeks = get_option_greek_data(symbol, expiry, option_type="PE")
        # Open interest changes
        ce_oi_change = get_oi_change(symbol, expiry, option_type="CE")
        pe_oi_change = get_oi_change(symbol, expiry, option_type="PE")
        ce_delta = ce_greeks.get("delta")
        pe_delta = pe_greeks.get("delta")
        if (
            ce_delta is not None
            and ce_oi_change is not None
            and ce_delta > 0.6
            and ce_oi_change > 0
        ):
            trend = "Bullish"
            reason = "CE delta > 0.6 and CE OI change > 0"
        elif (
            pe_delta is not None
            and pe_oi_change is not None
            and pe_delta < -0.6
            and pe_oi_change > 0
        ):
=======
# utils/trend_detector.py

#from trend_detector import detect_trend

from utils.greeks_handler import get_option_greek_data
from utils.oi_data import get_oi_change

# utils/trend_detector.py

def detect_trend(symbol, expiry):
    """
    Detects market trend (Bullish, Bearish, Sideways) based on CE/PE delta and OI change.
    Returns:
        dict: {
            "trend": "Bullish/Bearish/Sideways/Error",
            "reason": "...",
            "supporting_data": { ... }
        }
    """
    try:
        from utils.greeks_handler import get_option_greek_data
        from utils.oi_data import get_oi_change

        # Fetch CE and PE Greeks (delta)
        ce_greeks = get_option_greek_data(symbol, expiry, option_type="CE")
        pe_greeks = get_option_greek_data(symbol, expiry, option_type="PE")

        # Fetch OI change
        ce_oi_change = get_oi_change(symbol, expiry, option_type="CE")
        pe_oi_change = get_oi_change(symbol, expiry, option_type="PE")

        # Extract data
        ce_delta = ce_greeks.get("delta")
        pe_delta = pe_greeks.get("delta")

        # Logic
        if ce_delta is not None and ce_oi_change is not None and ce_delta > 0.6 and ce_oi_change > 0:
            trend = "Bullish"
            reason = "CE delta > 0.6 and CE OI change > 0"
        elif pe_delta is not None and pe_oi_change is not None and pe_delta < -0.6 and pe_oi_change > 0:
 main
            trend = "Bearish"
            reason = "PE delta < -0.6 and PE OI change > 0"
        else:
            trend = "Sideways"
            reason = "No strong CE/PE delta + OI change trigger"
# fix-bot-2025-07-24
=======

  main
        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": {
                "ce_delta": ce_delta,
                "ce_oi_change": ce_oi_change,
                "pe_delta": pe_delta,
 # fix-bot-2025-07-24
                "pe_oi_change": pe_oi_change,
            },
        }
    except Exception as exc:
        return {
            "trend": "Error",
            "reason": str(exc),
            "supporting_data": {},
        }
=======
                "pe_oi_change": pe_oi_change
            }
        }

    except Exception as e:
        return {
            "trend": "Error",
            "reason": str(e),
            "supporting_data": {}
        }

  main
