import requests
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SENSIBULL_CHAIN_API = "https://api.sensibull.com/v1/option_chain/"


def fetch_option_data(
    symbol: str = "NIFTY",
    strike: int = 25100,
    option_type: str = "CE",
    expiry: Optional[str] = None  # Format: "31JUL2025"
) -> Optional[Dict[str, float]]:
    """Fetch option data including Greeks and OI from Sensibull API."""
    try:
        response = requests.get(f"{SENSIBULL_CHAIN_API}{symbol.upper()}")
        if response.status_code != 200:
            logger.warning(f"Failed to fetch Sensibull data: {response.status_code}")
            return None

        data = response.json()
        if 'chains' not in data:
            logger.warning("No chains found in Sensibull response")
            return None

        # Iterate over option chains and find matching strike and type
        for chain in data['chains']:
            if (
                str(chain.get("strike")) == str(strike)
                and chain.get("type") == option_type
                and (expiry is None or chain.get("expiry") == expiry)
            ):
                return {
                    "ltp": chain.get("last_price", 0.0),
                    "oi": chain.get("open_interest", 0),
                    "iv": chain.get("iv", 0.0),
                    "delta": chain.get("delta", 0.0),
                    "theta": chain.get("theta", 0.0),
                    "gamma": chain.get("gamma", 0.0),
                    "vega": chain.get("vega", 0.0),
                    "strike": chain.get("strike"),
                    "expiry": chain.get("expiry"),
                    "type": chain.get("type"),
                }
        logger.warning("No matching strike/type found in Sensibull chain")
        return None

    except Exception as e:
        logger.error(f"Error fetching Sensibull option data: {e}")
        return None

