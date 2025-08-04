import logging
from datetime import datetime

import pandas as pd
import requests

from telegram_alerts import send_telegram_alert

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_MASTER_PATH = "nfo_scrip_master.csv"


def fetch_and_save_nfo_master_contract(path: str = SCRIP_MASTER_PATH) -> str:
    """Fetch the NFO master contract and save it to ``path``.

    The returned CSV contains only NFO segment records and includes a
    ``fetch_timestamp`` column used for cache freshness checks.
    """
    try:
        response = requests.get(SCRIP_MASTER_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        nfo_df = df[df["exch_seg"] == "NFO"].copy()
        nfo_df["fetch_timestamp"] = datetime.now()
        nfo_df.to_csv(path, index=False)
        return path
    except Exception as exc:  # noqa: BLE001
        logging.exception("master_contract_fetcher: failed to fetch NFO master contract: %s", exc)
        try:
            send_telegram_alert(f"master_contract_fetcher error: {exc}")
        except Exception:  # noqa: BLE001
            logging.exception("master_contract_fetcher: failed to send Telegram alert")
        raise
