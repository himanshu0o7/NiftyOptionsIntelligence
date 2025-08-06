"""
Test Greeks API with live Angel One credentials
"""
import os
from core.angel_api import AngelOneAPI
from core.options_greeks_api import OptionsGreeksAPI
from utils.logger import Logger

def test_live_greeks_api():
    """Test live Greeks API with real credentials"""
    logger = Logger()

    try:
        # Get credentials from environment
        api_key = os.getenv('ANGEL_API_KEY')
        client_code = os.getenv('ANGEL_CLIENT_CODE')
        password = os.getenv('ANGEL_PASSWORD')
        totp_secret = os.getenv('ANGEL_TOTP_SECRET')

        if not all([api_key, client_code, password, totp_secret]):
            logger.error("Missing Angel One credentials")
            return False

        # Initialize API client
        api_client = AngelOneAPI(api_key, client_code, password, totp_secret)

        # Connect to Angel One
        if not api_client.connect():
            logger.error("Failed to connect to Angel One API")
            return False

        logger.info("✅ Connected to Angel One API")

        # Initialize Greeks API
        greeks_api = OptionsGreeksAPI(api_client)

        # Test NIFTY Greeks
        nifty_greeks = greeks_api.get_option_greeks("NIFTY", "10JUL2025")

        if nifty_greeks:
            logger.info(f"✅ Retrieved {len(nifty_greeks)} NIFTY Greeks")

            # Display sample data
            for i, option in enumerate(nifty_greeks[:3]):
                logger.info(f"Sample {i+1}: Strike {option.get('strikePrice')} {option.get('optionType')} - "
                          f"Delta: {option.get('delta')}, Gamma: {option.get('gamma')}, "
                          f"Theta: {option.get('theta')}, Vega: {option.get('vega')}, "
                          f"IV: {option.get('impliedVolatility')}%")
        else:
            logger.error("❌ Failed to get NIFTY Greeks")

        # Test BANKNIFTY Greeks
        banknifty_greeks = greeks_api.get_option_greeks("BANKNIFTY", "09JUL2025")

        if banknifty_greeks:
            logger.info(f"✅ Retrieved {len(banknifty_greeks)} BANKNIFTY Greeks")
        else:
            logger.error("❌ Failed to get BANKNIFTY Greeks")

        # Test OI Buildup
        oi_data = greeks_api.get_oi_buildup_data()
        if oi_data:
            logger.info(f"✅ Retrieved {len(oi_data)} OI Buildup records")

        # Test PCR data
        pcr_data = greeks_api.get_pcr_data()
        if pcr_data:
            logger.info(f"✅ Retrieved {len(pcr_data)} PCR records")

        return True

    except Exception as e:
        logger.error(f"Greeks API test error: {e}")
        return False

if __name__ == "__main__":
    test_live_greeks_api()