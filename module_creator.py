import argparse
import logging
import os

from telegram_alerts import send_telegram_alert

TEMPLATES = {
    "session_manager.py": '''# Handles session re-use and token management
import json

def load_session():
    with open("session_tokens.json") as f:
        return json.load(f)
''',
    "order_executor.py": '''# Executes orders using SmartAPI
def place_order(order_data, token):
    # TODO: implement Angel One order placement
    pass
''',
    "greeks_handler.py": '''# Placeholder for Greeks calculations
def fetch_greeks(option_symbol):
    # TODO: Use Open Interest & Greeks here
    return {}
''',
    "risk_management/position_manager.py": '''# Manages risk and position limits
def check_position_limits():
    # TODO: Add risk control logic here
    pass
'''
}

logging.basicConfig(level=logging.INFO)


def create_modules(missing_files):
    for file in missing_files:
        try:
            dir_name = os.path.dirname(file)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(file, "w") as f:
                f.write(TEMPLATES.get(file, "# Empty Module\n"))
            logging.info("🛠 Created: %s", file)
        except Exception as e:  # pragma: no cover - broad except for robustness
            error_msg = f"module_creator: Error creating {file}: {e}"
            logging.error(error_msg)
            send_telegram_alert(f"⚠️ {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create placeholder modules.")
    parser.add_argument(
        "--files",
        help="Comma-separated list of files to create. Overrides auto-detection.",
    )
    args = parser.parse_args()

    if args.files:
        files = [f.strip() for f in args.files.split(",") if f.strip()]
        if files:
            create_modules(files)
    else:
        from autocode_checker import check_modules

        missing = check_modules()
        if missing:
            create_modules(missing)

