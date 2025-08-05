import argparse
import logging
import os

from telegram_alerts import send_telegram_alert
# codex/wrap-file-writing-logic-in-try/except
from utils.logger import default_logger as logger

# fix-bot-2025-07-24

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

# codex/wrap-file-writing-logic-in-try/except

logging.basicConfig(level=logging.INFO)

# fix-bot-2025-07-24

def create_modules(missing_files):
    """Create missing module files and directories.

    Args:
        missing_files (list[str]): List of file paths to create.

    Returns:
        dict: Summary of created files and failures.
    """
    summary = {"created": [], "failed": []}

    for file in missing_files:
# codex/wrap-file-writing-logic-in-try/except
        dir_name = os.path.dirname(file)
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as exc:
                msg = f"module_creator: failed to create directory '{dir_name}' for '{file}': {exc}"
                logger.error(msg)
                try:
                    send_telegram_alert(f"🛑 {msg}")
                except Exception:
                    logger.exception(
                        "module_creator: failed to send Telegram alert"
                    )
                summary["failed"].append(
                    {"file": file, "error": str(exc), "stage": "directory"}
                )
                continue

        try:
            with open(file, "w", encoding="utf-8") as f:
                f.write(TEMPLATES.get(os.path.basename(file), "# Empty Module\n"))
            logger.info(f"module_creator: created module '{file}'")
            summary["created"].append(file)
        except OSError as exc:
            msg = f"module_creator: failed to create file '{file}': {exc}"
            logger.error(msg)
            try:
                send_telegram_alert(f"🛑 {msg}")
            except Exception:
                logger.exception(
                    "module_creator: failed to send Telegram alert"
                )
            summary["failed"].append(
                {"file": file, "error": str(exc), "stage": "file"}
            )

    return summary


if __name__ == "__main__":
    from autocode_checker import check_modules

    missing = check_modules()
    if missing:
        result = create_modules(missing)
        print(result)

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
# fix-bot-2025-07-24

