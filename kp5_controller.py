"""KP5Bot master controller.

Aggregates modular components like ``self_audit`` and ``web_learner``
through a simple command-line interface and exposes functions for
Streamlit pages.
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict

from telegram_alerts import send_telegram_alert

import self_audit
import web_learner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("KP5Controller")

MODULES: Dict[str, Callable[[], str]] = {
    "self_audit": self_audit.run,
    "web_learner": web_learner.run,
}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="KP5Bot master controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in MODULES:
        subparsers.add_parser(name, help=f"Run the {name.replace('_', ' ')} module")
    return parser.parse_args(args)


def execute_command(command: str) -> str:
    """Execute the given command with logging and Telegram alerts."""
    if command not in MODULES:
        logger.error("Unknown command: %s", command)
        send_telegram_alert(f"❌ Unknown command attempted: {command}")
        raise ValueError(f"Unknown command: {command}")

    logger.info("Executing command: %s", command)
    try:
        result = MODULES[command]()
        logger.info("Command %s completed successfully", command)
        send_telegram_alert(f"✅ Command {command} completed successfully.")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while executing command %s: %s", command, exc)
        send_telegram_alert(f"❌ Error in {command}: {exc}")
        raise


def run_self_audit() -> str:
    """Expose self audit for Streamlit UI."""
    return execute_command("self_audit")


def run_web_learner() -> str:
    """Expose web learner for Streamlit UI."""
    return execute_command("web_learner")


def main(args: list[str] | None = None) -> None:
    """Entry point for command-line execution."""
    parsed = parse_args(args)
    execute_command(parsed.command)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
