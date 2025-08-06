"""Runtime safety guard for KP5Bot.

Monitors system metrics and provides hooks for additional safety checks.
"""

import logging
import os
import subprocess
from typing import Callable, Dict, Tuple

import psutil

from telegram_handler import send_alert


class ThresholdBreach(RuntimeError):
    """Exception raised when a monitored threshold is exceeded."""


class RiskGuard:
    """Monitors runtime metrics and enforces safety thresholds."""

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        file_churn_threshold: int = 10,
        log_size_threshold: int = 5 * 1024 * 1024,
        log_path: str = "trading_system.log",
    ):
        self.cpu_threshold = cpu_threshold
        self.file_churn_threshold = file_churn_threshold
        self.log_size_threshold = log_size_threshold
        self.log_path = log_path
        self.checks: Dict[str, Callable[[], Tuple[bool, str]]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.good_commit = self._get_current_commit()

    def register_check(
        self, name: str, check: Callable[[], Tuple[bool, str]]
    ) -> None:
        """Register an external safety check."""
        self.checks[name] = check

    def monitor(self) -> None:
        """Run all checks and handle any threshold breaches."""
        try:
            self._check_cpu()
            self._check_file_churn()
            self._check_log_size()
            self._run_custom_checks()
        except ThresholdBreach as exc:
            self._handle_breach(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected failures
            message = f"RiskGuard unexpected error: {exc}"
            self.logger.exception(message)
            try:
                send_alert(message)
            except Exception:
                pass
            raise

    # Internal checks -------------------------------------------------
    def _get_current_commit(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode()
                .strip()
            )
        except Exception as exc:  # pragma: no cover
            self.logger.error("RiskGuard failed to get commit: %s", exc)
            return ""

    def _check_cpu(self) -> None:
        usage = psutil.cpu_percent(interval=0.1)
        if usage > self.cpu_threshold:
            raise ThresholdBreach(
                f"CPU usage {usage:.2f}% exceeds {self.cpu_threshold}%"
            )

    def _check_file_churn(self) -> None:
        try:
            output = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            )
            changed_files = len(output.strip().splitlines())
            if changed_files > self.file_churn_threshold:
                raise ThresholdBreach(
                    f"File churn {changed_files} exceeds {self.file_churn_threshold}"
                )
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            self.logger.error("RiskGuard file churn check failed: %s", exc)

    def _check_log_size(self) -> None:
        if os.path.exists(self.log_path):
            size = os.path.getsize(self.log_path)
            if size > self.log_size_threshold:
                raise ThresholdBreach(
                    f"Log file {self.log_path} size {size} exceeds {self.log_size_threshold}"
                )

    def _run_custom_checks(self) -> None:
        for name, check in self.checks.items():
            ok, message = check()
            if not ok:
                raise ThresholdBreach(f"Custom check {name} failed: {message}")

    # Breach handling -------------------------------------------------
    def _handle_breach(self, message: str) -> None:
        try:
            send_alert(f"RiskGuard breach: {message}")
        except Exception:  # pragma: no cover
            self.logger.error("RiskGuard alert failed")
        if self.good_commit:
            try:
                subprocess.run(
                    ["git", "reset", "--hard", self.good_commit], check=True
                )
            except Exception as exc:  # pragma: no cover
                self.logger.error("RiskGuard rollback failed: %s", exc)
        raise RuntimeError(message)
