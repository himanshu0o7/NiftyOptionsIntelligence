"""Utility module that analyses repository changes and suggests improvements.

This module loads recent logs, diffs repository changes and performs a very
lightweight linting pass to suggest potential refactors.  If any of the required
core modules are missing, it delegates their creation to ``module_creator``.

All errors trigger Telegram alerts so that the maintainer is informed about
failures automatically.
"""

from __future__ import annotations

import os
import subprocess
from typing import List

from telegram_alerts import send_telegram_alert
from autocode_checker import check_modules
from module_creator import create_modules

LOG_PATH = "trading_system.log"


def load_logs(path: str = LOG_PATH) -> str:
    """Return contents of the log file if it exists."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: log load failed - {exc}")
        raise


def get_repo_diff(repo_path: str = ".") -> List[str]:
    """Return list of changed files according to git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        files = [f for f in result.stdout.splitlines() if f.endswith(".py")]
        return files
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: diff failed - {exc}")
        raise


def _lint_file(path: str) -> List[str]:
    """Run a tiny lint on ``path`` and return list of suggestions."""
    suggestions: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.rstrip("\n")
                if len(stripped) > 79:
                    suggestions.append(f"{path}:{line_no} line too long")
                if stripped != line.rstrip():
                    suggestions.append(f"{path}:{line_no} trailing whitespace")
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: lint failed for {path} - {exc}")
        raise
    return suggestions


def suggest_improvements(repo_path: str = ".") -> List[str]:
    """Return linting suggestions for changed files in ``repo_path``."""
    try:
        changed_files = get_repo_diff(repo_path)
        suggestions: List[str] = []
        for file in changed_files:
            suggestions.extend(_lint_file(os.path.join(repo_path, file)))
        if suggestions:
            send_telegram_alert(
                f"self_improver: improvements suggested for {len(suggestions)} issue(s)"
            )
        return suggestions
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: suggestion pass failed - {exc}")
        raise


def check_and_create_modules() -> List[str]:
    """Ensure mandatory modules exist and create them if missing."""
    try:
        missing = check_modules()
        if missing:
            create_modules(missing)
            send_telegram_alert(
                "self_improver: created missing modules " + ", ".join(missing)
            )
        return missing
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: module check failed - {exc}")
        raise


def main() -> None:
    """Entry point used when running as a script."""
    try:
        load_logs()
        suggest_improvements()
        check_and_create_modules()
    except Exception as exc:  # pragma: no cover - defensive
        send_telegram_alert(f"self_improver: unexpected failure - {exc}")


if __name__ == "__main__":
    main()
