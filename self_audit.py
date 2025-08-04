import os
import ast
import argparse
import logging
from typing import Dict, List

from telegram_alerts import send_telegram_alert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _attempt_import(module_name: str, file_path: str, missing: List[Dict], errors: List[Dict]):
    """Attempt to import a module and log failures."""
    try:
        __import__(module_name)
    except ImportError:
        missing.append({"module": module_name, "file": file_path})
        msg = f"Missing module {module_name} in {file_path}"
        logger.error(msg)
        send_telegram_alert(msg)
    except Exception as exc:  # pragma: no cover - unexpected errors
        msg = f"Error importing {module_name} in {file_path}: {exc}"
        logger.error(msg)
        send_telegram_alert(msg)
        errors.append({"module": module_name, "file": file_path, "error": str(exc)})


def scan_repository(base_path: str = ".") -> Dict[str, List[Dict]]:
    """Scan repository for missing modules and performance issues."""
    missing_modules: List[Dict] = []
    performance_issues: List[str] = []
    errors: List[Dict] = []

    for root, _, files in os.walk(base_path):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)

            try:
                if os.path.getsize(path) > 1_000_000:  # 1MB threshold
                    performance_issues.append(path)
                    logger.warning("Performance issue: %s exceeds size threshold", path)
            except Exception as exc:
                msg = f"Error checking size for {path}: {exc}"
                logger.error(msg)
                send_telegram_alert(msg)
                errors.append({"file": path, "error": str(exc)})

            try:
                with open(path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split(".")[0]
                            _attempt_import(module_name, path, missing_modules, errors)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        module_name = node.module.split(".")[0]
                        _attempt_import(module_name, path, missing_modules, errors)
            except Exception as exc:
                msg = f"Error scanning {path}: {exc}"
                logger.error(msg)
                send_telegram_alert(msg)
                errors.append({"file": path, "error": str(exc)})

    return {
        "missing_modules": missing_modules,
        "performance_issues": performance_issues,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Self audit utilities.")
    parser.add_argument("--scan", action="store_true", help="Run repository scan")
    parser.add_argument("--path", default=".", help="Path to scan")
    args = parser.parse_args()

    if args.scan:
        results = scan_repository(args.path)
        print(results)


if __name__ == "__main__":
    main()
