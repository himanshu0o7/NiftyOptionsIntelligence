"""Auto Planner module implementing planner-executor-critic loop.

This module uses a JSON file as memory to queue tasks. It exposes a simple
CLI to trigger a planning cycle. Fail-safes limit infinite loops and alerts are
sent on anomalies.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from telegram_alerts import send_telegram_alert

MODULE_NAME = "auto_planner"


@dataclass
class AutoPlanner:
    """Simple planner that reads and writes tasks to a JSON memory file."""

    memory_file: str = "planner_memory.json"
    max_iterations: int = 10
    max_tasks: int = 50
    memory: Dict[str, Any] = field(default_factory=lambda: {"tasks": [], "current_index": 0})

    def __post_init__(self) -> None:
        self.load_memory()

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def load_memory(self) -> None:
        """Load tasks from JSON memory file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            else:
                self.save_memory()
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: error loading memory - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")
            self.memory = {"tasks": [], "current_index": 0}

    def save_memory(self) -> None:
        """Persist memory to disk."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: error saving memory - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")

    # ------------------------------------------------------------------
    # Core loop components
    # ------------------------------------------------------------------
    def plan(self) -> Optional[str]:
        """Return the next pending action, if any."""
        try:
            tasks: List[Dict[str, Any]] = self.memory.get("tasks", [])
            for idx, task in enumerate(tasks):
                if task.get("status") == "pending":
                    self.memory["current_index"] = idx
                    self.save_memory()
                    return task.get("action")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: error in planner - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")
            return None

    def execute(self, action: str) -> Dict[str, Any]:
        """Execute a given action (placeholder implementation)."""
        try:
            print(f"Executing: {action}")
            idx = self.memory.get("current_index", 0)
            self.memory["tasks"][idx]["status"] = "done"
            self.save_memory()
            return {"action": action, "status": "done"}
        except Exception as exc:
            msg = f"{MODULE_NAME}: executor failure - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")
            return {"action": action, "status": "error", "error": str(exc)}

    def criticize(self, result: Dict[str, Any]) -> str:
        """Evaluate execution result and alert on anomalies."""
        try:
            if result.get("status") == "error":
                send_telegram_alert(f"❌ {MODULE_NAME} anomaly: {result}")
            return result.get("status", "unknown")
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: critic error - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")
            return "error"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def add_task(self, action: str) -> None:
        """Queue a new task for execution."""
        try:
            if len(self.memory.get("tasks", [])) >= self.max_tasks:
                msg = f"{MODULE_NAME}: task queue overflow"
                print(msg)
                send_telegram_alert(f"⚠️ {msg}")
                return
            self.memory.setdefault("tasks", []).append({"action": action, "status": "pending"})
            self.save_memory()
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: add_task failure - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")

    def cycle(self, iterations: int = 1) -> None:
        """Run planner-executor-critic loop."""
        try:
            for count in range(min(iterations, self.max_iterations)):
                action = self.plan()
                if not action:
                    print("No pending tasks.")
                    break
                result = self.execute(action)
                status = self.criticize(result)
                if status == "error":
                    break
            if iterations > self.max_iterations:
                send_telegram_alert(
                    f"⚠️ {MODULE_NAME}: iteration limit exceeded ({iterations} > {self.max_iterations})"
                )
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"{MODULE_NAME}: cycle error - {exc}"
            print(msg)
            send_telegram_alert(f"⚠️ {msg}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Planner CLI")
    parser.add_argument("--cycle", action="store_true", help="Run planning cycle")
    parser.add_argument("--iterations", type=int, default=1, help="Number of cycle iterations")
    parser.add_argument("--add", type=str, help="Add a new task action")
    args = parser.parse_args()

    planner = AutoPlanner()

    if args.add:
        planner.add_task(args.add)
    if args.cycle:
        planner.cycle(iterations=args.iterations)
    if not args.cycle and not args.add:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
