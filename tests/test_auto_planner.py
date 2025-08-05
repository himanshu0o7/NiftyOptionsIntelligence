"""Tests for the auto_planner module."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_planner import AutoPlanner


def test_plan_returns_next_pending_action(tmp_path):
    memory = {"tasks": [{"action": "task1", "status": "pending"}, {"action": "task2", "status": "pending"}]}
    mem_file = tmp_path / "mem.json"
    mem_file.write_text(json.dumps(memory))

    planner = AutoPlanner(memory_file=str(mem_file))
    action = planner.plan()
    assert action == "task1"


def test_cycle_marks_task_done(tmp_path):
    memory = {"tasks": [{"action": "task1", "status": "pending"}]}
    mem_file = tmp_path / "mem.json"
    mem_file.write_text(json.dumps(memory))

    planner = AutoPlanner(memory_file=str(mem_file))
    planner.cycle()
    data = json.loads(mem_file.read_text())
    assert data["tasks"][0]["status"] == "done"
