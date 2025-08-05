"""Unit tests for kp5_controller."""

import os
import sys

import pytest

import kp5_controller


def test_parse_args_self_audit():
    args = kp5_controller.parse_args(["self_audit"])
    assert args.command == "self_audit"


def test_execute_dispatch(monkeypatch):
    called = {}

    def fake_run():
        called["executed"] = True
        return "done"

    monkeypatch.setitem(kp5_controller.MODULES, "self_audit", fake_run)
    result = kp5_controller.execute_command("self_audit")
    assert called.get("executed") is True
    assert result == "done"
