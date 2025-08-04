"""Unit tests for self_improver module."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from self_improver import suggest_improvements  # noqa: E402


def test_suggest_improvements_detects_long_line(tmp_path):
    repo = tmp_path
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    sample = repo / "sample.py"
    sample.write_text("def foo():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.PIPE
    )
    sample.write_text(
        "def foo():\n    return '" + "x" * 90 + "'\n", encoding="utf-8"
    )
    suggestions = suggest_improvements(repo_path=str(repo))
    assert any("line too long" in s for s in suggestions)
