"""Unit tests for the foo module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import foo


def test_foo_returns_string():
    result = foo.foo()
    assert isinstance(result, str)


def test_foo_value():
    assert foo.foo() == "foo"
