"""Unit tests for the foo module."""

import pytest

import foo


def test_foo_returns_string():
    result = foo.foo()
    assert isinstance(result, str)


def test_foo_value():
    assert foo.foo() == "foo"
