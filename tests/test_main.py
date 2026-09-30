# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.

"""
tests/test_main.py
Unit tests for the intent extraction logic in backend/main.py (TODO 2).
"""

from backend.main import extract_budget


def test_extract_budget_under_phrase():
    """'under X' should correctly extract X as the budget."""
    assert extract_budget("phone under 20000") == 20000.0


def test_extract_budget_with_currency_symbol():
    """A ₹ symbol between the trigger word and the number should still work."""
    assert extract_budget("phone under ₹15000 with 5G") == 15000.0


def test_extract_budget_with_rs_prefix():
    """'Rs' before the number should still be handled correctly."""
    assert extract_budget("lipstick under Rs 400") == 400.0


def test_extract_budget_below_phrase():
    """'below X' should also be recognized as a budget phrase."""
    assert extract_budget("laptop below 50000") == 50000.0


def test_extract_budget_less_than_phrase():
    """'less than X' should also be recognized."""
    assert extract_budget("something less than 15000") == 15000.0


def test_extract_budget_budget_of_phrase():
    """'budget of X' should also be recognized."""
    assert extract_budget("budget of 10000") == 10000.0


def test_extract_budget_returns_none_when_absent():
    """A sentence with no budget phrase should return None, not crash or guess."""
    assert extract_budget("just a phone with good camera") is None


def test_extract_budget_is_case_insensitive():
    """Budget phrases should be recognized regardless of capitalization."""
    assert extract_budget("Phone UNDER 20000") == 20000.0
