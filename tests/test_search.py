#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
tests/test_search.py
--------------------
Tests for the Search Agent.

Run with:
    pytest tests/test_search.py -v
"""

import pytest
from agents.search import search_products, load_products


# ── Dataset loading ───────────────────────────────────────────────────────────

def test_load_products_returns_dataframe():
    df = load_products()
    assert len(df) > 0, "Product dataset should not be empty"


def test_load_products_has_required_columns():
    df = load_products()
    required = ["product_id", "product_name", "brand", "category", "store", "price", "mrp", "rating", "stock_status"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


# ── Keyword search ────────────────────────────────────────────────────────────

def test_search_returns_results_for_valid_query():
    results = search_products(query="phone")
    assert len(results) > 0, "Should return results for 'phone'"


def test_search_returns_empty_for_unknown_query():
    results = search_products(query="xyzproductthatdoesnotexist123")
    assert results == [], "Should return empty list for unknown query"


def test_search_is_case_insensitive():
    lower = search_products(query="phone")
    upper = search_products(query="PHONE")
    assert len(lower) == len(upper), "Search should be case insensitive"


# ── Budget filter ─────────────────────────────────────────────────────────────

def test_search_respects_budget():
    results = search_products(query="phone", max_budget=1500)
    for r in results:
        assert r["price"] <= 1500, f"Product price {r['price']} exceeds budget 1500"


def test_search_with_zero_budget_returns_empty():
    results = search_products(query="phone", max_budget=0)
    assert results == [], "Budget of 0 should return no results"


# ── Category filter ───────────────────────────────────────────────────────────

def test_search_filters_by_category():
    results = search_products(query="", category="Electronics")
    for r in results:
        assert r["category"].lower() == "electronics", f"Expected Electronics, got {r['category']}"


def test_search_category_is_case_insensitive():
    lower = search_products(query="", category="electronics")
    upper = search_products(query="", category="Electronics")
    assert len(lower) == len(upper), "Category filter should be case insensitive"


# ── Stock filter ──────────────────────────────────────────────────────────────

def test_search_excludes_out_of_stock():
    results = search_products(query="")
    for r in results:
        assert r["stock_status"] != "out_of_stock", "Out of stock products should be excluded"


# ── Combined filters ──────────────────────────────────────────────────────────

def test_search_combined_query_budget_category():
    results = search_products(query="serum", max_budget=600, category="Electronics")
    for r in results:
        assert r["price"] <= 600
        assert r["category"].lower() == "electronics"

# ── Fuzzy matching (TODO 1) ───────────────────────────────────────────────────

def test_search_fuzzy_matches_typo():
    """A slightly misspelled keyword should still find results, via fuzzy matching."""
    exact = search_products(query="phone")
    typo = search_products(query="phne")  # missing the "o"
    assert len(typo) > 0, "Fuzzy matching should still find results for a close typo"
    assert len(exact) > 0, "Sanity check: exact query should also return results"


def test_search_empty_query_does_not_crash():
    """An empty query string should not error, and should just skip keyword filtering."""
    results = search_products(query="")
    assert isinstance(results, list), "Empty query should still return a list, not crash"


def test_search_fuzzy_does_not_match_unrelated_word():
    """A word with no similarity to any product text should return no results,
    proving fuzzy matching isn't so loose it matches everything."""
    results = search_products(query="zzzxxqqjjbbnn")
    assert results == [], "Completely dissimilar keyword should return no matches"
