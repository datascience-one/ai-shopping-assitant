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
from agents.search import search_products, load_products, expand_query_with_synonyms


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

# ── Synonym expansion (TODO 2) ────────────────────────────────────────────────

def test_expand_query_synonyms_happy_path() -> None:
    """
    1. Happy Path Test: Validates that a standard smartphone query expands correctly.

    Verifies single-word lookups append their target synonym list definitions.
    """
    query = "budget mobile"
    result = expand_query_with_synonyms(query)

    # Original terms must still be present
    assert "budget" in result
    assert "mobile" in result

    # Synonyms from the active smartphone dictionary must be successfully appended
    assert "affordable" in result
    assert "cheap" in result
    assert "entry-level" in result
    assert "smartphone" in result


def test_expand_query_synonyms_empty_input() -> None:
    """
    2. Empty Input Test: Validates that empty strings or None values return cleanly.

    Ensures input guard thresholds return a string to maintain type safety boundaries.
    """
    assert expand_query_with_synonyms("") == ""
    assert expand_query_with_synonyms(None) == ""


def test_expand_query_synonyms_edge_case() -> None:
    """
    3. Edge Case Test: Evaluates multi-word boundaries and phrase lookup mechanics.

    Ensures that a compound smartphone string trigger like 'basic phone' maps
    back to its core key ('feature phone') and alternate listings bidirectionally.
    """
    query = "buying a basic phone"
    result = expand_query_with_synonyms(query)

    # Verifies internal space phrase segments expand successfully
    assert "feature phone" in result
    assert "keypad phone" in result
