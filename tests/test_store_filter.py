# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.
"""
tests/test_store_filter.py
---------------------------
Tests for the store filter in the /chat endpoint.
"""

# Standard imports
import pytest

# Third-party imports
from fastapi.testclient import TestClient

# Project imports
import backend.main as main

def make_product(name: str, store: str) -> dict:
    """Builds a minimal product dict with every field ProductResult needs."""
    return {
        "product_name": name, "brand": "X", "store": store,
        "price": 10000.0, "mrp": 12000.0, "rating": 4.0, "review_count": 10,
        "key_specs": "8GB", "pros": "", "cons": "", "score": 70.0,
    }


PRODUCTS = [
    make_product("Phone A1", "Store A"),
    make_product("Phone A2", "Store A"),
    make_product("Phone B1", "Store B"),
]


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaces search, compare and the LLM with simple stand-ins."""
    monkeypatch.setattr(main, "search_products", lambda **kwargs: list(PRODUCTS))
    monkeypatch.setattr(
        main, "compare_and_rank", lambda results, max_budget=None: results[:3]
    )
    monkeypatch.setattr(main, "get_recommendation", lambda *args, **kwargs: "reply")


def test_store_filter_keeps_only_selected_store() -> None:
    """Happy path: choosing Store A returns only Store A products."""
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": "phone", "stores": ["Store A"]})

    stores = {p["store"] for p in response.json()["top_products"]}
    assert response.status_code == 200
    assert stores == {"Store A"}


def test_no_store_filter_returns_all_stores() -> None:
    """Empty input: no stores field, or an empty list, searches every store."""
    client = TestClient(main.app)
    for body in ({"message": "phone"}, {"message": "phone", "stores": []}):
        response = client.post("/chat", json=body)
        assert len(response.json()["top_products"]) == 3


def test_store_filter_with_no_matching_products() -> None:
    """Edge case: a store with no products gives a 404, not a crash."""
    client = TestClient(main.app)
    response = client.post("/chat", json={"message": "phone", "stores": ["Store C"]})
    assert response.status_code == 404
