#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
tests/test_recommend.py
-----------------------
Tests for the Recommend Agent.

Note: LLM calls are NOT tested here — we test the prompt builder only.
      LLM integration tests require a live API key and are out of scope for unit tests.

Run with:
    pytest tests/test_recommend.py -v
"""

import pytest
from agents.recommend import build_prompt, get_recommendation, _call_gemini

# ── Sample data ───────────────────────────────────────────────────────────────

def make_product(name="Test Serum", store="Store A", price=499, score=82.5):
    return {
        "product_name": name,
        "brand": "TestBrand",
        "store": store,
        "price": price,
        "mrp": 599,
        "rating": 4.3,
        "review_count": 1200,
        "key_specs": "Volume: 30ml | Niacinamide 10%",
        "pros": "Effective, affordable",
        "cons": "May cause purging",
        "score": score,
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

def test_build_prompt_contains_query():
    query = "best serum for oily skin"
    products = [make_product()]
    prompt = build_prompt(query, products)
    assert query in prompt, "Prompt should contain the user query"


def test_build_prompt_contains_product_name():
    products = [make_product(name="Minimalist Serum")]
    prompt = build_prompt("serum", products)
    assert "Minimalist Serum" in prompt, "Prompt should contain the product name"


def test_build_prompt_contains_price():
    products = [make_product(price=499)]
    prompt = build_prompt("serum", products)
    assert "499" in prompt, "Prompt should contain the product price"


def test_build_prompt_contains_store():
    products = [make_product(store="Store B")]
    prompt = build_prompt("serum", products)
    assert "Store B" in prompt, "Prompt should contain the store name"


def test_build_prompt_numbers_products():
    products = [make_product(name=f"Product {i}", store=f"Store {s}") for i, s in enumerate(["A","B","C"])]
    prompt = build_prompt("serum", products)
    assert "1." in prompt
    assert "2." in prompt
    assert "3." in prompt


# ── Empty input ───────────────────────────────────────────────────────────────

def test_get_recommendation_returns_message_for_empty_products():
    result = get_recommendation("serum", [])
    assert isinstance(result, str), "Should return a string"
    assert len(result) > 0, "Should return a non-empty message"






 #── Gemini integration ────────────────────────────────────────────────────────


def test_call_gemini_returns_response_text(monkeypatch):
    class FakeResponse:
        text = "This is a Gemini recommendation."

    class FakeModels:
        def generate_content(self, model, contents):
            assert model == "gemini-3.6-flash"
            assert contents == "test prompt"
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "fake-key"
            self.models = FakeModels()

    monkeypatch.setattr("agents.recommend.config.GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("agents.recommend.genai.Client", FakeClient)

    result = _call_gemini("test prompt")

    assert result == "This is a Gemini recommendation."
