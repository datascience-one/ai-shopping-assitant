# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.

"""Tests for product-grounded follow-up quick replies in the Streamlit UI."""

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from agents import search as search_agent
import frontend.app as frontend_app
from frontend.app import (
    _build_local_comparison_rows,
    _build_follow_up_query,
    _cheaper_alternative_threshold,
    _generate_follow_up_suggestions,
    _suggestion_label,
    _queue_cheaper_search,
    _render_local_comparison,
    _select_local_comparison,
    _validate_follow_up_suggestions,
)


def make_product(price: float, key_specs: str = "") -> dict[str, object]:
    """Build a minimal returned-product record for suggestion tests."""
    return {"price": price, "key_specs": key_specs}


def test_generates_suggestions_from_available_specifications():
    """Suggest comparisons only when multiple products contain each spec."""
    products = [
        make_product(
            18000,
            "Rear Camera: 50 MP | Battery: 5000 mAh | RAM: 8 GB | "
            "Storage: 128 GB | Processor: Snapdragon 7 Gen 3 | "
            "Charging Speed: 33 W",
        ),
        make_product(
            22000,
            "Rear Camera: 64 MP | Battery: 5200 mAh | RAM: 12 GB | "
            "Storage: 256 GB | Processor: Dimensity 8300 | "
            "Fast Charging: 67W",
        ),
    ]

    suggestions = _generate_follow_up_suggestions(
        products, "Samsung smartphone under ₹30,000"
    )
    labels = [suggestion["label"] for suggestion in suggestions]

    assert "Compare rear camera details" in labels
    assert "Compare charging speeds" in labels
    assert "Compare battery capacities" in labels
    assert any(
        label.startswith("Show alternatives at or below ₹") for label in labels
    )
    assert all("smartphone alternatives" not in label for label in labels)
    assert len(suggestions) <= 4


def test_processor_comparison_is_prioritized_over_ram_storage_at_limit():
    """Keep processor details in the four-item set when both comparisons qualify."""
    products = [
        make_product(
            10000,
            "Rear Camera: 50 MP | Battery: 5000 mAh | RAM: 8 GB | "
            "Storage: 128 GB | Processor: Snapdragon 7 Gen 3",
        ),
        make_product(
            20000,
            "Rear Camera: 64 MP | Battery: 5200 mAh | RAM: 12 GB | "
            "Storage: 256 GB | Processor: Dimensity 8300",
        ),
    ]

    suggestions = _generate_follow_up_suggestions(
        products, "smartphone under 30000"
    )
    labels = [suggestion["label"] for suggestion in suggestions]

    assert len(suggestions) == 4
    assert "Compare processor details" in labels
    assert "Compare RAM and storage" not in labels


def test_omits_comparisons_when_specifications_are_unavailable():
    """Do not suggest comparisons with absent or one-product-only specs."""
    products = [
        make_product(18000, "Battery: 5000 mAh"),
        make_product(18000, "Rear Camera: 50 MP"),
    ]

    suggestions = _generate_follow_up_suggestions(products)

    assert suggestions == []


def test_battery_capacity_does_not_count_as_charging_speed():
    """Battery capacity alone cannot support a charging-speed suggestion."""
    products = [
        make_product(18000, "Battery: 5000 mAh"),
        make_product(22000, "Battery: 5200 mAh"),
    ]

    suggestions = _generate_follow_up_suggestions(products)

    assert "Compare charging speeds" not in [
        suggestion["label"] for suggestion in suggestions
    ]


def test_placeholder_specifications_are_not_treated_as_available():
    """Do not suggest comparisons for placeholder processor values."""
    products = [
        make_product(18000, "Processor: Unknown"),
        make_product(22000, "Processor: N/A"),
    ]

    suggestions = _generate_follow_up_suggestions(products)

    assert "Compare processor details" not in [
        suggestion["label"] for suggestion in suggestions
    ]


def test_cheaper_alternatives_use_a_data_based_cutoff():
    """Use a budget or the result median, and require products on both sides."""
    products = [
        make_product(10000),
        make_product(15000),
        make_product(24000),
    ]

    assert _cheaper_alternative_threshold(products, "", None) == 15000
    assert _cheaper_alternative_threshold(
        products, "smartphone under ₹12,000", None
    ) == 11999
    assert _cheaper_alternative_threshold(
        [make_product(10000), make_product(10000)], "", None
    ) is None

    suggestions = _generate_follow_up_suggestions(products, "smartphone")
    assert [item["label"] for item in suggestions] == [
        "Show alternatives at or below ₹15,000"
    ]


def test_cheaper_cutoff_wording_matches_inclusive_price_filter(monkeypatch):
    """A product priced exactly at the displayed maximum remains eligible."""
    products = pd.DataFrame([
        {
            "product_name": "Phone at cutoff",
            "brand": "Example",
            "subcategory": "Smartphone",
            "key_specs": "",
            "category": "Electronics",
            "price": 15000,
            "stock_status": "in_stock",
        }
    ])
    monkeypatch.setattr(search_agent, "load_products", lambda: products)

    results = search_agent.search_products(
        query="phone", max_budget=15000, category="Electronics"
    )

    assert [product["price"] for product in results] == [15000]
    assert _suggestion_label("cheaper", "15000") == (
        "Show alternatives at or below ₹15,000"
    )


def test_validation_removes_duplicate_suggestions():
    """Keep only one validated candidate for a repeated action."""
    products = [
        make_product(18000, "Rear Camera: 50 MP"),
        make_product(22000, "Rear Camera: 64 MP"),
    ]
    duplicate_candidates = [
        {"kind": "camera", "label": "Compare rear camera details"},
        {"kind": "camera", "label": "Compare rear camera details"},
    ]

    valid = _validate_follow_up_suggestions(duplicate_candidates, products)

    assert valid == [duplicate_candidates[0]]


def test_validation_requires_specs_for_multiple_products(caplog):
    """Reject camera comparison candidates unsupported by current results."""
    products = [
        make_product(18000, "Rear Camera: 50 MP"),
        make_product(22000, "Battery: 5200 mAh"),
    ]

    valid = _validate_follow_up_suggestions(
        [{"kind": "camera", "label": "Compare rear camera details"}],
        products,
    )

    assert valid == []
    assert "Omitting unsupported or invalid" in caplog.text


def test_empty_and_malformed_results_fall_back_to_no_suggestions(caplog):
    """Return no chips safely for empty products or invalid candidates."""
    assert _generate_follow_up_suggestions([]) == []

    products = [
        make_product(18000, "Rear Camera: 50 MP"),
        make_product(22000, "Rear Camera: 64 MP"),
    ]
    invalid_candidates = [
        {"kind": "cheaper", "label": "Invented cutoff", "threshold": "invalid"},
        {"kind": "camera"},
        {"kind": [], "label": "Malformed action"},
    ]
    valid = _validate_follow_up_suggestions(invalid_candidates, products)

    assert valid == []
    assert "Omitting unsupported or invalid" in caplog.text


def test_cheaper_query_remains_exactly_the_original_search_query():
    """Do not append refinement keywords to the backend's OR keyword query."""
    query = _build_follow_up_query(
        "Samsung smartphone under 30000",
        {"kind": "cheaper", "threshold": "15000"},
    )

    assert query == "Samsung smartphone under 30000"


def test_cheaper_button_queues_query_category_and_lower_budget(monkeypatch):
    """Cheaper follow-up retains exact query/category and lowers max_budget."""
    state = {}
    monkeypatch.setattr(frontend_app.st, "session_state", state)
    context = {
        "query": "Samsung smartphone under 30000",
        "budget": 30000,
        "category": "Electronics",
    }
    suggestion = {
        "kind": "cheaper",
        "label": "Show alternatives at or below ₹15,000",
        "threshold": "15000",
    }

    _queue_cheaper_search(suggestion, context)

    pending = state.pop("pending_follow_up")
    assert pending == {
        "query": "Samsung smartphone under 30000",
        "budget": 15000.0,
        "category": "Electronics",
    }
    calls = []
    monkeypatch.setattr(frontend_app.st, "spinner", lambda *_args: nullcontext())
    monkeypatch.setattr(
        frontend_app,
        "query_chat_api",
        lambda *args: calls.append(args) or {"recommendation": "", "top_products": []},
    )
    frontend_app._execute_search(
        "http://127.0.0.1:8000",
        pending["query"],
        pending["budget"],
        pending["category"],
    )
    assert calls == [
        (
            "http://127.0.0.1:8000",
            "Samsung smartphone under 30000",
            15000.0,
            "Electronics",
        )
    ]


def test_cheaper_follow_up_search_respects_lower_budget_and_inclusive_boundary(
    monkeypatch,
):
    """The queued cheaper-search cap filters results while retaining its limit."""
    products = pd.DataFrame(
        [
            {
                "product_name": "Samsung Phone Under Cheaper Limit",
                "brand": "Samsung", "category": "Electronics",
                "subcategory": "Smartphone", "store": "Store A",
                "price": 14999, "stock_status": "in_stock",
            },
            {
                "product_name": "Samsung Phone At Cheaper Limit",
                "brand": "Samsung", "category": "Electronics",
                "subcategory": "Smartphone", "store": "Store B",
                "price": 15000, "stock_status": "in_stock",
            },
            {
                "product_name": "Samsung Phone Above Cheaper Limit",
                "brand": "Samsung", "category": "Electronics",
                "subcategory": "Smartphone", "store": "Store A",
                "price": 15001, "stock_status": "in_stock",
            },
            {
                "product_name": "Samsung Phone Above Original Limit",
                "brand": "Samsung", "category": "Electronics",
                "subcategory": "Smartphone", "store": "Store B",
                "price": 30001, "stock_status": "in_stock",
            },
        ]
    )
    monkeypatch.setattr(search_agent, "load_products", lambda: products.copy())

    results = search_agent.search_products(
        "Samsung smartphone under 30000", max_budget=15000,
        category="Electronics",
    )

    assert {product["product_name"] for product in results} == {
        "Samsung Phone Under Cheaper Limit",
        "Samsung Phone At Cheaper Limit",
    }
    assert {product["store"] for product in results} == {"Store A", "Store B"}
    assert all(product["price"] <= 15000 for product in results)


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            requests.HTTPError(
                response=SimpleNamespace(
                    status_code=404,
                    json=lambda: {"detail": "No products found for this budget."},
                )
            ),
            "No products found for this budget.",
        ),
        (
            requests.ConnectionError("backend unavailable"),
            "Unable to connect to backend",
        ),
    ],
)
def test_failed_search_clears_previous_results_and_context(
    monkeypatch, error, expected_message
):
    """A failed search cannot leave old products, comparisons, or chips on screen."""
    state = {
        "last_result": {
            "recommendation": "Old recommendation",
            "top_products": [{"product_name": "Old phone"}],
        },
        "last_search_context": {"query": "old query", "budget": 30000},
        "active_follow_up_comparison": "camera",
    }
    errors = []
    monkeypatch.setattr(frontend_app.st, "session_state", state)
    monkeypatch.setattr(frontend_app.st, "spinner", lambda *_args: nullcontext())
    monkeypatch.setattr(frontend_app.st, "error", errors.append)

    def fail_search(*_args):
        raise error

    monkeypatch.setattr(frontend_app, "query_chat_api", fail_search)

    frontend_app._execute_search(
        "http://127.0.0.1:8000",
        "phone under 15000",
        15000,
        "Electronics",
    )

    assert "last_result" not in state
    assert "last_search_context" not in state
    assert state["active_follow_up_comparison"] is None
    assert any(expected_message in message for message in errors)


def test_local_comparison_button_does_not_queue_an_api_request(monkeypatch):
    """Specification actions only select a local view over current results."""
    state = {}
    monkeypatch.setattr(frontend_app.st, "session_state", state)
    context = {"query": "camera phones under 30000"}

    _select_local_comparison(
        {"kind": "camera", "label": "Compare rear camera details"},
        context,
    )

    assert state["active_follow_up_comparison"] == "camera"
    assert "pending_follow_up" not in state


def test_local_comparison_renders_values_from_existing_products(monkeypatch):
    """Focused comparison shows returned data without making an API call."""
    rendered = {}
    monkeypatch.setattr(frontend_app.st, "markdown", lambda text: rendered.setdefault("title", text))
    monkeypatch.setattr(frontend_app.st, "table", lambda rows: rendered.setdefault("rows", rows))
    products = [
        {
            "product_name": "Phone A",
            "price": 18000,
            "key_specs": "Rear Camera: 50 MP | Battery: 5000 mAh",
        },
        {
            "product_name": "Phone B",
            "price": 22000,
            "key_specs": "Rear Camera: 64 MP | Battery: 5200 mAh",
        },
    ]

    _render_local_comparison(products, "camera")

    assert rendered["rows"] == [
        {"Product": "Phone A", "Rear Camera": "50 MP", "Price": "₹18,000"},
        {"Product": "Phone B", "Rear Camera": "64 MP", "Price": "₹22,000"},
    ]


def test_local_comparison_supports_each_specification_action():
    """Render actual values for each locally supported specification action."""
    products = [
        {
            "product_name": "Phone A",
            "price": 18000,
            "key_specs": (
                "Rear Camera: 50 MP | Battery: 5000 mAh | RAM: 8 GB | "
                "Storage: 128 GB | Processor: Snapdragon 7 Gen 3 | "
                "Charging Speed: 33 W"
            ),
        },
        {
            "product_name": "Phone B",
            "price": 22000,
            "key_specs": (
                "Rear Camera: 64 MP | Battery: 5200 mAh | RAM: 12 GB | "
                "Storage: 256 GB | Processor: Dimensity 8300 | "
                "Fast Charging: 67 W"
            ),
        },
    ]

    expected_values = {
        "camera": "50 MP",
        "battery": "5000 mAh",
        "charging_speed": "33 W",
        "ram_storage": "8 GB / 128 GB",
        "processor": "Snapdragon 7 Gen 3",
    }
    for kind, expected in expected_values.items():
        _, rows = _build_local_comparison_rows(products, kind)
        assert rows[0][next(key for key in rows[0] if key not in {"Product", "Price"})] == expected
