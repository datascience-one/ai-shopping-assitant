# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)

"""
frontend/app.py
---------------
Streamlit UI for the AI Shopping Assistant.

Accepts natural language smartphone queries, communicates with the FastAPI
backend (/chat endpoint), renders scored product comparison cards across
stores, and displays the LLM recommendation.

To run:
    1. Start the backend:   uvicorn backend.main:app --reload
    2. Start the UI:        python -m streamlit run frontend/app.py --server.address 127.0.0.1

Students:
- The base UI layout, CSS cards, and FastAPI communication are implemented.
- The TODO items below guide you on extending the UI with multi-turn chat,
  streaming LLM responses, store filtering, and comparison exports.
- Do not remove existing function signatures or license headers.
"""

import html
import logging
import math
import os
import re
from typing import Any

import requests
import plotly.graph_objects as go
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
logger = logging.getLogger(__name__)


def inject_custom_css() -> None:
    """
    Injects custom CSS styling for modern product cards, score badges,
    responsive containers, and interactive bot greeting bubbles.
    """
    custom_css = """
        <style>
        .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
        .sub-header { font-size: 1rem; color: #64748B; margin-bottom: 1.2rem; }
        .bot-greeting-box {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-left: 5px solid #6366F1;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 0.8rem;
        }
        .bot-bubble-text {
            color: #1E293B;
            font-size: 1rem;
            line-height: 1.5;
        }
        .recommendation-box {
            background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
            border-left: 5px solid #0284C7;
            padding: 1.2rem;
            border-radius: 8px;
            margin-bottom: 1.8rem;
            color: #0F172A;
            line-height: 1.6;
        }
        .product-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .store-badge {
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #EEF2FF;
            color: #4F46E5;
        }
        .score-badge {
            padding: 0.25rem 0.6rem;
            font-size: 0.85rem;
            font-weight: 700;
            border-radius: 6px;
            background-color: #ECFDF5;
            color: #059669;
        }
        .price-text { font-size: 1.4rem; font-weight: 700; color: #0F172A; }
        .mrp-text { font-size: 0.9rem; color: #94A3B8; text-decoration: line-through; }
        .discount-text { font-size: 0.85rem; color: #16A34A; font-weight: 600; }
        .spec-item {
            font-size: 0.85rem;
            color: #475569;
            background-color: #F8FAFC;
            padding: 0.3rem 0.5rem;
            border-radius: 4px;
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def get_bot_avatar() -> str:
    """
    Returns the file path of the bot avatar image if present,
    or a default robot emoji.

    Returns:
        str: Path to avatar image or robot emoji.
    """
    candidates = [
        "frontend/assets/bot_avatar.png",
        "frontend/assets/bot_avatar.jpg",
        "frontend/assets/bot_avatar.jpeg",
        "frontend/assets/bot_avatar.webp",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "🤖"


def render_bot_greeting() -> str | None:
    """
    Renders an interactive greeting widget where the shopping bot asks
    the user what smartphone they need, with quick suggestion chips.

    Returns:
        str | None: Selected prompt string if clicked, else None.
    """
    avatar_src = get_bot_avatar()
    col1, col2 = st.columns([1, 7])
    with col1:
        if os.path.exists(avatar_src):
            st.image(avatar_src, width=95)
        else:
            st.markdown(
                '<div style="font-size: 3.5rem; text-align: center;">🤖</div>',
                unsafe_allow_html=True,
            )
    with col2:
        st.markdown(
            """
            <div class="bot-greeting-box">
                <div class="bot-bubble-text">
                    <strong>Shopping Assistant Bot:</strong><br>
                    <em>"Hey there! 👋 What kind of smartphone can I help you find today?
                    Tell me your budget, preferred brand, or must-have specs (camera, battery,
                    5G, gaming), and I'll search and score the best options across stores!"</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**⚡ Quick Suggestions to ask:**")
    pill_cols = st.columns(4)
    quick_suggestions = [
        ("🔥 Best under ₹25k", "Best phone under 25000"),
        ("📸 Top Camera Phone", "Best camera phone under 30000"),
        ("🔋 6000mAh Battery Beast", "Smartphone with 6000mAh battery"),
        ("⚡ High-Performance 5G", "Top 5G phone with fast processor"),
    ]
    selected_query: str | None = None
    for col, (label, query) in zip(pill_cols, quick_suggestions):
        with col:
            if st.button(label, key=f"quick_prompt_{label}", use_container_width=True):
                selected_query = query
    return selected_query


def check_backend_health(base_url: str = DEFAULT_API_BASE_URL) -> bool:
    """
    Checks if the FastAPI backend server is reachable.

    Args:
        base_url: The base URL of the FastAPI backend.

    Returns:
        bool: True if backend responds with status ok, False otherwise.
    """
    try:
        res = requests.get(f"{base_url}/health", timeout=2.0)
        return res.status_code == 200 and res.json().get("status") == "ok"
    except Exception:
        return False


def query_chat_api(
    base_url: str,
    message: str,
    max_budget: float | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Calls the POST /chat endpoint on the backend pipeline.

    Args:
        base_url: The base URL of the backend server.
        message: The user's query text (e.g. "Best phone under 20000").
        max_budget: Optional budget limit in INR.
        category: Optional product category filter.

    Returns:
        dict: The response payload containing recommendation and top_products.

    Raises:
        requests.HTTPError: If backend responds with an error code (e.g., 404).
        requests.ConnectionError: If backend server is offline or unreachable.
    """
    payload: dict[str, Any] = {"message": message}
    if max_budget is not None and max_budget > 0:
        payload["max_budget"] = max_budget
    if category and category.strip():
        payload["category"] = category.strip()

    response = requests.post(f"{base_url}/chat", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _render_sidebar_status(base_url: str) -> None:
    """Renders backend connection status in the sidebar."""
    if check_backend_health(base_url):
        st.sidebar.success("Backend: Online (http://127.0.0.1:8000)", icon="🟢")
    else:
        st.sidebar.error("Backend: Offline", icon="🔴")
        st.sidebar.caption("Run `uvicorn backend.main:app --reload` to start.")


def _render_example_queries() -> str | None:
    """Renders clickable quick example query buttons in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Example Queries")
    examples = [
        "Best Samsung phone under 25000",
        "5G phone with great battery backup",
        "Top camera phone under 30000",
        "Budget smartphone under 15000",
    ]
    for ex in examples:
        if st.sidebar.button(ex, use_container_width=True):
            return ex
    return None


def render_sidebar(base_url: str) -> tuple[float | None, str | None, str | None]:
    """
    Renders the Streamlit sidebar containing controls, filters, and status.

    Args:
        base_url: Backend API base URL.

    Returns:
        tuple: (max_budget, category, selected_example_query)
    """
    st.sidebar.title("⚙️ Settings & Filters")
    _render_sidebar_status(base_url)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters (Optional)")
    budget_on = st.sidebar.checkbox("Enable Budget Filter", value=False)
    max_budget = None
    if budget_on:
        max_budget = float(
            st.sidebar.number_input("Max Budget (₹)", 5000, 150000, 25000, 1000)
        )

    category = st.sidebar.text_input("Category Filter", value="").strip() or None
    example_query = _render_example_queries()
    return max_budget, category, example_query


def _build_card_html(prod: dict[str, Any], rank: int) -> str:
    """
    Builds HTML markup for a single product card with safe null handling.

    Args:
        prod: Product data dictionary.
        rank: Rank position (1, 2, 3).

    Returns:
        str: HTML markup string.
    """
    name = prod.get("product_name") or "Unknown Product"
    brand = prod.get("brand") or "Unknown Brand"
    store = prod.get("store") or "Store A"
    price = prod.get("price") or 0.0
    mrp = prod.get("mrp") or 0.0
    score = prod.get("score")
    rating = prod.get("rating") or 0.0
    reviews = prod.get("review_count") or 0
    specs = prod.get("key_specs") or "Standard specifications"

    disc = int(round((1 - price / mrp) * 100)) if (mrp and price and mrp > price > 0) else 0
    mrp_html = f'<span class="mrp-text">₹{mrp:,.0f}</span>' if (mrp and price and mrp > price) else ''
    disc_html = f'<span class="discount-text">({disc}% OFF)</span>' if disc > 0 else ''
    score_display = f"{score:.1f}/100" if isinstance(score, (int, float)) else "N/A"

    return f"""
    <div class="product-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span class="store-badge">🏪 {store}</span>
            <span class="score-badge">⭐ Score: {score_display}</span>
        </div>
        <h4 style="margin: 0.3rem 0; color: #1E293B;">#{rank} {name}</h4>
        <p style="color: #64748B; font-size: 0.85rem;">by <strong>{brand}</strong></p>
        <div style="margin-bottom: 0.8rem;">
            <span class="price-text">₹{price:,.0f}</span> {mrp_html} {disc_html}
        </div>
        <div style="margin-bottom: 0.8rem; font-size: 0.85rem; color: #D97706;">
            ★ {rating:.1f} / 5.0 <span style="color: #64748B;">({reviews:,} reviews)</span>
        </div>
        <div class="spec-item"><strong>Specs:</strong> {specs}</div>
    </div>
    """


def render_product_card(product: dict[str, Any], rank: int) -> None:
    """
    Renders a single product comparison card with expandable pros/cons.

    Args:
        product: Dictionary representing the ProductResult model.
        rank: The position rank (1, 2, 3) of the product.
    """
    st.markdown(_build_card_html(product, rank), unsafe_allow_html=True)
    pros, cons = product.get("pros", ""), product.get("cons", "")
    brand = product.get("brand", "")
    with st.expander(f"Pros & Cons for #{rank} {brand}"):
        if pros:
            st.markdown(f"**✅ Pros:** {pros}")
        if cons:
            st.markdown(f"**⚠️ Cons:** {cons}")


def _parse_product_specs(product: dict[str, Any]) -> dict[str, float | str | None]:
    """Extract comparable numeric specifications from a product's key_specs."""
    key_specs = str(product.get("key_specs") or "")
    specs: dict[str, float | str | None] = {
        "spec_ram": None,
        "spec_storage": None,
        "spec_battery": None,
        "spec_rear_camera": None,
        "spec_display_size": None,
        "processor": None,
        "charging_speed": None,
    }

    for field in ("charging_speed", "charging_speed_w", "charging_wattage"):
        charging_speed = _extract_wattage(product.get(field), numeric_field=True)
        if charging_speed is not None:
            specs["charging_speed"] = charging_speed
            break

    for item in key_specs.split("|"):
        item = item.strip()

        if ":" not in item:
            continue

        label, value = item.split(":", 1)
        label = label.strip().lower()
        value = value.strip()

        try:
            if label == "ram":
                specs["spec_ram"] = float(value.lower().replace("gb", "").strip())

            elif label == "storage":
                specs["spec_storage"] = float(
                    value.lower().replace("gb", "").strip()
                )

            elif label == "battery":
                specs["spec_battery"] = float(
                    value.lower().replace("mah", "").strip()
                )

            elif label == "display":
                specs["spec_display_size"] = float(
                    value.lower().replace("inches", "").replace("inch", "").strip()
                )

            elif label == "rear camera":
                camera_value = value.lower().replace("mp", "").strip()
                first_camera = camera_value.split("+")[0].strip()
                specs["spec_rear_camera"] = float(first_camera)

            elif label == "processor":
                specs["processor"] = value

            elif "charg" in label:
                charging_speed = _extract_wattage(value)
                if charging_speed is not None:
                    specs["charging_speed"] = charging_speed

        except ValueError:
            continue

    return specs


def _comparison_number(value: Any) -> float | None:
    """Return a finite numeric value, or None when it is unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_wattage(value: Any, numeric_field: bool = False) -> float | None:
    """Extract a positive wattage only from explicit charging data."""
    if numeric_field:
        numeric_value = _comparison_number(value)
        if numeric_value is not None and numeric_value > 0:
            return numeric_value
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:w|watts?)\b", str(value or ""), re.I)
    if not match:
        return None
    wattage = _comparison_number(match.group(1))
    return wattage if wattage is not None and wattage > 0 else None


def _charging_speed_available(product: dict[str, Any]) -> bool:
    """Return whether a product includes explicit charging-speed information."""
    for field in ("charging_speed", "charging_speed_w", "charging_wattage"):
        if _extract_wattage(product.get(field), numeric_field=True) is not None:
            return True

    for item in str(product.get("key_specs") or "").split("|"):
        if ":" not in item:
            continue
        label, value = (part.strip().lower() for part in item.split(":", 1))
        if "charg" in label and _extract_wattage(value) is not None:
            return True
    return False


def _spec_value_available(value: Any) -> bool:
    """Return whether a parsed numeric or text specification is meaningful."""
    number = _comparison_number(value)
    if number is not None:
        return number > 0
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "-",
            "n/a",
            "na",
            "none",
            "unknown",
            "not available",
            "not specified",
        }
    return False


def _specification_coverage(
    products: list[dict[str, Any]],
) -> dict[str, int]:
    """Count products with usable values for each comparable specification."""
    valid_products = [product for product in products if isinstance(product, dict)]
    parsed = [_parse_product_specs(product) for product in valid_products]
    return {
        "camera": sum(
            _spec_value_available(specs["spec_rear_camera"]) for specs in parsed
        ),
        "battery": sum(
            _spec_value_available(specs["spec_battery"]) for specs in parsed
        ),
        "ram_storage": sum(
            _spec_value_available(specs["spec_ram"])
            and _spec_value_available(specs["spec_storage"])
            for specs in parsed
        ),
        "processor": sum(
            _spec_value_available(specs["processor"])
            for specs in parsed
        ),
        "charging_speed": sum(
            _charging_speed_available(product) for product in valid_products
        ),
    }


def _extract_budget_from_query(query: str) -> float | None:
    """Extract an explicit under/below budget from the user's query."""
    match = re.search(
        r"\b(?:under|below|less than)\s*(?:₹|rs\.?\s*)?"
        r"([\d,]+(?:\.\d+)?)\s*(k)?\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    amount = _comparison_number(match.group(1).replace(",", ""))
    if amount is None:
        return None
    return amount * 1000 if match.group(2) else amount


def _cheaper_alternative_threshold(
    products: list[dict[str, Any]],
    original_query: str,
    original_budget: float | None,
) -> int | None:
    """Derive a lower request budget from returned prices and the prior budget."""
    prices = sorted(
        price
        for product in products
        if isinstance(product, dict)
        if (price := _comparison_number(product.get("price"))) is not None and price > 0
    )
    if len(set(prices)) < 2:
        return None

    middle = len(prices) // 2
    median_price = (
        prices[middle]
        if len(prices) % 2
        else (prices[middle - 1] + prices[middle]) / 2
    )
    budget = _comparison_number(original_budget)
    if budget is None or budget <= 0:
        budget = _extract_budget_from_query(original_query)
    threshold = math.floor(min(median_price, budget)) if budget else math.floor(median_price)
    if budget is not None and threshold >= budget:
        threshold = math.floor(budget) - 1
    if threshold <= 0:
        return None
    if not any(price < threshold for price in prices):
        return None
    if not any(price >= threshold for price in prices):
        return None
    return threshold


def _suggestion_label(kind: str, threshold: str | None = None) -> str | None:
    """Return the user-facing label for a supported suggestion type."""
    labels = {
        "camera": "Compare rear camera details",
        "charging_speed": "Compare charging speeds",
        "battery": "Compare battery capacities",
        "ram_storage": "Compare RAM and storage",
        "processor": "Compare processor details",
    }
    if not isinstance(kind, str):
        return None
    if kind == "cheaper":
        amount = _comparison_number(threshold)
        if amount is not None and amount > 0:
            return f"Show alternatives at or below ₹{int(amount):,}"
    return labels.get(kind)


def _validate_follow_up_suggestions(
    suggestions: list[dict[str, str]],
    products: list[dict[str, Any]],
    original_query: str = "",
    original_budget: float | None = None,
) -> list[dict[str, str]]:
    """Keep distinct suggestions only when returned products support them."""
    coverage = _specification_coverage(products)
    valid: list[dict[str, str]] = []
    seen: set[str] = set()
    threshold = _cheaper_alternative_threshold(products, original_query, original_budget)
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            logger.warning("Omitting malformed follow-up suggestion.")
            continue
        kind = suggestion.get("kind", "")
        if not isinstance(kind, str):
            logger.warning("Omitting follow-up suggestion with an invalid action.")
            continue
        candidate_threshold = suggestion.get("threshold")
        expected_label = _suggestion_label(kind, candidate_threshold)
        supported = (
            kind == "cheaper"
            and expected_label is not None
            and threshold is not None
            and _comparison_number(candidate_threshold) == threshold
        )
        if kind in coverage:
            supported = coverage[kind] >= 2 and expected_label is not None
        if not supported or suggestion.get("label") != expected_label:
            logger.warning("Omitting unsupported or invalid follow-up suggestion: %s", kind)
            continue
        if kind in seen:
            logger.info("Omitting duplicate follow-up suggestion: %s", kind)
            continue
        valid.append(suggestion)
        seen.add(kind)
    return valid


def _build_follow_up_candidates(
    products: list[dict[str, Any]],
    coverage: dict[str, int],
    original_query: str,
    original_budget: float | None,
) -> list[dict[str, str]]:
    """Create candidate actions from specification coverage and price data."""
    candidates: list[dict[str, str]] = []
    for kind in ("camera", "charging_speed", "battery", "processor", "ram_storage"):
        if coverage[kind] >= 2:
            label = _suggestion_label(kind)
            if label:
                candidates.append({"kind": kind, "label": label})
        else:
            logger.debug(
                "Excluding %s follow-up: available for fewer than two products.",
                kind,
            )

    threshold = _cheaper_alternative_threshold(products, original_query, original_budget)
    if threshold is not None:
        label = _suggestion_label("cheaper", str(threshold))
        if label:
            candidates.insert(1, {
                "kind": "cheaper",
                "label": label,
                "threshold": str(threshold),
            })
    else:
        logger.debug("Excluding cheaper-alternatives follow-up: no supported cutoff.")
    return candidates


def _generate_follow_up_suggestions(
    products: list[dict[str, Any]],
    original_query: str = "",
    original_budget: float | None = None,
) -> list[dict[str, str]]:
    """Select up to four follow-ups grounded in returned product data."""
    valid_products = [product for product in products if isinstance(product, dict)]
    if not valid_products:
        logger.info("No follow-up suggestions: the search returned no products.")
        return []
    if len(valid_products) != len(products):
        logger.warning("Ignoring malformed product results while generating follow-ups.")

    coverage = _specification_coverage(valid_products)
    prices = [
        _comparison_number(product.get("price"))
        for product in valid_products
        if _comparison_number(product.get("price")) is not None
    ]
    logger.info(
        "Follow-up evidence: products=%d, prices=%d, specification_coverage=%s",
        len(valid_products),
        len(prices),
        coverage,
    )
    candidates = _build_follow_up_candidates(
        valid_products, coverage, original_query, original_budget
    )
    validated = _validate_follow_up_suggestions(
        candidates, valid_products, original_query, original_budget
    )
    selected = validated[:4]
    logger.info(
        "Selected follow-up suggestions: %s",
        [suggestion["label"] for suggestion in selected],
    )
    return selected


def _build_follow_up_query(previous_query: str, suggestion: dict[str, str]) -> str:
    """Return the unchanged original query for the cheaper-results request."""
    if suggestion.get("kind") != "cheaper":
        return ""
    return previous_query.strip()


def _select_local_comparison(
    suggestion: dict[str, str], search_context: dict[str, Any]
) -> None:
    """Select a supported local comparison without calling the backend."""
    allowed_kinds = {"camera", "charging_speed", "battery", "ram_storage", "processor"}
    kind = suggestion.get("kind") if isinstance(suggestion, dict) else None
    if kind not in allowed_kinds:
        logger.warning("Ignoring unsupported local follow-up action.")
        return
    if not isinstance(search_context, dict) or not search_context.get("query"):
        logger.warning("Ignoring local follow-up because search context is missing.")
        return
    st.session_state["active_follow_up_comparison"] = kind


def _queue_cheaper_search(
    suggestion: dict[str, str], search_context: dict[str, Any]
) -> None:
    """Queue a cheaper search using the original query and category."""
    if not isinstance(suggestion, dict) or not isinstance(search_context, dict):
        logger.warning("Ignoring malformed follow-up click.")
        return
    previous_query = str(search_context.get("query") or "").strip()
    if not previous_query:
        logger.warning("Ignoring follow-up click because search context is missing.")
        return

    query = _build_follow_up_query(previous_query, suggestion)
    threshold = _comparison_number(suggestion.get("threshold"))
    if not query or threshold is None or threshold <= 0:
        logger.warning("Ignoring cheaper follow-up because its request is invalid.")
        return
    original_budget = _comparison_number(search_context.get("budget"))
    if original_budget is not None and threshold >= original_budget:
        logger.warning("Ignoring cheaper follow-up that does not lower the original budget.")
        return
    st.session_state["pending_follow_up"] = {
        "query": query,
        "budget": min(threshold, original_budget)
        if original_budget is not None
        else threshold,
        "category": search_context.get("category"),
    }


def _render_local_comparison(
    products: list[dict[str, Any]], kind: str | None
) -> None:
    """Render a focused comparison using only the currently returned products."""
    label, rows = _build_local_comparison_rows(products, kind)
    if len(rows) >= 2:
        st.markdown(f"#### {label} comparison")
        st.table(rows)


def _build_local_comparison_rows(
    products: list[dict[str, Any]], kind: str | None
) -> tuple[str, list[dict[str, str]]]:
    """Build focused rows from available product specs for one follow-up action."""
    spec_labels = {
        "camera": ("Rear Camera", "spec_rear_camera", "MP"),
        "charging_speed": ("Charging Speed", "charging_speed", "W"),
        "battery": ("Battery Capacity", "spec_battery", "mAh"),
        "processor": ("Processor", "processor", ""),
    }
    if kind == "ram_storage":
        label = "RAM / Storage"
        keys = ("spec_ram", "spec_storage")
    elif kind in spec_labels:
        label, key, unit = spec_labels[kind]
        keys = (key,)
    else:
        return "", []

    parsed_products = [
        (product, _parse_product_specs(product))
        for product in products
        if isinstance(product, dict)
    ]
    coverage = _specification_coverage([product for product, _ in parsed_products])
    required_coverage = (
        coverage["ram_storage"] if kind == "ram_storage" else coverage.get(kind, 0)
    )
    if required_coverage < 2:
        logger.info("Skipping local %s comparison: insufficient product data.", kind)
        return label, []

    rows = []
    for product, specs in parsed_products:
        values = [specs.get(key) for key in keys]
        if not all(_spec_value_available(value) for value in values):
            continue
        if kind == "ram_storage":
            value_text = f"{values[0]:g} GB / {values[1]:g} GB"
        elif kind == "processor":
            value_text = str(values[0])
        else:
            value_text = f"{values[0]:g} {unit}"
        rows.append({
            "Product": str(product.get("product_name") or "Product"),
            label: value_text,
            "Price": f"₹{_comparison_number(product.get('price')):,.0f}"
            if _comparison_number(product.get("price")) is not None
            else "N/A",
        })

    return label, rows


def render_follow_up_suggestions(
    response_data: dict[str, Any],
    search_context: dict[str, Any] | None = None,
) -> None:
    """Render validated follow-up buttons below the assistant response."""
    products = response_data.get("top_products", [])
    if not isinstance(products, list):
        logger.warning("Omitting follow-up suggestions: product results are invalid.")
        return
    context = search_context or {}
    query = str(context.get("query") or response_data.get("user_message") or "").strip()
    if not query:
        logger.warning("Omitting follow-up suggestions: original search context is missing.")
        return

    suggestions = _generate_follow_up_suggestions(
        products, query, _comparison_number(context.get("budget"))
    )
    if not suggestions:
        return

    st.markdown("**You might also try:**")
    columns = st.columns(min(len(suggestions), 4))
    for index, (column, suggestion) in enumerate(zip(columns, suggestions)):
        callback = (
            _queue_cheaper_search
            if suggestion["kind"] == "cheaper"
            else _select_local_comparison
        )
        with column:
            st.button(
                suggestion["label"],
                key=f"follow_up_{index}_{suggestion['kind']}",
                on_click=callback,
                args=(suggestion, context),
                width="stretch",
            )
    _render_local_comparison(
        products,
        st.session_state.get("active_follow_up_comparison"),
    )


def _build_comparison_rows(
    products: list[dict[str, Any]],
    parsed_specs: list[dict[str, float | str | None]],
) -> list[tuple[str, list[float | None], str, bool]]:
    """Build comparable specification values for the product table."""
    spec_rows = [
        ("RAM", "spec_ram", "GB"),
        ("Storage", "spec_storage", "GB"),
        ("Battery", "spec_battery", "mAh"),
        ("Rear Camera", "spec_rear_camera", "MP"),
        ("Display Size", "spec_display_size", "in"),
    ]
    rows = [
        (label, [_comparison_number(specs[key]) for specs in parsed_specs], unit, False)
        for label, key, unit in spec_rows
    ]
    rows.append(
        ("Score", [_comparison_number(product.get("score")) for product in products],
         "/100", False)
    )

    price_per_gb = []
    for product, specs in zip(products, parsed_specs):
        price = _comparison_number(product.get("price"))
        storage = _comparison_number(specs["spec_storage"])
        price_per_gb.append(
            price / storage
            if price is not None and price > 0 and storage is not None and storage > 0
            else None
        )
    rows.append(("Price per GB", price_per_gb, "₹/GB", True))
    return rows


def _comparison_cell(
    value: float | None, label: str, unit: str, best_value: float | None
) -> str:
    """Build an escaped table cell and highlight it when it is best in its row."""
    if value is None:
        display_value = "N/A"
    elif label == "Score":
        display_value = f"{value:.1f}{unit}"
    elif label == "Price per GB":
        display_value = f"₹{value:,.2f}/GB"
    else:
        display_value = f"{value:g} {unit}"

    is_best = value is not None and value == best_value
    background = "background-color: #DCFCE7; color: #166534;" if is_best else ""
    trophy = " 🏆" if is_best else ""
    return (
        f"<td style='padding: 0.65rem; border-top: 1px solid #E2E8F0; "
        f"{background}'>{html.escape(display_value)}{trophy}</td>"
    )


def _comparison_table_rows(
    rows: list[tuple[str, list[float | None], str, bool]],
) -> str:
    """Render comparison rows and mark the best available value in each row."""
    table_rows = []
    for label, values, unit, lowest_is_best in rows:
        available_values = [value for value in values if value is not None]
        best_value = None
        if available_values:
            best_value = min(available_values) if lowest_is_best else max(available_values)
        cells = "".join(
            _comparison_cell(value, label, unit, best_value) for value in values
        )
        table_rows.append(
            "<tr>"
            "<th style='padding: 0.65rem; text-align: left; "
            "border-top: 1px solid #E2E8F0;'>"
            f"{html.escape(label)}</th>{cells}</tr>"
        )
    return "".join(table_rows)


def render_spec_comparison(top_products: list[dict[str, Any]]) -> None:
    """Render a side-by-side comparison of the top products' specifications."""
    products = top_products[:3]
    if not products:
        return

    parsed_specs = [_parse_product_specs(product) for product in products]
    rows = _build_comparison_rows(products, parsed_specs)
    headers = "".join(
        "<th style='padding: 0.65rem; text-align: left;'>"
        f"#{index} {html.escape(str(product.get('product_name') or 'Product'))}"
        "</th>"
        for index, product in enumerate(products, start=1)
    )
    table_rows = _comparison_table_rows(rows)

    st.markdown("### 📋 Spec Comparison")
    st.markdown(
        "<div style='overflow-x: auto;'><table style='width: 100%; "
        "border-collapse: collapse;'><thead><tr>"
        "<th style='padding: 0.65rem; text-align: left;'>Specification</th>"
        f"{headers}</tr></thead><tbody>{table_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_spec_radar_chart(top_products: list[dict[str, Any]]) -> None:
    """Render an interactive radar chart comparing the top products."""
    if not top_products:
        return

    for product in top_products:
        product.update(_parse_product_specs(product))

    st.markdown("### 📊 Visual Specification Comparison")

    dimensions = [
        "Camera",
        "Battery",
        "Performance",
        "Display",
        "Value for Money",
    ]

    def performance_score(product: dict[str, Any]) -> float:
        """Estimate processor performance for visualization."""
        processor = str(product.get("processor") or "").lower()

        performance_tiers = [
            ("snapdragon 8 gen", 100),
            ("snapdragon 7s gen", 92),
            ("snapdragon 7 gen", 90),
            ("tensor g5", 95),
            ("tensor", 80),
            ("dimensity 9", 90),
            ("dimensity 8", 85),
            ("dimensity 7", 75),
            ("dimensity 6", 65),
            ("exynos 13", 70),
            ("exynos", 65),
            ("snapdragon 6", 65),
            ("snapdragon 4", 50),
            ("helio", 45),
            ("unisoc t7", 45),
            ("unisoc", 40),
        ]

        for processor_family, score in performance_tiers:
            if processor_family in processor:
                return float(score)

        return 50.0

    products_for_chart = top_products[:3]

    price_per_gb_values = []
    for product in products_for_chart:
        price = float(product.get("price") or 0)
        storage = float(product.get("spec_storage") or 0)

        if price > 0 and storage > 0:
            price_per_gb_values.append(price / storage)

    best_price_per_gb = min(price_per_gb_values) if price_per_gb_values else 0

    def radar_values(product: dict[str, Any]) -> list[float]:
        """Convert product specifications into 0-100 radar-chart scores."""
        camera = min(
            float(product.get("spec_rear_camera") or 0) / 200 * 100,
            100,
        )

        battery = min(
            float(product.get("spec_battery") or 0) / 7000 * 100,
            100,
        )

        display = min(
            float(product.get("spec_display_size") or 0) / 7.0 * 100,
            100,
        )

        price = float(product.get("price") or 0)
        storage = float(product.get("spec_storage") or 0)

        if price > 0 and storage > 0 and best_price_per_gb > 0:
            price_per_gb = price / storage
            value = min(100, (best_price_per_gb / price_per_gb) * 100)
        else:
            value = 0.0

        return [
            round(camera, 1),
            round(battery, 1),
            round(performance_score(product), 1),
            round(display, 1),
            round(value, 1),
        ]

    fig = go.Figure()

    for index, product in enumerate(products_for_chart, start=1):
        product_name = product.get("product_name") or f"Product {index}"
        values = radar_values(product)

        closed_dimensions = dimensions + [dimensions[0]]
        closed_values = values + [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=closed_values,
                theta=closed_dimensions,
                fill="toself",
                name=f"#{index} {product_name}",
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    "Score: %{r:.1f}/100"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            )
        ),
        showlegend=True,
        height=550,
        margin=dict(l=60, r=60, t=50, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_results(
    response_data: dict[str, Any],
    search_context: dict[str, Any] | None = None,
) -> None:
    """
    Renders the recommendation box and top product cards grid.

    Args:
        response_data: The backend ChatResponse payload.
    """
    recommendation = response_data.get("recommendation", "")
    top_products = response_data.get("top_products", [])

    st.markdown("### 💬 Assistant Recommendation")
    with st.chat_message("assistant", avatar=get_bot_avatar()):
        st.markdown(
            f'<div class="recommendation-box">{recommendation}</div>',
            unsafe_allow_html=True,
        )

    render_follow_up_suggestions(response_data, search_context)

    st.markdown("### 🏆 Top Ranked Products")
    if not top_products:
        st.info("No product cards returned.")
        return

    cols = st.columns(min(len(top_products), 3))
    for idx, (col, prod) in enumerate(zip(cols, top_products), start=1):
        with col:
            render_product_card(prod, rank=idx)

    render_spec_comparison(top_products)
    render_spec_radar_chart(top_products)


def _execute_search(
    base_url: str,
    query: str,
    budget: float | None,
    category: str | None,
) -> None:
    """
    Executes search query against backend API with spinner and error handling.

    Args:
        base_url: Backend API base URL.
        query: User's search query string.
        budget: Maximum budget filter value.
        category: Category filter value.
    """
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_search_context", None)
    st.session_state["active_follow_up_comparison"] = None

    with st.spinner("Searching stores, scoring specs, and generating recommendations..."):
        try:
            result = query_chat_api(base_url, query.strip(), budget, category)
            st.session_state["last_result"] = result
            st.session_state["active_follow_up_comparison"] = None
            st.session_state["last_search_context"] = {
                "query": query.strip(),
                "budget": budget,
                "category": category,
            }
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                detail = err.response.json().get("detail", "No products found.")
                st.error(f"🔍 {detail}")
            else:
                st.error(f"API Error: {err}")
        except requests.ConnectionError:
            st.error(
                "❌ Unable to connect to backend at http://127.0.0.1:8000.\n\n"
                "Please run: `uvicorn backend.main:app --reload`"
            )
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️", layout="wide")
    inject_custom_css()

    st.markdown('<div class="main-header">🛍️ AI Shopping Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-store intelligent smartphone comparison powered by agentic AI</div>',
        unsafe_allow_html=True,
    )

    max_budget, category, example_query = render_sidebar(DEFAULT_API_BASE_URL)
    bot_prompt = render_bot_greeting()

    if "current_query" not in st.session_state or example_query:
        st.session_state["current_query"] = example_query or ""
    if bot_prompt:
        st.session_state["current_query"] = bot_prompt

    with st.form(key="search_form", clear_on_submit=False):
        user_query = st.text_input(
            "What smartphone are you looking for?",
            value=st.session_state["current_query"],
            placeholder="e.g. Best 5G phone under 20000 with good camera",
        )
        submitted = st.form_submit_button("Search & Compare", use_container_width=False)

    pending_follow_up = st.session_state.pop("pending_follow_up", None)
    active_query = bot_prompt or example_query or (user_query if submitted else None)
    if pending_follow_up:
        _execute_search(
            DEFAULT_API_BASE_URL,
            pending_follow_up["query"],
            pending_follow_up["budget"],
            pending_follow_up["category"],
        )
    elif active_query and active_query.strip():
        _execute_search(DEFAULT_API_BASE_URL, active_query.strip(), max_budget, category)
    elif submitted and not user_query.strip():
        st.warning("Please enter a search query.")

    if st.session_state.get("last_result"):
        render_results(
            st.session_state["last_result"],
            st.session_state.get("last_search_context"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Multi-Turn Conversation History (Phase 3)
#   - Replace the static single-query form with a conversational interface
#     using `st.chat_message` and `st.chat_input`.
#   - Store message turns in `st.session_state["messages"] = []`.
#   - Pass a `session_id` to `query_chat_api()` so the backend can maintain
#     memory across multiple user queries (e.g. follow-up questions like:
#     "Which one of these has better low-light photography?").
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: LLM Streaming Recommendation Output
#   - Modify `render_results()` to stream the recommendation token-by-token
#     using `st.write_stream()` instead of rendering a static HTML box.
#   - Backend requirement: Add a Server-Sent Events (SSE) or streaming response
#     endpoint (e.g., `POST /chat/stream`) in `backend/main.py`.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Store Selector & Store Filter
#   - Add a multi-select filter in `render_sidebar()` for stores
#     (e.g., Store A: Amazon, Store B: Flipkart, Store C: Croma).
#   - Allow the user to select/deselect specific stores and pass the
#     store filter to the backend or filter results locally.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Spec Comparison Table & Value-for-Money Metrics
#   - In addition to product cards, render a side-by-side spec comparison table
#     comparing RAM, Storage, Battery, Camera, Display size, and Score.
#   - Highlight the best spec in each row (e.g. green background for highest
#     battery mAh or lowest price-per-GB).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 5: Export Comparison Results
#   - Add a download button (`st.download_button`) below the comparison cards
#     allowing users to export the current recommendation and product list
#     as a CSV or JSON summary.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 6: Interactive Chat Window & Multi-Turn Turns (`st.chat_message`)
#   - Implement full chat feed with user & bot messages:
#     `st.chat_message("user")` and `st.chat_message("assistant", avatar=get_bot_avatar())`.
#   - Preserve conversation flow and render historical product cards inline.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Implemented above: dynamic follow-up quick reply chips are generated from
# returned product specifications and prices in `render_follow_up_suggestions()`.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 8: Customizable Bot Avatar, Persona & Personality Selection
#   - Allow users to select bot personality via sidebar (e.g., 'Tech Geek',
#     'Bargain Hunter', 'Everyday User').
#   - Support uploading custom bot avatar images (`st.file_uploader`) into
#     `frontend/assets/bot_avatar.png` or switching avatars dynamically.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 9: Visual Spec Radar / Comparison Chart
#   - Render an interactive visual chart (e.g., using Streamlit's native chart
#     or Plotly) comparing the top 3 phones across 5 core dimensions:
#     Camera, Battery, Performance, Display, and Value for Money.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 10: Recommendation Feedback & Quality Rating
#   - Add an interactive rating widget (`st.feedback("thumbs")` or 1-5 stars)
#     under each recommendation to capture intern/user evaluation data.
#   - Store ratings for eval validation in `tests/test_evals.py`.
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
