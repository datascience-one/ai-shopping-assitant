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

import os
from typing import Any
import requests
import json
from datetime import datetime
import pandas as pd
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


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

EXPORT_COLUMNS = [
    "product_name", "brand", "store", "price", "mrp",
    "rating", "review_count", "score", "key_specs",
    "pros", "cons",
]


def _build_export_rows(top_products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flattens the top products into rows for CSV/JSON export.

    Args:
        top_products: Product dictionaries from the backend response.

    Returns:
        list[dict]: One row per product, with a rank plus the export columns.
    """
    rows = []
    for rank, prod in enumerate(top_products, start=1):
        row = {"rank": rank}
        for col in EXPORT_COLUMNS:
            if col == "key_specs":
                specs = prod.get("key_specs") or ""
                # Replace pipe separators with line breaks
                row[col] = specs.replace(" | ", "\n")
            else:
                row[col] = prod.get(col, "")
        rows.append(row)
    return rows

def _build_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """
    Converts export rows to CSV bytes that Excel opens correctly.

    Args:
        rows: Rows produced by _build_export_rows().

    Returns:
        bytes: UTF-8 (with BOM) CSV content, so the ₹ symbol displays properly.
    """
    df = pd.DataFrame(rows).rename(
        columns={"price": "price (₹)", "mrp": "mrp (₹)"}
    )
    return df.to_csv(index=False).encode("utf-8-sig")


def _build_json_text(recommendation: str, rows: list[dict[str, Any]]) -> str:
    """
    Builds a JSON summary containing the recommendation and product rows.

    Args:
        recommendation: The assistant's recommendation text.
        rows: Rows produced by _build_export_rows().

    Returns:
        str: Pretty-printed JSON string.
    """
    return json.dumps(
        {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "recommendation": recommendation,
            "products": rows,
        },
        indent=2,
        ensure_ascii=False,
    )


def render_export_buttons(response_data: dict[str, Any]) -> None:
    """
    Renders CSV and JSON download buttons for the current results.

    Args:
        response_data: The backend ChatResponse payload.
    """
    top_products = response_data.get("top_products", [])
    if not top_products:
        return

    rows = _build_export_rows(top_products)
    csv_data = _build_csv_bytes(rows)
    json_data = _build_json_text(response_data.get("recommendation", ""), rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.markdown("### :blue[:material/upload:] Export Comparison")
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        st.download_button(
            "Download CSV", data=csv_data,
            file_name=f"comparison_{stamp}.csv", mime="text/csv", icon=":material/description:", key="download_csv",
        )
    with col2:
        st.download_button(
            "Download JSON", data=json_data,
            file_name=f"comparison_{stamp}.json", mime="application/json", icon=":material/description:", key="download_json",
        )

def render_results(response_data: dict[str, Any]) -> None:
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

    st.markdown("### 🏆 Top Ranked Products")
    if not top_products:
        st.info("No product cards returned.")
        return

    cols = st.columns(min(len(top_products), 3))
    for idx, (col, prod) in enumerate(zip(cols, top_products), start=1):
        with col:
            render_product_card(prod, rank=idx)

    render_export_buttons(response_data)

def _execute_search(
    base_url: str, query: str, budget: float | None, category: str | None
) -> None:
    """
    Executes search query against backend API with spinner and error handling.

    Args:
        base_url: Backend API base URL.
        query: User's search query string.
        budget: Maximum budget filter value.
        category: Category filter value.
    """
    with st.spinner("Searching stores, scoring specs, and generating recommendations..."):
        try:
            result = query_chat_api(base_url, query.strip(), budget, category)
            st.session_state["last_result"] = result
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

    active_query = bot_prompt or example_query or (user_query if submitted else None)
    if active_query and active_query.strip():
        _execute_search(DEFAULT_API_BASE_URL, active_query.strip(), max_budget, category)
    elif submitted and not user_query.strip():
        st.warning("Please enter a search query.")

    if st.session_state.get("last_result"):
        render_results(st.session_state["last_result"])


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
# TODO 7: Dynamic Follow-Up Quick Reply Chips
#   - Below each assistant response, dynamically generate 3-4 clickable follow-up
#     chips based on the returned phones (e.g., "Compare camera details",
#     "Show cheaper alternatives under 18000", "Check battery charging speeds").
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
