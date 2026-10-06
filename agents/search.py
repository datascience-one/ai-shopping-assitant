#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##

"""
agents/search.py
----------------
Search Agent — filters products based on user query.

Phase 2 update: now reads from the live WooCommerce Store A REST API
instead of the static products.csv. Falls back to CSV automatically
if WooCommerce is unreachable or no API keys are configured — this
keeps Phase 1 workflows (and offline testing) working without changes.

This agent is responsible for:
- Loading products (live from WooCommerce, or from CSV as fallback)
- Parsing the user's intent (category, budget, keywords)
- Returning a filtered list of matching products

Students:
- The basic filtering logic is implemented below.
- TODO items show you where to extend and improve the logic.
- Do not change the function signatures — other agents depend on them.
"""

import pandas as pd
import os
import re
import requests
from requests.auth import HTTPBasicAuth

import config

# Path to the product dataset (fallback only)
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/products.csv")

# How long to wait for WooCommerce before falling back to CSV
WC_TIMEOUT_SECONDS = 5


# ─────────────────────────────────────────────────────────────────────────────
# WooCommerce live data loading (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _get_meta_value(meta_data: list[dict], key: str, default=""):
    """
    Reads a single value out of WooCommerce's meta_data list format.

    WooCommerce returns meta_data as a list like:
        [{"key": "dsa_brand", "value": "Samsung"}, ...]
    This helper finds the value for a given key, or returns a default.
    """
    for item in meta_data:
        if item.get("key") == key:
            return item.get("value", default)
    return default


def _map_wc_product_to_dict(wc_product: dict) -> dict:
    """
    Converts a single WooCommerce product response into our internal
    product dict shape — the same shape agents/compare.py expects,
    whether the data came from CSV or from WooCommerce.

    Args:
        wc_product: One product object from the WooCommerce REST API

    Returns:
        dict matching our standard product schema
    """
    meta = wc_product.get("meta_data", [])

    def num(val, cast=float, fallback=0):
        try:
            return cast(val)
        except (TypeError, ValueError):
            return fallback

    return {
        "product_id":       str(wc_product.get("id", "")),
        "product_name":     wc_product.get("name", ""),
        "brand":             _get_meta_value(meta, "dsa_brand"),
        "category":          _get_meta_value(meta, "dsa_category"),
        "subcategory":       _get_meta_value(meta, "dsa_subcategory"),
        "store":             _get_meta_value(meta, "dsa_store", "Store A"),
        "price":             num(wc_product.get("sale_price") or wc_product.get("price"), float, 0.0),
        "mrp":               num(wc_product.get("regular_price"), float, 0.0),
        "rating":            num(_get_meta_value(meta, "dsa_rating", "0"), float, 0.0),
        "review_count":      num(_get_meta_value(meta, "dsa_review_count", "0"), int, 0),
        "stock_status":      "out_of_stock" if wc_product.get("stock_status") == "outofstock" else "in_stock",
        "key_specs":         wc_product.get("short_description", "") or "",
        "pros":              _get_meta_value(meta, "dsa_pros"),
        "cons":              _get_meta_value(meta, "dsa_cons"),
        "spec_confidence":   _get_meta_value(meta, "dsa_spec_confidence", "low"),
        "spec_ram":          num(_get_meta_value(meta, "dsa_spec_ram", "0"), float, 0.0),
        "spec_storage":      num(_get_meta_value(meta, "dsa_spec_storage", "0"), float, 0.0),
        "spec_battery":      num(_get_meta_value(meta, "dsa_spec_battery", "0"), float, 0.0),
        "spec_rear_camera":  num(_get_meta_value(meta, "dsa_spec_rear_camera", "0"), float, 0.0),
        "spec_display_size": num(_get_meta_value(meta, "dsa_spec_display_size", "0"), float, 0.0),
    }


def _fetch_store_products(store_url: str, key: str, secret: str, per_page: int = 100) -> list[dict]:
    """
    Fetches all products from a single WooCommerce store, handling pagination.

    Args:
        store_url : Base URL of the store, e.g. http://localhost:8080
        key       : WooCommerce consumer key
        secret    : WooCommerce consumer secret
        per_page  : Products per page (WooCommerce max is 100)

    Returns:
        List of raw WooCommerce product dicts. Empty list on any failure.
    """
    if not key or not secret:
        return []

    all_products = []
    page = 1

    try:
        while True:
            response = requests.get(
                f"{store_url}/wp-json/wc/v3/products",
                params={"per_page": per_page, "page": page},
                auth=HTTPBasicAuth(key, secret),
                timeout=WC_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                break

            products = response.json()
            if not products:
                break

            all_products.extend(products)

            if len(products) < per_page:
                break
            page += 1

    except requests.exceptions.RequestException:
        # Store unreachable, timed out, or connection refused — caller falls back to CSV
        return []

    return all_products


def load_products_from_woocommerce() -> pd.DataFrame:
    """
    Loads products live from WooCommerce Store A (and B, C once configured).

    Returns:
        pd.DataFrame of products, or an empty DataFrame if no store
        is reachable or no API keys are configured.
    """
    all_rows = []

    stores = [
        (config.STORE_A_URL if hasattr(config, "STORE_A_URL") else config.WP_BASE_URL,
         config.WC_CONSUMER_KEY, config.WC_CONSUMER_SECRET),
    ]

    # TODO: Store B and C will be added here once they're populated —
    #   stores.append((config.STORE_B_URL, config.STORE_B_CONSUMER_KEY, config.STORE_B_CONSUMER_SECRET))
    #   stores.append((config.STORE_C_URL, config.STORE_C_CONSUMER_KEY, config.STORE_C_CONSUMER_SECRET))

    for store_url, key, secret in stores:
        raw_products = _fetch_store_products(store_url, key, secret)
        for wc_product in raw_products:
            all_rows.append(_map_wc_product_to_dict(wc_product))

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# CSV fallback (Phase 1 — used if WooCommerce is unreachable)
# ─────────────────────────────────────────────────────────────────────────────

def load_products_from_csv() -> pd.DataFrame:
    """Loads the product dataset from the local CSV file."""
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Unified loader — tries WooCommerce first, falls back to CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_products() -> pd.DataFrame:
    """
    Loads the product dataset — live from WooCommerce if reachable and
    configured, otherwise falls back to the local CSV.

    This means existing Phase 1 code and tests keep working unchanged,
    while Phase 2 environments with real WooCommerce keys automatically
    get live data with zero code changes elsewhere in the pipeline.

    Returns:
        pd.DataFrame: All products, from whichever source is available.
    """
    df = load_products_from_woocommerce()

    if df.empty:
        df = load_products_from_csv()

    return df

# ── Synonyms Dictionary (TODO 2) ─────────────────────────────────────────────
SYNONYMS = {
    # General terms
    "budget": ["affordable", "cheap", "entry-level"],

    # Smartphones
    "mobile": ["phone", "cellphone", "handset", "smartphone"],
    "feature phone": ["basic phone", "keypad phone"],
    "redmi": ["xiaomi"],
    "iqoo": ["vivo"],
    "iphone": ["apple"],
    "samsung": ["galaxy"],

    # Laptops
    "notebook": ["laptop", "ultrabook"],
    "pc": ["computer"],
    "macbook": ["apple"],

    # Earphones
    "earbuds": ["earphones", "headphones", "in-ear"],
    "wireless": ["bluetooth", "cordless"],
    "noise cancelling": ["ANC", "noise-cancelling"],


    # Skincare / Beauty
    "moisturiser": ["moisturizer", "hydrating"],
    "under eye cream": ["eye contour serum", "eye gel"],
    "sunscreen": ["sunblock", "spf"],
    "face wash": ["cleanser", "facewash"],
    "scrub": ["exfoliator", "exfoliating"],
    "serum": ["essence"],
    "beauty": ["skincare", "cosmetics", "makeup"]
}

def _has_phrase(text: str, phrase: str) -> bool:
    """
    Checks whether a word or phrase appears in text as whole words.

    Args:
        text  : Lowercased text to search in.
        phrase: Word or phrase to look for (case-insensitive).

    Returns:
        True only if the phrase is not part of a longer word.
    """
    return re.search(rf"\b{re.escape(phrase.lower())}\b", text) is not None



def expand_query_with_synonyms(query: str, category: str | None = None) -> str:
    """
    Expands a user query string with matching terms from the global synonyms dictionary.

    Args:
        query: Free-text search string from the user.
        category: Optional category filter string (not utilized by this lookup mapping).

    Returns:
        Expanded query string containing original text and unique added synonyms.
    """
    # Guard check for empty input while maintaining strict type boundaries
    if not query:
        return ""

    query_lower = query.lower()

    # Split the original query text into structural words
    expanded_terms = query.split()

    # Store lowercase versions in a set for reliable lookup evaluations
    seen_terms = {word.lower() for word in expanded_terms}

    # Scan the dictionary records directly to catch multi-word phrases and single words
    for key, synonyms_list in SYNONYMS.items():
        key_lower = key.lower()

        # Expand only when the key or one of its synonyms appears as a whole word/phrase
        if _has_phrase(query_lower, key_lower) or any(
            _has_phrase(query_lower, s) for s in synonyms_list
        ):
            for term in [key] + synonyms_list:
                if term.lower() not in seen_terms:
                    expanded_terms.append(term)
                    seen_terms.add(term.lower())

    return " ".join(expanded_terms)


def search_products(query: str, max_budget: float = None, category: str = None) -> list[dict]:
    """
    Filters products based on a search query, optional budget, and optional category.

    Args:
        query      : Free-text search string from the user (e.g. "phone under 15000")
        max_budget : Maximum price the user is willing to pay (optional)
        category   : Product category to filter by (optional, e.g. "Electronics")

    Returns:
        List of matching product dicts. Returns empty list if no matches found.

    Example:
        >>> results = search_products("samsung phone under 20000", max_budget=20000, category="Electronics")
    """
    df = load_products()

    if df.empty:
        return []

    # ── Category filter ──────────────────────────────────────────────────────
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    # ── Budget filter ────────────────────────────────────────────────────────
    if max_budget is not None:
        df = df[df["price"] <= max_budget]
    # ── Synonyms expansion (TODO 2) ──────────────────────────────────────────
    if query:
        query = expand_query_with_synonyms(query, category)
    # ── Keyword filter ───────────────────────────────────────────────────────
    # Searches across product name, brand, subcategory, and key_specs
    if query:
        keywords = query.lower().split()
        search_columns = ["product_name", "brand", "subcategory", "key_specs", "category"]

        def row_matches(row):
            combined = " ".join(str(row[col]).lower() for col in search_columns if col in row.index)
            return any(kw in combined for kw in keywords)

        df = df[df.apply(row_matches, axis=1)]

    # ── Stock filter ─────────────────────────────────────────────────────────
    if not df.empty:
        df = df[df["stock_status"] != "out_of_stock"]

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 1: Improve keyword matching
    #   - Right now it splits the query into words and does basic string matching.
    #   - Consider using fuzzy matching (e.g. rapidfuzz) so that
    #     "moisturizer" also matches "moisturiser", "mosturizer" etc.
    #   - Example: from rapidfuzz import fuzz
    #              fuzz.partial_ratio("moisturizer", "moisturiser") > 80
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 2: Handle synonyms and product-specific terms
    #   - "budget phone" should also match "affordable", "cheap", "entry-level"
    #   - Build a small synonym dictionary for common product terms
    #   - Example: synonyms = {"budget": ["affordable", "cheap", "entry-level"]}
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 3: Extract budget from the query automatically
    #   - Right now budget must be passed separately.
    #   - Add logic to detect "under 500", "below Rs 1000", "less than 2000"
    #     directly from the query string and set max_budget automatically.
    #   - Hint: use regex  re.search(r'under\s*(?:rs\.?\s*)?(\d+)', query, re.I)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 4: Add Store B and Store C once they're populated
    #   - Uncomment the stores.append(...) lines in load_products_from_woocommerce()
    #   - Add STORE_B_URL / STORE_B_CONSUMER_KEY / STORE_B_CONSUMER_SECRET
    #     (and _C_ equivalents) to your .env once Minu has those stores running
    #   - No other code changes needed — compare.py already groups by "store"
    # ─────────────────────────────────────────────────────────────────────────

    return df.to_dict(orient="records")
