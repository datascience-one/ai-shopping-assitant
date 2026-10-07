#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##

"""
data/load_products.py
---------------------
Populates Minu's WooCommerce store(s) with cleaned product data.

Reads data/products.csv (output of clean_products.py) and pushes
every product to the correct WooCommerce store via REST API.

Current setup (Phase 1-2):
    Store A → http://localhost:8080  (Amazon / Nandana's scraped data)
    Store B → http://localhost:8081  (Flipkart — coming in Phase 2)
    Store C → http://localhost:8082  (Meesho — coming in Phase 2)

NOTE: Minu's current docker-compose.yml only runs Store A on port 8080.
      Stores B and C will be added when the multi-store Docker setup is ready.

Usage:
    # Make sure Docker is running and WooCommerce is set up first:
    #   bash setup.sh
    #
    # Then run:
    python data/load_products.py

    # To load only Store A:
    python data/load_products.py --store A

    # To clear all products from a store before loading:
    python data/load_products.py --store A --clear

Students:
- STEP 1: Make sure setup.sh has been run and http://localhost:8080 is accessible
- STEP 2: Generate fresh WooCommerce API keys from the admin panel
          (WooCommerce → Settings → Advanced → REST API → Add Key)
- STEP 3: Add those keys to your .env file
- STEP 4: Run this script
- STEP 5: Open http://localhost:8080/wp-admin and verify products appear
"""

import os
import sys
import time
import argparse
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Store configuration
# ─────────────────────────────────────────────────────────────────────────────

STORES = {
    "A": {
        "name":       "Store A (Amazon)",
        "url":        os.getenv("STORE_A_URL", "http://localhost:8080"),
        "key":        os.getenv("WC_CONSUMER_KEY"),
        "secret":     os.getenv("WC_CONSUMER_SECRET"),
        "csv_filter": "Store A",   # matches the 'store' column in products.csv
    },
    "B": {
        "name":       "Store B (Flipkart)",
        "url":        os.getenv("STORE_B_URL", "http://localhost:8081"),
        "key":        os.getenv("STORE_B_CONSUMER_KEY"),
        "secret":     os.getenv("STORE_B_CONSUMER_SECRET"),
        "csv_filter": "Store B",
    },
    "C": {
        "name":       "Store C (Meesho)",
        "url":        os.getenv("STORE_C_URL", "http://localhost:8082"),
        "key":        os.getenv("STORE_C_CONSUMER_KEY"),
        "secret":     os.getenv("STORE_C_CONSUMER_SECRET"),
        "csv_filter": "Store C",
    },
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "products.csv")


# ─────────────────────────────────────────────────────────────────────────────
# WooCommerce helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_store_reachable(store_url: str) -> bool:
    """Checks if the WooCommerce store is running and reachable."""
    try:
        r = requests.get(f"{store_url}/wp-json/wc/v3", timeout=10)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_or_create_category(store_url: str, key: str, secret: str, category_name: str) -> int:
    """
    Returns the WooCommerce category ID for a given name.
    Creates the category if it doesn't exist yet.

    Args:
        store_url     : WooCommerce store base URL
        key           : WooCommerce consumer key
        secret        : WooCommerce consumer secret
        category_name : Category name e.g. "Electronics"

    Returns:
        int: WooCommerce category ID
    """
    url = f"{store_url}/wp-json/wc/v3/products/categories"
    auth = HTTPBasicAuth(key, secret)

    # Check if category already exists
    r = requests.get(url, params={"search": category_name, "per_page": 10}, auth=auth, timeout=15)
    if r.status_code == 200:
        existing = [c for c in r.json() if c["name"].lower() == category_name.lower()]
        if existing:
            return existing[0]["id"]

    # Create it
    r = requests.post(url, json={"name": category_name}, auth=auth, timeout=15)
    if r.status_code in (200, 201):
        cat_id = r.json()["id"]
        print(f"  Created category: {category_name} (ID: {cat_id})")
        return cat_id
    else:
        print(f"  Warning: Could not create category '{category_name}': {r.text[:100]}")
        return 0


def build_product_payload(row: pd.Series, category_id: int) -> dict:
    """
    Converts a row from products.csv into a WooCommerce product payload.

    Maps our internal schema to WooCommerce REST API fields.
    Spec data is stored in WooCommerce meta_data so the Search Agent
    can retrieve and use it in Phase 2.

    Args:
        row         : A single row from products.csv as a pandas Series
        category_id : WooCommerce category ID to assign the product to

    Returns:
        dict: WooCommerce product payload
    """
    # Build a clean short description from key specs
    short_desc = str(row.get("key_specs", ""))[:500] if row.get("key_specs") else ""

    # Full description
    description = str(row.get("description", ""))[:2000] if row.get("description") else short_desc

    # Price — WooCommerce expects strings
    price     = str(int(float(row.get("price", 0))))
    mrp       = str(int(float(row.get("mrp", 0)))) if row.get("mrp") else price

    # SKU — unique identifier per product per store
    sku = f"{row.get('product_id', '')}-{str(row.get('store', '')).replace(' ', '')}"

    payload = {
        "name":              str(row.get("product_name", ""))[:200],
        "type":              "simple",
        "regular_price":     mrp,          # original MRP
        "sale_price":        price,         # selling price (discounted)
        "sku":               sku,
        "description":       description,
        "short_description": short_desc,
        "manage_stock":      True,
        "stock_quantity":    100,           # default stock — Phase 2 can make this realistic
        "stock_status":      "instock" if row.get("stock_status") != "out_of_stock" else "outofstock",
        "categories":        [{"id": category_id}] if category_id else [],

        # Store all our structured data as WooCommerce meta fields
        # Phase 2 Search Agent reads these back via the REST API
        "meta_data": [
            {"key": "dsa_brand",          "value": str(row.get("brand", ""))},
            {"key": "dsa_category",       "value": str(row.get("category", ""))},
            {"key": "dsa_subcategory",    "value": str(row.get("subcategory", ""))},
            {"key": "dsa_store",          "value": str(row.get("store", ""))},
            {"key": "dsa_rating",         "value": str(row.get("rating", "0"))},
            {"key": "dsa_review_count",   "value": str(row.get("review_count", "0"))},
            {"key": "dsa_pros",           "value": str(row.get("pros", ""))},
            {"key": "dsa_cons",           "value": str(row.get("cons", ""))},
            {"key": "dsa_spec_confidence","value": str(row.get("spec_confidence", ""))},
            # Numeric spec fields for Compare Agent scoring
            {"key": "dsa_spec_ram",          "value": str(row.get("spec_ram", "0"))},
            {"key": "dsa_spec_storage",      "value": str(row.get("spec_storage", "0"))},
            {"key": "dsa_spec_battery",      "value": str(row.get("spec_battery", "0"))},
            {"key": "dsa_spec_rear_camera",  "value": str(row.get("spec_rear_camera", "0"))},
            {"key": "dsa_spec_display_size", "value": str(row.get("spec_display_size", "0"))},
        ]
    }

    # Add product image if available
    image_url = str(row.get("image_url", "")).strip()
    if image_url and image_url.startswith("http"):
        payload["images"] = [{"src": image_url, "alt": payload["name"]}]

    return payload


def push_product(store_url: str, key: str, secret: str, payload: dict) -> dict:
    """
    POSTs a single product to WooCommerce.

    Returns the API response dict.
    """
    url  = f"{store_url}/wp-json/wc/v3/products"
    auth = HTTPBasicAuth(key, secret)
    r    = requests.post(url, json=payload, auth=auth, timeout=30)

    try:
        return r.json()
    except Exception:
        return {"error": r.text, "status_code": r.status_code}


def clear_all_products(store_url: str, key: str, secret: str):
    """
    Deletes all existing products from the store.
    Used with --clear flag for a fresh load.
    """
    auth = HTTPBasicAuth(key, secret)
    url  = f"{store_url}/wp-json/wc/v3/products"
    page = 1

    print("  Clearing existing products...")
    while True:
        r = requests.get(url, params={"per_page": 100, "page": page}, auth=auth, timeout=30)
        products = r.json()

        print("\nDEBUG RESPONSE:")
        print(products)
        print("TYPE:", type(products))
        if not products:
            break
        for p in products:
            requests.delete(
                f"{url}/{p['id']}",
                params={"force": True},
                auth=auth,
                timeout=15
            )
            print(f"  Deleted: {p['name']}")
        page += 1
    print("  All products cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_store(store_key: str, clear: bool = False):
    """
    Loads products into a single WooCommerce store.

    Args:
        store_key : "A", "B", or "C"
        clear     : If True, deletes all existing products before loading
    """
    config = STORES[store_key]
    print(f"\n{'='*60}")
    print(f"Loading {config['name']}")
    print(f"URL: {config['url']}")
    print(f"{'='*60}")

    # ── Validate config ───────────────────────────────────────────────────────
    if not config["key"] or not config["secret"]:
        print(f"ERROR: Missing WooCommerce API keys for {config['name']}.")
        print(f"Add STORE_{store_key}_CONSUMER_KEY and STORE_{store_key}_CONSUMER_SECRET to .env")
        print(f"(For Store A use WC_CONSUMER_KEY and WC_CONSUMER_SECRET)")
        return

    # ── Check store is reachable ──────────────────────────────────────────────
    print(f"Checking store is reachable...")
    if not check_store_reachable(config["url"]):
        print(f"ERROR: Cannot reach {config['url']}")
        print(f"Make sure Docker is running: docker compose up -d")
        print(f"Then wait ~30 seconds and try again.")
        return
    print(f"Store is reachable.")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print(f"Run python data/clean_products.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    store_df = df[df["store"] == config["csv_filter"]]

    if store_df.empty:
        print(f"ERROR: No products found for '{config['csv_filter']}' in products.csv")
        print(f"Check that clean_products.py assigned the correct store name.")
        return

    print(f"Found {len(store_df)} products for {config['name']}")

    # ── Clear existing products ───────────────────────────────────────────────
    if clear:
        clear_all_products(config["url"], config["key"], config["secret"])

    # ── Create categories ─────────────────────────────────────────────────────
    print(f"\nSetting up categories...")
    category_ids = {}
    for category in store_df["category"].unique():
        cat_id = get_or_create_category(
            config["url"], config["key"], config["secret"], str(category)
        )
        category_ids[category] = cat_id

    # ── Push products ─────────────────────────────────────────────────────────
    print(f"\nPushing products...")
    success = 0
    failed  = 0
    skipped = 0

    for _, row in store_df.iterrows():
        product_name = str(row.get("product_name", "Unknown"))
        category     = str(row.get("category", ""))
        cat_id       = category_ids.get(category, 0)

        payload  = build_product_payload(row, cat_id)
        response = push_product(config["url"], config["key"], config["secret"], payload)

        if "id" in response:
            print(f"  OK  {product_name[:60]}")
            success += 1
        elif "code" in response and response["code"] == "product_invalid_sku":
            print(f"  SKIP (already exists)  {product_name[:50]}")
            skipped += 1
        else:
            error = response.get("message", str(response))[:80]
            print(f"  FAIL  {product_name[:50]}  |  {error}")
            failed += 1

        # Polite delay — avoid overwhelming the local server
        time.sleep(0.3)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print(f"Done loading {config['name']}")
    print(f"  Success : {success}")
    print(f"  Skipped : {skipped} (already existed)")
    print(f"  Failed  : {failed}")
    print(f"\nVerify at: {config['url']}/wp-admin")
    print(f"           {config['url']}/wp-json/wc/v3/products")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load products into WooCommerce stores")
    parser.add_argument(
        "--store",
        choices=["A", "B", "C", "all"],
        default="A",
        help="Which store to load (default: A)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing products before loading"
    )
    args = parser.parse_args()

    if args.store == "all":
        for store_key in ["A", "B", "C"]:
            load_store(store_key, clear=args.clear)
    else:
        load_store(args.store, clear=args.clear)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Update .env with real WooCommerce API keys
#   - Open http://localhost:8080/wp-admin
#   - Go to WooCommerce → Settings → Advanced → REST API
#   - Click Add Key, set permissions to Read/Write
#   - Copy Consumer Key and Consumer Secret into .env as:
#       WC_CONSUMER_KEY=ck_...
#       WC_CONSUMER_SECRET=cs_...
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Add Stores B and C to docker-compose.yml (Phase 2 — Minu's task)
#   - Currently only Store A runs on port 8080
#   - Minu needs to add Store B (port 8081) and Store C (port 8082)
#   - Once done, add to .env:
#       STORE_B_URL=http://localhost:8081
#       STORE_B_CONSUMER_KEY=ck_...
#       STORE_B_CONSUMER_SECRET=cs_...
#       STORE_C_URL=http://localhost:8082
#       STORE_C_CONSUMER_KEY=ck_...
#       STORE_C_CONSUMER_SECRET=cs_...
#   - Then run: python data/load_products.py --store all
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Phase 2 — Update agents/search.py to query WooCommerce live
#   - Replace pd.read_csv() with requests.get(WP_BASE_URL/wp-json/wc/v3/products)
#   - Read dsa_* meta_data fields back to reconstruct the product dict
#   - Query all 3 stores and merge results before passing to Compare Agent
# ─────────────────────────────────────────────────────────────────────────────
