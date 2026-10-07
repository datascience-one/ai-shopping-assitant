#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##

"""
agents/compare.py
-----------------
Compare Agent — scores and ranks filtered products, returning the top 3.

Scoring has two layers:
  1. Base score  — price, rating, popularity (works for all categories)
  2. Spec score  — category-specific numeric specs (smartphones: RAM, battery etc.)

The final score is a weighted combination of both layers.
Spec confidence (high/medium/low) adjusts how much the spec score counts.

Students:
- Base scoring is fully implemented.
- Spec scoring for smartphones is implemented.
- TODO items show where to extend for new categories and dynamic weighting.
- Do not change function signatures — Recommend Agent depends on them.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Spec scoring profiles (mirrors data/spec_profiles.py scoring_specs)
# ─────────────────────────────────────────────────────────────────────────────

SMARTPHONE_SCORING = {
    # spec_column     : (higher_is_better, max_value, weight_in_spec_score)
    "spec_ram":          (True,  16,   0.25),
    "spec_storage":      (True,  512,  0.20),
    "spec_battery":      (True,  7000, 0.30),
    "spec_rear_camera":  (True,  200,  0.15),
    "spec_display_size": (True,  7.0,  0.10),
}

# Confidence multiplier — how much spec score counts based on data reliability
CONFIDENCE_WEIGHT = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
    "":       0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Base scoring — works for all categories
# ─────────────────────────────────────────────────────────────────────────────

def score_base(
    product: dict,
    max_budget: float = None,
    query: str = ""
) -> float:
    """
    Scores a product on price, rating, and popularity.
    Returns a score between 0 and 100.

    Default weights:
        Price (discount from MRP) : 40%
        Rating                    : 40%
        Popularity (review count) : 20%

    Dynamic intent weights:
        Budget/cheap/affordable:
        Price 60%, Rating 25%, Popularity 15%

    Best/top rated/highest rated:
        Price 25%, Rating 60%, Popularity 15%

    If no supported intent is detected, the default weights are used.
    """

    price        = float(product.get("price", 0))
    mrp          = float(product.get("mrp", price))
    rating       = float(product.get("rating", 0))
    review_count = int(product.get("review_count", 0))

    # Price score — discount from MRP, capped at 50%
    discount_pct = ((mrp - price) / mrp) * 100 if mrp > 0 else 0
    price_score  = min(discount_pct, 50) * 2

    # Rating score — 0-5 scaled to 0-100
    rating_score = (rating / 5) * 100

    # Popularity — cap at 10,000 reviews
    popularity_score = min(review_count / 10000, 1) * 100

    query_lower = query.lower()

    price_weight = 0.40
    rating_weight = 0.40
    popularity_weight = 0.20

    budget_intent = any(
        phrase in query_lower
        for phrase in [
            "budget",
            "cheap",
            "cheapest",
            "affordable",
            "inexpensive",
            "low cost",
        ]
    )

    rating_intent = any(
        phrase in query_lower
        for phrase in [
            "best",
            "top",
            "top rated",
            "highest rated",
            "highly rated",
            "good rating",
            "well rated",
        ]
    )

    if budget_intent and rating_intent:
        # User wants an affordable product that is also well rated.
        price_weight = 0.50
        rating_weight = 0.40
        popularity_weight = 0.10

    elif budget_intent:
        price_weight = 0.60
        rating_weight = 0.25
        popularity_weight = 0.15

    elif rating_intent:
        price_weight = 0.25
        rating_weight = 0.60
        popularity_weight = 0.15

    return (
        (price_score * price_weight)
        + (rating_score * rating_weight)
        + (popularity_score * popularity_weight)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spec scoring — category specific
# ─────────────────────────────────────────────────────────────────────────────

def score_smartphone_specs(
    product: dict,
    query: str = ""
) -> float:
    """
    Scores a smartphone based on its numeric spec values.
    Returns a score between 0 and 100.

    Each spec is normalised against a maximum expected value,
    then combined using per-spec weights.

    Spec columns expected (added by clean_products.py):
        spec_ram, spec_storage, spec_battery,
        spec_rear_camera, spec_display_size

    Camera intent:
    Queries containing "camera", "photo", or "photography"
    increase the rear-camera spec weight from 15% to 40%.
    The remaining spec weights are adjusted so the total remains 100%.

    """
    query_lower = query.lower()

    scoring = SMARTPHONE_SCORING

    if any(word in query_lower for word in ["camera", "photo", "photography"]):
        scoring = {
            "spec_ram":          (True, 16,   0.15),
            "spec_storage":      (True, 512,  0.15),
            "spec_battery":      (True, 7000, 0.20),
            "spec_rear_camera":  (True, 200,  0.40),
            "spec_display_size": (True, 7.0,  0.10),
        }


    total = 0.0

    for col, (higher_is_better, max_val, weight) in scoring.items():
        val = float(product.get(col, 0))
        if max_val == 0:
            continue
        normalised = min(val / max_val, 1.0) * 100
        if not higher_is_better:
            normalised = 100 - normalised
        total += normalised * weight

    return round(total, 2)


def score_specs(
    product: dict,
    query: str = ""
) -> float:
    """
    Routes to the correct spec scorer based on category.
    Returns 0 if no spec scorer exists for this category.
    """
    category    = str(product.get("category", "")).lower()
    subcategory = str(product.get("subcategory", "")).lower()

    if category == "electronics" and subcategory == "smartphone":
        return score_smartphone_specs(product, query)
    # ─────────────────────────────────────────────────────────────────────────
    # TODO: Add new category scorers here
    #   elif subcategory == "laptop":
    #       return score_laptop_specs(product)
    #   elif subcategory in ["earphones", "headphones"]:
    #       return score_earphone_specs(product)
    # ─────────────────────────────────────────────────────────────────────────

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Combined scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_product(
    product: dict,
    max_budget: float = None,
    query: str = ""
) -> float:
    """
    Final score combining base score and spec score.

    Formula:
        final = (base_score * 0.60) + (spec_score * confidence_weight * 0.40)

    Spec confidence adjusts how much the spec score contributes:
        high   → full 40% weight on specs
        medium → 60% of that (24%)
        low    → 30% of that (12%)

    This prevents unreliable scraped spec data from dominating the score.

    Args:
        product    : Product dict from Search Agent
        max_budget : User's stated budget (currently unused, reserved for TODO)

    Returns:
        float: Score between 0 and 100
    """
    base = score_base(product, max_budget, query)
    spec = score_specs(product, query)

    confidence     = str(product.get("spec_confidence", "")).lower()
    conf_weight    = CONFIDENCE_WEIGHT.get(confidence, 0.3)

    final = (base * 0.60) + (spec * conf_weight * 0.40)
    return round(final, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def compare_and_rank(
    products: list[dict],
    max_budget: float = None,
    top_n: int = 3,
    query: str = ""
) -> list[dict]:
    """
    Scores all products and returns the top N, one per store where possible,
    always sorted by score descending.

    Selection strategy:
        1. Score every product.
        2. Pick the highest-scoring product from each distinct store
           (so results are diverse across stores when possible).
        3. If fewer than top_n stores are available, fill remaining
           slots with the next-highest-scoring products NOT already
           selected — this never re-adds a store that's already
           represented unless every other store has been exhausted.
        4. Always re-sort the final selection by score descending,
           since step 2's "one per store" pass can otherwise leave
           a lower-scoring product ahead of a higher-scoring one.

    Args:
        products   : Filtered product list from Search Agent
        max_budget : User's budget (passed to scorer)
        top_n      : How many products to return (default 3)

    Returns:
        List of top N products, each with a "score" field added,
        sorted by score descending.
    """
    if not products:
        return []

    # Score every product
    for p in products:
        p["score"] = score_product(p, max_budget, query)

    # Sort by score descending
    sorted_products = sorted(products, key=lambda x: x["score"], reverse=True)

    # ── Pass 1: best product per distinct store ─────────────────────────────
    seen_stores  = set()
    top_products = []
    selected_ids = set()

    for p in sorted_products:
        store = p.get("store")
        if store not in seen_stores:
            top_products.append(p)
            seen_stores.add(store)
            selected_ids.add(p.get("product_id"))
        if len(top_products) == top_n:
            break

    # ── Pass 2: fill remaining slots, never re-picking an already
    #            selected product, preferring untouched stores first ───────
    if len(top_products) < top_n:
        # First try products from stores not yet represented at all
        for p in sorted_products:
            if p.get("product_id") in selected_ids:
                continue
            if p.get("store") not in seen_stores:
                top_products.append(p)
                selected_ids.add(p.get("product_id"))
                seen_stores.add(p.get("store"))
            if len(top_products) == top_n:
                break

        # If still short, allow repeating a store, but never repeat
        # the exact same product
        if len(top_products) < top_n:
            for p in sorted_products:
                if p.get("product_id") in selected_ids:
                    continue
                top_products.append(p)
                selected_ids.add(p.get("product_id"))
                if len(top_products) == top_n:
                    break

    # ── Always re-sort the final selection by score descending ─────────────
    top_products = sorted(top_products, key=lambda x: x["score"], reverse=True)



    # ─────────────────────────────────────────────────────────────────────────
    # TODO 2: Value for money metric
    #   - Price per GB of storage: lower = better value
    #   - Price per mAh of battery: lower = better value
    #   - Add as a bonus to the final score for budget-tier products
    # ─────────────────────────────────────────────────────────────────────────

    return top_products
