#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
data/clean_products.py
----------------------
Data Cleaning Pipeline — transforms raw scraped CSVs into our standard schema.

Now category-aware: uses spec_profiles.py to handle different spec columns
per category rather than a one-size-fits-all approach.

Usage:
    python data/clean_products.py

Input:
    data/raw/amazon.csv      (smartphones from Amazon — Store A)
    data/raw/flipkart.csv    (smartphones from Flipkart — Store B)
    data/raw/meesho.csv      (smartphones from 3rd site — Store C)

Output:
    data/products.csv        (clean, merged, ready for agents)

Students:
- Column name mapping is in AMAZON_COLUMN_MAP / FLIPKART_COLUMN_MAP / MEESHO_COLUMN_MAP
- Spec profiles are in data/spec_profiles.py — edit there to add new categories
- TODO items show where to extend the pipeline
"""

import pandas as pd
import os
import re
import sys
import json

import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.spec_profiles import SPEC_PROFILES, get_profile

RAW_DIR    = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "products.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Column maps — maps each site's raw column names to our internal names
# These match Nandana's actual scraped columns
# ─────────────────────────────────────────────────────────────────────────────

AMAZON_COLUMN_MAP = {
    "product_name":       "product_name",
    "brand":              "brand",
    "selling_price":      "price",
    "mrp":                "mrp",
    "rating":             "rating",
    "review_count":       "review_count",
    "description":        "description",
    "ram":                "ram",
    "ram_source":         "ram_source",
    "storage":            "storage",
    "storage_source":     "storage_source",
    "processor":          "processor",
    "processor_source":   "processor_source",
    "battery":            "battery",
    "battery_source":     "battery_source",
    "display_size":       "display_size",
    "display_size_source":"display_size_source",
    "rear_camera":        "rear_camera",
    "rear_camera_source": "rear_camera_source",
    "front_camera":       "front_camera",
    "front_camera_source":"front_camera_source",
    "os":                 "os",
    "os_source":          "os_source",
    "ui_skin":            "ui_skin",
    "ui_skin_source":     "ui_skin_source",
}

# TODO: Update once Flipkart scraper is ready
FLIPKART_COLUMN_MAP = {
    "product_name":  "product_name",
    "brand":         "brand",
    "selling_price": "price",
    "mrp":           "mrp",
    "rating":        "rating",
    "review_count":  "review_count",
    "description":   "description",
    "ram":           "ram",
    "storage":       "storage",
    "processor":     "processor",
    "battery":       "battery",
    "display_size":  "display_size",
    "rear_camera":   "rear_camera",
    "os":            "os",
}

# TODO: Update once 3rd site scraper is ready
MEESHO_COLUMN_MAP = {
    "name":         "product_name",
    "brand":        "brand",
    "price":        "price",
    "mrp":          "mrp",
    "rating":       "rating",
    "reviews":      "review_count",
    "description":  "description",
    "ram":          "ram",
    "storage":      "storage",
    "processor":    "processor",
    "battery":      "battery",
    "display_size": "display_size",
    "rear_camera":  "rear_camera",
    "os":           "os",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_price(val) -> float:
    """Strips ₹ and commas, returns float. Returns 0.0 on failure."""
    if pd.isna(val) or str(val).strip() in ["N/A", "", "nan"]:
        return 0.0
    cleaned = re.sub(r"[₹,\s]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def clean_rating(val) -> float:
    """Extracts numeric rating. Handles '4.2 out of 5' style strings."""
    if pd.isna(val):
        return 0.0
    match = re.search(r"(\d+\.?\d*)", str(val))
    return float(match.group(1)) if match else 0.0


def clean_review_count(val) -> int:
    """Extracts integer review count. Handles '1,234 ratings' style strings."""
    if pd.isna(val):
        return 0
    cleaned = re.sub(r"[,\s]", "", str(val))
    match = re.search(r"(\d+)", cleaned)
    return int(match.group(1)) if match else 0


def clean_spec_value(val) -> str:
    """Cleans a spec value — strips whitespace, replaces N/A with empty."""
    if pd.isna(val) or str(val).strip() in ["N/A", "nan", ""]:
        return ""
    return str(val).strip()


def extract_numeric(val: str) -> float:
    """
    Extracts the first numeric value from a spec string.
    e.g. "6500 mAh" → 6500.0
         "6.72 inches" → 6.72
         "50 MP + 50 MP" → 50.0  (takes first number)
    """
    if not val:
        return 0.0
    match = re.search(r"(\d+\.?\d*)", str(val))
    return float(match.group(1)) if match else 0.0


def extract_ram_gb(val: str) -> float:
    """Extracts RAM in GB. Handles '8 GB', '12GB', '8 GB RAM'."""
    return extract_numeric(val)


def extract_storage_gb(val: str) -> float:
    """Extracts storage in GB. Handles '128 GB', '256GB'."""
    num = extract_numeric(val)
    if "TB" in str(val).upper():
        num *= 1024
    return num


def extract_battery_mah(val: str) -> float:
    """Extracts battery in mAh. Handles '6500 mAh', '7000mAh'."""
    return extract_numeric(val)


def extract_camera_mp(val: str) -> float:
    """Extracts primary camera MP. Takes the first number."""
    return extract_numeric(val)


def extract_display_inch(val: str) -> float:
    """Extracts display size in inches."""
    return extract_numeric(val)


def clean_product_name(name: str) -> str:
    """
    Shortens Amazon-style product titles.
    Removes colour variants and spec details crammed into the title.
    e.g. "REDMI Note 15 5G (Black, 8GB RAM 128GB Storage) | 108MP..."
      →  "Redmi Note 15 5G"
    """
    if not name:
        return name
    # Remove content in parentheses (colour, RAM info)
    name = re.sub(r'\(.*?\)', '', name)
    # Remove content after | or ,
    name = name.split('|')[0].split(',')[0]
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Spec confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def calculate_spec_confidence(row: pd.Series, profile: dict) -> str:
    """
    Calculates how reliable the spec data is for a product.

    Uses the _source columns from the scraper to determine if specs
    came from the structured spec table (most reliable), product title
    (reliable), or description text (least reliable).

    Returns: "high", "medium", or "low"
    """
    source_columns = profile.get("source_columns", {})
    trusted_sources = profile.get("trusted_sources", ["spec_table", "title"])

    if not source_columns:
        return "medium"

    reliable = 0
    total = 0

    for spec_col, source_col in source_columns.items():
        if source_col in row.index:
            source_val = str(row.get(source_col, "")).strip()
            if source_val and source_val not in ["N/A", "nan", ""]:
                total += 1
                if source_val in trusted_sources:
                    reliable += 1

    if total == 0:
        return "low"
    ratio = reliable / total
    if ratio >= 0.75:
        return "high"
    elif ratio >= 0.5:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Spec string builder
# ─────────────────────────────────────────────────────────────────────────────

def build_key_specs(row: pd.Series, profile: dict) -> str:
    """
    Builds a clean key_specs display string from structured spec columns.

    Uses the display_specs mapping from the profile to know which
    columns to include and what label to use.

    e.g. "RAM: 8GB | Storage: 128GB | Processor: Snapdragon 6 Gen 3 |
          Battery: 5520mAh | Display: 6.7in | Rear Camera: 108MP | OS: Android 15"
    """
    display_specs = profile.get("display_specs", {})
    parts = []

    for label, col in display_specs.items():
        val = clean_spec_value(row.get(col, ""))
        if val:
            parts.append(f"{label}: {val}")

    return " | ".join(parts) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# Numeric spec extraction for Compare Agent
# ─────────────────────────────────────────────────────────────────────────────

def extract_scoring_specs(row: pd.Series, profile: dict) -> dict:
    """
    Extracts numeric spec values used by the Compare Agent for scoring.

    Returns a dict of spec_name → numeric value.
    e.g. {"ram": 8.0, "storage": 128.0, "battery": 5520.0, ...}
    """
    scoring_specs = profile.get("scoring_specs", {})
    result = {}

    extractors = {
        "ram":          extract_ram_gb,
        "storage":      extract_storage_gb,
        "battery":      extract_battery_mah,
        "rear_camera":  extract_camera_mp,
        "display_size": extract_display_inch,
    }

    for spec_col, (unit, higher_is_better, max_val) in scoring_specs.items():
        raw_val = clean_spec_value(row.get(spec_col, ""))
        extractor = extractors.get(spec_col, extract_numeric)
        result[spec_col] = extractor(raw_val)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_and_map(filepath: str, column_map: dict, store_name: str) -> pd.DataFrame:
    """Loads raw CSV, maps columns to internal names, adds store label."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath} — skipping")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")

    available = {k: v for k, v in column_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["store"] = store_name
    return df


def process_smartphones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies smartphone-specific cleaning using the Electronics/Smartphones profile.
    Adds: category, subcategory, key_specs, spec_confidence, and numeric spec columns.
    """
    if df.empty:
        return df

    profile = SPEC_PROFILES["Electronics/Smartphones"]

    # Clean prices
    df["price"] = df["price"].apply(clean_price)
    df["mrp"]   = df["mrp"].apply(clean_price)
    # If MRP missing, use price (0% discount)
    df["mrp"] = df.apply(
        lambda r: r["price"] if r["mrp"] == 0 else r["mrp"], axis=1
    )

    # Clean ratings and reviews
    df["rating"]       = df["rating"].apply(clean_rating)
    df["review_count"] = df["review_count"].apply(clean_review_count)

    # Preserve original name for deduplication
    df["raw_product_name"] = df["product_name"]

    # Clean name only for display
    df["product_name"] = df["product_name"].apply(clean_product_name)

    # Set category fields
    df["category"]    = profile["category"]
    df["subcategory"] = profile["subcategory"]

    # Build key_specs string
    df["key_specs"] = df.apply(
        lambda r: build_key_specs(r, profile), axis=1
    )

    # Calculate spec confidence per row
    df["spec_confidence"] = df.apply(
        lambda r: calculate_spec_confidence(r, profile), axis=1
    )

    # Extract numeric scoring specs as separate columns
    # These are used by the Compare Agent for spec-based scoring
    scoring_specs = profile.get("scoring_specs", {})
    for spec_col in scoring_specs:
        extractor_map = {
            "ram":          extract_ram_gb,
            "storage":      extract_storage_gb,
            "battery":      extract_battery_mah,
            "rear_camera":  extract_camera_mp,
            "display_size": extract_display_inch,
        }
        extractor = extractor_map.get(spec_col, extract_numeric)
        raw_col = spec_col if spec_col in df.columns else None
        if raw_col:
            df[f"spec_{spec_col}"] = df[raw_col].apply(
                lambda v: extractor(clean_spec_value(str(v)))
            )
        else:
            df[f"spec_{spec_col}"] = 0.0

    # Add default columns without overwriting existing values
    if "pros" not in df.columns:
        df["pros"] = ""
    else:
        df["pros"] = df["pros"].fillna("")

    if "cons" not in df.columns:
        df["cons"] = ""
    else:
        df["cons"] = df["cons"].fillna("")

    if "stock_status" not in df.columns:
        df["stock_status"] = "in_stock"

    # Drop rows missing name or price
    df = df.dropna(subset=["product_name", "price"])
    df = df[df["price"] > 0]

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicates within the same store.

    Uses the ORIGINAL product name rather than the cleaned display name.
    This prevents different variants from collapsing together after
    clean_product_name() removes RAM/storage/color information.
    """

    before = len(df)

    df = df.drop_duplicates(
        subset=["raw_product_name", "brand", "store", "price"]
    )

    after = len(df)

    print(
        f"Deduplication: {before} -> {after} rows "
        f"({before - after} removed)"
    )

    return df

def _normalise_evidence(text: str) -> str:
    """Normalize text for comparing evidence with product information."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _extract_numeric_values(text: str) -> set[str]:
    """Extract normalized numeric values from text."""
    values = re.findall(r"(?<!\d)\d[\d,]*(?:\.\d+)?", text)
    normalized = set()

    for value in values:
        try:
            normalized.add(format(float(value.replace(",", "")), ".12g"))
        except ValueError:
            continue

    return normalized


def _validate_generated_item(
    item: dict,
    row: pd.Series,
) -> list[str]:
    """Validate generated pros and cons against product information."""
    errors = []

    if not isinstance(item, dict):
        return ["The product result must be a JSON object."]

    source_columns = {
        "product_name",
        "description",
        "key_specs",
        "ram",
        "storage",
        "processor",
        "battery",
        "display_size",
        "rear_camera",
        "front_camera",
        "os",
        "ui_skin",
    }

    source_columns.update(
        column
        for column in row.index
        if column.startswith("spec_")
    )

    source_texts = []

    for column in source_columns:
        if column not in row.index:
            continue

        value = row.get(column, "")

        if pd.notna(value) and str(value).strip():
            source_texts.append(str(value).strip())

    source_numbers = _extract_numeric_values(" ".join(source_texts))

    for field, minimum, maximum in [
        ("pros", 2, 3),
        ("cons", 1, 2),
    ]:
        entries = item.get(field)

        if not isinstance(entries, list):
            errors.append(f"{field} must be a list.")
            continue

        if not minimum <= len(entries) <= maximum:
            errors.append(
                f"{field} must contain {minimum} to {maximum} items."
            )

        for position, entry in enumerate(entries):
            label = f"{field}[{position}]"

            if not isinstance(entry, dict):
                errors.append(
                    f"{label} must contain a claim and supporting evidence."
                )
                continue

            claim = entry.get("claim")
            evidence = entry.get("evidence")

            if not isinstance(claim, str) or not claim.strip():
                errors.append(f"{label} has an empty or invalid claim.")
                continue

            if not isinstance(evidence, str) or not evidence.strip():
                errors.append(f"{label} has no supporting evidence.")
                continue

            if re.search(r"\bnan\b", claim, re.IGNORECASE):
                errors.append(f"{label} contains the forbidden text 'nan'.")

            if re.search(r"\bnan\b", evidence, re.IGNORECASE):
                errors.append(f"{label} evidence contains the text 'nan'.")

            normalized_evidence = _normalise_evidence(evidence)

            # Check that claim terms are supported by the evidence.
            stop_words = {
                "a", "an", "the", "is", "are", "was", "were",
                "of", "for", "with", "and", "or", "to", "in",
                "on", "by", "from", "has", "have", "had",
                "this", "that", "it",
            }

            def claim_tokens(text: str) -> set[str]:
                tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))

                # Treat common negative forms as equivalent.
                if tokens.intersection({"no", "without", "excluded"}):
                    tokens.difference_update({"no", "without", "excluded"})
                    tokens.add("not")

                return tokens - stop_words

            unsupported_terms = (
                claim_tokens(claim) - claim_tokens(evidence)
            )

            if unsupported_terms:
                errors.append(
                    f"{label} contains terms not supported by its evidence: "
                    f"{', '.join(sorted(unsupported_terms))}."
                )

            if not any(
                normalized_evidence in _normalise_evidence(source)
                for source in source_texts
            ):
                errors.append(
                    f"{label} evidence does not match the product information."
                )

            claim_numbers = _extract_numeric_values(claim)
            evidence_numbers = _extract_numeric_values(evidence)

            if not claim_numbers.issubset(source_numbers):
                errors.append(
                    f"{label} contains a number not found in the product "
                    "description or specifications."
                )

            if not claim_numbers.issubset(evidence_numbers):
                errors.append(
                    f"{label} contains a number not supported by its evidence."
                )

    return errors

def _request_validated_batch(
    client,
    products: list[dict],
    df: pd.DataFrame,
    max_attempts: int = 3,
) -> tuple[dict, dict]:
    """Generate and validate product claims, retrying invalid results."""
    pending = {
        str(product["row_id"]): {
            "product": product,
            "errors": [],
        }
        for product in products
    }
    validated = {}

    index_by_row_id = {
        str(index): index for index in df.index
    }

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break

        current_products = [
            state["product"]
            for state in pending.values()
        ]

        retry_feedback = {
            row_id: state["errors"]
            for row_id, state in pending.items()
            if state["errors"]
        }

        feedback_text = ""
        if retry_feedback:
            feedback_text = (
                "\nFix these specific validation errors from "
                "the previous attempt:\n"
                + json.dumps(retry_feedback, ensure_ascii=False)
                + "\n"
            )

        prompt = f"""
You are a careful product-data assistant.

Generate pros and cons only from the supplied product information.
Return valid JSON with this structure:
{{
  "products": [
    {{
      "row_id": "copy the input row_id exactly",
      "pros": [
        {{"claim": "concise advantage",
          "evidence": "exact source excerpt"}}
      ],
      "cons": [
        {{"claim": "realistic limitation",
          "evidence": "exact source excerpt"}}
      ]
    }}
  ]
}}

Rules:
1. Return 2 to 3 pros and 1 to 2 cons per product.
2. Every evidence value must be an exact excerpt from one
   supplied source field.
3. Do not invent product specifications or unsupported claims.
4. Every number in a claim must match the source information
   and its supporting evidence.
5. Never include the text "nan" in a claim or evidence.
6. Return JSON only, without Markdown fences.
{feedback_text}
Products:
{json.dumps(current_products, ensure_ascii=False)}
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            result = json.loads(response.text or "{}")

            if not isinstance(result, dict):
                raise ValueError("Response must be a JSON object.")

            generated = result.get("products")

            if not isinstance(generated, list):
                raise ValueError(
                    "JSON response must contain a products list."
                )

            by_row_id = {
                str(item["row_id"]): item
                for item in generated
                if isinstance(item, dict) and "row_id" in item
            }

        except Exception as exc:
            error = (
                f"Request or JSON parsing failed: "
                f"{type(exc).__name__}: {exc}"
            )
            for state in pending.values():
                state["errors"] = [error]
            continue

        for row_id in list(pending):
            item = by_row_id.get(row_id)

            if item is None:
                pending[row_id]["errors"] = [
                    "Response omitted this product row_id."
                ]
                continue

            if row_id not in index_by_row_id:
                pending[row_id]["errors"] = [
                    "Could not match row_id to the original product."
                ]
                continue

            errors = _validate_generated_item(
                item,
                df.loc[index_by_row_id[row_id]],
            )

            if errors:
                pending[row_id]["errors"] = errors
                continue

            validated[row_id] = {
                "item": item,
                "attempt": attempt,
            }
            del pending[row_id]

    return validated, pending


def generate_pros_cons(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pros and cons for products missing them, using Gemini."""

    if "pros_cons_status" not in df.columns:
        df["pros_cons_status"] = ""

    if "description" not in df.columns:
        df["pros_cons_status"] = "skipped_missing_description"
        print("Description column not found; skipping pros/cons generation.")
        return df

    if "pros" not in df.columns:
        df["pros"] = ""
    else:
        df["pros"] = df["pros"].fillna("")

    if "cons" not in df.columns:
        df["cons"] = ""
    else:
        df["cons"] = df["cons"].fillna("")

    # Select only products that have a description and are missing pros or cons.
    missing = (
        df["description"].fillna("").astype(str).str.strip().ne("")
        & (
            df["pros"].astype(str).str.strip().eq("")
            | df["cons"].astype(str).str.strip().eq("")
        )
    )
    indices = df.index[missing].tolist()

    if not indices:
        print("No products need pros/cons generation.")
        return df

    if not config.GEMINI_API_KEY:
        print("Gemini API key is missing; skipping pros/cons generation.")
        return df

    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # Process products in small batches to reduce API calls.
    batch_size = 10

    passed_first_time = 0
    passed_after_retries = 0
    flagged_for_review = 0

    specification_columns = [
        "key_specs",
        "ram",
        "storage",
        "processor",
        "battery",
        "display_size",
        "rear_camera",
        "front_camera",
        "os",
        "ui_skin",
    ]

    specification_columns.extend(
        column
        for column in df.columns
        if column.startswith("spec_")
    )

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        products = []

        index_by_row_id = {
            str(idx): idx for idx in batch_indices
        }

        for idx in batch_indices:
            specifications = {
                column: str(df.at[idx, column]).strip()
                for column in specification_columns
                if column in df.columns
                and pd.notna(df.at[idx, column])
                and str(df.at[idx, column]).strip()
            }

            products.append({
                "row_id": str(idx),
                "product_name": str(df.at[idx, "product_name"]),
                "description": str(df.at[idx, "description"])[:1500],
                "specifications": specifications,
            })

        # Request results and validate before accepting any claims.
        validated, pending = _request_validated_batch(
            client=client,
            products=products,
            df=df,
            max_attempts=3,
        )

        # Save only results that passed validation.
        for row_id, result in validated.items():
            idx = index_by_row_id[row_id]
            item = result["item"]
            attempt = result["attempt"]

            pros = item["pros"]
            cons = item["cons"]

            # Preserve any existing values.
            if not str(df.at[idx, "pros"]).strip():
                df.at[idx, "pros"] = "; ".join(
                    entry["claim"].strip()
                    for entry in pros
                )

            if not str(df.at[idx, "cons"]).strip():
                df.at[idx, "cons"] = "; ".join(
                    entry["claim"].strip()
                    for entry in cons
                )

            if attempt == 1:
                df.at[idx, "pros_cons_status"] = (
                    "generated_first_attempt"
                )
                passed_first_time += 1
            else:
                df.at[idx, "pros_cons_status"] = (
                    "generated_after_retry"
                )
                passed_after_retries += 1

        # Never save unvalidated generated content.
        for row_id in pending:
            idx = index_by_row_id[row_id]
            df.at[idx, "pros_cons_status"] = "needs_review"
            flagged_for_review += 1

        print(
            f"Processed batch {start // batch_size + 1}: "
            f"{len(validated)} validated, "
            f"{len(pending)} flagged."
        )

    print(
        "Pros/cons generation summary: "
        f"passed first time={passed_first_time}, "
        f"passed after retries={passed_after_retries}, "
        f"flagged for review={flagged_for_review}."
    )

    return df



def add_product_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sequential product IDs."""
    df = df.reset_index(drop=True)
    df["product_id"] = ["P" + str(i + 1).zfill(3) for i in range(len(df))]
    return df


def main():
    print("Starting data cleaning pipeline...\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    amazon_df   = load_and_map(
        os.path.join(RAW_DIR, "amazon.csv"),   AMAZON_COLUMN_MAP,   "Store A"
    )
    flipkart_df = load_and_map(
        os.path.join(RAW_DIR, "flipkart.csv"), FLIPKART_COLUMN_MAP, "Store B"
    )
    meesho_df   = load_and_map(
        os.path.join(RAW_DIR, "meesho.csv"),   MEESHO_COLUMN_MAP,   "Store C"
    )

    all_dfs = [df for df in [amazon_df, flipkart_df, meesho_df] if not df.empty]

    if not all_dfs:
        print("No raw data found. Add CSVs to data/raw/ and try again.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMerged: {len(merged)} rows from {len(all_dfs)} stores\n")

    # ── Process by category ───────────────────────────────────────────────────
    # Right now all data is smartphones — process with smartphone profile
    # When new categories arrive, add elif blocks here with their own processor
    cleaned = process_smartphones(merged)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    cleaned = deduplicate(cleaned)

    # ── Generate pros and cons ────────────────────────────────────────────────
    cleaned = generate_pros_cons(cleaned)

    # raw_product_name was only needed for deduplication
    if "raw_product_name" in cleaned.columns:
        cleaned = cleaned.drop(columns=["raw_product_name"])

    # ── Add IDs ───────────────────────────────────────────────────────────────
    cleaned = add_product_ids(cleaned)
    # ── Final column order ────────────────────────────────────────────────────
    base_columns = [
        "product_id", "product_name", "brand", "category", "subcategory",
        "store", "price", "mrp", "rating", "review_count",
        "stock_status", "key_specs", "spec_confidence",
        "pros", "cons", "pros_cons_status",
    ]
    # Add numeric spec columns — exclude spec_confidence since it's
    # already in base_columns (it's a category label, not a numeric spec)
    spec_cols = [
        c for c in cleaned.columns
        if c.startswith("spec_") and c != "spec_confidence"
    ]
    column_order = base_columns + spec_cols
    existing = [c for c in column_order if c in cleaned.columns]
    cleaned = cleaned[existing]

    # ── Save ──────────────────────────────────────────────────────────────────
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total products: {len(cleaned)}")
    print(f"\nSpec confidence breakdown:")
    print(cleaned["spec_confidence"].value_counts().to_string())
    print(f"\nNumeric spec columns added:")
    for col in spec_cols:
        print(f"  {col}: {cleaned[col].describe()['mean']:.1f} avg")

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 1: Update FLIPKART_COLUMN_MAP and MEESHO_COLUMN_MAP
    #   - Once Flipkart and Meesho scrapers are ready, check actual column names
    #     and update the maps at the top of this file
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 2: Add new category processors
    #   - When laptops/earphones/skincare data arrives, add a new profile in
    #     data/spec_profiles.py and call it here:
    #     e.g. laptop_cleaned = process_laptops(laptop_df)
    #          final = pd.concat([cleaned, laptop_cleaned])
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
