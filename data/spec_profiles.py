#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
data/spec_profiles.py
---------------------
Category-specific spec profiles.

Each profile defines:
- which columns to extract from the raw scraped CSV
- which sources are trusted for scoring
- how to build the key_specs string
- how to score specs numerically for the Compare Agent

When a new category is added (laptops, earphones, skincare etc.),
add a new profile here. Nothing else needs to change.

Students:
- To add a new category: copy the Smartphones profile below,
  update the fields to match your scraped columns, and add it
  to SPEC_PROFILES with the correct category key.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Spec Profiles — one per category
# ─────────────────────────────────────────────────────────────────────────────

SPEC_PROFILES = {

    "Electronics/Smartphones": {

        # Raw CSV columns to keep (besides product_name, brand, price, mrp, rating, review_count)
        "spec_columns": [
            "ram", "storage", "processor", "battery",
            "display_size", "rear_camera", "front_camera", "os", "ui_skin"
        ],

        # Source columns for each spec — used to calculate confidence
        # Format: spec_column → its source column in the raw CSV
        "source_columns": {
            "ram":          "ram_source",
            "storage":      "storage_source",
            "processor":    "processor_source",
            "battery":      "battery_source",
            "display_size": "display_size_source",
            "rear_camera":  "rear_camera_source",
            "front_camera": "front_camera_source",
            "os":           "os_source",
        },

        # Sources considered reliable for scoring
        "trusted_sources": ["spec_table", "title"],

        # Specs used for numeric scoring in the Compare Agent
        # Format: spec_column → (unit, higher_is_better, max_value_for_normalisation)
        "scoring_specs": {
            "ram":          ("GB",   True,  16),    # 16GB = max expected
            "storage":      ("GB",   True,  512),   # 512GB = max expected
            "battery":      ("mAh",  True,  7000),  # 7000mAh = max expected
            "rear_camera":  ("MP",   True,  200),   # 200MP = max expected
            "display_size": ("in",   True,  7.0),   # 7 inch = max expected
        },

        # How to build the key_specs display string
        # Format: label → spec_column
        "display_specs": {
            "RAM":          "ram",
            "Storage":      "storage",
            "Processor":    "processor",
            "Battery":      "battery",
            "Display":      "display_size",
            "Rear Camera":  "rear_camera",
            "OS":           "os",
        },

        # Subcategory label to apply
        "subcategory": "Smartphone",

        # Category label to apply
        "category": "Electronics",
    },
        "Electronics/Laptops": {

        "spec_columns": [
            "ram", "storage", "processor", "display_size",
            "battery_life_hrs", "weight"
        ],

        "source_columns": {
            "ram":              "ram_source",
            "storage":          "storage_source",
            "processor":        "processor_source",
            "display_size":     "display_size_source",
            "battery_life_hrs": "battery_life_hrs_source",
            "weight":           "weight_source",
        },

        "trusted_sources": ["spec_table", "title"],

        "scoring_specs": {
            "ram":              ("GB",  True,  64),
            "storage":          ("GB",  True,  2000),
            "battery_life_hrs": ("hrs", True,  20),
            "weight":           ("kg",  False, 3),
        },

        "display_specs": {
            "RAM":          "ram",
            "Storage":      "storage",
            "Processor":    "processor",
            "Display":      "display_size",
            "Battery Life": "battery_life_hrs",
            "Weight":       "weight",
        },

        "subcategory": "Laptop",
        "category": "Electronics",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TODO: Add Laptops profile
    #   "Electronics/Laptops": {
    #       "spec_columns": ["ram", "storage", "processor", "display_size", "battery_life_hrs", "weight"],
    #       "scoring_specs": {
    #           "ram":              ("GB",  True,  64),
    #           "storage":          ("GB",  True,  2000),
    #           "battery_life_hrs": ("hrs", True,  20),
    #           "weight":           ("kg",  False, 3),   # lower is better
    #       },
    #       ...
    #   }
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO: Add Skincare profile
    #   "Skincare/Serum": {
    #       "spec_columns": ["volume_ml", "key_ingredient", "skin_type", "spf"],
    #       "scoring_specs": {
    #           "volume_ml": ("ml", True, 100),
    #       },
    #       ...
    #   }
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TODO: Add Earphones profile
    #   "Electronics/Earphones": {
    #       "spec_columns": ["battery_hrs", "driver_size_mm", "connectivity", "water_resistance"],
    #       "scoring_specs": {
    #           "battery_hrs":     ("hrs", True, 60),
    #           "driver_size_mm":  ("mm",  True, 15),
    #       },
    #       ...
    #   }
    # ─────────────────────────────────────────────────────────────────────────
}


def get_profile(category: str, subcategory: str = "") -> dict:
    """
    Returns the spec profile for a given category/subcategory.
    Falls back to a generic profile if no specific one exists.

    Args:
        category    : e.g. "Electronics"
        subcategory : e.g. "Smartphones"

    Returns:
        dict: spec profile
    """
    key = f"{category}/{subcategory}" if subcategory else category
    if key in SPEC_PROFILES:
        return SPEC_PROFILES[key]

    # Try category only
    for profile_key in SPEC_PROFILES:
        if profile_key.startswith(category):
            return SPEC_PROFILES[profile_key]

    return {}
