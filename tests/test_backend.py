# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.

import math

from backend.main import normalize_product_text


def test_normalize_product_text_converts_nan_to_empty_strings():
    product = {
        "product_name": "Test Phone",
        "pros": math.nan,
        "cons": math.nan,
    }

    result = normalize_product_text(product)

    assert result["pros"] == ""
    assert result["cons"] == ""
