# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.

import pandas as pd

from data.clean_products import generate_pros_cons


def test_preserves_existing_pros_and_cons():
    df = pd.DataFrame({
        "product_name": ["Test Phone"],
        "description": ["A smartphone with a large display."],
        "pros": ["Good display"],
        "cons": ["Expensive"],
    })

    result = generate_pros_cons(df)

    assert result.loc[0, "pros"] == "Good display"
    assert result.loc[0, "cons"] == "Expensive"


def test_skips_when_description_is_missing():
    df = pd.DataFrame({
        "product_name": ["Test Phone"],
        "pros": [""],
        "cons": [""],
    })

    result = generate_pros_cons(df)

    assert result.loc[0, "pros"] == ""
    assert result.loc[0, "cons"] == ""
    assert result.loc[0, "pros_cons_status"] == "skipped_missing_description"


import pandas as pd
from unittest.mock import MagicMock, patch

from data.clean_products import generate_pros_cons


@patch("config.GEMINI_API_KEY", "test-key")
def test_retries_when_battery_number_is_incorrect():
    description = (
        "A smartphone with an OLED display, a 5000mAh battery, "
        "and no charger included in the box."
    )

    df = pd.DataFrame({
        "product_name": ["Test Phone"],
        "description": [description],
        "pros": [""],
        "cons": [""],
    })

    # First response: incorrect battery capacity.
    bad_response = MagicMock()
    bad_response.text = (
        '{"products": [{"row_id": "0", '
        '"pros": ['
        '{"claim": "6000mAh battery", "evidence": "' + description + '"},'
        '{"claim": "OLED display", "evidence": "' + description + '"}], '
        '"cons": ['
        '{"claim": "Charger is not included", '
        '"evidence": "' + description + '"}]}]}'
    )

    # Second response: battery capacity matches the description.
    valid_response = MagicMock()
    valid_response.text = (
        '{"products": [{"row_id": "0", '
        '"pros": ['
        '{"claim": "5000mAh battery", "evidence": "' + description + '"},'
        '{"claim": "OLED display", "evidence": "' + description + '"}], '
        '"cons": ['
        '{"claim": "Charger is not included", '
        '"evidence": "' + description + '"}]}]}'
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = [
            bad_response,
            valid_response,
        ]

        result = generate_pros_cons(df)

    # The first response must be rejected and the retry accepted.
    assert mock_client.models.generate_content.call_count == 2

    assert result.loc[0, "pros"] == (
        "5000mAh battery; OLED display"
    )
    assert result.loc[0, "cons"] == "Charger is not included"
    assert result.loc[0, "pros_cons_status"] == "generated_after_retry"

    # Confirm the retry prompt contains the validation feedback.
    retry_prompt = (
        mock_client.models.generate_content.call_args_list[1]
        .kwargs["contents"]
    )
    assert "number not found" in retry_prompt

@patch("config.GEMINI_API_KEY", "test-key")
def test_flags_for_review_after_three_failed_attempts():
    description = (
        "A smartphone with a 5000mAh battery and an OLED display. "
        "No charger included in the box."
    )

    df = pd.DataFrame({
        "product_name": ["Test Phone"],
        "description": [description],
        "pros": [""],
        "cons": [""],
    })

    # Gemini repeatedly returns an incorrect battery capacity.
    bad_response = MagicMock()
    bad_response.text = (
        '{"products": [{"row_id": "0", '
        '"pros": ['
        '{"claim": "6000mAh battery", "evidence": "' + description + '"},'
        '{"claim": "OLED display", "evidence": "' + description + '"}], '
        '"cons": ['
        '{"claim": "Charger is not included", '
        '"evidence": "' + description + '"}]}]}'
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = [
            bad_response,
            bad_response,
            bad_response,
        ]

        result = generate_pros_cons(df)

    # The API must never be called more than three times.
    assert mock_client.models.generate_content.call_count == 3

    # Rejected generated content must not be saved.
    assert result.loc[0, "pros"] == ""
    assert result.loc[0, "cons"] == ""

    # Failed validation must be flagged for manual review.
    assert result.loc[0, "pros_cons_status"] == "needs_review"

@patch("config.GEMINI_API_KEY", "test-key")
def test_valid_response_passes_first_attempt_and_reports_summary(capsys):
    description = (
        "A smartphone with an OLED display and a 5000mAh battery. "
        "No charger included in the box."
    )

    df = pd.DataFrame({
        "product_name": ["Test Phone"],
        "description": [description],
        "pros": [""],
        "cons": [""],
    })

    mock_response = MagicMock()
    mock_response.text = (
        '{"products": [{"row_id": "0", '
        '"pros": ['
        '{"claim": "OLED display", "evidence": "' + description + '"},'
        '{"claim": "5000mAh battery", "evidence": "' + description + '"}], '
        '"cons": ['
        '{"claim": "Charger is not included", '
        '"evidence": "' + description + '"}]}]}'
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = generate_pros_cons(df)

    assert mock_client.models.generate_content.call_count == 1
    assert result.loc[0, "pros_cons_status"] == "generated_first_attempt"

    output = capsys.readouterr().out
    assert "passed first time=1" in output
    assert "passed after retries=0" in output
    assert "flagged for review=0" in output
