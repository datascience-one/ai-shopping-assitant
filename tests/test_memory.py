#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
tests/test_memory.py
---------------------
Tests for the conversation memory and session_id handling of the /chat endpoint.
"""

# Standard imports
import pytest

# Third-party imports
from fastapi.testclient import TestClient

# Project imports
import backend.main as main
from backend.main import (
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_CHARS,
    _conversation_history,
    add_to_history,
    get_history,
)

FAKE_PRODUCT = {
    "product_name": "Phone A", "brand": "X", "store": "Store A",
    "price": 10000.0, "mrp": 12000.0, "rating": 4.2, "review_count": 10,
    "key_specs": "8GB", "pros": "", "cons": "", "score": 80.0,
}


@pytest.fixture(autouse=True)
def clear_history() -> None:
    """Starts every test with an empty conversation history."""
    _conversation_history.clear()


def test_keeps_only_latest_messages() -> None:
    """After adding 8 messages, only the latest 6 are kept."""
    for i in range(8):
        add_to_history("s1", "user", f"message {i}")

    history = get_history("s1")
    assert len(history) == MAX_HISTORY_MESSAGES
    assert history[0]["text"] == "message 2"
    assert history[-1]["text"] == "message 7"


def test_sessions_stay_separate() -> None:
    """Two session ids never see each other's messages."""
    add_to_history("s1", "user", "first chat")
    add_to_history("s2", "user", "second chat")

    assert get_history("s1") == [{"role": "user", "text": "first chat"}]
    assert get_history("s2") == [{"role": "user", "text": "second chat"}]


def test_new_session_returns_empty_list() -> None:
    """A session id that was never used has an empty history."""
    assert get_history("never-used") == []


def test_long_message_is_trimmed() -> None:
    """A stored message is cut to MAX_MESSAGE_CHARS characters."""
    add_to_history("s1", "user", "a" * 900)
    assert len(get_history("s1")[0]["text"]) == MAX_MESSAGE_CHARS


def test_chat_remembers_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two /chat requests with the same session id share one history."""
    seen_histories: list[list[dict] | None] = []

    def fake_recommendation(
        query: str,
        top_products: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """Stands in for Gemini: records the history it receives and returns a fixed reply."""
        seen_histories.append(history)
        return f"reply to {query}"

    monkeypatch.setattr(main, "search_products", lambda **kwargs: [FAKE_PRODUCT])
    monkeypatch.setattr(
        main, "compare_and_rank", lambda results, max_budget=None: [FAKE_PRODUCT]
    )
    monkeypatch.setattr(main, "get_recommendation", fake_recommendation)

    client = TestClient(main.app)
    first = client.post("/chat", json={"message": "best phone"})
    session_id = first.json()["session_id"]
    assert session_id  # the server created one

    second = client.post(
        "/chat", json={"message": "which one?", "session_id": session_id}
    )
    assert second.json()["session_id"] == session_id
    assert seen_histories[0] == []
    assert [m["text"] for m in seen_histories[1]] == [
        "best phone",
        "reply to best phone",
    ]


def test_chat_rejects_too_long_session_id() -> None:
    """A session id over the length limit is refused with a 422 error."""
    client = TestClient(main.app)
    response = client.post(
        "/chat", json={"message": "hi", "session_id": "x" * 65}
    )
    assert response.status_code == 422
