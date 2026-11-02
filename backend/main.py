#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
backend/main.py
---------------
FastAPI server — the entry point for the AI Shopping Assistant.

Endpoints:
    POST /chat   — accepts a user message, runs the full agent pipeline,
                   returns a product recommendation
    GET  /health — health check

To run:
    uvicorn backend.main:app --reload

Students:
- The /chat endpoint wires together Search → Compare → Recommend.
- TODO items show you what to extend once the base is working.
"""

import logging
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.search import search_products
from agents.compare import compare_and_rank
from agents.recommend import get_recommendation

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# Logs are printed to the terminal where uvicorn is running.
# In Phase 4 this will be replaced by LangSmith tracing.
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Shopping Assistant",
    description="An agentic shopping assistant that searches, compares, and recommends products.",
    version="0.1.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

MAX_SESSION_ID_LENGTH = 64


class ChatRequest(BaseModel):
    message: str
    max_budget: float = None      # optional — user can specify budget
    category: str = None          # optional — user can specify category
    session_id: str | None = Field(default=None, max_length=MAX_SESSION_ID_LENGTH)


class ProductResult(BaseModel):
    product_name: str
    brand: str
    store: str
    price: float
    mrp: float
    rating: float
    review_count: int
    key_specs: str
    pros: str
    cons: str
    score: float = None


class ChatResponse(BaseModel):
    user_message: str
    recommendation: str           # natural language response from LLM
    top_products: list[ProductResult]
    session_id: str | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Conversation memory
# In-memory for now. To switch to Redis later, only the bodies of
# add_to_history() and get_history() need to change.
# _save_exchange() calls those two, so it stays the same.
# Sessions never expire here, so the dict grows until the server restarts.
# Phase 3: Redis with a TTL (time-to-live) removes old sessions automatically.
# ─────────────────────────────────────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 500
_conversation_history: dict[str, list[dict[str, str]]] = {}


def add_to_history(session_id: str, role: str, text: str) -> None:
    """
    Saves one message to a session's conversation history.

    Args:
        session_id: Identifies one chat session.
        role: Who wrote the message, "user" or "assistant".
        text: The message text.
    """
    history = _conversation_history.setdefault(session_id, [])
    history.append({"role": role, "text": text[:MAX_MESSAGE_CHARS]})
    del history[:-MAX_HISTORY_MESSAGES]  # keep only the latest messages


def get_history(session_id: str) -> list[dict[str, str]]:
    """
    Returns the saved messages for a session, oldest first.

    Args:
        session_id: Identifies one chat session.

    Returns:
        list[dict]: Messages with "role" and "text". Empty if the session is new.
    """
    return list(_conversation_history.get(session_id, []))

def _save_exchange(session_id: str, message: str, recommendation: str) -> None:
    """
    Saves a user message and the assistant reply to the session history.

    Args:
        session_id: Identifies one chat session.
        message: The user's message.
        recommendation: The assistant's reply.
    """
    add_to_history(session_id, "user", message)
    add_to_history(session_id, "assistant", recommendation)
    logger.info(
        f"MEMORY   | session={session_id} "
        f"| messages={len(get_history(session_id))}"
    )
# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint — runs the full agent pipeline.

    Pipeline:
        1. Search Agent  — filters products based on query + budget + category
        2. Compare Agent — scores and ranks filtered products, picks top 3
        3. Recommend Agent — sends top 3 to LLM, returns natural language reply

    Args:
        request: ChatRequest with user message, optional budget, category and session_id

    Returns:
        ChatResponse with LLM recommendation, top product details and session_id
    """

    pipeline_start = time.time()

    logger.info(f"QUERY    | message='{request.message}' | budget={request.max_budget} | category={request.category}")

    # ── Step 1 — Search ───────────────────────────────────────────────────────
    t0 = time.time()
    search_results = search_products(
        query=request.message,
        max_budget=request.max_budget,
        category=request.category,
    )
    search_time = time.time() - t0

    logger.info(f"SEARCH   | found={len(search_results)} products | time={search_time:.3f}s")

    if not search_results:
        logger.warning(f"SEARCH   | no results for query='{request.message}'")
        raise HTTPException(
            status_code=404,
            detail="No products found matching your query. Try different keywords or a higher budget."
        )

    # ── Step 2 — Compare ──────────────────────────────────────────────────────
    t0 = time.time()
    top_products = compare_and_rank(search_results, max_budget=request.max_budget)
    compare_time = time.time() - t0

    logger.info(f"COMPARE  | top={len(top_products)} products | time={compare_time:.3f}s")
    for i, p in enumerate(top_products, 1):
        logger.info(
            f"           #{i} {p['product_name']} | store={p['store']} "
            f"| price={p['price']} | score={p.get('score', 'N/A')}"
        )

    # ── Step 3 — Recommend ────────────────────────────────────────────────────
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id)
    t0 = time.time()
    recommendation = get_recommendation(request.message, top_products, history)
    llm_time = time.time() - t0

    total_time = time.time() - pipeline_start

    logger.info(f"LLM      | time={llm_time:.3f}s")
    logger.info(f"COMPLETE | total_time={total_time:.3f}s")
    _save_exchange(session_id, request.message, recommendation)
    return ChatResponse(
        user_message=request.message,
        recommendation=recommendation,
        top_products=[ProductResult(**p) for p in top_products],
        session_id=session_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Add conversation memory
#   - Right now every /chat call is stateless — the assistant has no memory
#     of previous messages in the same session.
#   - Add a session_id field to ChatRequest and maintain a conversation history
#     dict in memory (or Redis later).
#   - Pass the conversation history to the LLM so it can handle follow-ups
#     like "which of these is best for dry skin?" after a previous query.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Add intent extraction
#   - Before calling the Search Agent, add an intent extraction step that
#     parses the user's message and automatically fills max_budget and category.
#   - Example: "I need a good lipstick under Rs 400" should automatically set
#     max_budget=400 and category="Beauty" without the user specifying them.
#   - You can use the LLM itself or a simple regex-based extractor for this.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Connect to live WooCommerce stores
#   - Right now the system reads from a static CSV file.
#   - In Phase 2, replace the CSV with live WooCommerce API calls.
#   - Endpoint: GET {WP_BASE_URL}/wp-json/wc/v3/products
#   - See add_users.py for the auth pattern (consumer key + secret).
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Replace logging with LangSmith tracing (Phase 4)
#   - pip install langsmith
#   - Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true in .env
#   - Wrap the pipeline with @traceable decorator from langsmith
#   - Every Search, Compare, Recommend step becomes a traced span
#   - View full traces at https://smith.langchain.com
# ─────────────────────────────────────────────────────────────────────────────
