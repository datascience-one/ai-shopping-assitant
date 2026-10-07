#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##

"""
agents/recommend.py
-------------------
Recommend Agent — sends top products to an LLM and gets a natural language recommendation.

This agent is responsible for:
- Building a prompt from the top ranked products
- Calling the configured LLM provider (Gemini / OpenAI / Anthropic)
- Returning a clean, conversational recommendation response

Students:
- The prompt builder is implemented below.
- You need to complete the LLM call for your chosen provider (see TODOs).
- Only implement the provider your team is using — leave others as stubs.

Testing note:
- Until an LLM provider is implemented, get_recommendation() returns a
  clearly-labeled placeholder response instead of crashing. This lets
  anyone test the Search and Compare agents end-to-end through /chat
  before the LLM integration task is picked up.
"""

from google import genai
import config


def build_prompt(user_query: str, top_products: list[dict]) -> str:
    """
    Builds the prompt to send to the LLM.

    Args:
        user_query   : Original query from the user
        top_products : Top ranked products from the Compare Agent

    Returns:
        str: Formatted prompt string
    """
    product_lines = []
    for i, p in enumerate(top_products, start=1):
        product_lines.append(
            f"{i}. {p['product_name']} by {p['brand']}\n"
            f"   Store: {p['store']} | Price: Rs {p['price']} (MRP: Rs {p['mrp']})\n"
            f"   Rating: {p['rating']}/5 ({p['review_count']} reviews)\n"
            f"   Specs: {p['key_specs']}\n"
            f"   Pros: {p['pros']}\n"
            f"   Cons: {p['cons']}\n"
            f"   Score: {p.get('score', 'N/A')}/100"
        )

    products_text = "\n\n".join(product_lines)

    prompt = f"""You are a helpful shopping assistant. A user is looking for: "{user_query}"

Based on the available products, here are the top recommendations:

{products_text}

Please give a friendly, concise recommendation in 3–5 sentences.
- Mention which product offers the best value and why.
- Point out any trade-offs the user should know about.
- Suggest which store to buy from based on the price.
- Do not make up any information not provided above.
"""
    return prompt


def _build_fallback_response(provider: str, top_products: list[dict]) -> str:
    """
    Returns a clearly-labeled placeholder response when the LLM provider
    has not been implemented yet (_call_gemini etc. raises NotImplementedError).

    This allows the Search and Compare agents to be tested end-to-end
    through the /chat endpoint without requiring the LLM integration
    task to be finished first.

    Args:
        provider     : The configured LLM_PROVIDER name (gemini/openai/anthropic)
        top_products : Top ranked products from the Compare Agent

    Returns:
        str: A placeholder message listing the top products with a note
             that the LLM call still needs to be implemented.
    """
    product_summary = "\n".join(
        f"  {i + 1}. {p['product_name']} ({p['store']}) "
        f"— Rs {p['price']} — score {p.get('score', 'N/A')}"
        for i, p in enumerate(top_products)
    )
    return (
        f"[LLM not yet connected — implement _call_{provider}() in agents/recommend.py]\n\n"
        f"Pipeline is working correctly. Top {len(top_products)} results found:\n"
        f"{product_summary}"
    )


def get_recommendation(user_query: str, top_products: list[dict]) -> str:
    """
    Sends the top products to the configured LLM and returns a recommendation.

    If the configured LLM provider has not been implemented yet, returns a
    clearly-labeled placeholder instead of crashing — this keeps the
    Search and Compare agents testable through /chat at all times.

    Args:
        user_query   : Original query from the user
        top_products : Top ranked products from the Compare Agent

    Returns:
        str: Natural language recommendation from the LLM, or a
             placeholder message if the LLM integration is not yet built.

    Raises:
        ValueError: If LLM_PROVIDER is not recognised
    """
    if not top_products:
        return "Sorry, I could not find any products matching your request. Try a different search."

    prompt = build_prompt(user_query, top_products)
    provider = config.LLM_PROVIDER.lower()

    if provider not in ("gemini", "openai", "anthropic"):
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Set it to gemini, openai, or anthropic in .env"
        )

    try:
        if provider == "gemini":
            return _call_gemini(prompt)
        elif provider == "openai":
            return _call_openai(prompt)
        elif provider == "anthropic":
            return _call_anthropic(prompt)
    except NotImplementedError:
        return _build_fallback_response(provider, top_products)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider Implementations
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """
    Calls the Google Gemini API and returns the response text.

    TODO:
        1. Install the SDK:       pip install google-genai
        2. Import and call:
               from google import genai
               client = genai.Client(api_key=config.GEMINI_API_KEY)
               response = client.models.generate_content(
                   model="gemini-3.6-flash",
                   contents=prompt
               )
               return response.text
    """
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text

def _call_openai(prompt: str) -> str:
    """
    Calls the OpenAI API and returns the response text.

    TODO:
        1. Install the SDK:       pip install openai
        2. Import and call:
               from openai import OpenAI
               client = OpenAI(api_key=config.OPENAI_API_KEY)
               response = client.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[{"role": "user", "content": prompt}]
               )
               return response.choices[0].message.content
    """
    raise NotImplementedError("OpenAI integration not yet implemented. See TODO in _call_openai()")


def _call_anthropic(prompt: str) -> str:
    """
    Calls the Anthropic Claude API and returns the response text.

    TODO:
        1. Install the SDK:       pip install anthropic
        2. Import and call:
               import anthropic
               client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
               message = client.messages.create(
                   model="claude-haiku-4-5-20251001",
                   max_tokens=512,
                   messages=[{"role": "user", "content": prompt}]
               )
               return message.content[0].text
    """
    raise NotImplementedError("Anthropic integration not yet implemented. See TODO in _call_anthropic()")
