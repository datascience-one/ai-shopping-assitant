#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##
"""
config.py
---------
Loads all environment variables from .env and exposes them
as a single config object used across the project.

Students: Do not hardcode any keys or URLs here.
          Add new config values to .env.example and load them below.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# WooCommerce
# ─────────────────────────────────────────────
WP_BASE_URL = os.getenv("WP_BASE_URL", "http://localhost:8080")
WP_USERNAME = os.getenv("WP_USERNAME", "admin")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")
WC_CONSUMER_KEY = os.getenv("WC_CONSUMER_KEY")
WC_CONSUMER_SECRET = os.getenv("WC_CONSUMER_SECRET")

# Store A is the same store as above (kept for naming consistency with load_products.py)
STORE_A_URL = os.getenv("STORE_A_URL", WP_BASE_URL)

# Store B and C — populated once Minu adds them to docker-compose.yml (Phase 2)
STORE_B_URL = os.getenv("STORE_B_URL", "http://localhost:8081")
STORE_B_CONSUMER_KEY = os.getenv("STORE_B_CONSUMER_KEY")
STORE_B_CONSUMER_SECRET = os.getenv("STORE_B_CONSUMER_SECRET")

STORE_C_URL = os.getenv("STORE_C_URL", "http://localhost:8082")
STORE_C_CONSUMER_KEY = os.getenv("STORE_C_CONSUMER_KEY")
STORE_C_CONSUMER_SECRET = os.getenv("STORE_C_CONSUMER_SECRET")

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai | anthropic
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")
