"""appconfig.py
Project-level settings for the GyanVerse Agent.

Named `appconfig` (not `config`) so it never collides with the pre-existing
retrieval module `retrieval/config.py`, which is imported as bare `config`
inside the retrieval package.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Google Cloud / Vertex AI ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- Gemini model used by the ADK agent + tool calls ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- Outputs & persistence ---
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
SESSION_COLLECTION = os.getenv("SESSION_COLLECTION", "sessions")
LOCAL_STATE_DIR = os.getenv("LOCAL_STATE_DIR", ".state")

# --- Firestore ---
# Set FIRESTORE_EMULATOR=true to use the Firestore emulator (no cloud billing).
FIRESTORE_EMULATOR = os.getenv("FIRESTORE_EMULATOR", "").lower() in ("1", "true", "yes")

# --- Web fallback for fetch_supplementary ---
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
FETCH_TIMEOUT_SECONDS = int(os.getenv("FETCH_TIMEOUT_SECONDS", "8"))