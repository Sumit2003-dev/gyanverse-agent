"""agent/llm.py
Thin helper for direct Gemini calls used by the agent tools
(identify_gaps, fetch_supplementary fallback, generate_study_guide).

Two backends, selected automatically:
  - Gemini API key  (GOOGLE_API_KEY set)  -> free tier, no GCP billing
  - Vertex AI       (GOOGLE_CLOUD_PROJECT set) -> enterprise
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import appconfig as config

_client = None
_MAX_RETRIES = 5


def _make_client():
    global _client
    if _client is not None:
        return _client
    from google import genai

    if config.GOOGLE_API_KEY:
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
        return _client
    if not config.GOOGLE_CLOUD_PROJECT:
        raise ValueError(
            "Neither GOOGLE_API_KEY nor GOOGLE_CLOUD_PROJECT is set. Add one to your "
            ".env file (GOOGLE_API_KEY = free tier, no billing)."
        )
    _client = genai.Client(
        vertexai=True,
        project=config.GOOGLE_CLOUD_PROJECT,
        location=config.GOOGLE_CLOUD_LOCATION,
    )
    return _client


def generate(prompt: str, temperature: float = 0.2, max_output_tokens: int = 4096) -> str:
    """Send a single-turn prompt to Gemini and return the text response."""
    client = _make_client()
    from google.genai import errors

    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": max_output_tokens},
            )
            return (response.text or "").strip()
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2


def generate_json(prompt: str, temperature: float = 0.1, max_output_tokens: int = 4096):
    """Ask Gemini for JSON output and parse it robustly."""
    raw = generate(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}