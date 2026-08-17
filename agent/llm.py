"""agent/llm.py
Thin helper for direct Gemini calls (Vertex AI) used by the agent tools
(identify_gaps, fetch_supplementary fallback, generate_study_guide).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import appconfig as config

_initialized = False


def _ensure_vertex():
    global _initialized
    if _initialized:
        return
    if not config.GOOGLE_CLOUD_PROJECT:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not set. Add it to your .env file and authenticate with "
            "`gcloud auth application-default login` before running the agent."
        )
    import vertexai

    vertexai.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_LOCATION)
    _initialized = True


def generate(prompt: str, temperature: float = 0.2, max_output_tokens: int = 4096) -> str:
    """Send a single-turn prompt to Gemini and return the text response."""
    _ensure_vertex()
    from vertexai.generative_models import GenerativeModel

    model = GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
    )
    return (response.text or "").strip()


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