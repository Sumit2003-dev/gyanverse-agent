"""retrieval/config.py
Settings for the GyanVerse retrieval pipeline (FAISS index, embedding backend).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Embeddings ---
# gemini      -> Vertex AI text embeddings (Google-native, used for the hackathon build)
# openai      -> OpenAI text-embedding-3-small (the pre-existing GyanVerse AI backend)
# local       -> Sentence Transformers "all-MiniLM-L6-v2" (fully offline; needs torch)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "gemini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-005")      # Vertex AI embedding model
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Legacy LLM (used only by the pre-existing rag_engine.answer()) ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Google Cloud (Vertex AI) ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 4

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
INDEX_NAME = os.getenv("INDEX_NAME", "gyanverse_index")