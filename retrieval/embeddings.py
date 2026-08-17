"""retrieval/embeddings.py
Pluggable embedding backend for the GyanVerse retrieval pipeline.

Backends:
  - gemini : Vertex AI text embeddings (text-embedding-005)
  - openai : OpenAI text-embedding-3-small (pre-existing GyanVerse AI backend)
  - local  : Sentence Transformers all-MiniLM-L6-v2 (fully offline)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    faiss_l2 = None
    try:
        import faiss

        faiss_l2 = faiss
    except Exception:  # pragma: no cover - faiss is a hard dependency
        pass
    if faiss_l2 is not None:
        faiss_l2.normalize_L2(vectors)
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
    return vectors


def _embed_gemini(texts):
    if not config.GOOGLE_CLOUD_PROJECT:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not set. Add it to your .env file, then run "
            "`gcloud auth application-default login` (or set GOOGLE_APPLICATION_CREDENTIALS)."
        )
    import vertexai
    from vertexai.language_models import TextEmbeddingModel

    vertexai.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_LOCATION)
    model = TextEmbeddingModel.from_pretrained(config.EMBED_MODEL)
    results = model.get_embeddings(texts)
    return [r.values for r in results]


def _embed_openai(texts):
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file to use the openai backend.")
    import openai

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _embed_local(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'local' embedding backend needs `pip install sentence-transformers torch`. "
            "Install it, or switch EMBED_BACKEND to 'gemini' or 'openai'."
        ) from exc
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_texts(texts, backend=None):
    """Embed a list of strings into a normalized float32 matrix."""
    backend = backend or config.EMBED_BACKEND
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    if backend == "gemini":
        raw = _embed_gemini(texts)
    elif backend == "openai":
        raw = _embed_openai(texts)
    elif backend in ("local", "sentence_transformers"):
        raw = _embed_local(texts)
    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {backend!r}")
    return _normalize(np.array(raw, dtype="float32"))