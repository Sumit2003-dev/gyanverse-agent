"""
rag_engine.py
Loads the FAISS index and retrieves relevant context for a query.
Adapted from the pre-existing GyanVerse AI component to use a pluggable embedding backend.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import faiss

import config
from vectorstore_impl import load_faiss_index
from embeddings import embed_texts


class RAGEngine:
    def __init__(self, index_dir=None):
        index_dir = index_dir or config.VECTORSTORE_DIR
        index_path = os.path.join(index_dir, config.INDEX_NAME + ".index")
        meta_path = os.path.join(index_dir, config.INDEX_NAME + ".pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Index or metadata not found. Run `python retrieval/ingest.py` to build the index.\n"
                f"Missing: {index_path if not os.path.exists(index_path) else meta_path}"
            )

        self.index = load_faiss_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.texts = meta.get("texts", [])
        self.metadatas = meta.get("metadatas", [])

    def retrieve(self, query: str, k: int = None):
        k = k or config.TOP_K
        q_emb = embed_texts([query])
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append(
                {
                    "page_content": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(score),
                }
            )
        return results


def engine_exists(index_dir=None):
    index_dir = index_dir or config.VECTORSTORE_DIR
    index_path = os.path.join(index_dir, config.INDEX_NAME + ".index")
    return os.path.exists(index_path)