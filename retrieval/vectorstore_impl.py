import os
import faiss
import numpy as np
from typing import List


def save_faiss_index(embeddings: np.ndarray, index_path: str):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    index = faiss.read_index(index_path)
    return index


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += chunk_size - overlap
    return chunks
