"""
ingest.py
Loads documents from DATA_DIR, splits them into chunks, embeds them,
and builds/saves a FAISS index.

Adapted from the pre-existing GyanVerse AI component to use the pluggable
embedding backend (default: Gemini text-embeddings via Vertex AI).

Usage:
    python retrieval/ingest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
from typing import List

from pypdf import PdfReader
from docx import Document as DocxDocument
import numpy as np
import faiss

import config
from vectorstore_impl import save_faiss_index, chunk_text
from embeddings import embed_texts


def load_documents(data_dir: str):
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if os.path.isdir(path):
            continue

        text = ""
        if fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"Failed to load PDF {fname}: {e}")
                continue
        elif fname.lower().endswith(".docx"):
            try:
                doc = DocxDocument(path)
                for p in doc.paragraphs:
                    text += p.text + "\n"
            except Exception as e:
                print(f"Failed to load DOCX {fname}: {e}")
                continue
        elif fname.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Failed to load TXT {fname}: {e}")
                continue
        else:
            print(f"Skipping unsupported file: {fname}")
            continue

        if text.strip():
            docs.append({"source": fname, "text": text})
            print(f"Loaded: {fname}")
    return docs


def chunk_documents(docs: List[dict]):
    chunks = []
    for doc in docs:
        pieces = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, p in enumerate(pieces):
            chunks.append({"page_content": p, "metadata": {"source": doc["source"], "chunk": i}})
    return chunks


def build_index(chunks: List[dict]):
    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks with backend='{config.EMBED_BACKEND}' ...")
    try:
        embeddings = embed_texts(texts)
    except Exception as exc:
        print(f"Embedding failed: {exc}")
        return

    if embeddings.shape[1] == 0:
        print("No embeddings produced. Nothing to save.")
        return

    index_path = os.path.join(config.VECTORSTORE_DIR, config.INDEX_NAME + ".index")
    meta_path = os.path.join(config.VECTORSTORE_DIR, config.INDEX_NAME + ".pkl")

    save_faiss_index(embeddings, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metas}, f)

    print(f"Saved index to {index_path} and metadata to {meta_path}")


if __name__ == "__main__":
    if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        print(f"Put your PDFs/DOCX/TXT files in the '{config.DATA_DIR}/' folder first.")
        exit(1)

    docs = load_documents(config.DATA_DIR)
    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    build_index(chunks)