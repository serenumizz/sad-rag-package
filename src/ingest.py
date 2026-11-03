"""
Step 1 of the RAG pipeline: embed every document and save the vectors to disk.

Run once (or whenever data/project_docs.py changes):
    python src/ingest.py

Requires: pip install openai numpy
Requires env var: OPENAI_API_KEY
"""
import json
import os
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from project_docs import DOCUMENTS  # noqa: E402

EMBED_MODEL = "text-embedding-3-small"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index.json")


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running ingest.py")

    client = OpenAI(api_key=api_key)

    texts = [f"{d['title']}\n{d['text']}" for d in DOCUMENTS]
    print(f"Embedding {len(texts)} documents with {EMBED_MODEL}...")
    vectors = embed_texts(client, texts)

    index = [
        {**doc, "embedding": vec}
        for doc, vec in zip(DOCUMENTS, vectors)
    ]

    with open(OUT_PATH, "w") as f:
        json.dump(index, f)

    print(f"Saved {len(index)} embedded chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
