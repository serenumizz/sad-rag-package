"""
Shared retrieval logic: load the embedded index, embed a query, and return
the top-k most similar chunks by cosine similarity.
"""
import json
import os

import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index.json")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Retriever:
    def __init__(self, client: OpenAI, index_path: str = INDEX_PATH):
        self.client = client
        with open(index_path) as f:
            self.index = json.load(f)
        for item in self.index:
            item["_vec"] = np.array(item["embedding"], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def top_k(self, query: str, k: int = 3) -> list[dict]:
        q_vec = self.embed_query(query)
        scored = [
            {**{kk: vv for kk, vv in item.items() if kk not in ("_vec", "embedding")},
             "score": _cosine_sim(q_vec, item["_vec"])}
            for item in self.index
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
