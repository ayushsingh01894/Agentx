"""
memory.py
Module 4 - Agent Memory Systems
Implements:
 - Working memory (current session state)
 - Episodic memory (log of past research sessions)
 - Semantic/vector memory (lightweight embedding + cosine similarity retrieval,
   swap `_embed` for sentence-transformers/OpenAI embeddings in production)
"""
import json
import math
import random
from pathlib import Path

EPISODIC_FILE = Path(__file__).parent.parent / "episodic_memory.json"


class WorkingMemory:
    """Holds state for the current run only."""
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)


def _embed(text: str, dim: int = 32):
    """Deterministic pseudo-embedding for demo purposes (no external deps).
    Replace with a real embedding model (e.g. sentence-transformers) in production."""
    random.seed(abs(hash(text.lower())) % (2 ** 32))
    return [random.random() for _ in range(dim)]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-8)


class SemanticMemory:
    """Simple in-memory vector store. Swap with FAISS/Chroma/Qdrant for production."""
    def __init__(self):
        self._store: list[tuple[str, list[float]]] = []

    def add(self, text: str):
        self._store.append((text, _embed(text)))

    def retrieve(self, query: str, top_k: int = 3):
        if not self._store:
            return []
        q_emb = _embed(query)
        scored = [(t, _cosine(q_emb, e)) for t, e in self._store]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]


class EpisodicMemory:
    """Persists a record of every research session to disk (JSON file)."""
    def __init__(self):
        self._episodes = self._load()

    def _load(self):
        if EPISODIC_FILE.exists():
            with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def log_episode(self, topic: str, report: str):
        self._episodes.append({"topic": topic, "report_preview": report[:200]})
        with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
            json.dump(self._episodes, f, indent=2)

    def past_topics(self):
        return [e["topic"] for e in self._episodes]
