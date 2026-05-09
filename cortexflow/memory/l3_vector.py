"""
L3 Vector Store — Semantic / Episodic Memory Layer.

This layer allows agents to recall relevant experiences from their history 
using vector similarity (embeddings) instead of keyword matching.
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Any, Optional, List
import structlog

from cortexflow.memory.base import MemoryLayer

logger = structlog.get_logger(__name__)


class MemoryDocument:
    """A semantically indexed record."""
    def __init__(
        self, 
        key: str, 
        content: str, 
        embedding: List[float], 
        metadata: dict[str, Any] | None = None
    ) -> None:
        self.key = key
        self.content = content
        self.embedding = np.array(embedding, dtype=np.float32)
        self.metadata = metadata or {}
        self.score: float = 0.0


class L3VectorStore(MemoryLayer):
    """
    High-performance Semantic Memory Layer.
    Uses Cosine Similarity for episodic retrieval.
    """

    def __init__(self, embedding_provider: Any | None = None) -> None:
        self._store: dict[str, MemoryDocument] = {}
        self._embedding_provider = embedding_provider
        logger.info("memory.l3.init", mode="semantic")

    async def read(self, key: str) -> Any | None:
        doc = self._store.get(key)
        return doc.content if doc else None

    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Stores content with an embedding. 
        If value is a tuple (content, embedding), it uses the provided embedding.
        Otherwise, it tries to generate one via the provider.
        """
        content = ""
        embedding = []
        metadata = {}

        if isinstance(value, dict):
            content = value.get("content", str(value))
            embedding = value.get("embedding", [])
            metadata = value.get("metadata", {})
        else:
            content = str(value)

        # If no embedding is provided, we'd normally call the embedding_provider
        # For now, if missing, we use a zero-vector or a mock
        if not embedding:
            logger.debug("memory.l3.auto_vectorization", key=key)
            # In production: embedding = await self._embedding_provider.embed(content)
            embedding = [0.0] * 1536 # Default size

        self._store[key] = MemoryDocument(
            key=key, 
            content=content, 
            embedding=embedding, 
            metadata=metadata
        )

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[MemoryDocument]:
        """
        Retrieves the most semantically similar documents.
        Complexity: O(n) for local search. Scalable via ChromaDB/pgvector.
        """
        if not self._store:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        
        results = []
        for doc in self._store.values():
            # Cosine Similarity
            dot_product = np.dot(q_vec, doc.embedding)
            doc_norm = np.linalg.norm(doc.embedding)
            
            if q_norm == 0 or doc_norm == 0:
                score = 0.0
            else:
                score = dot_product / (q_norm * doc_norm)
            
            doc.score = float(score)
            results.append(doc)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def flush(self, prefix: str | None = None) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    async def health(self) -> dict[str, Any]:
        return {
            "layer": "L3_vector",
            "status": "ready",
            "vectors_cached": len(self._store),
            "engine": "cortex-vector-local"
        }
