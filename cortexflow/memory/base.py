"""
MemoryLayer — Abstract base for all CortexFlow memory tiers.

The memory system is a 4-level hierarchy inspired by CPU cache design:

  L1 Registers  — WorkingMemory (in-process dict, sub-ms)
  L2 Cache       — RedisCache (network, ~1ms)
  L3 Main Memory — VectorStore (semantic search, ~10ms)
  L4 Archive     — ReasoningArchive (cold storage, async write)

Each layer implements this interface. The MemoryVault facade routes
read/write operations across layers automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryLayer(ABC):
    """Abstract base class for all memory tiers."""

    @abstractmethod
    async def read(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if not found."""
        ...

    @abstractmethod
    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Persist a value.
        :param key: Storage key
        :param value: Serializable value
        :param ttl: Time-to-live in seconds. None = no expiry.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from this layer."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists without fetching the value."""
        ...

    @abstractmethod
    async def flush(self, prefix: str | None = None) -> int:
        """
        Clear entries. If prefix given, only clear matching keys.
        Returns number of keys removed.
        """
        ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return health metrics for this layer (latency, size, etc.)."""
        ...
