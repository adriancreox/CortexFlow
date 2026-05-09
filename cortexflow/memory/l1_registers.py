"""
L1 Registers — Working Memory (Volatile, In-Process)

The fastest memory tier. Sub-millisecond access.
Lives only for the duration of an agent's active execution context.
Cleared when the agent returns to IDLE or when summarization runs.

Think of it as the agent's CPU registers — the values it's actively
working with right now.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import structlog

from cortexflow.memory.base import MemoryLayer

logger = structlog.get_logger(__name__)


class L1WorkingMemory(MemoryLayer):
    """
    In-process dictionary with LRU eviction and size cap.
    Thread-safe for single async event loop use.
    """

    def __init__(self, max_entries: int = 128) -> None:
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    async def read(self, key: str) -> Any | None:
        if key not in self._store:
            self._misses += 1
            return None

        value, expires_at = self._store[key]

        # Check TTL expiry
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            self._misses += 1
            return None

        # Move to end (LRU: most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return value

    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = time.monotonic() + ttl if ttl is not None else None

        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self._max_entries:
            # Evict LRU (first item)
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("l1.evict", key=evicted_key, size=len(self._store))

        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        if key not in self._store:
            return False
        _, expires_at = self._store[key]
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            return False
        return True

    async def flush(self, prefix: str | None = None) -> int:
        if prefix is None:
            count = len(self._store)
            self._store.clear()
            return count
        keys_to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._store[k]
        return len(keys_to_delete)

    async def health(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "layer": "L1_registers",
            "size": len(self._store),
            "max_size": self._max_entries,
            "hit_rate": round(hit_rate, 3),
            "hits": self._hits,
            "misses": self._misses,
        }

    def get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous read for use inside the CVM hot path (no await overhead)."""
        if key not in self._store:
            return default
        value, expires_at = self._store[key]
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            return default
        return value

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous write for use inside the CVM hot path."""
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self._max_entries:
            self._store.popitem(last=False)
        self._store[key] = (value, None)
