"""
L2 Cache — Redis-backed Short-Term Memory

Network-accessible, shared across processes and machines. ~1ms latency.
Graceful degradation: falls back silently if Redis is unreachable.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from cortexflow.memory.base import MemoryLayer

logger = structlog.get_logger(__name__)


class L2RedisCache(MemoryLayer):
    """Redis cache with JSON serialization and namespace isolation."""

    DEFAULT_TTL = 3600  # 1 hour

    def __init__(self, redis_client: Any, namespace: str = "default") -> None:
        self._redis = redis_client
        self._namespace = namespace
        self._available = True
        self._hits = 0
        self._misses = 0
        self._errors = 0

    def _key(self, key: str) -> str:
        return f"cf:l2:{self._namespace}:{key}"

    async def read(self, key: str) -> Any | None:
        if not self._available:
            return None
        try:
            raw = await self._redis.get(self._key(key))
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(raw)
        except Exception as e:
            self._errors += 1
            self._available = False
            logger.warning("l2.read.error", key=key, error=str(e))
            return None

    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._available:
            return
        try:
            serialized = json.dumps(value, default=str)
            effective_ttl = ttl if ttl is not None else self.DEFAULT_TTL
            await self._redis.setex(self._key(key), effective_ttl, serialized)
        except Exception as e:
            self._errors += 1
            self._available = False
            logger.warning("l2.write.error", key=key, error=str(e))

    async def delete(self, key: str) -> None:
        if not self._available:
            return
        try:
            await self._redis.delete(self._key(key))
        except Exception as e:
            logger.warning("l2.delete.error", key=key, error=str(e))

    async def exists(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            return bool(await self._redis.exists(self._key(key)))
        except Exception:
            return False

    async def flush(self, prefix: str | None = None) -> int:
        if not self._available:
            return 0
        try:
            pattern = self._key(f"{prefix}*" if prefix else "*")
            keys = await self._redis.keys(pattern)
            if not keys:
                return 0
            return await self._redis.delete(*keys)
        except Exception as e:
            logger.warning("l2.flush.error", error=str(e))
            return 0

    async def health(self) -> dict[str, Any]:
        latency_ms = None
        try:
            start = time.perf_counter()
            await self._redis.ping()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            self._available = True
        except Exception:
            self._available = False

        total = self._hits + self._misses
        return {
            "layer": "L2_redis_cache",
            "available": self._available,
            "latency_ms": latency_ms,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "namespace": self._namespace,
        }

    async def ping(self) -> bool:
        try:
            await self._redis.ping()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False
