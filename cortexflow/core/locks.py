"""
DistributedLock — Atomic, reentrant distributed locking for CortexFlow.

Problem: When multiple events arrive simultaneously for the same agent,
only ONE CVM tick should execute at a time. Without locking, you get
race conditions where two ticks read the same snapshot and produce
conflicting state updates.

Solution: Redis SET NX EX (atomic acquire) + Lua script (atomic release).
Fallback: asyncio.Lock for in-memory / test environments.

The lock is always acquired by the Scheduler before waking an agent.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

# Lua script: release lock only if caller owns it (atomic compare-and-delete)
_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class LockAcquisitionError(Exception):
    """Raised when a distributed lock cannot be acquired within the timeout."""

    def __init__(self, lock_key: str, timeout: float) -> None:
        super().__init__(f"Could not acquire lock '{lock_key}' within {timeout}s")
        self.lock_key = lock_key
        self.timeout = timeout


class InMemoryLock:
    """
    asyncio.Lock-backed distributed lock for single-process environments.
    Used in tests and local dev when Redis is not available.
    """

    _locks: dict[str, asyncio.Lock] = {}
    _owners: dict[str, str] = {}

    @classmethod
    def _get_lock(cls, key: str) -> asyncio.Lock:
        if key not in cls._locks:
            cls._locks[key] = asyncio.Lock()
        return cls._locks[key]

    @asynccontextmanager
    async def acquire(
        self,
        key: str,
        ttl: int = 30,
        timeout: float = 10.0,
    ) -> AsyncIterator[str]:
        lock = self._get_lock(key)
        token = str(uuid.uuid4())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            self._owners[key] = token
            logger.debug("lock.acquired", key=key, backend="memory")
            yield token
        except asyncio.TimeoutError:
            raise LockAcquisitionError(key, timeout)
        finally:
            if key in self._owners and self._owners[key] == token:
                lock.release()
                del self._owners[key]
                logger.debug("lock.released", key=key, backend="memory")

    @classmethod
    def reset(cls) -> None:
        """Clear all locks. For testing only."""
        cls._locks.clear()
        cls._owners.clear()


class RedisDistributedLock:
    """
    Redis-backed distributed lock using SET NX EX pattern.
    Safe for multi-process and multi-node deployments.
    """

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    @asynccontextmanager
    async def acquire(
        self,
        key: str,
        ttl: int = 30,
        timeout: float = 10.0,
        retry_interval: float = 0.05,
    ) -> AsyncIterator[str]:
        token = str(uuid.uuid4())
        lock_key = f"cortexflow:lock:{key}"
        deadline = time.monotonic() + timeout
        acquired = False

        while time.monotonic() < deadline:
            result = await self._redis.set(  # type: ignore[union-attr]
                lock_key, token, nx=True, ex=ttl
            )
            if result:
                acquired = True
                logger.debug("lock.acquired", key=lock_key, backend="redis")
                break
            await asyncio.sleep(retry_interval)

        if not acquired:
            raise LockAcquisitionError(lock_key, timeout)

        try:
            yield token
        finally:
            await self._redis.eval(_RELEASE_SCRIPT, 1, lock_key, token)  # type: ignore[union-attr]
            logger.debug("lock.released", key=lock_key, backend="redis")


# Default lock backend — replaced by runtime based on config
_default_lock: InMemoryLock | RedisDistributedLock = InMemoryLock()


def get_lock() -> InMemoryLock | RedisDistributedLock:
    """Get the globally configured lock backend."""
    return _default_lock


def configure_lock(lock: InMemoryLock | RedisDistributedLock) -> None:
    """Configure the global lock backend. Called during runtime startup."""
    global _default_lock
    _default_lock = lock
