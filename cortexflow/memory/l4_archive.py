"""
L4 Archive — Immutable Reasoning Audit Log.

The "Black Box" of CortexFlow. Every decision, every tool call, 
and every event is archived here for compliance and Cortex-Pulse observability.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import structlog

from cortexflow.memory.base import MemoryLayer

logger = structlog.get_logger(__name__)


class L4Archive(MemoryLayer):
    """
    Partitioned, non-blocking audit archive.
    """

    def __init__(self, base_dir: str = ".cortexflow/vault/archive", retention_days: int = 30) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        self._running = False
        self._worker_task: asyncio.Task[None] | None = None
        logger.info("memory.l4.init", storage="partitioned-jsonl")

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._flush_worker())

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            # Wait for queue to drain
            while not self._queue.empty():
                await asyncio.sleep(0.1)
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _flush_worker(self) -> None:
        """Processes the archive queue in the background and runs cleanup."""
        # Initial cleanup
        await self._cleanup_old_logs()
        
        while self._running:

            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._write_record(record)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("memory.l4.flush_failed", error=str(e))

    async def _write_record(self, record: dict[str, Any]) -> None:
        """Writes record to the daily partitioned file."""
        # Partitioning by date: YYYY-MM-DD
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = self._base_dir / f"audit-{day}.jsonl"
        
        line = json.dumps(record, default=str) + "\n"
        
        # Use thread pool for blocking I/O
        await asyncio.to_thread(self._append_to_file, file_path, line)

    def _append_to_file(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Archives a reasoning step or event.
        'key' is typically the agent_id or event_id.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subject": key,
            "data": value,
            "version": "1.0"
        }
        await self._queue.put(record)

    async def read(self, key: str) -> Any | None:
        """L4 is write-optimized. Use Cortex-Pulse for reading."""
        return None

    async def delete(self, key: str) -> None:
        """Immutability enforcement."""
        logger.warning("memory.l4.delete_denied", key=key, reason="Archive is immutable")

    async def exists(self, key: str) -> bool:
        return False

    async def flush(self, prefix: str | None = None) -> int:
        """Force-flush the write queue (checkpoint)."""
        await self.checkpoint()
        return 0

    async def checkpoint(self) -> None:
        """Ensures all currently queued records are written to disk."""
        await self._queue.join()
        logger.info("memory.l4.checkpoint_reached")


    async def _cleanup_old_logs(self) -> None:
        """Deletes log files older than retention_days."""
        try:
            now = datetime.now(timezone.utc).timestamp()
            max_age_seconds = self._retention_days * 86400
            
            for file in self._base_dir.glob("audit-*.jsonl"):
                file_age = now - file.stat().st_mtime
                if file_age > max_age_seconds:
                    logger.info("memory.l4.retention_cleanup", file=file.name, age_days=int(file_age / 86400))
                    file.unlink()
        except Exception as e:
            logger.error("memory.l4.cleanup_failed", error=str(e))

    async def health(self) -> dict[str, Any]:

        return {
            "layer": "L4_archive",
            "storage": "partitioned_jsonl",
            "queue_depth": self._queue.qsize(),
            "directory": str(self._base_dir)
        }
