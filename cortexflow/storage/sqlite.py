"""
SQLiteSnapshotStore — Local file-based persistence for CortexFlow.
"""

from __future__ import annotations

import sqlite3
import json
import asyncio
from typing import Optional, List
import structlog

from cortexflow.storage.base import SnapshotStore
from cortexflow.core.snapshot import StateSnapshot

logger = structlog.get_logger(__name__)


class SQLiteSnapshotStore(SnapshotStore):
    """
    Persistent storage using a local SQLite database.
    Optimized for single-process but multi-agent persistence.
    """

    def __init__(self, db_path: str = ".cortexflow/vault/state.db") -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        logger.info("storage.sqlite.init", path=db_path)

    async def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            
            self._conn = await asyncio.to_thread(sqlite3.connect, self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            await self._init_db()
        return self._conn

    async def _init_db(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS snapshots (
            agent_id TEXT PRIMARY KEY,
            agent_name TEXT,
            snapshot_id TEXT,
            data JSON,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        await asyncio.to_thread(self._conn.execute, query)
        await asyncio.to_thread(self._conn.commit)

    async def save(self, snapshot: StateSnapshot) -> None:
        conn = await self._get_conn()
        query = """
        INSERT INTO snapshots (agent_id, agent_name, snapshot_id, data, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(agent_id) DO UPDATE SET
            snapshot_id = excluded.snapshot_id,
            data = excluded.data,
            updated_at = CURRENT_TIMESTAMP;
        """
        data_json = snapshot.model_dump_json()
        await asyncio.to_thread(
            conn.execute, 
            query, 
            (snapshot.agent_id, snapshot.agent_name, snapshot.snapshot_id, data_json)
        )
        await asyncio.to_thread(conn.commit)
        logger.debug("storage.sqlite.saved", agent_id=snapshot.agent_id, snapshot_id=snapshot.snapshot_id)

    async def load(self, agent_id: str) -> Optional[StateSnapshot]:
        conn = await self._get_conn()
        query = "SELECT data FROM snapshots WHERE agent_id = ?"
        cursor = await asyncio.to_thread(conn.execute, query, (agent_id,))
        row = await asyncio.to_thread(cursor.fetchone)
        
        if not row:
            return None
            
        return StateSnapshot.model_validate_json(row["data"])

    async def list_agents(self) -> List[str]:
        conn = await self._get_conn()
        query = "SELECT agent_id FROM snapshots"
        cursor = await asyncio.to_thread(conn.execute, query)
        rows = await asyncio.to_thread(cursor.fetchall)
        return [row["agent_id"] for row in rows]

    async def delete(self, agent_id: str) -> None:
        conn = await self._get_conn()
        query = "DELETE FROM snapshots WHERE agent_id = ?"
        await asyncio.to_thread(conn.execute, query, (agent_id,))
        await asyncio.to_thread(conn.commit)

    async def close(self) -> None:
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
