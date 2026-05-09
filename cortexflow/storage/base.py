"""
SnapshotStore — Persistence layer for CortexFlow Agent states.

Allows agents to survive process restarts by storing their StateSnapshots
in a persistent database (SQLite, Postgres, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List

from cortexflow.core.snapshot import StateSnapshot


class SnapshotStore(ABC):
    """
    Abstract interface for snapshot persistence.
    """

    @abstractmethod
    async def save(self, snapshot: StateSnapshot) -> None:
        """Persist a snapshot to storage."""
        ...

    @abstractmethod
    async def load(self, agent_id: str) -> Optional[StateSnapshot]:
        """Load the latest snapshot for an agent."""
        ...

    @abstractmethod
    async def list_agents(self) -> List[str]:
        """Return a list of all agent IDs in storage."""
        ...

    @abstractmethod
    async def delete(self, agent_id: str) -> None:
        """Remove all data for an agent."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""
        ...
