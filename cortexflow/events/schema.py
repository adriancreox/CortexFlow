"""
CortexEvent Schema — The standard event language of the CortexFlow mesh.

Every interaction in the system — a user message, a tool result, a timer tick,
a state update — is expressed as a CortexEvent. This uniformity is what makes
the runtime observable, replayable, and exactly-once safe.

Event IDs use ULID (Universally Unique Lexicographically Sortable Identifier)
so they are both globally unique AND time-ordered without a database sequence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_ulid() -> str:
    return str(ULID())


class EventType(str, Enum):
    """
    Standard event vocabulary for CortexFlow.
    Custom events can be defined as strings with the 'CUSTOM:' prefix.
    """

    # Core lifecycle
    AGENT_TICK = "AGENT_TICK"           # Scheduled heartbeat / wake signal
    AGENT_START = "AGENT_START"         # Agent initialized
    AGENT_STOP = "AGENT_STOP"           # Agent gracefully stopped
    AGENT_ERROR = "AGENT_ERROR"         # Unhandled error in tick
    AGENT_LOOP_DETECTED = "AGENT_LOOP_DETECTED"  # Iteration limit hit

    # Communication
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"  # External message (user/API)
    MESSAGE_SENT = "MESSAGE_SENT"          # Agent sent a message

    # Tool execution
    TOOL_CALL = "TOOL_CALL"               # Agent requested a tool
    TOOL_RESULT = "TOOL_RESULT"           # Tool returned a result
    TOOL_ERROR = "TOOL_ERROR"             # Tool execution failed

    # State
    STATE_UPDATE = "STATE_UPDATE"         # Snapshot committed
    STATE_RESTORE = "STATE_RESTORE"       # Snapshot loaded from archive

    # Workflow orchestration
    WORKFLOW_START = "WORKFLOW_START"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"

    # Memory
    MEMORY_SUMMARIZE = "MEMORY_SUMMARIZE"  # Background summarization triggered
    MEMORY_EVICT = "MEMORY_EVICT"          # L1 eviction event

    # Custom
    CUSTOM = "CUSTOM"


class EventPriority(int, Enum):
    """Processing priority. Lower number = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class CortexEvent(BaseModel):
    """
    The fundamental unit of communication in CortexFlow.

    Design guarantees:
    - event_id is ULID: globally unique + time-sorted
    - idempotency_key: safe to deliver the same event multiple times
    - Exactly-once delivery is enforced by the broker using idempotency_key
    """

    # Identity & ordering
    event_id: str = Field(default_factory=_new_ulid)
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Routing
    type: EventType | str
    source_agent: str | None = None       # None = external/system
    target_agent: str | None = None       # None = broadcast
    tool_name: str | None = None          # For tool-related events
    workflow_id: str | None = None

    # Content
    payload: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = Field(default_factory=_utcnow)
    correlation_id: str | None = None     # For tracing chains of events
    retry_count: int = 0
    max_retries: int = 3

    # Delivery tracking
    delivered_at: datetime | None = None
    acked_at: datetime | None = None

    @classmethod
    def tick(cls, agent_id: str, **payload: Any) -> "CortexEvent":
        """Factory: create a standard heartbeat tick for an agent."""
        return cls(
            type=EventType.AGENT_TICK,
            target_agent=agent_id,
            payload=payload,
            priority=EventPriority.NORMAL,
        )

    @classmethod
    def message(cls, target_agent: str, content: str, source: str | None = None) -> "CortexEvent":
        """Factory: create a user/external message event."""
        return cls(
            type=EventType.MESSAGE_RECEIVED,
            source_agent=source,
            target_agent=target_agent,
            payload={"content": content},
            priority=EventPriority.HIGH,
        )

    @classmethod
    def tool_result(
        cls,
        target_agent: str,
        call_id: str,
        tool_name: str,
        result: Any,
        error: str | None = None,
    ) -> "CortexEvent":
        """Factory: create a tool execution result event."""
        return cls(
            type=EventType.TOOL_RESULT if not error else EventType.TOOL_ERROR,
            target_agent=target_agent,
            tool_name=tool_name,
            payload={"call_id": call_id, "result": result, "error": error},
            priority=EventPriority.HIGH,
        )

    @classmethod
    def custom(
        cls,
        name: str,
        target_agent: str | None = None,
        **payload: Any,
    ) -> "CortexEvent":
        """Factory: create a custom domain event."""
        return cls(
            type=f"CUSTOM:{name}",
            target_agent=target_agent,
            payload=payload,
        )

    model_config = {"frozen": True}
