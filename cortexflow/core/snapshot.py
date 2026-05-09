"""
StateSnapshot — The serializable DNA of every CortexFlow agent.

A StateSnapshot is an immutable, complete representation of an agent's mind
at a precise point in time. It can be:
  - Persisted to SQLite / Postgres for crash recovery
  - Replicated across nodes for distributed execution
  - Diffed to generate reasoning audit trails
  - Serialized for time-travel debugging in the Cognitive Dashboard

Design principle: if you can serialize the mind, you can scale it to 1M instances.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from ulid import ULID


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_ulid() -> str:
    return str(ULID())


class AgentStatus(str, Enum):
    """Lifecycle states of a CortexFlow agent."""

    IDLE = "idle"          # Waiting for events
    RUNNING = "running"    # Inside a CVM tick
    PAUSED = "paused"      # Deliberately suspended
    ERROR = "error"        # Failed, needs intervention
    TERMINATED = "terminated"  # Permanently stopped


class ToolCall(BaseModel):
    """A request by the agent to invoke an external tool."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"frozen": True}


class ReasoningStep(BaseModel):
    """A single step in the agent's chain-of-thought."""

    step_id: str = Field(default_factory=_new_ulid)
    thought: str
    action: str | None = None
    observation: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = {"frozen": True}


class Mailbox(BaseModel):
    """
    The agent's message queue. Events land here and are processed FIFO.
    Bounded to prevent unbounded memory growth.
    """

    MAX_SIZE: int = 1000
    pending: list[str] = Field(default_factory=list)  # event IDs awaiting processing
    processed: list[str] = Field(default_factory=list)  # for idempotency checking

    def has_pending(self) -> bool:
        return len(self.pending) > 0

    def enqueue(self, event_id: str) -> None:
        if len(self.pending) >= self.MAX_SIZE:
            raise OverflowError(
                f"Agent mailbox is full ({self.MAX_SIZE} messages). "
                "The agent may be stuck in a loop or overwhelmed."
            )
        if event_id not in self.processed:
            self.pending.append(event_id)

    def dequeue(self) -> str | None:
        if not self.pending:
            return None
        event_id = self.pending.pop(0)
        self.processed.append(event_id)
        # Trim processed log to avoid unbounded growth
        if len(self.processed) > 10_000:
            self.processed = self.processed[-5_000:]
        return event_id

    model_config = {"frozen": False}


class WorkingMemory(BaseModel):
    """
    L1 Registers — volatile in-process context for the current tick.
    Cleared between major reasoning cycles.
    Acts as the agent's 'scratchpad'.
    """

    MAX_ENTRIES: int = 128
    store: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0  # estimated tokens used

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if len(self.store) >= self.MAX_ENTRIES and key not in self.store:
            # Evict oldest key (FIFO)
            oldest = next(iter(self.store))
            del self.store[oldest]
        self.store[key] = value

    def clear(self) -> None:
        self.store.clear()
        self.token_count = 0

    model_config = {"frozen": False}


class StateSnapshot(BaseModel):
    """
    The complete, serializable state of a CortexFlow agent at a moment in time.

    This is the unit of scale. A running agent IS its StateSnapshot.
    To migrate an agent across machines: serialize this, send it, deserialize it.
    To debug an agent: load a historical snapshot and replay from there.

    Immutability: After a CVM tick, a NEW snapshot is produced.
    The old one is archived. This gives you a complete audit trail.
    """

    # Identity
    agent_id: str
    agent_name: str
    snapshot_id: str = Field(default_factory=_new_ulid)

    # Lifecycle
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    max_iterations: int = 50  # Infinite loop guard

    # Communication
    mailbox: Mailbox = Field(default_factory=Mailbox)

    # Memory (L1 only — L2/L3/L4 are external services)
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)

    # Reasoning audit trail
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    trace_truncated: bool = False
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)


    # Timing
    created_at: datetime = Field(default_factory=_utcnow)
    last_tick_at: datetime | None = None
    last_event_at: datetime | None = None

    # Metadata (arbitrary key-value, user-defined)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Error tracking
    last_error: str | None = None
    consecutive_errors: int = 0

    @field_validator("iteration")
    @classmethod
    def iteration_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Iteration count cannot be negative")
        return v

    def is_looping(self) -> bool:
        """Detect if the agent has exceeded its iteration limit."""
        return self.iteration >= self.max_iterations

    def next_iteration(self) -> "StateSnapshot":
        """Return a new snapshot with incremented iteration counter."""
        return self.model_copy(
            update={
                "iteration": self.iteration + 1,
                "last_tick_at": _utcnow(),
                "snapshot_id": _new_ulid(),
            }
        )

    def with_status(self, status: AgentStatus) -> "StateSnapshot":
        return self.model_copy(update={"status": status, "snapshot_id": _new_ulid()})

    def with_error(self, error: str) -> "StateSnapshot":
        return self.model_copy(
            update={
                "status": AgentStatus.ERROR,
                "last_error": error,
                "consecutive_errors": self.consecutive_errors + 1,
                "snapshot_id": _new_ulid(),
            }
        )

    def append_reasoning_step(self, step: ReasoningStep) -> "StateSnapshot":
        new_trace = self.reasoning_trace + [step]
        truncated = False
        # Cap trace at 200 steps to prevent unbounded growth
        if len(new_trace) > 200:
            new_trace = new_trace[-200:]
            truncated = True
        return self.model_copy(
            update={
                "reasoning_trace": new_trace, 
                "trace_truncated": truncated or self.trace_truncated,
                "snapshot_id": _new_ulid()
            }
        )


    model_config = {"frozen": True}
