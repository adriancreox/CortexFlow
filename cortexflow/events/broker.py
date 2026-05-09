"""
EventBroker — Abstract interface for the CortexFlow Event Mesh.

The broker is the nervous system of the runtime. All inter-agent
communication and state transitions flow through it.

Design guarantees provided by all implementations:
  - Exactly-once delivery via idempotency_key deduplication
  - Ordered delivery within a single agent's mailbox
  - Non-blocking publish (fire and forget with ACK tracking)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from cortexflow.events.schema import CortexEvent, EventType

# Type alias for async event handler functions
EventHandler = Callable[[CortexEvent], Coroutine[Any, Any, None]]


class EventBroker(ABC):
    """Abstract base class for all event broker implementations."""

    @abstractmethod
    async def publish(self, event: CortexEvent) -> None:
        """
        Publish an event to the mesh.
        Returns immediately. Delivery is asynchronous.
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        pattern: str | EventType,
        handler: EventHandler,
        group: str | None = None,
    ) -> str:
        """
        Subscribe to events matching a pattern or exact type.
        Returns a subscription ID for later unsubscription.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by ID."""
        ...

    @abstractmethod
    async def ack(self, event_id: str) -> None:
        """Acknowledge successful processing of an event."""
        ...

    @abstractmethod
    async def nack(self, event_id: str, reason: str) -> None:
        """Negative-acknowledge an event (will be retried if under max_retries)."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Initialize the broker and start listening."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the broker."""
        ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return broker health metrics."""
        ...
