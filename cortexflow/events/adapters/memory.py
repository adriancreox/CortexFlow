"""
InMemoryBroker — Zero-infrastructure event broker for local dev and tests.

Uses asyncio queues internally. No Redis, no NATS, nothing external.
This is what makes the CortexFlow 'Hello World' work with zero setup.

Exactly-once guarantee: idempotency_key deduplication via in-memory set.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any

import structlog

from cortexflow.events.broker import EventBroker, EventHandler
from cortexflow.events.schema import CortexEvent, EventType

logger = structlog.get_logger(__name__)


class InMemoryBroker(EventBroker):
    """
    Asyncio-backed event broker. Single-process only.
    Perfect for unit tests, examples, and local development.
    """

    def __init__(self) -> None:
        # pattern/type → list of (subscription_id, handler)
        self._subscriptions: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)
        # For agent-specific routing
        self._agent_queues: dict[str, asyncio.Queue[CortexEvent]] = {}
        # Idempotency: processed event IDs
        self._processed: set[str] = set()
        self._published_count = 0
        self._delivered_count = 0
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._main_queue: asyncio.Queue[CortexEvent] = asyncio.Queue()

    async def start(self) -> None:
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("broker.start", backend="in-memory")

    async def stop(self) -> None:
        self._running = False
        if self._dispatch_task:
            self._dispatch_task.cancel()
        logger.info("broker.stop", backend="in-memory")

    async def publish(self, event: CortexEvent) -> None:
        """Enqueue event for async dispatch."""
        if event.idempotency_key in self._processed:
            logger.debug("broker.dedup", event_id=event.event_id)
            return
        await self._main_queue.put(event)
        self._published_count += 1
        logger.debug(
            "broker.published",
            event_id=event.event_id,
            type=event.type,
            target=event.target_agent,
        )

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._main_queue.get(), timeout=0.1)
                await self._route(event)
                self._main_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("broker.dispatch.error", error=str(e))

    async def _route(self, event: CortexEvent) -> None:
        """Route an event to all matching subscribers."""
        self._processed.add(event.idempotency_key)
        # Cap dedup set to prevent unbounded growth
        if len(self._processed) > 100_000:
            self._processed = set(list(self._processed)[-50_000:])

        matched = False

        # Exact type match
        type_key = str(event.type)
        for sub_id, handler in self._subscriptions.get(type_key, []):
            asyncio.create_task(self._safe_call(handler, event))  # noqa: RUF006
            matched = True

        # Wildcard "*" subscribers (receive all events)
        for sub_id, handler in self._subscriptions.get("*", []):
            asyncio.create_task(self._safe_call(handler, event))  # noqa: RUF006
            matched = True

        # Agent-specific queue routing
        if event.target_agent and event.target_agent in self._agent_queues:
            await self._agent_queues[event.target_agent].put(event)
            matched = True

        if matched:
            self._delivered_count += 1

    async def _safe_call(self, handler: EventHandler, event: CortexEvent) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.error("broker.handler.error", handler=handler.__name__, error=str(e))

    async def subscribe(
        self,
        pattern: str | EventType,
        handler: EventHandler,
        group: str | None = None,
    ) -> str:
        sub_id = str(uuid.uuid4())
        key = str(pattern)
        self._subscriptions[key].append((sub_id, handler))
        logger.debug("broker.subscribe", pattern=key, sub_id=sub_id)
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        for key in self._subscriptions:
            self._subscriptions[key] = [
                (sid, h) for sid, h in self._subscriptions[key]
                if sid != subscription_id
            ]

    async def ack(self, event_id: str) -> None:
        pass  # In-memory broker has no external ACK mechanism

    async def nack(self, event_id: str, reason: str) -> None:
        logger.warning("broker.nack", event_id=event_id, reason=reason)

    def register_agent_queue(self, agent_id: str) -> asyncio.Queue[CortexEvent]:
        """Register a dedicated queue for an agent. Used by the Scheduler."""
        if agent_id not in self._agent_queues:
            self._agent_queues[agent_id] = asyncio.Queue()
        return self._agent_queues[agent_id]

    async def health(self) -> dict[str, Any]:
        return {
            "backend": "in-memory",
            "running": self._running,
            "queue_depth": self._main_queue.qsize(),
            "subscriptions": sum(len(v) for v in self._subscriptions.values()),
            "published": self._published_count,
            "delivered": self._delivered_count,
            "dedup_set_size": len(self._processed),
        }
