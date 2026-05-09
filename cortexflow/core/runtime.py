"""
CortexRuntime — The top-level orchestrator.

Wires together: Scheduler + CVM + MemoryVault + EventBroker.
Entry point for all CortexFlow applications.

Usage:
    runtime = CortexRuntime()
    runtime.use_broker(InMemoryBroker())
    await runtime.start()
    await runtime.run_agent(my_agent)
    await runtime.emit(CortexEvent.message("agent-1", "Hello!"))
    await runtime.stop()
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog

from cortexflow.core.cvm import CognitiveVM
from cortexflow.core.locks import InMemoryLock, RedisDistributedLock, configure_lock
from cortexflow.core.scheduler import Scheduler
from cortexflow.core.snapshot import AgentStatus, StateSnapshot
from cortexflow.events.adapters.memory import InMemoryBroker
from cortexflow.events.broker import EventBroker
from cortexflow.events.schema import CortexEvent, EventType
from cortexflow.memory.l4_archive import L4Archive
from cortexflow.memory.vault import MemoryVault
from cortexflow.providers.base import LLMProvider
from cortexflow.providers.tools import ToolRegistry

logger = structlog.get_logger(__name__)


class CortexRuntime:
    """
    The CortexFlow runtime. One instance per application.

    Configuration is done via method chaining before calling start().
    After start(), the runtime is immutable — reconfigure and restart.
    """

    def __init__(self, concurrency: int = 16) -> None:
        self._tool_registry = ToolRegistry()
        self._cvm = CognitiveVM(tool_registry=self._tool_registry)
        self._broker: EventBroker = InMemoryBroker()
        self._scheduler = Scheduler(
            broker=self._broker,
            tool_registry=self._tool_registry,
            cvm=self._cvm,
            concurrency=concurrency
        )
        self._vaults: dict[str, MemoryVault] = {}
        self._archive = L4Archive()
        self._storage: SnapshotStore | None = None
        self._shield: Shield | None = None
        self._started = False




    # ── Configuration ────────────────────────────────────────────────────────

    def use_broker(self, broker: EventBroker) -> "CortexRuntime":
        """Set the event broker. Must be called before start()."""
        if self._started:
            raise RuntimeError("Cannot reconfigure a running runtime. Call stop() first.")
        self._broker = broker
        return self

    def use_redis(self, redis_client: Any) -> "CortexRuntime":
        """Configure Redis for L2 cache and distributed locks."""
        from cortexflow.core.locks import RedisDistributedLock
        configure_lock(RedisDistributedLock(redis_client))
        return self

    def use_storage(self, storage: SnapshotStore) -> "CortexRuntime":
        """Set the snapshot storage backend for agent persistence."""
        self._storage = storage
        self._scheduler._storage = storage
        return self

    def use_shield(self, shield: Shield) -> "CortexRuntime":
        """Set the governance shield to enforce policies."""
        self._shield = shield
        self._scheduler._shield = shield
        return self



    def tools(self) -> ToolRegistry:
        """Access the tool registry to register tools."""
        return self._tool_registry

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> "CortexRuntime":
        """Start the runtime. Must be awaited before any agent is run."""
        await self._broker.start()
        await self._archive.start()
        await self._scheduler.start()


        # Wire broker events to scheduler
        await self._broker.subscribe("*", self._on_event)

        self._started = True
        logger.info(
            "runtime.start",
            broker=type(self._broker).__name__,
        )
        return self

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        await self._scheduler.stop()
        await self._archive.stop()
        await self._broker.stop()

        # Flush L4 archives
        for vault in self._vaults.values():
            if vault._l4:
                await vault._l4.stop()
        self._started = False
        logger.info("runtime.stop")

    async def __aenter__(self) -> "CortexRuntime":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── Agent Management ─────────────────────────────────────────────────────

    async def spawn(
        self,
        definition: Any,  # AgentDefinition from sdk.agent
        agent_id: str | None = None,
    ) -> str:
        """
        Spawn a new agent instance from a definition.
        Returns the agent_id of the new instance.
        """
        if not self._started:
            raise RuntimeError("Runtime not started. Call await runtime.start() first.")

        aid = agent_id or f"{definition.name}-{str(uuid.uuid4())[:8]}"

        # Try to resume from storage
        snapshot = None
        if self._storage:
            snapshot = await self._storage.load(aid)
            if snapshot:
                logger.info("runtime.resume", agent_id=aid, snapshot_id=snapshot.snapshot_id)

        if not snapshot:
            snapshot = StateSnapshot(
                agent_id=aid,
                agent_name=definition.name,
                max_iterations=definition.max_iterations,
                metadata={
                    "system_prompt": definition.instructions,
                    "token_budget": definition.token_budget,
                    "max_history_steps": definition.max_history_steps,
                },
            )


        vault = MemoryVault(agent_id=aid, l4=self._archive)
        self._vaults[aid] = vault


        self._scheduler.register(snapshot, definition.provider, vault, definition.allowed_scopes)
        logger.info("runtime.spawn", agent_id=aid, name=definition.name)
        return aid

    async def kill(self, agent_id: str) -> None:
        """Terminate an agent and free its resources."""
        self._scheduler.unregister(agent_id)
        self._vaults.pop(agent_id, None)
        logger.info("runtime.kill", agent_id=agent_id)

    def get_snapshot(self, agent_id: str) -> StateSnapshot | None:
        """Get the current state snapshot of an agent (for the Cortex-Pulse dashboard)."""
        acb = self._scheduler.get_acb(agent_id)
        return acb.snapshot if acb else None

    # ── Communication ────────────────────────────────────────────────────────

    async def emit(self, event: CortexEvent) -> None:
        """Emit an event into the mesh."""
        await self._broker.publish(event)

    async def send(self, agent_id: str, message: str) -> None:
        """Convenience: send a text message to a specific agent."""
        event = CortexEvent.message(target_agent=agent_id, content=message)
        await self.emit(event)

    async def _on_event(self, event: CortexEvent) -> None:
        """Internal: route broker events to the scheduler."""
        await self._scheduler.dispatch(event)

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return runtime-wide metrics for the Cortex-Pulse dashboard."""
        return {
            "runtime": {
                "started": self._started,
                "agents": len(self._vaults),
            },
            "scheduler": self._scheduler.stats(),
        }

    async def health(self) -> dict[str, Any]:
        broker_health = await self._broker.health()
        vault_health = {}
        for aid, vault in self._vaults.items():
            vault_health[aid] = await vault.health()
        return {
            "broker": broker_health,
            "vaults": vault_health,
            "scheduler": self._scheduler.stats(),
        }
