# Copyright (c) 2026 CortexFlow / Adrian Creox. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Cognitive Scheduler — El Kernel de CortexFlow.
"""


from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from cortexflow.core.cvm import CognitiveVM, LoopDetectedError, TickResult
from cortexflow.core.locks import LockAcquisitionError, get_lock
from cortexflow.core.snapshot import AgentStatus, StateSnapshot
from cortexflow.events.schema import CortexEvent, EventType


if TYPE_CHECKING:
    from cortexflow.providers.base import LLMProvider
    from cortexflow.providers.tools import ToolRegistry
    from cortexflow.events.broker import EventBroker
    from cortexflow.memory.vault import MemoryVault
    from cortexflow.storage.base import SnapshotStore
    from cortexflow.core.shield import Shield

logger = structlog.get_logger(__name__)


@dataclass
class AgentRegistration:
    """ACB (Agent Control Block) - Estado del proceso en el Kernel."""
    agent_id: str
    agent_name: str
    snapshot: StateSnapshot
    provider: "LLMProvider"
    vault: "MemoryVault"
    is_processing: bool = False

    total_ticks: int = 0
    total_tokens: int = 0
    allowed_scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: __import__("time").monotonic())



class Scheduler:
    """
    El Kernel que gestiona el ciclo de vida de los agentes y la ejecución de acciones.
    """

    def __init__(
        self,
        broker: "EventBroker",
        tool_registry: "ToolRegistry",
        cvm: CognitiveVM | None = None,
        storage: "SnapshotStore" | None = None,
        shield: "Shield" | None = None,
        concurrency: int = 16,
    ) -> None:
        self._broker = broker
        self._tool_registry = tool_registry
        self._cvm = cvm or CognitiveVM(tool_registry=tool_registry)
        self._storage = storage
        self._shield = shield
        self._concurrency = concurrency

        self._registry: dict[str, AgentRegistration] = {}
        self._vaults: dict[str, "MemoryVault"] = {}
        self._semaphore = asyncio.Semaphore(concurrency)
        self._running = False
        self._worker_task: asyncio.Task[None] | None = None
        self._dispatch_queue: asyncio.Queue[tuple[str, CortexEvent]] = asyncio.Queue()
        self._ticks_processed = 0


    def register(
        self, 
        snapshot: StateSnapshot, 
        provider: "LLMProvider",
        vault: "MemoryVault",
        allowed_scopes: list[str] | None = None
    ) -> AgentRegistration:

        # Wrap provider with CostGuard if budget exists
        budget = snapshot.metadata.get("token_budget")
        if budget:
            from cortexflow.core.cost import CostGuard
            provider = CostGuard(provider, budget=budget)

        acb = AgentRegistration(
            agent_id=snapshot.agent_id,
            agent_name=snapshot.agent_name,
            snapshot=snapshot,
            provider=provider,
            vault=vault,
            allowed_scopes=allowed_scopes or [],
        )


        self._registry[snapshot.agent_id] = acb
        logger.info("scheduler.register", agent_id=snapshot.agent_id)
        return acb

    def unregister(self, agent_id: str) -> None:
        """Elimina un agente del registro."""
        self._registry.pop(agent_id, None)
        logger.info("scheduler.unregister", agent_id=agent_id)

    def get_acb(self, agent_id: str) -> AgentRegistration | None:
        """Obtiene el ACB de un agente."""
        return self._registry.get(agent_id)

    async def dispatch(self, event: CortexEvent) -> None:
        """Encola eventos para su procesamiento por el worker loop."""
        target = event.target_agent
        if target and target in self._registry:
            await self._dispatch_queue.put((target, event))
        elif not target:
            for agent_id in list(self._registry.keys()):
                await self._dispatch_queue.put((agent_id, event))

    async def start(self) -> None:
        self._running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info("scheduler.start", concurrency=self._concurrency)

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
        logger.info("scheduler.stop", total_ticks=self._ticks_processed)

    async def _process_loop(self) -> None:
        while self._running:
            try:
                agent_id, event = await asyncio.wait_for(
                    self._dispatch_queue.get(), timeout=0.1
                )
                asyncio.create_task(self._wake(agent_id, event))
                self._dispatch_queue.task_done()
            except asyncio.TimeoutError:
                continue

    async def _wake(self, agent_id: str, event: CortexEvent) -> None:
        """Ciclo vital: Lock -> CVM -> Tool Runner -> Commit State."""
        async with self._semaphore:
            lock = get_lock()
            try:
                # TTL de 30s para evitar bloqueos huerfanos si el proceso cae
                async with lock.acquire(key=f"agent:{agent_id}", ttl=30):
                    acb = self._registry.get(agent_id)
                    if not acb or acb.is_processing:
                        return

                    acb.is_processing = True
                    try:
                        # 1. EJECUCIÓN COGNITIVA
                        result: TickResult = await self._cvm.tick(
                            snapshot=acb.snapshot,
                            event=event,
                            provider=acb.provider,
                        )
                        
                        # 2. COMMIT DEL NUEVO ESTADO
                        acb.snapshot = result.snapshot
                        acb.total_ticks += 1
                        acb.total_tokens += result.tokens_used
                        self._ticks_processed += 1

                        # 3. ARCHIVE REASONING (L4)
                        asyncio.create_task(
                            acb.vault.archive_reasoning(
                                snapshot_id=result.snapshot.snapshot_id,
                                reasoning=result.reasoning_step
                            )
                        )

                        # 4. ACKNOWLEDGE EVENT
                        await self._broker.ack(event.event_id)

                        # 5. PERSIST STATE (Inmortality)
                        if self._storage:
                            await self._storage.save(acb.snapshot)



                        # 6. SHIELD GOVERNANCE
                        tool_calls = result.tool_calls_to_dispatch
                        if self._shield and tool_calls:
                            tool_calls = await self._shield.authorize(
                                agent_id=agent_id,
                                tool_calls=tool_calls,
                                context={"snapshot": acb.snapshot}
                            )

                        # 7. TOOL RUNNER
                        if tool_calls:
                            for tool_call in tool_calls:
                                try:
                                    output = await self._tool_registry.call(
                                        tool_name=tool_call.tool_name, 
                                        arguments=tool_call.arguments,
                                        caller_scopes=acb.allowed_scopes
                                    )

                                    res_event = CortexEvent.tool_result(
                                        target_agent=agent_id,
                                        call_id=tool_call.call_id,
                                        tool_name=tool_call.tool_name, 
                                        result=output
                                    )
                                    await self._broker.publish(res_event)
                                    
                                except Exception as e:
                                    logger.error("scheduler.tool_exec.failed", tool=tool_call.tool_name, error=str(e))
                                    err_event = CortexEvent.tool_result(
                                        target_agent=agent_id,
                                        call_id=tool_call.call_id,
                                        tool_name=tool_call.tool_name,
                                        result=None,
                                        error=str(e)
                                    )
                                    await self._broker.publish(err_event)


                    except LoopDetectedError as e:
                        acb.snapshot = acb.snapshot.with_error(str(e)).with_status(AgentStatus.PAUSED)
                    except Exception as e:
                        logger.error("scheduler.tick.error", agent=agent_id, error=str(e))
                        # NACK: Tell the broker something went wrong
                        await self._broker.nack(event.event_id, str(e))
                        acb.snapshot = acb.snapshot.with_error(str(e))
                    finally:
                        acb.is_processing = False


            except LockAcquisitionError:
                # Si el agente está bloqueado por otro worker, re-intentamos más tarde
                await asyncio.sleep(0.5)
                await self._dispatch_queue.put((agent_id, event))

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "agents_registered": len(self._registry),
            "ticks_processed": self._ticks_processed,
            "concurrency_limit": self._concurrency,
            "dispatch_queue_depth": self._dispatch_queue.qsize(),
            "agents": {
                agent_id: {
                    "status": acb.snapshot.status,
                    "ticks": acb.total_ticks,
                    "tokens": acb.total_tokens,
                    "is_processing": acb.is_processing,
                    "iteration": acb.snapshot.iteration,
                }
                for agent_id, acb in self._registry.items()
            },
        }
