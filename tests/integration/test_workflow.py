"""
Integration test — Full end-to-end workflow using InMemoryBroker.

No Redis, no API keys, no external services.
Tests the complete stack: Runtime → Scheduler → CVM → Memory
"""

from __future__ import annotations

import asyncio

import pytest

from cortexflow import CortexRuntime, defineAgent
from cortexflow.core.snapshot import AgentStatus
from cortexflow.events.schema import CortexEvent
from cortexflow.sdk.testing import MockProvider


@pytest.fixture
async def runtime() -> CortexRuntime:
    rt = CortexRuntime()
    await rt.start()
    yield rt
    await rt.stop()


@pytest.mark.asyncio
async def test_spawn_and_send_message(runtime: CortexRuntime) -> None:
    provider = MockProvider(responses=["Hello back!"])
    agent_def = defineAgent(
        name="integration-agent",
        instructions="You are a helpful agent.",
        provider=provider,
    )
    agent_id = await runtime.spawn(agent_def)
    assert agent_id.startswith("integration-agent-")

    await runtime.send(agent_id, "Hello!")
    await asyncio.sleep(0.3)

    snapshot = runtime.get_snapshot(agent_id)
    assert snapshot is not None
    assert snapshot.iteration == 1


@pytest.mark.asyncio
async def test_multiple_agents_isolated(runtime: CortexRuntime) -> None:
    provider_a = MockProvider(responses=["Response from A"])
    provider_b = MockProvider(responses=["Response from B"])

    agent_a = defineAgent(name="agent-a", instructions="Agent A", provider=provider_a)
    agent_b = defineAgent(name="agent-b", instructions="Agent B", provider=provider_b)

    id_a = await runtime.spawn(agent_a)
    id_b = await runtime.spawn(agent_b)

    await runtime.send(id_a, "Message to A")
    await runtime.send(id_b, "Message to B")
    await asyncio.sleep(0.5)

    snap_a = runtime.get_snapshot(id_a)
    snap_b = runtime.get_snapshot(id_b)

    assert snap_a is not None and snap_b is not None
    assert snap_a.agent_id != snap_b.agent_id
    assert snap_a.iteration == 1
    assert snap_b.iteration == 1


@pytest.mark.asyncio
async def test_agent_accumulates_iterations(runtime: CortexRuntime) -> None:
    provider = MockProvider(responses=["R1", "R2", "R3"])
    agent_def = defineAgent(name="multi-tick", instructions="Keep going", provider=provider)
    agent_id = await runtime.spawn(agent_def)

    for i in range(3):
        await runtime.send(agent_id, f"Message {i}")
        await asyncio.sleep(0.2)

    snapshot = runtime.get_snapshot(agent_id)
    assert snapshot is not None
    assert snapshot.iteration == 3


@pytest.mark.asyncio
async def test_runtime_stats_accurate(runtime: CortexRuntime) -> None:
    provider = MockProvider(responses=["Response"])
    agent_def = defineAgent(name="stats-agent", instructions="Test", provider=provider)
    agent_id = await runtime.spawn(agent_def)

    stats_before = runtime.stats()
    assert stats_before["runtime"]["agents"] == 1

    await runtime.send(agent_id, "Hello")
    await asyncio.sleep(0.3)

    stats_after = runtime.stats()
    agent_stats = stats_after["scheduler"]["agents"].get(agent_id)
    assert agent_stats is not None
    assert agent_stats["ticks"] >= 1


@pytest.mark.asyncio
async def test_kill_removes_agent(runtime: CortexRuntime) -> None:
    provider = MockProvider()
    agent_def = defineAgent(name="kill-me", instructions="Test", provider=provider)
    agent_id = await runtime.spawn(agent_def)

    await runtime.kill(agent_id)

    snapshot = runtime.get_snapshot(agent_id)
    assert snapshot is None


@pytest.mark.asyncio
async def test_runtime_context_manager() -> None:
    provider = MockProvider(responses=["OK"])
    agent_def = defineAgent(name="ctx-test", instructions="Test", provider=provider)

    async with CortexRuntime() as rt:
        agent_id = await rt.spawn(agent_def)
        await rt.send(agent_id, "Hello")
        await asyncio.sleep(0.3)
        snap = rt.get_snapshot(agent_id)
        assert snap is not None
