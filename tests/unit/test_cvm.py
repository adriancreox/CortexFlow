"""
Unit tests for CognitiveVM — the core execution loop.
All tests use MockProvider — no API key required.
"""

from __future__ import annotations

import pytest

from cortexflow.core.cvm import CognitiveVM, LoopDetectedError
from cortexflow.core.snapshot import AgentStatus, StateSnapshot
from cortexflow.events.schema import CortexEvent
from cortexflow.providers.base import ToolCall
from cortexflow.sdk.testing import MockProvider


@pytest.fixture
def cvm() -> CognitiveVM:
    return CognitiveVM()


@pytest.fixture
def snapshot() -> StateSnapshot:
    return StateSnapshot(
        agent_id="cvm-test-001",
        agent_name="test",
        metadata={"system_prompt": "You are a test agent."},
    )


@pytest.mark.asyncio
async def test_tick_produces_new_snapshot(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(responses=["Hello, world!"])
    event = CortexEvent.message("cvm-test-001", "Hi!")
    result = await cvm.tick(snapshot, event, provider)
    assert result.success
    assert result.snapshot.iteration == 1
    assert result.snapshot.snapshot_id != snapshot.snapshot_id


@pytest.mark.asyncio
async def test_tick_increments_iteration(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(responses=["Response"])
    event = CortexEvent.message("cvm-test-001", "Hello")
    result = await cvm.tick(snapshot, event, provider)
    assert result.snapshot.iteration == 1


@pytest.mark.asyncio
async def test_tick_records_reasoning_step(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(responses=["I am thinking..."])
    event = CortexEvent.message("cvm-test-001", "Think!")
    result = await cvm.tick(snapshot, event, provider)
    assert len(result.snapshot.reasoning_trace) == 1
    assert result.snapshot.reasoning_trace[0].thought == "I am thinking..."


@pytest.mark.asyncio
async def test_tick_with_tool_calls(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    tool_call = ToolCall(call_id="tc-001", tool_name="search_web", arguments={"query": "AI"})
    provider = MockProvider(tool_calls=[[tool_call]])
    event = CortexEvent.message("cvm-test-001", "Search for AI news")
    result = await cvm.tick(snapshot, event, provider)
    assert result.success
    assert len(result.tool_calls_to_dispatch) == 1
    assert result.tool_calls_to_dispatch[0].tool_name == "search_web"
    assert result.snapshot.status == AgentStatus.RUNNING


@pytest.mark.asyncio
async def test_loop_detection_raises(cvm: CognitiveVM) -> None:
    looping_snapshot = StateSnapshot(
        agent_id="loop-agent",
        agent_name="looper",
        iteration=50,  # Already at max
        max_iterations=50,
    )
    provider = MockProvider()
    event = CortexEvent.tick("loop-agent")
    with pytest.raises(LoopDetectedError):
        await cvm.tick(looping_snapshot, event, provider)


@pytest.mark.asyncio
async def test_provider_error_sets_error_status(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(fail_after=0)
    event = CortexEvent.message("cvm-test-001", "Hello")
    result = await cvm.tick(snapshot, event, provider)
    assert not result.success
    assert result.snapshot.status == AgentStatus.ERROR
    assert result.snapshot.last_error is not None


@pytest.mark.asyncio
async def test_tick_stores_response_in_working_memory(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(responses=["The answer is 42"])
    event = CortexEvent.message("cvm-test-001", "What is the answer?")
    result = await cvm.tick(snapshot, event, provider)
    last = result.snapshot.working_memory.get("last_response")
    assert last == "The answer is 42"


@pytest.mark.asyncio
async def test_tick_is_deterministic_with_same_input(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider1 = MockProvider(responses=["Answer A"])
    provider2 = MockProvider(responses=["Answer A"])
    event = CortexEvent.message("cvm-test-001", "Same question")
    result1 = await cvm.tick(snapshot, event, provider1)
    result2 = await cvm.tick(snapshot, event, provider2)
    r1 = result1.snapshot.working_memory.get("last_response")
    r2 = result2.snapshot.working_memory.get("last_response")
    assert r1 == r2


@pytest.mark.asyncio
async def test_multiple_ticks_chain_correctly(cvm: CognitiveVM, snapshot: StateSnapshot) -> None:
    provider = MockProvider(responses=["Tick 1", "Tick 2", "Tick 3"])
    current = snapshot
    for i in range(3):
        event = CortexEvent.message("cvm-test-001", f"Message {i}")
        result = await cvm.tick(current, event, provider)
        current = result.snapshot
    assert current.iteration == 3
    assert len(current.reasoning_trace) == 3
