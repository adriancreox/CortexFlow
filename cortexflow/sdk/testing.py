"""
MockProvider — LLM mock for testing and CI/CD.
AgentTestHarness — Full agent test environment.

These tools make it possible to test CortexFlow agents without:
  - An internet connection
  - An API key
  - Any external infrastructure

Usage in tests:
    from cortexflow.sdk.testing import MockProvider, AgentTestHarness

    mock = MockProvider(responses=["I found the answer: 42"])
    harness = AgentTestHarness(agent_def, provider=mock)

    result = await harness.send("What is 6 times 7?")
    assert "42" in result.last_response
    assert result.snapshot.status == AgentStatus.IDLE
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cortexflow.core.cvm import CognitiveVM, TickResult
from cortexflow.core.scheduler import Scheduler
from cortexflow.core.snapshot import AgentStatus, StateSnapshot
from cortexflow.events.adapters.memory import InMemoryBroker
from cortexflow.events.schema import CortexEvent
from cortexflow.providers.base import (
    Completion,
    CompletionRequest,
    LLMProvider,
    ToolCall,
)
from cortexflow.sdk.agent import AgentDefinition


class MockProvider(LLMProvider):
    """
    Deterministic LLM mock that cycles through predefined responses.
    Records all requests for assertion in tests.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        tool_calls: list[list[ToolCall]] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._responses = responses or ["Mock response."]
        self._tool_calls_sequence = tool_calls or []
        self._call_index = 0
        self._fail_after = fail_after
        self.recorded_requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Completion:
        self.recorded_requests.append(request)

        if self._fail_after is not None and self._call_index >= self._fail_after:
            from cortexflow.providers.base import ProviderError
            raise ProviderError("Mock failure triggered", provider="mock")

        idx = self._call_index % len(self._responses)
        response_text = self._responses[idx]

        tc: list[ToolCall] = []
        if self._tool_calls_sequence and self._call_index < len(self._tool_calls_sequence):
            tc = self._tool_calls_sequence[self._call_index]

        self._call_index += 1

        return Completion(
            content=response_text if not tc else None,
            tool_calls=tc,
            input_tokens=len(" ".join(m.content for m in request.messages)) // 4,
            output_tokens=len(response_text) // 4,
            model="mock-v1",
            stop_reason="tool_use" if tc else "stop",
        )

    async def health(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock-v1"

    def reset(self) -> None:
        self._call_index = 0
        self.recorded_requests.clear()

    @property
    def call_count(self) -> int:
        return self._call_index


@dataclass
class HarnessResult:
    """The result of a test harness interaction."""
    snapshot: StateSnapshot
    tick_result: TickResult
    last_response: str | None
    events_emitted: list[CortexEvent] = field(default_factory=list)


class AgentTestHarness:
    """
    A fully isolated test environment for a single CortexFlow agent.
    No external dependencies required — everything runs in-process.

    Usage:
        harness = AgentTestHarness(my_agent_def)
        result = await harness.send("Hello!")
        assert result.last_response is not None
        assert result.snapshot.iteration == 1
    """

    def __init__(
        self,
        definition: AgentDefinition,
        provider: LLMProvider | None = None,
    ) -> None:
        self._definition = definition
        self._provider = provider or MockProvider()
        self._cvm = CognitiveVM()
        self._emitted: list[CortexEvent] = []

        import uuid
        self._agent_id = f"test-{definition.name}-{str(uuid.uuid4())[:6]}"
        self._snapshot = StateSnapshot(
            agent_id=self._agent_id,
            agent_name=definition.name,
            max_iterations=definition.max_iterations,
            metadata={"system_prompt": definition.instructions},
        )

    async def send(self, message: str) -> HarnessResult:
        """Send a message and run one tick. Returns the result."""
        event = CortexEvent.message(
            target_agent=self._agent_id,
            content=message,
        )
        return await self._tick(event)

    async def tick(self, event: CortexEvent) -> HarnessResult:
        """Run a tick with a custom event."""
        return await self._tick(event)

    async def _tick(self, event: CortexEvent) -> HarnessResult:
        result = await self._cvm.tick(
            snapshot=self._snapshot,
            event=event,
            provider=self._provider,
        )
        self._snapshot = result.snapshot
        last_response = result.completion.content if result.completion else None
        return HarnessResult(
            snapshot=self._snapshot,
            tick_result=result,
            last_response=last_response,
            events_emitted=self._emitted.copy(),
        )

    def assert_status(self, expected: AgentStatus) -> None:
        assert self._snapshot.status == expected, (
            f"Expected status {expected}, got {self._snapshot.status}"
        )

    def assert_no_error(self) -> None:
        assert self._snapshot.last_error is None, (
            f"Agent has error: {self._snapshot.last_error}"
        )

    def assert_iteration(self, n: int) -> None:
        assert self._snapshot.iteration == n, (
            f"Expected iteration {n}, got {self._snapshot.iteration}"
        )

    @property
    def snapshot(self) -> StateSnapshot:
        return self._snapshot
