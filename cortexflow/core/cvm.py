"""
CognitiveVM (CVM) — The execution loop of CortexFlow.

The CVM implements the core cognitive cycle:
    Input → Reason → Action → Observe → Commit

This is the "CPU" of an agent. It takes an immutable StateSnapshot,
processes one event, and produces a NEW StateSnapshot. It never mutates
in place — this is what enables time-travel debugging and distributed
state replication.

Design principles:
  - Pure function core: tick(snapshot, event, provider) -> TickResult
  - No I/O inside the reasoning loop (tool I/O is async and external)
  - Infinite loop detection via iteration counter + configurable backoff
  - Every reasoning step is recorded for audit trail
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from cortexflow.core.snapshot import (
    AgentStatus,
    ReasoningStep,
    StateSnapshot,
    ToolCall,
)
from cortexflow.events.schema import CortexEvent, EventType
from cortexflow.providers.base import (
    BudgetExceededError,
    Completion,
    CompletionRequest,
    LLMProvider,
    Message,
    ProviderError,
    ToolSchema,
)

if TYPE_CHECKING:
    from cortexflow.providers.tools import ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class TickResult:
    """
    The output of a single CVM tick.

    Contains the new agent state, any tool calls to dispatch,
    any events to emit, and performance metrics.
    """

    snapshot: StateSnapshot
    tool_calls_to_dispatch: list[ToolCall] = field(default_factory=list)
    events_to_emit: list[CortexEvent] = field(default_factory=list)
    completion: Completion | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    reasoning_step: ReasoningStep | None = None
    success: bool = True
    error: str | None = None



class LoopDetectedError(Exception):
    """Raised when an agent exceeds its max_iterations guard."""

    def __init__(self, agent_id: str, iteration: int) -> None:
        super().__init__(
            f"Agent '{agent_id}' exceeded max iterations ({iteration}). "
            "Possible infinite loop detected. Agent halted."
        )
        self.agent_id = agent_id
        self.iteration = iteration


class CognitiveVM:
    """
    The Cognitive Virtual Machine.

    Stateless by design — all state lives in the StateSnapshot.
    The CVM is a pure transformation engine:
        (snapshot + event + provider) → new_snapshot + side_effects
    """

    def __init__(self, tool_registry: "ToolRegistry | None" = None) -> None:
        self._tools = tool_registry

    async def tick(
        self,
        snapshot: StateSnapshot,
        event: CortexEvent,
        provider: LLMProvider,
    ) -> TickResult:
        """
        Execute one cognitive cycle for an agent.

        This is the hot path. It should be fast, deterministic, and safe.
        LLM calls are the only source of latency here.
        """
        start = time.perf_counter()
        log = logger.bind(
            agent_id=snapshot.agent_id,
            agent_name=snapshot.agent_name,
            event_type=event.type,
            iteration=snapshot.iteration,
        )

        log.info("cvm.tick.start")

        # ── Guard: Infinite loop detection ─────────────────────────────────
        if snapshot.is_looping():
            log.error("cvm.loop.detected", max_iterations=snapshot.max_iterations)
            raise LoopDetectedError(snapshot.agent_id, snapshot.iteration)

        # ── Phase 1: INPUT — Construct the reasoning context ───────────────
        messages = self._build_context(snapshot, event)

        # ── Phase 2: REASON — Call the LLM ────────────────────────────────
        tool_schemas = self._get_tool_schemas()
        request = CompletionRequest(
            messages=messages,
            tools=tool_schemas,
            token_budget=snapshot.metadata.get("token_budget"),
        )

        try:
            completion = await provider.complete(request)
            log.info(
                "cvm.reason.complete",
                model=completion.model,
                tokens=completion.total_tokens,
                stop_reason=completion.stop_reason,
            )
        except BudgetExceededError as e:
            log.warning("cvm.budget.exceeded", error=str(e))
            error_snapshot = snapshot.with_error(str(e))
            return TickResult(
                snapshot=error_snapshot,
                success=False,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except ProviderError as e:
            log.error("cvm.provider.error", error=str(e), provider=e.provider)
            error_snapshot = snapshot.with_error(str(e))
            return TickResult(
                snapshot=error_snapshot,
                success=False,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # ── Phase 3: ACTION — Parse what the agent wants to do ────────────
        reasoning_step = ReasoningStep(
            thought=completion.content or "(tool call — no text)",
            action=self._describe_action(completion),
        )

        tool_calls_to_dispatch: list[ToolCall] = []
        if completion.has_tool_calls:
            for tc in completion.tool_calls:
                tool_call = ToolCall(
                    call_id=tc.call_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                tool_calls_to_dispatch.append(tool_call)
                log.info("cvm.tool.dispatched", tool=tc.tool_name, call_id=tc.call_id)

        # ── Phase 4: OBSERVE — Update snapshot with results ───────────────
        new_snapshot = (
            snapshot
            .next_iteration()
            .append_reasoning_step(reasoning_step)
        )

        # Store last response in working memory for context continuity
        new_snapshot.working_memory.set("last_response", completion.content)
        new_snapshot.working_memory.set("last_event_type", str(event.type))

        if tool_calls_to_dispatch:
            new_snapshot = new_snapshot.model_copy(
                update={
                    "status": AgentStatus.RUNNING,
                    "pending_tool_calls": tool_calls_to_dispatch,
                }
            )
        else:
            new_snapshot = new_snapshot.with_status(AgentStatus.IDLE)

        # ── Phase 5: COMMIT ────────────────────────────────────────────────
        duration = (time.perf_counter() - start) * 1000
        log.info(
            "cvm.tick.complete",
            duration_ms=round(duration, 2),
            new_iteration=new_snapshot.iteration,
            status=new_snapshot.status,
            tool_calls=len(tool_calls_to_dispatch),
        )

        return TickResult(
            snapshot=new_snapshot,
            tool_calls_to_dispatch=tool_calls_to_dispatch,
            completion=completion,
            duration_ms=duration,
            tokens_used=completion.total_tokens,
            reasoning_step=reasoning_step,
            success=True,
        )


    def _build_context(self, snapshot: StateSnapshot, event: CortexEvent) -> list[Message]:
        """
        Construct the message list for the LLM from the agent's state and event.
        This is the "framing" of the agent's reality.
        """
        messages: list[Message] = []

        # System prompt from agent definition (stored in metadata)
        system_prompt = snapshot.metadata.get("system_prompt", "You are a helpful AI agent.")
        messages.append(Message(role="system", content=system_prompt))

        # Recent reasoning history (last N steps for context window management)
        max_history = snapshot.metadata.get("max_history_steps", 10)
        recent_steps = snapshot.reasoning_trace[-max_history:]
        for step in recent_steps:
            if step.thought:
                messages.append(Message(role="assistant", content=step.thought))
            if step.observation:
                messages.append(Message(role="user", content=f"[Observation] {step.observation}"))

        # Current event as the new user message
        event_content = self._format_event(event)
        messages.append(Message(role="user", content=event_content))

        return messages

    def _format_event(self, event: CortexEvent) -> str:
        """Format an event into a human-readable message for the LLM."""
        if event.type == EventType.MESSAGE_RECEIVED:
            return event.payload.get("content", "")
        if event.type == EventType.TOOL_RESULT:
            result = event.payload.get("result")
            error = event.payload.get("error")
            if error:
                return f"[Tool Error] {error}"
            return f"[Tool Result] {result}"
        if event.type == EventType.AGENT_TICK:
            return event.payload.get("prompt", "Continue your task.")
        return f"[Event: {event.type}] {event.payload}"

    def _get_tool_schemas(self) -> list[ToolSchema]:
        """Get JSON schemas for all registered tools."""
        if self._tools is None:
            return []
        return self._tools.get_schemas()

    def _describe_action(self, completion: Completion) -> str | None:
        """Produce a human-readable action description for the reasoning trace."""
        if completion.has_tool_calls:
            calls = ", ".join(tc.tool_name for tc in completion.tool_calls)
            return f"call_tools({calls})"
        if completion.is_complete:
            return "respond"
        return "continue"

    def _extract_observation(self, tool_calls: list[ToolCall]) -> str | None:
        """Format completed tool results as an observation string."""
        if not tool_calls:
            return None
        parts = []
        for tc in tool_calls:
            if tc.result is not None:
                parts.append(f"{tc.tool_name}: {tc.result}")
            elif tc.error:
                parts.append(f"{tc.tool_name} ERROR: {tc.error}")
        return "\n".join(parts) if parts else None
