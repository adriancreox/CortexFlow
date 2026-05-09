"""
Unit tests for StateSnapshot — the DNA of every CortexFlow agent.

Tests cover: serialization, state transitions, mailbox, working memory,
loop detection, and immutability guarantees.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cortexflow.core.snapshot import (
    AgentStatus,
    Mailbox,
    ReasoningStep,
    StateSnapshot,
    WorkingMemory,
)


class TestStateSnapshot:
    def _make_snapshot(self, **kwargs: object) -> StateSnapshot:
        return StateSnapshot(
            agent_id="test-001",
            agent_name="test-agent",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_default_status_is_idle(self) -> None:
        s = self._make_snapshot()
        assert s.status == AgentStatus.IDLE

    def test_next_iteration_increments(self) -> None:
        s = self._make_snapshot()
        s2 = s.next_iteration()
        assert s2.iteration == 1
        assert s.iteration == 0  # original unchanged

    def test_snapshot_is_immutable(self) -> None:
        s = self._make_snapshot()
        with pytest.raises(Exception):
            s.iteration = 99  # type: ignore[misc]

    def test_with_status_returns_new_snapshot(self) -> None:
        s = self._make_snapshot()
        s2 = s.with_status(AgentStatus.RUNNING)
        assert s2.status == AgentStatus.RUNNING
        assert s.status == AgentStatus.IDLE

    def test_with_error_sets_error_fields(self) -> None:
        s = self._make_snapshot()
        s2 = s.with_error("Something went wrong")
        assert s2.status == AgentStatus.ERROR
        assert s2.last_error == "Something went wrong"
        assert s2.consecutive_errors == 1

    def test_consecutive_errors_accumulate(self) -> None:
        s = self._make_snapshot()
        s = s.with_error("Error 1")
        s = s.with_error("Error 2")
        assert s.consecutive_errors == 2

    def test_loop_detection(self) -> None:
        s = self._make_snapshot(max_iterations=3)
        for _ in range(3):
            s = s.next_iteration()
        assert s.is_looping()

    def test_no_loop_below_max(self) -> None:
        s = self._make_snapshot(max_iterations=50)
        s = s.next_iteration()
        assert not s.is_looping()

    def test_append_reasoning_step(self) -> None:
        s = self._make_snapshot()
        step = ReasoningStep(thought="I should search for this.", action="call_tools(search)")
        s2 = s.append_reasoning_step(step)
        assert len(s2.reasoning_trace) == 1
        assert s2.reasoning_trace[0].thought == "I should search for this."

    def test_reasoning_trace_capped_at_200(self) -> None:
        s = self._make_snapshot()
        for i in range(210):
            step = ReasoningStep(thought=f"Step {i}")
            s = s.append_reasoning_step(step)
        assert len(s.reasoning_trace) <= 200

    def test_snapshot_serializes_to_json(self) -> None:
        s = self._make_snapshot()
        data = s.model_dump()
        serialized = json.dumps(data, default=str)
        assert "agent_id" in serialized
        assert "test-001" in serialized

    def test_snapshot_round_trips(self) -> None:
        s = self._make_snapshot(metadata={"key": "value"})
        data = s.model_dump()
        s2 = StateSnapshot(**data)
        assert s2.agent_id == s.agent_id
        assert s2.metadata == {"key": "value"}

    def test_snapshot_id_changes_on_mutation(self) -> None:
        s = self._make_snapshot()
        s2 = s.next_iteration()
        assert s.snapshot_id != s2.snapshot_id

    def test_negative_iteration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._make_snapshot(iteration=-1)


class TestMailbox:
    def test_enqueue_dequeue(self) -> None:
        mb = Mailbox()
        mb.enqueue("event-001")
        mb.enqueue("event-002")
        assert mb.dequeue() == "event-001"
        assert mb.dequeue() == "event-002"
        assert mb.dequeue() is None

    def test_idempotency(self) -> None:
        mb = Mailbox()
        mb.enqueue("event-001")
        mb.dequeue()
        mb.enqueue("event-001")  # Already processed
        assert mb.dequeue() is None

    def test_overflow_raises(self) -> None:
        mb = Mailbox()
        mb.MAX_SIZE = 3
        mb.enqueue("e1")
        mb.enqueue("e2")
        mb.enqueue("e3")
        with pytest.raises(OverflowError):
            mb.enqueue("e4")


class TestWorkingMemory:
    def test_set_get(self) -> None:
        wm = WorkingMemory()
        wm.set("key", "value")
        assert wm.get("key") == "value"

    def test_missing_key_returns_default(self) -> None:
        wm = WorkingMemory()
        assert wm.get("missing", "default") == "default"

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.set("key", "value")
        wm.clear()
        assert wm.get("key") is None

    def test_eviction_at_max(self) -> None:
        wm = WorkingMemory()
        wm.MAX_ENTRIES = 3
        wm.set("a", 1)
        wm.set("b", 2)
        wm.set("c", 3)
        wm.set("d", 4)  # should evict "a"
        assert wm.get("a") is None
        assert wm.get("d") == 4
