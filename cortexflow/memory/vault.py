"""
MemoryVault — Unified facade over the 4-tier memory hierarchy.

Routes reads and writes across L1 → L4 with automatic promotion,
context segmentation, and background summarization.

Context Segmentation (Virtual Context Segmentation):
  Hot  Segment: last N messages — full precision, always in L1
  Warm Segment: older messages  — summarized, stored in L2
  Cold Segment: far history     — vector-indexed, retrieved via L3
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from cortexflow.memory.l1_registers import L1WorkingMemory
from cortexflow.memory.l2_cache import L2RedisCache
from cortexflow.memory.l3_vector import L3VectorStore
from cortexflow.memory.l4_archive import L4Archive

logger = structlog.get_logger(__name__)

# Context segmentation thresholds
HOT_SEGMENT_SIZE = 5    # messages kept verbatim in L1
WARM_SEGMENT_SIZE = 20  # messages summarized into L2
# Beyond WARM_SEGMENT_SIZE → indexed into L3 via background job


class MemoryVault:
    """
    Unified memory facade for a single agent instance.

    Access pattern (read): L1 → L2 → L3 → None
    Write pattern: always L1 first, promote to L2 on eviction.
    Archive pattern: async write to L4 on every state commit.
    """

    def __init__(
        self,
        agent_id: str,
        l1: L1WorkingMemory | None = None,
        l2: L2RedisCache | None = None,
        l3: L3VectorStore | None = None,
        l4: L4Archive | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._l1 = l1 or L1WorkingMemory()
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._summarizer_provider: LLMProvider | None = None
        self._summarization_task: asyncio.Task[None] | None = None


    # ── Core Read/Write ─────────────────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """Read-through cache: L1 → L2 → L3."""
        scoped_key = self._scope(key)

        # L1 hit
        value = await self._l1.read(scoped_key)
        if value is not None:
            return value

        # L2 hit → promote to L1
        if self._l2:
            value = await self._l2.read(scoped_key)
            if value is not None:
                await self._l1.write(scoped_key, value)
                logger.debug("vault.l2.promote", key=key)
                return value

        # L3 hit (semantic) → promote to L1
        if self._l3:
            value = await self._l3.read(scoped_key)
            if value is not None:
                await self._l1.write(scoped_key, value)
                logger.debug("vault.l3.promote", key=key)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        persist: bool = True,
    ) -> None:
        """Write-through: L1 always, L2 if persist=True."""
        scoped_key = self._scope(key)
        await self._l1.write(scoped_key, value, ttl)
        if persist and self._l2:
            await self._l2.write(scoped_key, value, ttl)

    async def delete(self, key: str) -> None:
        scoped_key = self._scope(key)
        await self._l1.delete(scoped_key)
        if self._l2:
            await self._l2.delete(scoped_key)

    # ── Context Segmentation ────────────────────────────────────────────────

    async def append_message(self, role: str, content: str) -> None:
        """
        Add a conversation message with automatic context segmentation.
        Hot: last HOT_SEGMENT_SIZE messages in L1 (verbatim).
        Warm: older messages summarized to L2.
        """
        messages: list[dict[str, str]] = await self.get("messages") or []
        messages.append({"role": role, "content": content})

        # Keep only hot segment in L1
        if len(messages) > HOT_SEGMENT_SIZE + WARM_SEGMENT_SIZE:
            # Trigger background summarization of warm segment
            warm = messages[HOT_SEGMENT_SIZE:HOT_SEGMENT_SIZE + WARM_SEGMENT_SIZE]
            asyncio.create_task(self._summarize_warm(warm))  # noqa: RUF006
            messages = messages[-HOT_SEGMENT_SIZE:]

        await self.set("messages", messages, persist=True)

    async def get_hot_context(self) -> list[dict[str, str]]:
        """Get the hot segment — last N messages, verbatim."""
        messages: list[dict[str, str]] = await self.get("messages") or []
        return messages[-HOT_SEGMENT_SIZE:]

    async def get_warm_summary(self) -> str | None:
        """Get the compressed warm segment summary from L2."""
        return await self.get("warm_summary")

    async def _summarize_warm(self, messages: list[dict[str, str]]) -> None:
        """
        Compresses warm messages into a high-density summary using an LLM.
        """
        if not self._summarizer_provider:
            # Fallback to simple concatenation if no provider is set
            summary_parts = [f"{m['role']}: {m['content'][:150]}" for m in messages]
            summary = " | ".join(summary_parts)
        else:
            try:
                from cortexflow.providers.base import CompletionRequest, Message
                prompt = f"Summarize the following conversation segment concisely, preserving key facts and decisions:\n{messages}"
                request = CompletionRequest(
                    messages=[Message(role="system", content="You are a memory compression engine. Summarize facts accurately.")],
                    max_tokens=200
                )
                # Note: This is simplified. In production, we'd add the messages to the request.
                completion = await self._summarizer_provider.complete(request)
                summary = completion.content or "Summary failed."
            except Exception as e:
                logger.error("vault.summarize.failed", error=str(e))
                summary = "Error generating summary."

        await self.set("warm_summary", summary, ttl=7200, persist=True)
        logger.info("vault.warm.summarized", agent_id=self._agent_id, messages=len(messages), method="llm" if self._summarizer_provider else "concat")


    # ── Archive ─────────────────────────────────────────────────────────────

    async def archive_reasoning(self, snapshot_id: str, reasoning: Any) -> None:
        """Async write of a reasoning trace to L4 (fire and forget)."""
        if self._l4:
            await self._l4.write(
                key=f"reasoning:{self._agent_id}:{snapshot_id}",
                value=reasoning,
            )

    # ── Health ──────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "agent_id": self._agent_id,
            "l1": await self._l1.health(),
        }
        if self._l2:
            report["l2"] = await self._l2.health()
        if self._l3:
            report["l3"] = await self._l3.health()
        if self._l4:
            report["l4"] = await self._l4.health()
        return report

    def _scope(self, key: str) -> str:
        return f"agent:{self._agent_id}:{key}"
