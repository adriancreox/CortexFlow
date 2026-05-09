"""
Hello Agent — CortexFlow in 15 lines.

The simplest possible CortexFlow agent. No tools, no workflows.
Just an agent that remembers what you said.

Run:
    pip install cortexflow
    python examples/hello_agent.py

Expected output:
    ✅ Agent 'assistant-abc123' started
    📤 Sending: "What is the Actor Model in computer science?"
    🤖 [mock] Response: I can help with that...
    📊 Stats: {'ticks': 1, 'tokens': 42, ...}
"""

from __future__ import annotations

import asyncio

from cortexflow import CortexRuntime, defineAgent
from cortexflow.events.schema import CortexEvent
from cortexflow.sdk.testing import MockProvider  # swap for OpenAIProvider in production

# ── Define the agent ──────────────────────────────────────────────────────────

assistant = defineAgent(
    name="assistant",
    instructions="""
    You are a helpful, knowledgeable assistant.
    Answer questions concisely and accurately.
    When you don't know something, say so honestly.
    """,
    provider=MockProvider(responses=[
        "The Actor Model is a concurrent computation model where 'actors' are the "
        "universal primitives of computation. Each actor can: receive messages, "
        "create new actors, send messages to other actors, and determine how to "
        "respond to the next message it receives. CortexFlow's agent architecture "
        "is directly inspired by this model."
    ]),
    token_budget=50_000,
)


# ── Run ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("⚡ CortexFlow — Hello Agent Example\n")

    async with CortexRuntime() as runtime:
        # Spawn a live agent instance from the definition
        agent_id = await runtime.spawn(assistant)
        print(f"✅ Agent started: '{agent_id}'")

        # Send a message — this emits a MESSAGE_RECEIVED event
        question = "What is the Actor Model in computer science?"
        print(f"\n📤 Sending: \"{question}\"")
        await runtime.send(agent_id, question)

        # Give the runtime a moment to process the tick
        await asyncio.sleep(0.5)

        # Inspect the agent's state (this is the Cortex-Pulse API)
        snapshot = runtime.get_snapshot(agent_id)
        if snapshot:
            last_response = snapshot.working_memory.get("last_response")
            print(f"\n🤖 Response:\n{last_response}")

        # Runtime stats — useful for the observability dashboard
        stats = runtime.stats()
        scheduler_stats = stats["scheduler"]
        agent_stats = scheduler_stats["agents"].get(agent_id, {})
        print(f"\n📊 Agent stats:")
        print(f"   Ticks:     {agent_stats.get('ticks', 0)}")
        print(f"   Tokens:    {agent_stats.get('tokens', 0)}")
        print(f"   Status:    {agent_stats.get('status', 'unknown')}")

    print("\n✅ Runtime stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
