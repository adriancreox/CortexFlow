"""
Stateful Negotiator — An agent that remembers across sessions.

Demonstrates L2 Redis cache for cross-session memory.
The agent tracks negotiation history and adapts its strategy
based on previous interactions — even after a restart.

This is the example that shows why CortexFlow is different:
the agent's memory outlives the process.

Run:
    pip install "cortexflow[openai]"
    python examples/stateful_negotiator.py
"""

from __future__ import annotations

import asyncio

from cortexflow import CortexRuntime, defineAgent
from cortexflow.events.schema import CortexEvent
from cortexflow.memory.vault import MemoryVault
from cortexflow.sdk.testing import MockProvider


negotiator = defineAgent(
    name="negotiator",
    instructions="""
    You are an expert sales negotiator. Your goal: close deals at maximum value.

    Strategy guidelines:
    - Start at list price. Never volunteer discounts.
    - If pushed, offer value-adds before price reductions.
    - Track concessions you've made — never repeat them without getting something back.
    - If a prospect is a returning buyer, acknowledge the relationship.
    - Your BATNA (Best Alternative To Negotiated Agreement): walk away at 20% discount.

    Remember: every interaction is a data point. Learn from each session.
    """,
    provider=MockProvider(responses=[
        "Thank you for your interest. Our solution is priced at $50,000/year, "
        "which reflects the full value of our enterprise-grade reliability and support.",

        "I understand budget constraints. Before we discuss pricing, let me share "
        "what our top clients get from the ROI perspective. Most see 3x return in "
        "year one. That said, I can offer an extended payment plan — 12 monthly "
        "installments instead of annual. Would that help?",

        "I appreciate your candor. Given our previous conversation and the volume "
        "you're considering, I'm authorized to offer $45,000 with a 2-year commitment. "
        "That's our best structured offer. This is the same terms we gave [Enterprise Client]. "
        "Should I send the contract?",
    ]),
    memory_config={"persist_session": True},
)


async def simulate_session(session_num: int, messages: list[str]) -> None:
    """Simulate one negotiation session."""
    print(f"\n{'─'*60}")
    print(f"📞 Session {session_num}")
    print(f"{'─'*60}")

    async with CortexRuntime() as runtime:
        agent_id = await runtime.spawn(negotiator, agent_id="negotiator-main")

        for msg in messages:
            print(f"\n👤 Prospect: {msg}")
            await runtime.send(agent_id, msg)
            await asyncio.sleep(0.4)

            snapshot = runtime.get_snapshot(agent_id)
            if snapshot:
                response = snapshot.working_memory.get("last_response")
                if response:
                    print(f"🤝 Agent:    {response[:300]}...")


async def main() -> None:
    print("⚡ CortexFlow — Stateful Negotiator Demo")
    print("Demonstrates cross-tick memory and adaptive behavior\n")

    # Session 1: Initial contact
    await simulate_session(1, [
        "Hi, I'm interested in your enterprise plan. What's the price?",
        "That's above our budget. Can we do better?",
    ])

    # Session 2: Follow-up (agent remembers previous session via L2)
    await simulate_session(2, [
        "We spoke last week. We've got budget approval but need to be at $44K.",
    ])

    print(f"\n{'─'*60}")
    print("✅ Demo complete.")
    print("\nKey CortexFlow features demonstrated:")
    print("  • Agent state persists across multiple runtime sessions")
    print("  • Working memory tracks conversation context")
    print("  • Multiple ticks chain coherently")
    print("  • Clean async context manager lifecycle")


if __name__ == "__main__":
    asyncio.run(main())
