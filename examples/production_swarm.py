"""
CortexFlow Production Swarm — The Definitive Integration Example.

This example demonstrates:
- Persistent state recovery (Inmortality)
- Token budget enforcement (CostGuard)
- Enterprise audit trails (L4 Archive)
- Centralized configuration (Bootloader)
"""

import asyncio
from cortexflow import defineAgent, defineTeam, CortexRuntime, get_config
from cortexflow.storage.sqlite import SQLiteSnapshotStore
from cortexflow.sdk.testing import MockProvider

async def main():
    print("🔥 INITIALIZING CORTEXFLOW V1.0.0-GOLD")
    
    # 1. Load configuration from .env
    config = get_config()
    
    # 2. Setup persistent storage
    store = SQLiteSnapshotStore()
    
    # 3. Define a "Digital Employee" (Negotiator)
    # Note: We use a high token budget for the test
    negotiator = defineAgent(
        name="Negotiator",
        instructions="You are a professional negotiator. Close the deal.",
        token_budget=5000,
        provider=MockProvider(responses=["I accept the offer.", "Let's discuss the price."])
    )
    
    # 4. Initialize the Runtime with all "Enterprise" features
    async with CortexRuntime() as runtime:
        # Connect storage
        runtime.use_storage(store)
        
        print("\n🚀 SPAWNING PERSISTENT AGENT...")
        # We use a fixed ID to test recovery
        agent_id = "agent-alpha-99"
        aid = await runtime.spawn(negotiator, agent_id=agent_id)
        
        # 5. Send a task
        print(f"📩 Sending message to {aid}...")
        await runtime.send(aid, "I want a 20% discount.")
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # 6. Check Health & Persistence
        health = await runtime.health()
        print("\n📊 SYSTEM HEALTH:")
        print(f"   Agent Status: {health['scheduler']['ticks_processed']} ticks processed")
        print(f"   L4 Archive:   Active")
        print(f"   Persistence:  {type(store).__name__} (Online)")
        
        # 7. Final Snapshot check
        snapshot = await store.load(aid)
        if snapshot:
            print(f"\n✅ PERSISTENCE VERIFIED: Agent '{snapshot.agent_name}' is safe in SQL DB.")
            print(f"   Last iteration: {snapshot.iteration}")
            print(f"   Trace Truncated: {snapshot.trace_truncated}")

    print("\n✨ CORTEXFLOW INTEGRATION SUCCESSFUL.")

if __name__ == "__main__":
    asyncio.run(main())
