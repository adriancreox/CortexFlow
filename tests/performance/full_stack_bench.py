"""
Full-Stack Benchmark — Testing Kernel + L3 Memory + L4 Archival.
"""

import asyncio
import time
import structlog
from cortexflow import defineAgent, CortexRuntime
from cortexflow.sdk.testing import MockProvider

async def run_benchmark(ticks=50):
    print(f"🚀 Starting Full-Stack Benchmark ({ticks} ticks)...")
    
    # 1. Define agent with a provider
    agent_def = defineAgent(
        name="bench-agent",
        instructions="Benchmark performance test.",
        provider=MockProvider(responses=["Benchmark OK"])
    )

    async with CortexRuntime() as runtime:
        # 2. Spawn agent (Initializes Vault L1-L4)
        agent_id = await runtime.spawn(agent_def)
        
        start_time = time.perf_counter()
        
        # 3. Stress the Event Mesh and Scheduler
        for i in range(ticks):
            await runtime.send(agent_id, f"Tick {i}")
        
        # 4. Wait for processing to complete
        # Since it's all local/mock, it should be very fast
        while True:
            stats = runtime.stats()
            processed = stats["scheduler"]["ticks_processed"]
            if processed >= ticks:
                break
            await asyncio.sleep(0.01)
            
        duration = time.perf_counter() - start_time
        
        print("\n📊 BENCHMARK RESULTS:")
        print(f"   Total Time:   {duration:.4f}s")
        print(f"   Avg Latency:  {(duration/ticks)*1000:.2f}ms/tick")
        print(f"   Throughput:   {ticks/duration:.1f} ticks/sec")
        
        # Check L4 Archive Health
        health = await runtime.health()
        print(f"   L4 Storage:   {health['vaults'][agent_id]['l4']['storage']}")
        print(f"   L4 Queue:     {health['vaults'][agent_id]['l4']['queue_depth']}")

if __name__ == "__main__":
    asyncio.run(run_benchmark(50))
