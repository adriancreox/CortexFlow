"""
demo-agent — Powered by CortexFlow
"""
import asyncio
from cortexflow import defineAgent, CortexRuntime
from cortexflow.providers.openai import OpenAIProvider

# 1. Define your agent blueprint
agent = defineAgent(
    name="my-agent",
    instructions="You are a helpful AI process running inside CortexFlow.",
    provider=OpenAIProvider(model='gpt-4o'),
    allowed_scopes=["internet"]
)

async def main():
    # 2. Start the Cognitive Runtime
    async with CortexRuntime() as runtime:
        # 3. Spawn a stateful process
        agent_id = await runtime.spawn(agent)
        print(f"🚀 Agent Process Started: {agent_id}")

        # 4. Interact with the event mesh
        await runtime.send(agent_id, "System check: are you operational?")
        
        # Keep running to see the logs
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
