"""
SDR Lead Qualifier — A concrete example of a "Digital Employee" built on CortexFlow.

Objective:
Given a company name, the agent:
1. Researches the company via web search.
2. Evaluates the lead based on custom criteria.
3. Saves a Qualification Report to the local filesystem.
"""

import asyncio
import structlog
from cortexflow import defineAgent, CortexRuntime
from cortexflow.tools.standard import std_tools
from cortexflow.sdk.testing import MockProvider

# Configure logging for better visibility in the terminal
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

# 1. DEFINE THE AGENT BLUEPRINT
# This is a pure data structure. No I/O happens here.
sdr_agent = defineAgent(
    name="sdr-analyst",
    instructions="""
    You are a professional B2B Sales Development Representative (SDR).
    Your goal is to qualify leads for a specialized AI automation agency.
    
    CRITERIA:
    - Target: Tech companies, E-commerce, or Legal firms.
    - Size: 10-200 employees.
    
    WORKFLOW:
    1. Search for the company name using 'web_search'.
    2. Analyze if they fit the criteria.
    3. Write a report named 'lead_<company_name>.md' using 'write_file'.
    4. Provide a final Lead Score (1-10).
    """,
    # In a real app, use OpenAIProvider, AnthropicProvider, etc.
    # Here we use MockProvider for demonstration.
    provider=MockProvider(),
    # Grant access to the required tools from the Standard Library
    tools=[std_tools.web_search, std_tools.write_file],
    # SECURITY: Only grant the scopes needed for this job
    allowed_scopes=["internet", "filesystem:write"]
)

async def main():
    print("⚡ Starting CortexFlow SDR Lead Qualifier Demo...")
    
    # 2. INITIALIZE THE RUNTIME
    # This starts the Event Mesh and the Scheduler
    async with CortexRuntime() as runtime:
        
        # 3. SPAWN THE AGENT PROCESS
        # This allocates memory and registers the agent in the Kernel
        agent_id = await runtime.spawn(sdr_agent)
        print(f"🚀 SDR Process active: {agent_id}")

        # 4. EMIT A WORK REQUEST
        # We send a message to the agent's mailbox via the Event Mesh
        print("📤 Dispatching task: Research 'Creox Studios'...")
        await runtime.send(agent_id, "Research Creox Studios and qualify them as a lead.")

        # 5. WAIT FOR EXECUTION
        # The Kernel will wake the agent, handle the reasoning cycles,
        # execute the search tool, and perform the file write.
        await asyncio.sleep(2)

        # 6. INSPECT FINAL STATE
        # We can peek into the agent's memory at any time
        snapshot = runtime.get_snapshot(agent_id)
        if snapshot:
            print("\n📊 EXECUTION SUMMARY:")
            print(f"   Status:     {snapshot.status}")
            print(f"   Iterations: {snapshot.iteration}")
            print(f"   Last Logic: {snapshot.working_memory.get('last_response')[:200]}...")
            
    print("\n✅ Demo complete. Check the 'vault/' directory for the generated report.")

if __name__ == "__main__":
    asyncio.run(main())
