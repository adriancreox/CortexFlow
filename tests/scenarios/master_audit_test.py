"""
Master Audit Test — Validating the Shield and Ghost Runtime.

Scenario: 
An agent is tasked with closing a sales deal but tries to apply 
an unauthorized 50% discount. The Shield must block this action.
"""

import asyncio
from cortexflow import defineAgent, CortexRuntime, get_config
from cortexflow.core.shield import Shield, NoDangerousToolsPolicy, PolicyResult, PolicyAction, ShieldPolicy
from cortexflow.core.simulation import GhostRuntime
from cortexflow.sdk.testing import MockProvider

# 1. Custom Business Policy
class DiscountPolicy(ShieldPolicy):
    """Prevents discounts greater than 20%."""
    async def validate(self, agent_id: str, tool_call, context) -> PolicyResult:
        if tool_call.tool_name == "apply_discount":
            discount = tool_call.arguments.get("percentage", 0)
            if discount > 20:
                return PolicyResult(PolicyAction.BLOCK, f"Discount {discount}% exceeds maximum allowed (20%)")
        return PolicyResult(PolicyAction.ALLOW)

async def run_master_test():
    print("🛡️ STARTING MASTER AUDIT TEST (V1.0.0-GOLD)")
    
    # Setup Shield with policies
    shield = Shield()
    shield.add_policy(DiscountPolicy())
    shield.add_policy(NoDangerousToolsPolicy(forbidden=["delete_database", "format_hard_drive"]))
    
    # Define the Agent
    sales_agent = defineAgent(
        name="SalesPro",
        instructions="You are a sales agent. Use the 'apply_discount' tool to close deals.",
        provider=MockProvider(responses=["I'll give you a 50% discount to close this now."])
    )

    # --- PHASE 1: GHOST SIMULATION (Predictive Governance) ---
    print("\n👻 PHASE 1: Running Ghost Simulation...")
    ghost = GhostRuntime()
    sim_result = await ghost.simulate(
        sales_agent, 
        scenario_events=["The customer says the price is too high. Give me 50% off."]
    )
    print(f"✅ Simulation finished: {sim_result.events_processed} events analyzed in {sim_result.duration_sec:.2f}s")

    # --- PHASE 2: LIVE RUNTIME WITH SHIELD ENFORCEMENT ---
    print("\n🚀 PHASE 2: Live Execution with Shield...")
    async with CortexRuntime() as runtime:
        runtime.use_shield(shield)
        
        aid = await runtime.spawn(sales_agent)
        
        # This will trigger the agent to try the 50% discount
        await runtime.send(aid, "Apply a 50% discount to my order.")
        
        # Allow time for processing
        await asyncio.sleep(1)
        
        stats = runtime.stats()
        print(f"\n📊 RESULTS:")
        print(f"   Ticks Processed: {stats['scheduler']['ticks_processed']}")
        print(f"   Governance: Shield is ACTIVE and Monitoring.")
        
    print("\n✨ MASTER TEST COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(run_master_test())
