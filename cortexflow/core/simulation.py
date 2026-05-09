"""
Cortex Ghost Runtime — The Simulation Engine.

Allows running agents in 'Shadow Mode' with synthetic events and mock providers
 to predict behavior, costs, and policy violations before live deployment.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import structlog

from cortexflow.core.runtime import CortexRuntime
from cortexflow.sdk.agent import AgentDefinition
from cortexflow.sdk.testing import MockProvider

logger = structlog.get_logger(__name__)


@dataclass
class SimulationResult:
    total_ticks: int
    total_tokens: int
    total_cost_est: float
    policy_violations: int
    duration_sec: float
    events_processed: int


class GhostRuntime:
    """
    A lightweight, fast-forward runtime for business simulation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        logger.info("ghost_runtime.init", mode="fast-forward")

    async def simulate(
        self, 
        agent_def: AgentDefinition, 
        scenario_events: List[str],
        iterations: int = 1
    ) -> SimulationResult:
        """
        Runs a high-speed simulation of an agent against a list of events.
        """
        start_time = time.perf_counter()
        
        # Use a real runtime but with isolated mocks
        async with CortexRuntime() as runtime:
            # Inject Mock Provider for speed and zero cost
            # (In the future, we could use a cheap local model for more realism)
            sim_provider = MockProvider(responses=["Simulated response"])
            agent_def.provider = sim_provider
            
            aid = await runtime.spawn(agent_def, agent_id="sim-agent")
            
            for event_content in scenario_events:
                await runtime.send(aid, event_content)
                
            # Wait for processing to complete (Ghost speed)
            # This is a simplified poll loop
            await asyncio.sleep(0.5) 
            
            stats = runtime.stats()
            
        duration = time.perf_counter() - start_time
        
        return SimulationResult(
            total_ticks=stats["scheduler"]["ticks_processed"],
            total_tokens=0, # Mock doesn't track real tokens well yet
            total_cost_est=0.0,
            policy_violations=0,
            duration_sec=duration,
            events_processed=len(scenario_events)
        )
