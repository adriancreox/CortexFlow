"""
Cortex Swarm — High-level Multi-Agent Orchestration.

Provides the 'defineTeam' abstraction to coordinate multiple agents 
via the EventMesh without manual event handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import structlog

from cortexflow.sdk.agent import AgentDefinition
from cortexflow.core.runtime import CortexRuntime

logger = structlog.get_logger(__name__)


@dataclass
class TeamMember:
    """A member of a swarm with a specific role."""
    agent: AgentDefinition
    role: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Team:
    """
    A collection of agents working together.
    Handles automatic spawning and routing.
    """

    def __init__(self, name: str, leader: AgentDefinition, workers: List[AgentDefinition]) -> None:
        self.name = name
        self.leader_def = leader
        self.worker_defs = workers
        self._runtime: Optional[CortexRuntime] = None
        self._member_ids: Dict[str, str] = {}

    async def deploy(self, runtime: CortexRuntime) -> None:
        """Spawns all team members into the runtime."""
        self._runtime = runtime
        
        # Spawn Leader
        leader_id = await runtime.spawn(self.leader_def, agent_id=f"{self.name}-leader")
        self._member_ids["leader"] = leader_id
        
        # Spawn Workers
        for i, worker in enumerate(self.worker_defs):
            worker_id = await runtime.spawn(worker, agent_id=f"{self.name}-worker-{i}")
            self._member_ids[f"worker-{i}"] = worker_id
            
        logger.info("team.deployed", name=self.name, total_members=len(self._member_ids))

    async def task(self, input_data: str) -> None:
        """Assigns a task to the team (via the leader)."""
        if "leader" not in self._member_ids:
            raise RuntimeError("Team not deployed. Call await team.deploy(runtime) first.")
        
        await self._runtime.send(self._member_ids["leader"], input_data)


def defineTeam(name: str, leader: AgentDefinition, workers: List[AgentDefinition]) -> Team:
    """SDK Helper to create a multi-agent team."""
    return Team(name=name, leader=leader, workers=workers)
