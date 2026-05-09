"""
Cortex Shield — The Governance and Policy Engine for CortexFlow.

The Shield acts as a constitutional layer that intercepts every action
requested by an agent and validates it against a set of business policies
BEFORE execution.
"""

from __future__ import annotations

import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
from enum import Enum

from cortexflow.core.snapshot import ToolCall

logger = structlog.get_logger(__name__)


class PolicyAction(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


@dataclass
class PolicyResult:
    action: PolicyAction
    reason: Optional[str] = None


class ShieldPolicy(ABC):
    """Base class for all governance policies."""
    @abstractmethod
    async def validate(self, agent_id: str, tool_call: ToolCall, context: Dict[str, Any]) -> PolicyResult:
        ...


class Shield:
    """
    The main Governance Engine. 
    Intercepts ToolCalls and enforces business logic.
    """

    def __init__(self) -> None:
        self._policies: List[ShieldPolicy] = []
        logger.info("shield.init", status="active")

    def add_policy(self, policy: ShieldPolicy) -> None:
        self._policies.append(policy)
        logger.info("shield.policy_added", type=type(policy).__name__)

    async def authorize(self, agent_id: str, tool_calls: List[ToolCall], context: Dict[str, Any]) -> List[ToolCall]:
        """
        Validates a batch of tool calls.
        Returns ONLY the authorized ones or raises an exception if blocked.
        """
        authorized = []
        
        for tc in tool_calls:
            allowed = True
            for policy in self._policies:
                result = await policy.validate(agent_id, tc, context)
                
                if result.action == PolicyAction.BLOCK:
                    logger.warning(
                        "shield.violation.blocked", 
                        agent_id=agent_id, 
                        tool=tc.tool_name, 
                        reason=result.reason
                    )
                    allowed = False
                    break
                
                if result.action == PolicyAction.REQUIRE_APPROVAL:
                    logger.info(
                        "shield.approval_required", 
                        agent_id=agent_id, 
                        tool=tc.tool_name
                    )
                    # For now, we block it and signal approval needed (simplified)
                    allowed = False
                    break
            
            if allowed:
                authorized.append(tc)
                
        return authorized

# --- Standard Library of Policies ---

class NoDangerousToolsPolicy(ShieldPolicy):
    """Prevents execution of forbidden tools in specific environments."""
    def __init__(self, forbidden: List[str]) -> None:
        self._forbidden = forbidden

    async def validate(self, agent_id: str, tool_call: ToolCall, context: Dict[str, Any]) -> PolicyResult:
        if tool_call.tool_name in self._forbidden:
            return PolicyResult(PolicyAction.BLOCK, f"Tool '{tool_call.tool_name}' is globally forbidden.")
        return PolicyResult(PolicyAction.ALLOW)
