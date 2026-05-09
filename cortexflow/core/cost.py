"""
CostGuard — Middleware for token budget enforcement.
"""

from __future__ import annotations

import structlog
from typing import Dict

from cortexflow.providers.base import (
    LLMProvider, 
    CompletionRequest, 
    Completion, 
    BudgetExceededError
)

logger = structlog.get_logger(__name__)


class CostGuard(LLMProvider):
    """
    Middleware that tracks and enforces token budgets for an agent.
    Wraps any LLMProvider and raises BudgetExceededError if the limit is hit.
    """

    def __init__(self, provider: LLMProvider, budget: int | None = None) -> None:
        self._provider = provider
        self._budget = budget
        self._used = 0
        logger.info("cost_guard.init", provider=provider.name, budget=budget)

    @property
    def name(self) -> str:
        return f"cost-guard({self._provider.name})"

    async def complete(self, request: CompletionRequest) -> Completion:
        # Check if budget is already hit
        if self._budget is not None and self._used >= self._budget:
            logger.error("cost_guard.violation.pre_flight", used=self._used, budget=self._budget)
            raise BudgetExceededError(self._used, self._budget, self.name)

        # Execute call
        completion = await self._provider.complete(request)

        # Update tracking
        self._used += completion.total_tokens
        
        # Check if call just pushed us over
        if self._budget is not None and self._used > self._budget:
            logger.warning("cost_guard.violation.post_flight", used=self._used, budget=self._budget)
            # We allow the last completion to return, but the next one will fail.
            # Alternatively, we could throw here if we wanted strict enforcement.

        return completion

    async def health(self) -> bool:
        return await self._provider.health()

    @property
    def tokens_used(self) -> int:
        return self._used
