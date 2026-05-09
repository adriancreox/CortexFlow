"""
Middleware for LLM Providers — CostGuard and LatencyGuard.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cortexflow.providers.base import (
    BudgetExceededError,
    Completion,
    CompletionRequest,
    LLMProvider,
    ProviderError,
)

logger = structlog.get_logger(__name__)


class ProviderMiddleware(LLMProvider):
    """Base class for provider wrappers/decorators."""

    def __init__(self, provider: LLMProvider) -> None:
        self._inner = provider

    async def complete(self, request: CompletionRequest) -> Completion:
        return await self._inner.complete(request)

    async def health(self) -> bool:
        return await self._inner.health()

    @property
    def name(self) -> str:
        return self._inner.name


class CostGuard(ProviderMiddleware):
    """
    Tracks token usage and enforces hard budgets per agent/session.
    """

    def __init__(self, provider: LLMProvider, total_budget: int | None = None) -> None:
        super().__init__(provider)
        self._total_budget = total_budget
        self._used_tokens = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        # 1. Budget check (pre-flight)
        budget = request.token_budget or self._total_budget
        if budget and self._used_tokens >= budget:
            raise BudgetExceededError(self._used_tokens, budget, self.name)

        # 2. Execute
        completion = await self._inner.complete(request)

        # 3. Track usage
        self._used_tokens += completion.total_tokens
        
        logger.info(
            "cost_guard.usage",
            provider=self.name,
            added=completion.total_tokens,
            total=self._used_tokens,
            budget=budget,
        )
        return completion


class LatencyGuard(ProviderMiddleware):
    """
    Enforces timeouts and exponential backoff retries for provider calls.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_attempts: int = 3,
        max_latency_seconds: float = 60.0,
    ) -> None:
        super().__init__(provider)
        self._max_attempts = max_attempts
        self._max_latency = max_latency_seconds

    async def complete(self, request: CompletionRequest) -> Completion:
        # Use tenacity for robust retries on common provider errors
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((ProviderError, TimeoutError)),
            reraise=True,
        )

        start_time = time.perf_counter()
        try:
            async for attempt in retryer:
                with attempt:
                    return await self._inner.complete(request)
        finally:
            duration = time.perf_counter() - start_time
            if duration > self._max_latency:
                logger.warning(
                    "latency_guard.slow_call",
                    provider=self.name,
                    duration_s=round(duration, 2),
                    limit_s=self._max_latency,
                )
