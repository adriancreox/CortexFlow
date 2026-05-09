"""
LLMProvider — Unified interface over all language model providers.

The key insight: OpenAI, Anthropic, and local models all do the same thing
(turn messages into text) but express it differently. This layer normalizes
everything into a single, provider-agnostic schema so:

  1. You can swap providers without touching agent code.
  2. The CostGuard and LatencyGuard middleware work uniformly.
  3. Tool calling is validated consistently — no hallucinated parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """A single turn in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None   # For tool result messages
    name: str | None = None


@dataclass(frozen=True)
class ToolSchema:
    """JSON Schema definition of a tool callable by the agent."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    """
    Normalized output from any LLM provider.
    The CVM works exclusively with this type — never with raw provider responses.
    """

    content: str | None                       # Text response (may be None if only tool calls)
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "unknown"
    stop_reason: str = "stop"                 # "stop" | "tool_use" | "length" | "error"
    raw: dict[str, Any] = field(default_factory=dict)  # Original provider response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_complete(self) -> bool:
        return self.stop_reason == "stop" and not self.has_tool_calls


@dataclass(frozen=True)
class CompletionRequest:
    """Everything needed to call any LLM provider."""

    messages: list[Message]
    tools: list[ToolSchema] = field(default_factory=list)
    model: str | None = None         # None = provider default
    temperature: float = 0.7
    max_tokens: int = 4096
    token_budget: int | None = None  # Hard cap enforced by CostGuard middleware
    timeout_seconds: float = 30.0


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> Completion:
        """
        Send a completion request and return a normalized Completion.
        Must raise ProviderError on failure (never return partial results silently).
        """
        ...

    @abstractmethod
    async def health(self) -> bool:
        """Check if the provider is reachable."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'openai-gpt-4o')."""
        ...


class ProviderError(Exception):
    """Raised when an LLM provider call fails unrecoverably."""

    def __init__(self, message: str, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class BudgetExceededError(ProviderError):
    """Raised by CostGuard when the token budget is exhausted."""

    def __init__(self, used: int, budget: int, provider: str) -> None:
        super().__init__(
            f"Token budget exceeded: used {used}, budget {budget}",
            provider=provider,
        )
        self.used = used
        self.budget = budget
