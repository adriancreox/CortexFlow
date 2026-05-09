"""
defineAgent — The primary SDK entry point for CortexFlow developers.

This is what the developer touches. Everything else is the engine.

Usage:
    from cortexflow import defineAgent, CortexRuntime
    from cortexflow.providers.openai import OpenAIProvider

    researcher = defineAgent(
        name="researcher",
        instructions="You are a world-class research analyst...",
        provider=OpenAIProvider(model="gpt-4o"),
        tools=[search_web, read_url],
        token_budget=50_000,
    )

    async with CortexRuntime() as runtime:
        agent_id = await runtime.spawn(researcher)
        await runtime.send(agent_id, "Research the latest developments in quantum computing")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortexflow.providers.base import LLMProvider
from cortexflow.providers.tools import ToolRegistry


@dataclass
class AgentDefinition:
    """
    The blueprint for a CortexFlow agent.
    Immutable by design — spawn() uses this to create runtime instances.
    """

    # Identity
    name: str
    instructions: str

    # Execution
    provider: LLMProvider
    max_iterations: int = 50
    max_history_steps: int = 10

    # Budget
    token_budget: int | None = None  # None = unlimited

    # Tools
    tool_registry: ToolRegistry | None = None

    # Memory config
    memory_config: dict[str, Any] = field(default_factory=dict)

    # Security
    allowed_scopes: list[str] = field(default_factory=list)

    # Metadata
    tags: list[str] = field(default_factory=list)

    description: str = ""


def defineAgent(
    name: str,
    instructions: str,
    provider: LLMProvider,
    tools: list[Any] | None = None,
    max_iterations: int = 50,
    token_budget: int | None = None,
    max_history_steps: int = 10,
    description: str = "",
    tags: list[str] | None = None,
    memory_config: dict[str, Any] | None = None,
    allowed_scopes: list[str] | None = None,
) -> AgentDefinition:

    """
    Define a CortexFlow agent.

    This is a pure value — no side effects, no I/O. Pass it to
    runtime.spawn() to create a live, stateful agent instance.

    Args:
        name:              Human-readable agent identifier
        instructions:      The agent's system prompt / persona
        provider:          LLM provider (OpenAI, Anthropic, Ollama, Mock)
        tools:             List of registered tool functions
        max_iterations:    Infinite loop guard (default: 50)
        token_budget:      Hard token cap per session (None = unlimited)
        max_history_steps: How many reasoning steps to include in context
        description:       Optional description for the Cortex-Pulse dashboard
        tags:              Labels for filtering in observability tools
        memory_config:     Override default memory tier settings
        allowed_scopes:    Security: List of authorized capability scopes


    Returns:
        AgentDefinition — pass to runtime.spawn()

    Example:
        agent = defineAgent(
            name="analyst",
            instructions="Analyze financial data and identify trends.",
            provider=OpenAIProvider(model="gpt-4o"),
            token_budget=100_000,
        )
    """
    return AgentDefinition(
        name=name,
        instructions=instructions,
        provider=provider,
        max_iterations=max_iterations,
        token_budget=token_budget,
        max_history_steps=max_history_steps,
        description=description,
        tags=tags or [],
        memory_config=memory_config or {},
        allowed_scopes=allowed_scopes or [],
    )
