"""
Reactive Workflow SDK — Event-driven agent orchestration.

Workflows are the coordination layer above individual agents.
They listen for events and orchestrate multi-agent interactions.

Instead of "chains" (rigid, sequential), CortexFlow uses
Reactive Flows (event-driven, composable, resilient).

Usage:
    from cortexflow.sdk.workflow import defineWorkflow

    billing = defineWorkflow(
        name="billing_recovery",
        trigger="payment.failed",
    )

    @billing.on("payment.failed")
    async def handle_failed_payment(event, runtime):
        agent_id = await runtime.spawn(support_closer)
        await runtime.send(agent_id, f"Recover payment for {event.payload['user_id']}")
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from cortexflow.events.schema import CortexEvent

if TYPE_CHECKING:
    from cortexflow.core.runtime import CortexRuntime

logger = structlog.get_logger(__name__)

WorkflowHandler = Callable[["CortexEvent", "CortexRuntime"], Coroutine[Any, Any, None]]


@dataclass
class WorkflowStep:
    event_pattern: str
    handler: WorkflowHandler


@dataclass
class WorkflowDefinition:
    name: str
    trigger: str
    steps: list[WorkflowStep] = field(default_factory=list)
    description: str = ""


class Workflow:
    """
    A reactive workflow builder.

    Workflows subscribe to events and orchestrate agents in response.
    Each .on() call adds a reactive step.
    """

    def __init__(self, name: str, trigger: str, description: str = "") -> None:
        self._definition = WorkflowDefinition(
            name=name,
            trigger=trigger,
            description=description,
        )

    def on(
        self,
        event_pattern: str,
    ) -> Callable[[WorkflowHandler], WorkflowHandler]:
        """Decorator: register a handler for a specific event pattern."""

        def decorator(fn: WorkflowHandler) -> WorkflowHandler:
            self._definition.steps.append(
                WorkflowStep(event_pattern=event_pattern, handler=fn)
            )
            logger.debug("workflow.step.registered", name=self._definition.name, pattern=event_pattern)
            return fn

        return decorator

    def pipe(self, *agent_definitions: Any) -> "Workflow":
        """
        Chain agents sequentially: output of one becomes input of next.
        Returns self for chaining.
        """
        # Sequential pipeline registration — implemented via event chaining
        # Full implementation in v0.2.0 with streaming support
        logger.info("workflow.pipe.registered", agents=len(agent_definitions))
        return self

    @property
    def definition(self) -> WorkflowDefinition:
        return self._definition

    async def mount(self, runtime: "CortexRuntime") -> None:
        """Register all workflow handlers with the runtime's event broker."""
        for step in self._definition.steps:
            await runtime._broker.subscribe(
                pattern=step.event_pattern,
                handler=lambda event, h=step.handler: h(event, runtime),
            )
        logger.info("workflow.mounted", name=self._definition.name, steps=len(self._definition.steps))


def defineWorkflow(
    name: str,
    trigger: str,
    description: str = "",
) -> Workflow:
    """
    Define a reactive CortexFlow workflow.

    Args:
        name:        Unique workflow identifier
        trigger:     The primary event type that activates this workflow
        description: Human-readable description for the dashboard

    Returns:
        Workflow builder — use .on() to add event handlers

    Example:
        billing = defineWorkflow("billing_recovery", trigger="payment.failed")

        @billing.on("payment.failed")
        async def recover(event, runtime):
            agent_id = await runtime.spawn(support_agent)
            await runtime.send(agent_id, str(event.payload))
    """
    return Workflow(name=name, trigger=trigger, description=description)


def on(event_type: str) -> Callable[[WorkflowHandler], WorkflowHandler]:
    """
    Module-level decorator for simple event binding (no Workflow object needed).

    Usage:
        @on("payment.failed")
        async def handle(event, runtime):
            ...
    """
    def decorator(fn: WorkflowHandler) -> WorkflowHandler:
        fn._cortexflow_event = event_type  # type: ignore[attr-defined]
        return fn
    return decorator
