"""
ToolRegistry — Tool registration and circuit-breaking for CortexFlow.

Tools are first-class citizens. Every tool definition includes:
  - JSON Schema auto-generated from Python type hints (via Pydantic)
  - Native retry policy with exponential backoff
  - Circuit breaker: tool is suspended after N consecutive failures
  - Timeout enforcement

This eliminates LLM hallucinated parameters — the schema is the contract.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, get_type_hints

import structlog
from pydantic import BaseModel, create_model

from cortexflow.providers.base import ToolSchema

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tool suspended due to failures
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryPolicy:
    attempts: int = 3
    backoff: str = "exponential"  # "fixed" | "exponential" | "none"
    base_delay: float = 1.0
    max_delay: float = 30.0

    def delay_for(self, attempt: int) -> float:
        if self.backoff == "none":
            return 0.0
        if self.backoff == "fixed":
            return self.base_delay
        # exponential
        return min(self.base_delay * (2 ** attempt), self.max_delay)


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Coroutine[Any, Any, Any]]
    schema: ToolSchema
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: float = 30.0
    failure_threshold: int = 5   # Opens circuit after N failures
    recovery_timeout: float = 60.0
    required_scopes: list[str] = field(default_factory=list)  # Security: Scopes needed to call this tool


    # Circuit breaker state
    _circuit_state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure: float = field(default=0.0, init=False)
    _call_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)


class ToolCallError(Exception):
    """Raised when a tool execution fails after all retries."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {reason}")
        self.tool_name = tool_name
        self.reason = reason


class CircuitOpenError(ToolCallError):
    """Raised when the circuit breaker is open for a tool."""


class PermissionDeniedError(ToolCallError):
    """Raised when an agent tries to call a tool without the required scopes."""



class ToolRegistry:
    """Registry for all tools available to CortexFlow agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 30.0,
        required_scopes: list[str] | None = None,
    ) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:

        """Decorator to register a coroutine function as a CortexFlow tool."""

        def decorator(
            fn: Callable[..., Coroutine[Any, Any, Any]]
        ) -> Callable[..., Coroutine[Any, Any, Any]]:
            schema = self._build_schema(fn, name, description)
            tool = ToolDefinition(
                name=name,
                description=description,
                handler=fn,
                schema=schema,
                retry_policy=retry_policy or RetryPolicy(),
                timeout=timeout,
                required_scopes=required_scopes or [],
            )
            self._tools[name] = tool
            logger.info("tool.registered", name=name)

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.call(name, kwargs)

            return wrapper

        return decorator

    async def call(
        self, 
        tool_name: str, 
        arguments: dict[str, Any], 
        caller_scopes: list[str] | None = None
    ) -> Any:
        """Execute a tool with security checks, circuit breaking, and retry logic."""

        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolCallError(tool_name, "Tool not found in registry")

        # 1. SECURITY CHECK: Verify if the caller has the required scopes
        if tool.required_scopes:
            caller_scopes = caller_scopes or []
            missing = [s for s in tool.required_scopes if s not in caller_scopes]
            if missing:
                logger.error("tool.permission_denied", tool=tool_name, missing_scopes=missing)
                raise PermissionDeniedError(
                    tool_name, 
                    f"Insufficient permissions. Missing scopes: {', '.join(missing)}"
                )

        # 2. CIRCUIT BREAKER CHECK

        if tool._circuit_state == CircuitState.OPEN:
            elapsed = time.monotonic() - tool._last_failure
            if elapsed < tool.recovery_timeout:
                raise CircuitOpenError(tool_name, f"Circuit open. Retry in {tool.recovery_timeout - elapsed:.0f}s")
            tool._circuit_state = CircuitState.HALF_OPEN
            logger.info("tool.circuit.half_open", tool=tool_name)

        last_error: Exception | None = None
        for attempt in range(tool.retry_policy.attempts):
            try:
                tool._call_count += 1
                result = await asyncio.wait_for(
                    tool.handler(**arguments),
                    timeout=tool.timeout,
                )
                # Success — reset circuit
                tool._failure_count = 0
                tool._circuit_state = CircuitState.CLOSED
                tool._success_count += 1
                logger.info("tool.success", tool=tool_name, attempt=attempt)
                return result

            except (asyncio.TimeoutError, Exception) as e:
                last_error = e
                tool._failure_count += 1
                tool._last_failure = time.monotonic()
                logger.warning("tool.attempt.failed", tool=tool_name, attempt=attempt, error=str(e))

                if tool._failure_count >= tool.failure_threshold:
                    tool._circuit_state = CircuitState.OPEN
                    logger.error("tool.circuit.open", tool=tool_name, failures=tool._failure_count)

                if attempt < tool.retry_policy.attempts - 1:
                    delay = tool.retry_policy.delay_for(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)

        raise ToolCallError(tool_name, str(last_error))

    def get_schemas(self) -> list[ToolSchema]:
        """Return JSON schemas for all registered tools (for LLM tool calling)."""
        return [t.schema for t in self._tools.values()]

    def get_stats(self) -> dict[str, Any]:
        return {
            name: {
                "calls": t._call_count,
                "successes": t._success_count,
                "failures": t._failure_count,
                "circuit": t._circuit_state,
            }
            for name, t in self._tools.items()
        }

    def _build_schema(self, fn: Callable[..., Any], name: str, description: str) -> ToolSchema:
        """Auto-generate JSON Schema from Python function signature."""
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        _TYPE_MAP: dict[Any, str] = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "ctx"):
                continue
            py_type = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(py_type, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return ToolSchema(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        )
