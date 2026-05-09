"""
Anthropic Provider — CortexFlow adapter for Claude models.
"""

from __future__ import annotations

from typing import Any

import structlog

from cortexflow.providers.base import (
    Completion,
    CompletionRequest,
    LLMProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolSchema,
)

logger = structlog.get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude adapter.

    Supports: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
    Requires: anthropic>=0.28 (pip install cortexflow[anthropic])

    Usage:
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-...",  # or ANTHROPIC_API_KEY
        )
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install it with: pip install cortexflow[anthropic]"
            )
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key, **kwargs)

    async def complete(self, request: CompletionRequest) -> Completion:
        system_prompt, messages = self._split_messages(request.messages)
        tools = self._convert_tools(request.tools) if request.tools else []

        try:
            response = await self._client.messages.create(
                model=request.model or self._model,
                system=system_prompt,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout_seconds,
            )
        except Exception as e:
            raise ProviderError(str(e), provider="anthropic") from e

        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        import json
        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    call_id=block.id,
                    tool_name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        stop_map = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
        }

        return Completion(
            content=content_text,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=stop_map.get(response.stop_reason or "end_turn", "stop"),
            raw={},
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"anthropic-{self._model}"

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        result = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                result.append({"role": m.role, "content": m.content})
        return system, result

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]
