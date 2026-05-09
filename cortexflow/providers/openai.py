"""
OpenAI Provider — CortexFlow adapter for OpenAI GPT models.

Normalizes OpenAI's response format into CortexFlow's unified
Completion schema. Handles tool_calls extraction transparently.
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


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider adapter.

    Supports: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    Requires: openai>=1.30 (pip install cortexflow[openai])

    Usage:
        provider = OpenAIProvider(
            model="gpt-4o",
            api_key="sk-...",  # or set OPENAI_API_KEY env var
        )
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install it with: pip install cortexflow[openai]"
            )

        self._model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    async def complete(self, request: CompletionRequest) -> Completion:
        messages = self._convert_messages(request.messages)
        tools = self._convert_tools(request.tools) if request.tools else None

        try:
            response = await self._client.chat.completions.create(
                model=request.model or self._model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=request.timeout_seconds,
            )
        except Exception as e:
            raise ProviderError(str(e), provider="openai") from e

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json
                tool_calls.append(ToolCall(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                ))

        logger.debug(
            "openai.complete",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return Completion(
            content=msg.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            stop_reason=choice.finish_reason or "stop",
            raw=response.model_dump(),
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"openai-{self._model}"

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.name:
                entry["name"] = m.name
            result.append(entry)
        return result

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
