"""
Ollama Provider — Local model adapter for air-gapped / dev environments.
Supports any model running via Ollama (Llama 3, Mistral, Qwen, Gemma, etc.)
"""

from __future__ import annotations

from typing import Any

import structlog

from cortexflow.providers.base import (
    Completion,
    CompletionRequest,
    LLMProvider,
    ProviderError,
    ToolCall,
)

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama (local) provider adapter.

    Requires Ollama running locally: https://ollama.com
    Requires: pip install cortexflow[ollama]

    Usage:
        provider = OllamaProvider(model="llama3")
        # Or with custom host:
        provider = OllamaProvider(model="mistral", host="http://localhost:11434")
    """

    DEFAULT_MODEL = "llama3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
    ) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "Ollama provider requires the ollama package. "
                "Install it with: pip install cortexflow[ollama]"
            )
        self._model = model
        self._host = host
        self._ollama = ollama

    async def complete(self, request: CompletionRequest) -> Completion:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        try:
            import httpx
            async with httpx.AsyncClient(base_url=self._host, timeout=request.timeout_seconds) as client:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": request.model or self._model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": request.temperature},
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            raise ProviderError(str(e), provider="ollama") from e

        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval = data.get("prompt_eval_count", 0)

        return Completion(
            content=content,
            tool_calls=[],
            input_tokens=prompt_eval,
            output_tokens=eval_count,
            model=data.get("model", self._model),
            stop_reason="stop",
            raw=data,
        )

    async def health(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(base_url=self._host, timeout=3.0) as client:
                r = await client.get("/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama-{self._model}"
