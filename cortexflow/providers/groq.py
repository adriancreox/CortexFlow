"""
Groq Provider — CortexFlow adapter for ultra-fast Groq LPU inference.
"""

from __future__ import annotations

from typing import Any

from cortexflow.providers.openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """
    Groq provider adapter.
    
    Optimized for ultra-low latency inference using Groq LPUs.
    Uses the OpenAI-compatible API.
    
    Usage:
        provider = GroqProvider(
            model="llama-3.3-70b-versatile",
            api_key="gsk_...",
        )
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return f"groq-{self._model}"
