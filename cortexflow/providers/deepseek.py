"""
DeepSeek Provider — CortexFlow adapter for DeepSeek models.
"""

from __future__ import annotations

from typing import Any

from cortexflow.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """
    DeepSeek provider adapter.
    
    Compatible with DeepSeek-V3 and DeepSeek-R1.
    Uses the OpenAI-compatible API.
    
    Usage:
        provider = DeepSeekProvider(
            model="deepseek-chat",
            api_key="sk-...",
        )
    """

    DEFAULT_MODEL = "deepseek-chat"
    BASE_URL = "https://api.deepseek.com"

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
        return f"deepseek-{self._model}"
