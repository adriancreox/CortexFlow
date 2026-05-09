"""
Unit tests for new providers — DeepSeek and Groq.
"""

from __future__ import annotations

import pytest

from cortexflow.providers.deepseek import DeepSeekProvider
from cortexflow.providers.groq import GroqProvider


def test_deepseek_initialization() -> None:
    provider = DeepSeekProvider(api_key="sk-test")
    assert provider.name == "deepseek-deepseek-chat"
    assert str(provider._client.base_url).rstrip("/") == "https://api.deepseek.com"


def test_groq_initialization() -> None:
    provider = GroqProvider(api_key="gsk-test")
    assert provider.name == "groq-llama-3.3-70b-versatile"
    assert str(provider._client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"


def test_provider_agnosticism() -> None:
    # Verify both can be instantiated and follow LLMProvider interface
    providers = [
        DeepSeekProvider(api_key="test"),
        GroqProvider(api_key="test")
    ]
    for p in providers:
        assert hasattr(p, "complete")
        assert hasattr(p, "health")
        assert p.name.startswith(("deepseek-", "groq-"))
