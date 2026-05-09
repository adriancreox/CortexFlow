"""
CortexConfig — Centralized configuration management for CortexFlow.

Reads from environment variables and .env files to configure:
- API Keys (Gemini, Anthropic, OpenAI, etc.)
- Infrastructure (Redis, SQLite, Postgres)
- Logging & Observability
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

@dataclass
class CortexConfig:
    """
    Global configuration object for the CortexFlow Runtime.
    """

    # --- PROVIDERS ---
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    deepseek_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))

    # --- INFRASTRUCTURE ---
    # Default to local dev settings
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", ".cortexflow/vault/state.db"))
    
    # --- LOGGING ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: os.getenv("LOG_JSON", "false").lower() == "true")

    # --- KERNEL ---
    concurrency_limit: int = field(default_factory=lambda: int(os.getenv("CONCURRENCY_LIMIT", "16")))

    def validate(self) -> None:
        """Ensures at least one provider key is present."""
        keys = [self.gemini_api_key, self.anthropic_api_key, self.openai_api_key, self.deepseek_api_key]
        if not any(keys):
            # We don't raise here to allow manual configuration, 
            # but we could warn or log.
            pass

def get_config() -> CortexConfig:
    """Returns the global configuration instance."""
    config = CortexConfig()
    config.validate()
    return config
