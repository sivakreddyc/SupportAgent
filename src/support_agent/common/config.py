"""Runtime configuration.

Precedence (highest first): explicit kwargs > env vars `SA_<FIELD>` > config/settings.yaml > defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_QUEUE_MAX_SIZE = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            self._data = {str(k).lower(): v for k, v in raw.items()}

    def get_field_value(self, field, field_name):  # pragma: no cover - unused by pydantic for dict sources
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if k in self.settings_cls.model_fields}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SA_", extra="ignore", case_sensitive=False)

    # queue / workers
    queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE
    worker_count: int = 4
    max_retries: int = 3
    ticket_timeout_seconds: float = 60.0
    retry_backoff_base_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0
    shutdown_grace_seconds: float = 10.0

    # scheduler / poller
    poll_interval_seconds: float = 15.0
    poll_jitter_ratio: float = 0.1
    review_dispatch_interval_seconds: float = 3.0

    # browser / ticket UI
    browser_headless: bool = True
    browser_page_pool: int = 3
    browser_timeout_ms: int = 10_000
    ticket_ui_url: str = "http://localhost:8001"
    ticket_ui_user: str = "agent"
    ticket_ui_password: str = "agent"
    artifacts_dir: str = "artifacts"

    # user info service
    user_service_url: str = "http://localhost:8002"
    user_service_api_key: str = "dev-key"
    user_service_timeout_seconds: float = 2.0
    user_cache_ttl_seconds: float = 60.0

    # llm
    llm_provider: Literal["mock", "anthropic"] = "mock"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 4096
    llm_server_fallback: bool = True
    mock_llm_latency_ms: int = 50

    # storage / observability
    db_path: str = "data/support_agent.db"
    log_level: str = "INFO"
    log_file: str = "logs/agent.jsonl"
    metrics_sample_interval_seconds: float = 5.0
    ops_api_host: str = "127.0.0.1"
    ops_api_port: int = 8003

    # policy
    policy_file: str = str(PROJECT_ROOT / "config" / "policy.yaml")

    @field_validator("queue_max_size", mode="before")
    @classmethod
    def _default_queue_size(cls, v: Any) -> int:
        """Unset / empty / non-positive means the default (1000)."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return DEFAULT_QUEUE_MAX_SIZE
        try:
            n = int(v)
        except (TypeError, ValueError):
            return DEFAULT_QUEUE_MAX_SIZE
        return n if n > 0 else DEFAULT_QUEUE_MAX_SIZE

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            _YamlSource(settings_cls, DEFAULT_SETTINGS_FILE),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
