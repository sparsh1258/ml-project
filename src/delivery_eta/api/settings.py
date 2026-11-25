"""Settings, read from environment variables that start with ETA_ (see .env.example)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ETA_", extra="ignore")

    model_dir: Path = Path("artifacts/model")
    # When set, predictions and metrics need one of these in the X-API-Key header.
    api_keys: Annotated[frozenset[str], NoDecode] = frozenset()
    # Other websites allowed to call the API from a browser. Empty: only our own page.
    cors_origins: Annotated[frozenset[str], NoDecode] = frozenset()
    docs: bool = True
    metrics: bool = True
    max_batch: int = 500
    max_body_bytes: int = 1_000_000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("api_keys")
    @classmethod
    def _long_enough(cls, keys: frozenset[str]) -> frozenset[str]:
        if any(len(key) < 16 for key in keys):
            raise ValueError("every API key must be at least 16 characters")
        return keys
