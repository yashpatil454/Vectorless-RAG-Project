from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    google_api_key: str = ""
    model_name: str = "gemini-1.5-flash"

    # Storage paths
    upload_dir: Path = Path("data/uploads")
    parsed_dir: Path = Path("data/parsed")
    index_store_dir: Path = Path("data/index_store")
    history_file: Path = Path("data/history.json")

    # Reasoning loop
    max_hops: int = 3

    # FastAPI ↔ Streamlit
    api_base_url: str = "http://localhost:8000"

    def ensure_dirs(self) -> None:
        """Create all required data directories if they do not exist."""
        for d in (self.upload_dir, self.parsed_dir, self.index_store_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
