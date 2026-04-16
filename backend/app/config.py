from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Multilingual Laptop Review Intelligence System"
    app_version: str = "1.0.0"
    raw_dataset_path: Path = Field(
        default=ROOT_DIR / "data" / "raw" / "final_dataset.csv",
        alias="RAW_DATASET_PATH",
    )
    artifact_dir: Path = Field(
        default=ROOT_DIR / "artifacts",
        alias="ARTIFACT_DIR",
    )
    powerbi_export_dir: Path = Field(
        default=ROOT_DIR / "powerbi",
        alias="POWERBI_EXPORT_DIR",
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    enable_fake_mode: bool = Field(default=False, alias="ENABLE_FAKE_MODE")
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
