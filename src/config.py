"""Configuration management."""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # Documents
    documents_path: str = Field(default="./data/documents")
    max_document_size_mb: int = Field(default=50)

    # Vector Store
    vector_dimension: int = Field(default=1536)
    similarity_threshold: float = Field(default=0.3)
    max_evidence_items: int = Field(default=8)

    # Pathway
    pathway_host: str = Field(default="localhost")
    pathway_port: int = Field(default=8765)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings