"""Configuration management for Clinical Evidence Copilot."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


"""Configuration management for Clinical Evidence Copilot."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Document Processing
    documents_path: str = Field(default="./data/documents", description="Path to documents folder")
    max_document_size_mb: int = Field(default=50, description="Maximum document size in MB")
    supported_formats: list[str] = Field(default=["pdf", "html", "xml"], description="Supported document formats")
    
    # Vector Store Configuration
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    vector_dimension: int = Field(default=1536, description="Vector dimension")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Query Processing
    max_query_length: int = Field(default=1000, description="Maximum query length")
    response_timeout_seconds: int = Field(default=30, description="Response timeout in seconds")
    max_evidence_items: int = Field(default=10, description="Maximum evidence items to return")
    
    # Medical Domain Settings
    evidence_hierarchy: dict[str, int] = Field(
        default={
            "systematic_review": 1,
            "meta_analysis": 2,
            "randomized_controlled_trial": 3,
            "cohort_study": 4,
            "case_control_study": 5,
            "case_series": 6,
            "case_report": 7,
            "expert_opinion": 8
        },
        description="Evidence hierarchy ranking"
    )
    
    # Security and Compliance
    enable_hipaa_logging: bool = Field(default=True, description="Enable HIPAA logging")
    log_level: str = Field(default="INFO", description="Log level")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings