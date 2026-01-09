# src/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import sys
from loguru import logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    model_name: str = "gpt-4-turbo-preview"

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # FAISS Configuration
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10

    # Application Settings
    log_level: str = "INFO"
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Paths
    data_dir: Path = Path("/app/data")
    raw_dir: Path = Path("/app/data/raw")
    processed_dir: Path = Path("/app/data/processed")
    indices_dir: Path = Path("/app/data/indices")
    logs_dir: Path = Path("/app/logs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


def setup_logging(log_level: str = "INFO"):
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    logger.add(
        "/app/logs/rag_{time}.log",
        rotation="500 MB",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


# Initialize logging
settings = get_settings()
setup_logging(settings.log_level)
