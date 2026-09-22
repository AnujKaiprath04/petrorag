"""
PetroRAG Backend Core Configuration Module
Centralized, type-safe settings management using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project root (up 3 levels from backend/app/core)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """
    PetroRAG Application Settings.
    Reads environment variables and .env file.
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Application
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "PetroRAG"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Storage Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "data" / "sample"
    GROUND_TRUTH_DIR: Path = PROJECT_ROOT / "data" / "ground_truth"

    # Relational Database
    DATABASE_URL: str = Field(
        default=f"sqlite:///{PROJECT_ROOT}/data/petrorag.db",
        description="SQLAlchemy database connection string. Uses SQLite local fallback or PostgreSQL."
    )
    DB_ECHO: bool = False

    # Qdrant Vector Database
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_STORAGE_PATH: Path = PROJECT_ROOT / "data" / "qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "petrorag_knowledge_base"
    QDRANT_PREFER_GRPC: bool = False

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_NORMALIZE: bool = True

    # OCR Configuration
    OCR_ENGINE: Literal["tesseract", "paddleocr"] = "tesseract"
    TESSERACT_CMD: str = "tesseract"
    OCR_CONFIDENCE_THRESHOLD: float = 60.0
    OCR_CHAR_DENSITY_THRESHOLD: float = 50.0

    # Chunking Configuration
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64
    MIN_CHUNK_LENGTH: int = 30
    MAX_CHUNK_LENGTH: int = 2000

    # Part 2 Hybrid Retrieval & BM25 Configuration
    BM25_INDEX_PATH: Path = PROJECT_ROOT / "data" / "processed" / "bm25_index.json"
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.55
    SPARSE_WEIGHT: float = 0.45
    RETRIEVAL_CANDIDATE_POOL_SIZE: int = 30
    DEFAULT_TOP_K: int = 5

    # Part 2 Reranker Configuration
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16

    # Part 2 Context & Prompt Configuration
    CONTEXT_TOKEN_BUDGET: int = 2048
    PRESERVE_TECHNICAL_PARAMS: bool = True
    COMPRESSION_ENABLED: bool = True
    COMPRESSION_TARGET_RATIO: float = 0.65
    COMPRESSION_MIN_SENTENCE_SCORE: float = 0.15

    # Part 2 LLM Provider Configuration
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Part 2 Grounding & Abstention Thresholds
    GROUNDING_THRESHOLD: float = 0.70
    ABSTENTION_RETRIEVAL_THRESHOLD: float = 0.30

    def ensure_directories(self) -> None:
        """Create required data and storage directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.SAMPLE_DATA_DIR,
            self.GROUND_TRUTH_DIR,
            self.QDRANT_STORAGE_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_directories()
