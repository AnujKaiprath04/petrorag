"""
Unit Test: Module 1 Environment & Configuration Verification
Tests settings loading, directory creation, logger operation, and dependency availability.
"""

import logging
from src.core.config import settings
from src.core.logging import setup_logger, log_stage_event


def test_settings_load():
    """Verify that settings are loaded with correct types and default values."""
    assert settings.APP_NAME == "PetroRAG"
    assert settings.EMBEDDING_MODEL_NAME is not None
    assert settings.DEFAULT_CHUNK_SIZE > 0
    assert settings.DEFAULT_CHUNK_OVERLAP >= 0
    assert settings.QDRANT_COLLECTION_NAME == "petrorag_knowledge_base"


def test_directories_exist():
    """Verify that required data directories are auto-created by settings."""
    assert settings.DATA_DIR.exists()
    assert settings.RAW_DATA_DIR.exists()
    assert settings.PROCESSED_DATA_DIR.exists()
    assert settings.SAMPLE_DATA_DIR.exists()
    assert settings.GROUND_TRUTH_DIR.exists()


def test_structured_logger(caplog):
    """Verify that the logger outputs formatted logs without errors."""
    test_logger = setup_logger("test_module1")
    with caplog.at_level(logging.INFO):
        log_stage_event(
            logger_instance=test_logger,
            stage="ENVIRONMENT_INIT",
            status="SUCCESS",
            document_id="DOC-INIT-001",
            details={"database": "configured", "api_key": "secret12345"}
        )
    # Ensure sensitive keys are redacted
    assert "***REDACTED***" in caplog.text
    assert "secret12345" not in caplog.text
    assert "DOC-INIT-001" in caplog.text


def test_qdrant_client_availability():
    """Verify that qdrant_client can be imported and initialized in local memory mode."""
    from qdrant_client import QdrantClient
    client = QdrantClient(location=":memory:")
    collections = client.get_collections()
    assert collections is not None


def test_document_libraries_import():
    """Verify that document processing libraries import properly."""
    import fitz  # PyMuPDF
    import docx
    import pdfplumber
    import sqlalchemy
    assert fitz.__version__ is not None
    assert docx.__version__ is not None
    assert pdfplumber.__version__ is not None
    assert sqlalchemy.__version__ is not None
