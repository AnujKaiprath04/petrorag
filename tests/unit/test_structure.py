"""
Unit Test: Module 2 Project Structure & Architecture Verification
Verifies directory layout, package initialization, module import paths, and FastAPI health check.
"""

from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_package_structure_exists():
    """Verify that all modular directories specified in Module 2 exist."""
    required_dirs = [
        "backend/app/api",
        "backend/app/core",
        "backend/app/models",
        "backend/app/schemas",
        "backend/app/services",
        "ingestion/loaders",
        "ingestion/ocr",
        "ingestion/preprocessing",
        "ingestion/metadata",
        "ingestion/chunking",
        "ingestion/pipeline",
        "embeddings",
        "vectorstore",
        "data/raw",
        "data/processed",
        "data/sample",
        "data/ground_truth",
        "scripts",
        "docs",
        "docker",
        "tests",
    ]
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        assert full_path.exists(), f"Missing expected directory: {dir_path}"
        assert full_path.is_dir(), f"Expected a directory: {dir_path}"


def test_package_inits():
    """Verify that Python packages contain an __init__.py file."""
    package_dirs = [
        "backend",
        "backend/app",
        "backend/app/api",
        "backend/app/core",
        "backend/app/models",
        "backend/app/schemas",
        "backend/app/services",
        "ingestion",
        "ingestion/loaders",
        "ingestion/ocr",
        "ingestion/preprocessing",
        "ingestion/metadata",
        "ingestion/chunking",
        "ingestion/pipeline",
        "embeddings",
        "vectorstore",
    ]
    for pkg in package_dirs:
        init_file = PROJECT_ROOT / pkg / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in package: {pkg}"


def test_module_imports():
    """Verify that key project modules import cleanly."""
    import backend.app.core.config as cfg
    import backend.app.core.logging as log
    import src.core.config as src_cfg
    import src.core.logging as src_log

    assert cfg.settings.APP_NAME == "PetroRAG"
    assert src_cfg.settings.APP_NAME == "PetroRAG"
    assert log.logger is not None
    assert src_log.logger is not None


def test_fastapi_health_endpoint():
    """Verify that FastAPI application initializes and health check responds 200 OK."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "PetroRAG"
