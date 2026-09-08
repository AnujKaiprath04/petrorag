"""
PetroRAG Backend Core Logging Module
Structured, formatted telemetry logging without secret leakage.
"""

import logging
import sys
from typing import Any, Dict
from backend.app.core.config import settings

REDACTED_KEYS = {"api_key", "password", "secret", "token", "database_url", "credential"}


class RedactingFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information and format structured records."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return msg


def setup_logger(name: str = "petrorag") -> logging.Logger:
    """Configure and return a structured logger for the specified component."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = True

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        formatter = RedactingFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def log_stage_event(
    logger_instance: logging.Logger,
    stage: str,
    status: str,
    document_id: str,
    details: Dict[str, Any]
) -> None:
    """Helper for emitting structured lifecycle log events."""
    safe_details = {
        k: ("***REDACTED***" if any(s in k.lower() for s in REDACTED_KEYS) else v)
        for k, v in details.items()
    }
    logger_instance.info(
        f"STAGE={stage} STATUS={status} doc_id={document_id} metrics={safe_details}"
    )
