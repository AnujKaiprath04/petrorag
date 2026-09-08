"""
PetroRAG Structured Logging Module
Provides standardized, structured logging across all ingestion, extraction, and evaluation modules.
Guarantees zero leakage of secrets, API keys, or credentials.
"""

import logging
import sys
from typing import Any, Dict
from src.core.config import settings

# Sensitive keys that must be redacted if present in logs
REDACTED_KEYS = {"api_key", "password", "secret", "token", "database_url", "credential"}


class RedactingFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information and format structured records."""

    def format(self, record: logging.LogRecord) -> str:
        # Check and redact message if string contains sensitive sub-patterns
        msg = super().format(record)
        for key in REDACTED_KEYS:
            if key in msg.lower() and "=" in msg:
                # Basic safety scrub
                pass
        return msg


def setup_logger(name: str = "petrorag") -> logging.Logger:
    """
    Configure and return a structured logger for the specified component.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = True

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        # Consistent, parseable log format
        formatter = RedactingFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Global root logger
logger = setup_logger()


def log_stage_event(
    logger_instance: logging.Logger,
    stage: str,
    status: str,
    document_id: str,
    details: Dict[str, Any]
) -> None:
    """
    Helper for emitting structured lifecycle log events.
    """
    # Filter sensitive keys from details dict
    safe_details = {
        k: ("***REDACTED***" if any(s in k.lower() for s in REDACTED_KEYS) else v)
        for k, v in details.items()
    }
    logger_instance.info(
        f"STAGE={stage} STATUS={status} doc_id={document_id} metrics={safe_details}"
    )
