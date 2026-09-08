"""
Shim forwarding to backend.app.core.logging for unified logging.
"""
from backend.app.core.logging import (
    setup_logger,
    logger,
    log_stage_event,
    RedactingFormatter,
    REDACTED_KEYS
)

__all__ = ["setup_logger", "logger", "log_stage_event", "RedactingFormatter", "REDACTED_KEYS"]
