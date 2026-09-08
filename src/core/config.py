"""
Shim forwarding to backend.app.core.config for unified configuration.
"""
from backend.app.core.config import Settings, settings, PROJECT_ROOT

__all__ = ["Settings", "settings", "PROJECT_ROOT"]
