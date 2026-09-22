"""
PetroRAG Core Package
Exports configuration, logging, interfaces, and shared data transfer models.
"""

from src.core.config import Settings, settings, PROJECT_ROOT
from src.core.logging import setup_logger, logger, log_stage_event
from src.core.interfaces import (
    RetrievalChannel,
    RetrievedChunk,
    RerankedChunk,
    BuiltContext,
    PromptBundle,
    GenerationConfig,
    LLMResponse,
    Citation,
    ClaimSupportStatus,
    ClaimVerification,
    GroundingResult,
    BaseRetriever,
    BaseReranker,
    BaseContextBuilder,
    BasePromptBuilder,
    BaseLLMProvider,
    BaseCitationEngine,
    BaseGroundingEvaluator,
)

__all__ = [
    "Settings",
    "settings",
    "PROJECT_ROOT",
    "setup_logger",
    "logger",
    "log_stage_event",
    "RetrievalChannel",
    "RetrievedChunk",
    "RerankedChunk",
    "BuiltContext",
    "PromptBundle",
    "GenerationConfig",
    "LLMResponse",
    "Citation",
    "ClaimSupportStatus",
    "ClaimVerification",
    "GroundingResult",
    "BaseRetriever",
    "BaseReranker",
    "BaseContextBuilder",
    "BasePromptBuilder",
    "BaseLLMProvider",
    "BaseCitationEngine",
    "BaseGroundingEvaluator",
]
