"""
PetroRAG Domain-Specific Exceptions
Structured exception hierarchy for query validation, retrieval, reranking, and verification.
"""

from typing import Optional


class PetroRAGException(Exception):
    """Base exception for all PetroRAG system components."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PETRORAG_ERROR"


class QueryValidationError(PetroRAGException):
    """Raised when an incoming query fails security or structural validation."""
    def __init__(self, message: str, error_code: str = "QUERY_VALIDATION_ERROR"):
        super().__init__(message, error_code)


class PromptInjectionError(QueryValidationError):
    """Raised when an input exhibits adversarial prompt injection or instruction override patterns."""
    def __init__(self, message: str = "Prompt injection attempt detected."):
        super().__init__(message, error_code="PROMPT_INJECTION_DETECTED")


class RetrievalException(PetroRAGException):
    """Raised when an error occurs during dense, sparse, or hybrid retrieval."""
    def __init__(self, message: str, error_code: str = "RETRIEVAL_ERROR"):
        super().__init__(message, error_code)


class LLMGenerationError(PetroRAGException):
    """Raised when an LLM provider fails to generate a response."""
    def __init__(self, message: str, error_code: str = "LLM_GENERATION_ERROR"):
        super().__init__(message, error_code)


class AbstentionTriggered(PetroRAGException):
    """Raised or handled when the system determines an answer cannot be reliably grounded."""
    def __init__(self, reason: str, barrier: str = "GROUNDING"):
        super().__init__(f"Abstention triggered by {barrier}: {reason}", error_code=f"ABSTENTION_{barrier}")
        self.reason = reason
        self.barrier = barrier
