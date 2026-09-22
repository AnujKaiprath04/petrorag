"""
PetroRAG Query Validation & Security Guard Layer (Module 2.2)
Validates input size, malformed characters, system command execution attempts,
and indirect/direct prompt injections while preserving Oil & Gas technical terminology.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.exceptions import (
    QueryValidationError,
    PromptInjectionError,
)
from src.core.logging import logger


class QueryValidationResult(BaseModel):
    """Structured report returned by QueryValidator."""
    is_valid: bool
    sanitized_query: str
    original_query: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    detected_flags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Regex patterns for adversarial instruction injection
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|prompts|rules|protocols|safeguards|constraints)",
    r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions|prompts|rules|protocols)",
    r"reveal\s+(the\s+)?(system\s+prompt|developer\s+mode|internal\s+instructions|credentials|api\s*keys?)",
    r"(show|print|output|display|dump)\s+(the\s+)?(full\s+)?(system\s+prompt|developer\s+instructions|system\s+instructions|system\s+prompt\s+instructions|hidden\s+credentials|api\s*keys?)",
    r"you\s+are\s+now\s+(in\s+developer\s+mode|DAN|unrestricted|an\s+unfiltered\s+AI)",
    r"(bypass|override|disable)\s+(all\s+)?(safety|guardrails|filters|content\s+policies|equipment\s+safety\s+limits)",
    r"(system\s*:\s*system\s*prompt)",
    r"<\s*script\s*>",
    r"jailbreak",
]

# Regex patterns for malicious system or execution commands
SYSTEM_COMMAND_PATTERNS = [
    r"(\brm\s+-rf\b|\bdel\s+/[sS]\b)",
    r"(\bdrop\s+table\b|\bunion\s+select\b|\btruncate\s+table\b)",
    r"(\beval\s*\(|\bexec\s*\(|\bos\.system\s*\(|\bsubprocess\.Popen\b)",
    r"(\bformat\s+[c-zC-Z]:\b)",
    r"(\bcurl\b|\bwget\b)\s+https?://",
    r"\b(execute\s+bash\s+script|bash\s+script|sh\s+script|powershell\s+script)\b",
]

# Common invisible and zero-width unicode characters to strip
ZERO_WIDTH_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\ufeff",  # zero-width no-break space (BOM)
]


class QueryValidator:
    """
    Validates, sanitizes, and normalizes user queries for the PetroRAG engine.
    Ensures security against prompt injection and malicious commands while
    preserving technical tags, numbers, units, and equipment IDs.
    """

    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 1000,
        max_words: int = 200,
        strict_injection_block: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_words = max_words
        self.strict_injection_block = strict_injection_block

        # Precompile regexes for optimal latency
        self._injection_regexes = [
            re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS
        ]
        self._sys_cmd_regexes = [
            re.compile(p, re.IGNORECASE) for p in SYSTEM_COMMAND_PATTERNS
        ]

    def normalize_text(self, text: str) -> str:
        """
        Normalize unicode, strip hidden zero-width artifacts, and collapse whitespace
        while strictly preserving engineering notation, units, and equipment tags.
        """
        if not text:
            return ""

        # Normalize unicode to NFKC standard
        normalized = unicodedata.normalize("NFKC", text)

        # Remove zero-width characters
        for zw in ZERO_WIDTH_CHARS:
            normalized = normalized.replace(zw, "")

        # Replace smart quotes with standard ASCII quotes
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")

        # Strip unprintable control characters except standard whitespace (\n, \t, \r)
        cleaned_chars = []
        for ch in normalized:
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in "\n\t\r":
                continue
            cleaned_chars.append(ch)
        normalized = "".join(cleaned_chars)

        # Collapse excessive whitespace and newlines, preserving single spaces
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"[\r\n]+", "\n", normalized)
        return normalized.strip()

    def validate(self, query: Optional[str]) -> QueryValidationResult:
        """
        Execute comprehensive validation on query and return structured report.
        """
        if query is None:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                original_query="",
                error_code="EMPTY_QUERY",
                error_message="Query cannot be None.",
                detected_flags=["empty_input"]
            )

        sanitized = self.normalize_text(query)

        # 1. Check empty or whitespace only
        if not sanitized:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                original_query=query,
                error_code="EMPTY_QUERY",
                error_message="Query cannot be empty or contain only whitespace.",
                detected_flags=["empty_input"]
            )

        # 2. Check null bytes or severe encoding artifacts
        if "\x00" in query:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                original_query=query,
                error_code="MALFORMED_INPUT",
                error_message="Query contains invalid null bytes or malformed binary encoding.",
                detected_flags=["null_bytes"]
            )

        # 3. Length checks
        if len(sanitized) < self.min_length:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query=sanitized,
                original_query=query,
                error_code="QUERY_TOO_SHORT",
                error_message=f"Query is too short ({len(sanitized)} chars). Minimum length is {self.min_length}.",
                detected_flags=["too_short"],
                metadata={"length": len(sanitized)}
            )

        if len(sanitized) > self.max_length:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query=sanitized[:self.max_length],
                original_query=query,
                error_code="QUERY_TOO_LONG",
                error_message=f"Query exceeds maximum character length of {self.max_length} ({len(sanitized)} chars).",
                detected_flags=["too_long"],
                metadata={"length": len(sanitized), "max_allowed": self.max_length}
            )

        words = sanitized.split()
        if len(words) > self.max_words:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query=sanitized,
                original_query=query,
                error_code="QUERY_TOO_LONG",
                error_message=f"Query word count exceeds threshold ({len(words)} > {self.max_words}).",
                detected_flags=["too_many_words"],
                metadata={"word_count": len(words), "max_allowed": self.max_words}
            )

        detected_flags = []

        # 4. Check for system commands
        for pattern in self._sys_cmd_regexes:
            if pattern.search(sanitized):
                detected_flags.append("system_command")
                return QueryValidationResult(
                    is_valid=False,
                    sanitized_query=sanitized,
                    original_query=query,
                    error_code="MALICIOUS_INPUT",
                    error_message="Query contains unauthorized system or command execution syntax.",
                    detected_flags=detected_flags,
                    metadata={"matched_pattern": pattern.pattern}
                )

        # 5. Check for prompt injection
        for pattern in self._injection_regexes:
            if pattern.search(sanitized):
                detected_flags.append("prompt_injection")
                if self.strict_injection_block:
                    return QueryValidationResult(
                        is_valid=False,
                        sanitized_query=sanitized,
                        original_query=query,
                        error_code="PROMPT_INJECTION_DETECTED",
                        error_message="Query matches prompt injection or instruction override signatures.",
                        detected_flags=detected_flags,
                        metadata={"matched_pattern": pattern.pattern}
                    )

        # Valid Query
        return QueryValidationResult(
            is_valid=True,
            sanitized_query=sanitized,
            original_query=query,
            detected_flags=detected_flags,
            metadata={
                "char_length": len(sanitized),
                "word_count": len(words),
            }
        )

    def validate_or_throw(self, query: Optional[str]) -> str:
        """
        Validate query and return sanitized string. Raises QueryValidationError if invalid.
        """
        res = self.validate(query)
        if not res.is_valid:
            if res.error_code == "PROMPT_INJECTION_DETECTED":
                raise PromptInjectionError(res.error_message or "Prompt injection detected.")
            raise QueryValidationError(
                message=res.error_message or "Query validation failed.",
                error_code=res.error_code or "VALIDATION_FAILED"
            )
        return res.sanitized_query
