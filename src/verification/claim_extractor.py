"""
PetroRAG Atomic Claim Extraction Engine (Module 2.21)
Deconstructs complex technical responses into atomic, independently verifiable
factual propositions for claim-level NLI grounding evaluation.
"""

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.generation.context_compressor import (
    PARAM_PATTERN,
    SAFETY_PATTERN,
    STANDARD_PATTERN,
    TAG_PATTERN,
    split_sentences,
)


class AtomicClaim(BaseModel):
    """An individual atomic factual claim extracted from generated text."""
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., description="Deconstructed factual proposition")
    original_sentence: str = Field(..., description="Source sentence before decomposition")
    entities: List[str] = Field(default_factory=list, description="Equipment tags and standards")
    parameters: List[str] = Field(default_factory=list, description="Technical quantities and measurements")
    is_safety_critical: bool = Field(default=False)


# Conversational / administrative meta-prefixes to strip from claims
_META_PREFIXES = [
    r"^(?:based on(?: the)? (?:provided )?(?:documents?|manuals?|text|context),?)\s*",
    r"^(?:according to(?: the)? (?:provided )?(?:documents?|manuals?|text|context),?)\s*",
    r"^(?:as (?:stated|mentioned|noted) in (?:the )?(?:documents?|manuals?|text),?)\s*",
    r"^(?:please note that,?)\s*",
    r"^(?:it is (?:important|mandatory|recommended) to note that,?)\s*",
    r"^(?:in summary,?)\s*",
    r"^(?:in conclusion,?)\s*",
    r"^(?:overall,?)\s*",
    r"^\*\*Direct Answer\*\*:\s*",
    r"^\*\*Supporting Evidence\*\*:\s*",
    r"^\*\*Engineering Analysis\*\*:\s*",
    r"^\*\*Limitations\*\*:\s*",
    r"^\*\*Sources\*\*:\s*",
]


class ClaimExtractor:
    """
    Deconstructs generated answers into atomic factual claims.
    Handles coordinate conjunctions, subordinate clauses, bullet points,
    and technical parameter association.
    """

    def __init__(self, min_claim_length: int = 15):
        self.min_claim_length = min_claim_length

    def _strip_meta_prefixes(self, text: str) -> str:
        """Removes conversational framing from sentences."""
        cleaned = text.strip()
        for prefix in _META_PREFIXES:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()
        # Also remove trailing citation markers from claim text (e.g. "[1]", "[DOC-01]")
        cleaned = re.sub(r"\[[^\]]+\]", "", cleaned).strip()
        return cleaned

    def _split_compound_sentence(self, sentence: str) -> List[str]:
        """
        Splits a compound sentence into constituent clauses using conjunctions
        and semicolons, ensuring each clause carries sufficient technical context.
        """
        cleaned = self._strip_meta_prefixes(sentence)
        if len(cleaned) < self.min_claim_length:
            return []

        # Split on semicolons or bullet points
        raw_clauses = re.split(r";\s*|\s*•\s*|\s*-\s+", cleaned)
        sub_clauses: List[str] = []

        for clause in raw_clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Split on strong coordinate conjunctions joining independent clauses
            # e.g., ", and ", ", while ", ", whereas "
            conjunction_parts = re.split(r",\s+(?:and|while|whereas|but)\s+", clause, flags=re.IGNORECASE)
            for part in conjunction_parts:
                part = part.strip()
                if len(part) >= self.min_claim_length:
                    sub_clauses.append(part)

        # Fallback to cleaned sentence if no valid sub-clauses were derived
        if not sub_clauses and len(cleaned) >= self.min_claim_length:
            sub_clauses.append(cleaned)

        return sub_clauses

    def extract_claims(self, text: str) -> List[AtomicClaim]:
        """
        Decomposes answer text into a list of typed AtomicClaims with audited entities.
        """
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        claims: List[AtomicClaim] = []
        claim_counter = 1

        for sent in sentences:
            # Skip pure markdown headers or section markers
            if sent.startswith("#") or sent.startswith("---"):
                continue

            clauses = self._split_compound_sentence(sent)
            for clause in clauses:
                # Extract domain entities and parameters
                tags = TAG_PATTERN.findall(clause)
                standards = STANDARD_PATTERN.findall(clause)
                params = PARAM_PATTERN.findall(clause)
                safety = bool(SAFETY_PATTERN.findall(clause))

                all_entities = list(set(tags + standards))
                all_params = list(set(params))

                # Capitalize first character and ensure punctuation
                formatted_claim = clause[0].upper() + clause[1:] if len(clause) > 1 else clause.upper()
                if not formatted_claim.endswith("."):
                    formatted_claim += "."

                claims.append(AtomicClaim(
                    claim_id=f"clm-{claim_counter:03d}",
                    claim_text=formatted_claim,
                    original_sentence=sent.strip(),
                    entities=all_entities,
                    parameters=all_params,
                    is_safety_critical=safety
                ))
                claim_counter += 1

        logger.info(f"Extracted {len(claims)} atomic claims from text ({len(sentences)} sentences).")
        return claims
