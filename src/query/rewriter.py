"""
PetroRAG Query Rewriting Layer (Module 2.5)
Transforms natural-language conversational queries into concise, high-density,
retrieval-friendly queries for vector and lexical retrieval engines without
hallucinating unsupported entities.
"""

from abc import ABC, abstractmethod
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from src.query.intent import QueryIntent
from src.query.entities import QueryEntities, RuleBasedEntityExtractor


class RewrittenQueries(BaseModel):
    """Container holding original query and generated retrieval formulations."""
    original_query: str
    primary_rewritten: str
    rewritten_candidates: List[str] = Field(default_factory=list)
    strategy_used: str = "rule_based"


class BaseQueryRewriter(ABC):
    """Abstract interface for query rewriting."""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        entities: Optional[QueryEntities] = None
    ) -> RewrittenQueries:
        """Transform natural language question into retrieval-friendly formulations."""
        pass


# Conversational filler prefixes to strip
CONVERSATIONAL_STOP_PREFIXES = [
    r"^(please\s+)?(tell\s+me|explain\s+to\s+me|can\s+you\s+tell\s+me|could\s+you\s+explain|i\s+want\s+to\s+know|i\s+would\s+like\s+to\s+understand)\s+",
    r"^(do\s+you\s+know|what\s+is\s+the\s+reason\s+(why|for))\s+",
    r"^(kindly\s+provide|provide\s+information\s+on)\s+",
    r"^(in\s+your\s+opinion|according\s+to\s+documentation)\s+",
]

# Wh-question transform patterns
WH_TRANSFORMS = [
    (r"\bwhy\s+(is|are|did|does)\s+", ""),
    (r"\bhow\s+(to|do\s+i|can\s+we)\s+", ""),
    (r"\bwhat\s+(causes|caused|is\s+the\s+cause\s+of)\s+", ""),
    (r"\bwhat\s+(is|are)\s+the\s+(procedure\s+for|steps\s+for|guidelines\s+for)\s+", ""),
    (r"\bwhat\s+(is|are)\s+the\s+", ""),
    (r"\bwhere\s+(is|are|can\s+i\s+find)\s+the\s+", ""),
]


class RuleBasedQueryRewriter(BaseQueryRewriter):
    """
    Deterministic domain-aware query rewriter.
    Deconstructs natural language syntax, extracts core domain subjects and predicates,
    and constructs search-optimized variants while strictly preserving existing
    equipment tags, values, and standards without introducing unsupported entities.
    """

    def __init__(self, entity_extractor: Optional[RuleBasedEntityExtractor] = None):
        self.entity_extractor = entity_extractor or RuleBasedEntityExtractor()

    def _strip_conversational_filler(self, text: str) -> str:
        cleaned = text.strip()
        for pat in CONVERSATIONAL_STOP_PREFIXES:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        for pat, replacement in WH_TRANSFORMS:
            cleaned = re.sub(pat, replacement, cleaned, flags=re.IGNORECASE)
        # Remove trailing question mark and punctuation
        cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
        return cleaned

    def rewrite(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        entities: Optional[QueryEntities] = None
    ) -> RewrittenQueries:
        if not entities:
            entities = self.entity_extractor.extract(query)

        stripped = self._strip_conversational_filler(query)
        candidates: List[str] = []

        # Identify key anchor tokens
        anchors: List[str] = []
        if entities.equipment_ids:
            anchors.extend(entities.equipment_ids)
        if entities.equipment_types:
            anchors.extend(entities.equipment_types)
        if entities.wells:
            anchors.extend(entities.wells)
        if entities.fields:
            anchors.extend(entities.fields)
        if entities.standards:
            anchors.extend(entities.standards)
        if entities.parameters:
            anchors.extend(entities.parameters)

        # Base clean formulation
        clean_base = stripped.lower()

        # Generate targeted domain formulations based on intent or query characteristics
        is_troubleshooting = (
            intent == QueryIntent.TROUBLESHOOTING
            or any(c in clean_base for c in ["high", "trip", "leak", "vibration", "abnormal", "fault", "failure", "increasing"])
        )
        is_maintenance = (
            intent == QueryIntent.MAINTENANCE
            or any(c in clean_base for c in ["maintain", "maintenance", "replace", "inspect", "service", "schedule"])
        )
        is_safety = (
            intent == QueryIntent.SAFETY
            or any(c in clean_base for c in ["safety", "loto", "esd", "hazard", "h2s", "permit"])
        )

        if is_troubleshooting:
            # Reorder adjective if present (e.g. "compressor vibration high" -> "compressor high vibration")
            phrase = clean_base
            if "vibration high" in phrase:
                phrase = phrase.replace("vibration high", "high vibration")
            elif "pressure high" in phrase:
                phrase = phrase.replace("pressure high", "high pressure")
            elif "temperature high" in phrase:
                phrase = phrase.replace("temperature high", "high temperature")

            # Variant 1: Core issue + causes
            candidates.append(f"{phrase} causes")
            # Variant 2: Core issue + troubleshooting
            candidates.append(f"{re.sub(r' (high|low|abnormal)', '', phrase)} troubleshooting")
            # Variant 3: Abnormal condition + maintenance
            candidates.append(f"{re.sub(r' (high|low)', ' abnormal', phrase)} maintenance")
            # Variant 4: Failure modes
            candidates.append(f"{re.sub(r' (high|low|abnormal)', '', phrase)} failure modes")

        elif is_maintenance:
            candidates.append(f"{clean_base} procedure")
            candidates.append(f"{clean_base} maintenance guidelines")
            candidates.append(f"{clean_base} inspection checklist")
            candidates.append(f"{clean_base} standard operating procedure")

        elif is_safety:
            candidates.append(f"{clean_base} safety protocol")
            candidates.append(f"{clean_base} standard procedure")
            candidates.append(f"{clean_base} emergency isolation")

        else:
            # Generic technical or equipment QA
            candidates.append(clean_base)
            if entities.equipment_types:
                eq_type = entities.equipment_types[0]
                candidates.append(f"{eq_type} specifications operating limits")
            candidates.append(f"{clean_base} operating manual")
            candidates.append(f"{clean_base} technical description")

        # Deduplicate while preserving order
        unique_candidates: List[str] = []
        for c in candidates:
            # Clean extra spaces
            norm = re.sub(r"\s+", " ", c).strip()
            if norm and norm not in unique_candidates:
                unique_candidates.append(norm)

        primary = unique_candidates[0] if unique_candidates else query

        return RewrittenQueries(
            original_query=query,
            primary_rewritten=primary,
            rewritten_candidates=unique_candidates,
            strategy_used="rule_based"
        )
