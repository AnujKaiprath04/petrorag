"""
PetroRAG Contradiction Detection Engine (Module 2.26)
Analyzes cross-document discrepancies and intra-context operational conflicts.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.core.interfaces import BuiltContext, RerankedChunk
from src.generation.context_compressor import PARAM_PATTERN, TAG_PATTERN
from src.verification.grounding_evaluator import OPPOSING_OPERATIONAL_STATES


class ConflictPair(BaseModel):
    """Pairwise conflicting statements across documents."""
    chunk_a_id: str
    chunk_b_id: str
    conflict_type: str = Field(..., description="OPERATIONAL_STATE, PARAMETER_DISCREPANCY")
    statement_a: str
    statement_b: str
    resolution_guidance: Optional[str] = None


class ContradictionReport(BaseModel):
    """Comprehensive cross-document consistency assessment."""
    has_conflicts: bool = Field(default=False)
    conflict_count: int = Field(default=0)
    conflicts: List[ConflictPair] = Field(default_factory=list)
    requires_manual_reconciliation: bool = Field(default=False)


class ContradictionDetector:
    """
    Cross-checks retrieved candidate chunks for intra-context discrepancies
    before prompt injection and generation.
    """

    def detect_conflicts(self, chunks: List[RerankedChunk]) -> ContradictionReport:
        """
        Scans all pairwise chunks in candidate pool for contradictory operational states.
        """
        conflicts: List[ConflictPair] = []

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                ca = chunks[i]
                cb = chunks[j]

                text_a = ca.text.lower()
                text_b = cb.text.lower()

                # Check for opposing operational states on common equipment or topics
                for state_a, state_b in OPPOSING_OPERATIONAL_STATES:
                    pat_a = r"\b" + re.escape(state_a) + r"\b"
                    pat_b = r"\b" + re.escape(state_b) + r"\b"

                    if re.search(pat_a, text_a) and re.search(pat_b, text_b):
                        # Verify that they share at least one common entity/equipment tag
                        tags_a = set(TAG_PATTERN.findall(ca.text))
                        tags_b = set(TAG_PATTERN.findall(cb.text))
                        common_tags = tags_a.intersection(tags_b)

                        if common_tags or len(chunks) <= 3:
                            tag_str = ", ".join(common_tags) if common_tags else "common equipment"
                            conflicts.append(ConflictPair(
                                chunk_a_id=ca.chunk_id,
                                chunk_b_id=cb.chunk_id,
                                conflict_type="OPERATIONAL_STATE",
                                statement_a=f"{ca.chunk_id}: asserts '{state_a}' for {tag_str}",
                                statement_b=f"{cb.chunk_id}: asserts '{state_b}' for {tag_str}",
                                resolution_guidance=f"Check document revision dates for {tag_str} to identify latest superseding standard."
                            ))
                            break

        return ContradictionReport(
            has_conflicts=len(conflicts) > 0,
            conflict_count=len(conflicts),
            conflicts=conflicts,
            requires_manual_reconciliation=len(conflicts) > 0
        )
