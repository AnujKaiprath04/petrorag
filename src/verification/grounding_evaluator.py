"""
PetroRAG Grounding Verification & Hallucination Detection Engine (Modules 2.22 - 2.23)
Evaluates claim-level Natural Language Inference (NLI) entailment, validates
factual parameters against source context, and detects technical hallucinations.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.core.config import settings
from src.core.interfaces import (
    BaseGroundingEvaluator,
    BuiltContext,
    Citation,
    ClaimSupportStatus,
    ClaimVerification,
    GroundingResult,
    RerankedChunk,
)
from src.core.logging import logger
from src.verification.citation_engine import _compute_overlap, _tokenize
from src.verification.claim_extractor import AtomicClaim, ClaimExtractor


OPPOSING_OPERATIONAL_STATES: List[Tuple[str, str]] = [
    ("normally open", "normally closed"),
    ("fail open", "fail closed"),
    ("open", "closed"),
    ("energized", "de-energized"),
    ("permitted", "prohibited"),
    ("allowed", "forbidden"),
    ("mandatory", "optional"),
    ("enabled", "disabled"),
    ("active", "inactive"),
    ("pressurized", "depressurized"),
]


class GroundingEvaluator(BaseGroundingEvaluator):
    """
    Evaluates factual grounding of generated answers at the atomic claim level.
    Performs parameter-exact verification, contradiction detection, and hallucination scoring.
    """

    def __init__(
        self,
        grounding_threshold: float = settings.GROUNDING_THRESHOLD,
        claim_extractor: Optional[ClaimExtractor] = None
    ):
        self.grounding_threshold = grounding_threshold
        self.claim_extractor = claim_extractor or ClaimExtractor()

    def _verify_single_claim(
        self,
        claim: AtomicClaim,
        candidate_chunks: List[RerankedChunk]
    ) -> ClaimVerification:
        """
        Evaluates entailment status of an individual atomic claim against candidate chunks.
        Checks for exact technical parameter matches, equipment tag containment, and semantic overlap.
        """
        claim_tokens = _tokenize(claim.claim_text)
        claim_params = set(claim.parameters)
        claim_entities = set(claim.entities)

        best_overlap = 0.0
        best_chunk: Optional[RerankedChunk] = None
        best_status = ClaimSupportStatus.UNSUPPORTED
        best_confidence = 0.0
        reasoning = ""

        all_context_text = " ".join(c.text for c in candidate_chunks)

        # 1. Parameter Validation: Check if numbers in the claim exist anywhere in context
        if claim_params:
            missing_params = [p for p in claim_params if p.lower() not in all_context_text.lower()]
            if missing_params:
                return ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    status=ClaimSupportStatus.UNSUPPORTED,
                    confidence=0.95,
                    reasoning=f"Unsupported numerical parameters not found in context: {missing_params}"
                )

        claim_lower = claim.claim_text.lower()

        # 2. Evaluate each chunk for supporting evidence and contradictions
        for chunk in candidate_chunks:
            chunk_tokens = _tokenize(chunk.text)
            overlap = _compute_overlap(claim_tokens, chunk_tokens)
            chunk_lower = chunk.text.lower()

            # Check for direct operational contradiction against this chunk
            for state_a, state_b in OPPOSING_OPERATIONAL_STATES:
                pattern_a = r"\b" + re.escape(state_a) + r"\b"
                pattern_b = r"\b" + re.escape(state_b) + r"\b"

                if (
                    re.search(pattern_a, claim_lower)
                    and re.search(pattern_b, chunk_lower)
                    and not re.search(pattern_a, chunk_lower)
                    and overlap >= 0.35
                ):
                    return ClaimVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=ClaimSupportStatus.CONTRADICTED,
                        confidence=0.98,
                        supporting_chunk_ids=[chunk.chunk_id],
                        reasoning=f"Direct operational contradiction: claim specifies '{state_a}' but chunk {chunk.chunk_id} specifies '{state_b}'."
                    )
                elif (
                    re.search(pattern_b, claim_lower)
                    and re.search(pattern_a, chunk_lower)
                    and not re.search(pattern_b, chunk_lower)
                    and overlap >= 0.35
                ):
                    return ClaimVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        status=ClaimSupportStatus.CONTRADICTED,
                        confidence=0.98,
                        supporting_chunk_ids=[chunk.chunk_id],
                        reasoning=f"Direct operational contradiction: claim specifies '{state_b}' but chunk {chunk.chunk_id} specifies '{state_a}'."
                    )

            # Check parameter presence in this specific chunk
            has_all_params = all(p.lower() in chunk_lower for p in claim_params)
            has_all_entities = all(e.lower() in chunk_lower for e in claim_entities)

            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk

                if has_all_params and has_all_entities and overlap >= 0.40:
                    best_status = ClaimSupportStatus.SUPPORTED
                    best_confidence = min(1.0, round(0.70 + overlap * 0.30, 2))
                    reasoning = f"Fully supported by chunk {chunk.chunk_id} with exact parameters."
                elif (has_all_params or has_all_entities) and overlap >= 0.25:
                    best_status = ClaimSupportStatus.PARTIALLY_SUPPORTED
                    best_confidence = 0.65
                    reasoning = f"Partially supported by chunk {chunk.chunk_id}."

        if best_status == ClaimSupportStatus.UNSUPPORTED:
            best_confidence = 0.80
            reasoning = "No evidence chunk provides sufficient entailment for this claim."

        supporting_chunks = [best_chunk.chunk_id] if best_chunk and best_status in [ClaimSupportStatus.SUPPORTED, ClaimSupportStatus.PARTIALLY_SUPPORTED] else []

        return ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            status=best_status,
            confidence=best_confidence,
            supporting_chunk_ids=supporting_chunks,
            reasoning=reasoning
        )

    async def evaluate_grounding(
        self,
        answer_text: str,
        context: BuiltContext,
        citations: Optional[List[Citation]] = None
    ) -> GroundingResult:
        """
        Implements BaseGroundingEvaluator:
        Deconstructs answer into atomic claims, verifies each against context,
        and computes aggregate grounding score and hallucination status.
        """
        if not answer_text or not answer_text.strip():
            return GroundingResult(
                grounding_score=0.0,
                total_claims=0,
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                is_grounded=False,
                hallucination_detected=False,
                abstention_recommended=True
            )

        claims = self.claim_extractor.extract_claims(answer_text)
        if not claims:
            return GroundingResult(
                grounding_score=1.0,
                total_claims=0,
                supported_claims=0,
                partially_supported_claims=0,
                unsupported_claims=0,
                is_grounded=True,
                hallucination_detected=False,
                abstention_recommended=False
            )

        verified_claims: List[ClaimVerification] = []
        supported_count = 0
        partially_supported_count = 0
        unsupported_count = 0
        hallucination_detected = False

        for claim in claims:
            verif = self._verify_single_claim(claim, context.chunks)
            verified_claims.append(verif)

            if verif.status == ClaimSupportStatus.SUPPORTED:
                supported_count += 1
            elif verif.status == ClaimSupportStatus.PARTIALLY_SUPPORTED:
                partially_supported_count += 1
            elif verif.status in [ClaimSupportStatus.UNSUPPORTED, ClaimSupportStatus.CONTRADICTED]:
                unsupported_count += 1
                if verif.status == ClaimSupportStatus.CONTRADICTED or claim.parameters:
                    hallucination_detected = True

        total = len(verified_claims)
        effective_score = (supported_count + 0.5 * partially_supported_count) / total if total > 0 else 0.0
        effective_score = round(effective_score, 4)

        is_grounded = effective_score >= self.grounding_threshold and not hallucination_detected
        abstention_recommended = not is_grounded

        logger.info(
            f"Grounding Evaluation: score={effective_score}, total={total}, "
            f"supp={supported_count}, part={partially_supported_count}, unsupp={unsupported_count}, "
            f"grounded={is_grounded}, hallucination={hallucination_detected}"
        )

        return GroundingResult(
            grounding_score=effective_score,
            total_claims=total,
            supported_claims=supported_count,
            partially_supported_claims=partially_supported_count,
            unsupported_claims=unsupported_count,
            is_grounded=is_grounded,
            claims=verified_claims,
            hallucination_detected=hallucination_detected,
            abstention_recommended=abstention_recommended
        )

    def evaluate_grounding_sync(
        self,
        answer_text: str,
        context: BuiltContext,
        citations: Optional[List[Citation]] = None
    ) -> GroundingResult:
        """Synchronous wrapper for offline benchmarking."""
        import asyncio
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                self.evaluate_grounding(answer_text, context, citations)
            ).result()
