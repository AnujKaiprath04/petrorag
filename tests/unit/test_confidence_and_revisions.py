"""
Unit Test: Modules 2.25 - 2.27 Confidence, Contradictions & Revision Awareness
Verifies:
1. Multi-component confidence estimation and categorical calibration.
2. Cross-document contradiction detection for opposing operational states.
3. Document revision parsing, chronological resolution, and supersession penalties.
"""

import pytest
from src.core.interfaces import GroundingResult, RerankedChunk
from src.verification.confidence_estimator import (
    ConfidenceAssessment,
    ConfidenceEstimator,
    ConfidenceLevel,
)
from src.verification.contradiction_detector import ContradictionDetector, ContradictionReport
from src.verification.revision_manager import RevisionManager


def test_confidence_estimator_calibration():
    """Verify confidence calculation across clean, moderate, and hallucinated responses."""
    estimator = ConfidenceEstimator()

    # Case 1: High confidence clean response
    clean_grounding = GroundingResult(
        grounding_score=1.0,
        total_claims=3,
        supported_claims=3,
        unsupported_claims=0,
        is_grounded=True,
        hallucination_detected=False
    )
    conf_clean = estimator.estimate_confidence(
        retrieval_score=0.95,
        grounding_result=clean_grounding,
        citation_precision=1.0
    )
    assert conf_clean.level == ConfidenceLevel.HIGH
    assert conf_clean.composite_score >= 0.85
    assert conf_clean.is_actionable_for_field is True

    # Case 2: Hallucinated response drops to UNRELIABLE
    hallucinated_grounding = GroundingResult(
        grounding_score=0.30,
        total_claims=3,
        supported_claims=1,
        unsupported_claims=2,
        is_grounded=False,
        hallucination_detected=True
    )
    conf_hallucinated = estimator.estimate_confidence(
        retrieval_score=0.90,
        grounding_result=hallucinated_grounding,
        citation_precision=0.50
    )
    assert conf_hallucinated.level == ConfidenceLevel.UNRELIABLE
    assert conf_hallucinated.is_actionable_for_field is False
    assert any("Hallucination" in r for r in conf_hallucinated.risk_factors)

    # Case 3: Abstained response
    conf_abstained = estimator.estimate_confidence(
        retrieval_score=0.20,
        grounding_result=clean_grounding,
        is_abstained=True
    )
    assert conf_abstained.level == ConfidenceLevel.UNRELIABLE
    assert conf_abstained.composite_score == 0.0


def test_contradiction_detector_operational_states():
    """Verify detection of opposing operational states across chunks for common equipment."""
    detector = ContradictionDetector()

    c1 = RerankedChunk(
        chunk_id="chk-sop-2018",
        document_id="DOC-VALVE-OLD",
        document_title="Valve Procedures 2018.pdf",
        text="Isolation valve XV-101 is normally open during baseline hydrocarbon flow.",
        initial_score=0.8,
        reranker_score=0.85,
        rank=1
    )
    c2 = RerankedChunk(
        chunk_id="chk-sop-2023",
        document_id="DOC-VALVE-NEW",
        document_title="Valve Procedures 2023.pdf",
        text="Emergency isolation valve XV-101 is normally closed during hazardous operations.",
        initial_score=0.8,
        reranker_score=0.88,
        rank=2
    )

    report = detector.detect_conflicts([c1, c2])

    assert isinstance(report, ContradictionReport)
    assert report.has_conflicts is True
    assert report.conflict_count == 1
    assert "XV-101" in report.conflicts[0].statement_a


def test_revision_manager_supersession_and_recency_boost():
    """Verify document family grouping, recency score boosting, and supersession marking."""
    manager = RevisionManager()

    c_old = RerankedChunk(
        chunk_id="chk-comp-rev1",
        document_id="DOC-C101-REV1",
        document_title="Centrifugal Compressor Operating Manual Rev 1.pdf",
        text="Compressor C-101 operating envelope (2018 Release). Discharge pressure: 40.0 bar.",
        initial_score=0.80,
        reranker_score=0.85,
        rank=1,
        metadata={"document_title": "Centrifugal Compressor Operating Manual Rev 1"}
    )
    c_new = RerankedChunk(
        chunk_id="chk-comp-rev3",
        document_id="DOC-C101-REV3",
        document_title="Centrifugal Compressor Operating Manual Rev 3.pdf",
        text="Compressor C-101 operating envelope (2023 Release). Discharge pressure: 42.5 bar.",
        initial_score=0.80,
        reranker_score=0.87,
        rank=2,
        metadata={"document_title": "Centrifugal Compressor Operating Manual Rev 3"}
    )

    resolved = manager.resolve_revisions([c_old, c_new], boost_multiplier=1.20)

    assert len(resolved) == 2
    # Rev 3 should be rank 1 with boosted score
    top_chunk = resolved[0]
    assert top_chunk.chunk_id == "chk-comp-rev3"
    assert top_chunk.metadata.get("is_latest_revision") is True
    assert top_chunk.reranker_score > 0.87

    # Rev 1 should be marked superseded
    older_chunk = resolved[1]
    assert older_chunk.chunk_id == "chk-comp-rev1"
    assert older_chunk.metadata.get("is_superseded") is True
    assert older_chunk.metadata.get("superseded_by") == "Rev 3"
    assert older_chunk.reranker_score < 0.85
