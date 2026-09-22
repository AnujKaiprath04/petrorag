"""
Unit Test: Module 2.19 - 2.21 Citations, Lineage & Claim Extraction Verification
Verifies:
1. Citation parsing, chunk resolution, and verbatim snippet extraction.
2. Fallback auto-citation synthesis when model omits explicit brackets.
3. Citation validation metrics (precision, validity, hallucinated citation detection).
4. SourceTraceabilityAuditor SHA-256 cryptographic lineage and tamper detection.
5. Atomic claim decomposition, entity tagging, and conversational prefix stripping.
"""

import pytest
from src.core.interfaces import BuiltContext, Citation, RerankedChunk
from src.verification.citation_engine import CitationEngine
from src.verification.claim_extractor import AtomicClaim, ClaimExtractor
from src.verification.lineage import LineageManifest, SourceTraceabilityAuditor


def _create_test_context() -> BuiltContext:
    c1 = RerankedChunk(
        chunk_id="chk-c101",
        document_id="DOC-SOP-C101",
        document_title="Centrifugal Compressor Operations Manual.pdf",
        text=(
            "Centrifugal compressor C-101 normal operating discharge pressure is 42.5 bar at 8500 RPM. "
            "High vibration alarm triggers at 4.5 mm/s per ISO 10816-3."
        ),
        initial_score=0.91,
        reranker_score=0.98,
        rank=1,
        page_number=14,
        section_title="4.1 Operating Conditions"
    )
    c2 = RerankedChunk(
        chunk_id="chk-s101",
        document_id="DOC-SOP-S101",
        document_title="Three Phase Separator Manual.pdf",
        text=(
            "Production separator S-101 operating pressure is 45.0 bar. "
            "High-high level emergency trip activates at 82% liquid volume."
        ),
        initial_score=0.85,
        reranker_score=0.92,
        rank=2,
        page_number=22,
        section_title="3.2 Level Protection"
    )
    return BuiltContext(
        context_text=f"[DOC-SOP-C101]\n{c1.text}\n\n---\n\n[DOC-SOP-S101]\n{c2.text}",
        chunks=[c1, c2],
        token_count=60,
        total_candidates_evaluated=2,
        pruned_chunks_count=0
    )


def test_citation_engine_bracket_parsing_and_resolution():
    """Verify parsing numeric and structured citation markers with snippet extraction."""
    engine = CitationEngine()
    context = _create_test_context()

    answer = (
        "Compressor C-101 operates at 42.5 bar [1]. "
        "Separator S-101 high-high level emergency trip occurs at 82% [2]."
    )

    citations = engine.extract_citations(answer, context)

    assert len(citations) == 2
    assert citations[0].citation_id == "[1]"
    assert citations[0].document_id == "DOC-SOP-C101"
    assert citations[0].page_number == 14
    assert "42.5 bar" in citations[0].text_snippet
    assert citations[0].relevance_score == 0.98

    assert citations[1].citation_id == "[2]"
    assert citations[1].document_id == "DOC-SOP-S101"
    assert citations[1].page_number == 22
    assert "82%" in citations[1].text_snippet


def test_citation_engine_auto_cite_fallback():
    """Verify fallback automatic citation assignment when model output lacks bracketed markers."""
    engine = CitationEngine()
    context = _create_test_context()

    unannotated_answer = (
        "The normal operating discharge pressure for compressor C-101 is 42.5 bar at 8500 RPM. "
        "Production separator S-101 operating pressure is maintained at 45.0 bar."
    )

    citations = engine.extract_citations(unannotated_answer, context)

    assert len(citations) >= 2
    doc_ids = {c.document_id for c in citations}
    assert "DOC-SOP-C101" in doc_ids
    assert "DOC-SOP-S101" in doc_ids


def test_citation_validation_metrics():
    """Verify citation precision and hallucination detection."""
    engine = CitationEngine()
    context = _create_test_context()

    valid_citation = Citation(
        citation_id="[1]",
        document_id="DOC-SOP-C101",
        document_title="Centrifugal Compressor Operations Manual.pdf",
        page_number=14,
        chunk_id="chk-c101",
        text_snippet="Centrifugal compressor C-101 normal operating discharge pressure is 42.5 bar",
        relevance_score=0.98
    )
    hallucinated_citation = Citation(
        citation_id="[99]",
        document_id="DOC-NONEXISTENT",
        document_title="Fake Manual.pdf",
        page_number=999,
        chunk_id="chk-fake-99",
        text_snippet="Fake data not in context",
        relevance_score=0.1
    )

    metrics = engine.validate_citations(
        citations=[valid_citation, hallucinated_citation],
        answer_text="Compressor C-101 operates at 42.5 bar.",
        context=context
    )

    assert metrics["total_citations"] == 2
    assert metrics["valid_citations"] == 1
    assert metrics["hallucinated_citations"] == 1
    assert metrics["citation_precision"] == 0.5


def test_source_traceability_lineage_and_tamper_detection():
    """Verify cryptographic SHA-256 lineage manifest generation and integrity verification."""
    auditor = SourceTraceabilityAuditor()
    engine = CitationEngine()
    context = _create_test_context()

    answer = "Compressor C-101 normal discharge pressure is 42.5 bar [1]."
    citations = engine.extract_citations(answer, context)

    manifest = auditor.build_lineage(answer, citations, context)

    assert isinstance(manifest, LineageManifest)
    assert manifest.total_citations == 1
    assert len(manifest.records) == 1
    rec = manifest.records[0]
    assert rec.chunk_id == "chk-c101"
    assert rec.page_number == 14
    assert len(rec.chunk_sha256) == 64  # Valid SHA-256 hex string

    # Verify integrity passes with genuine context
    assert auditor.verify_manifest_integrity(manifest, context) is True

    # Tamper with context text and verify integrity failure
    tampered_chunk = RerankedChunk(
        chunk_id="chk-c101",
        document_id="DOC-SOP-C101",
        text="TAMPERED ILLEGITIMATE TEXT HERE",
        initial_score=0.9,
        reranker_score=0.98,
        rank=1
    )
    tampered_context = BuiltContext(
        context_text="Tampered",
        chunks=[tampered_chunk],
        token_count=10,
        total_candidates_evaluated=1,
        pruned_chunks_count=0
    )
    assert auditor.verify_manifest_integrity(manifest, tampered_context) is False


def test_atomic_claim_extraction_and_entity_tagging():
    """Verify deconstruction of compound sentences, prefix stripping, and entity tagging."""
    extractor = ClaimExtractor()

    text = (
        "**Direct Answer**: Centrifugal compressor C-101 operates at 42.5 bar, "
        "and high vibration alarm triggers at 4.5 mm/s per ISO 10816-3. "
        "Emergency shutdown (ESD) occurs if vibration exceeds 7.1 mm/s."
    )

    claims = extractor.extract_claims(text)

    # Should deconstruct into 3 atomic claims:
    # 1. C-101 operates at 42.5 bar
    # 2. High vibration alarm triggers at 4.5 mm/s per ISO 10816-3
    # 3. Emergency shutdown occurs if vibration exceeds 7.1 mm/s
    assert len(claims) >= 3
    assert all(isinstance(c, AtomicClaim) for c in claims)

    # Prefix stripping
    assert not any("**Direct Answer**" in c.claim_text for c in claims)

    # Entity and parameter audit
    all_entities = [e for c in claims for e in c.entities]
    all_params = [p for c in claims for p in c.parameters]

    assert "C-101" in all_entities
    assert any("ISO 10816-3" in e for e in all_entities)
    assert any("42.5 bar" in p for p in all_params)
    assert any("4.5 mm/s" in p for p in all_params)
    assert any("7.1 mm/s" in p for p in all_params)

    # Safety critical flag
    assert any(c.is_safety_critical for c in claims)
