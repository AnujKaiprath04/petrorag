"""
Unit Test: Module 2.18 Response Schema Verification
Verifies:
1. PetroRAGRequest validation, field constraints, and default values.
2. PetroRAGResponse serialization, JSON schema compliance, and nested metadata.
3. Citation, Grounding, Abstention, and Telemetry data contract round-tripping.
4. Top-level convenience properties (is_grounded, is_abstained, confidence).
"""

import json
import pytest
from backend.app.schemas.rag_response import (
    AbstentionInfo,
    CitationInfo,
    ClaimVerificationInfo,
    GroundingMetadata,
    PetroRAGRequest,
    PetroRAGResponse,
    SourceDocumentInfo,
    TelemetryMetadata,
)


def test_rag_request_schema():
    """Verify request parameters and default settings."""
    req = PetroRAGRequest(
        query="Why did compressor C-101 trip on vibration?"
    )
    assert req.top_k == 5
    assert req.enable_reranking is True
    assert req.enable_compression is True
    assert req.temperature == 0.0

    # Custom request
    req2 = PetroRAGRequest(
        query="What is the high pressure trip for S-101?",
        top_k=10,
        metadata_filters={"equipment": "S-101"}
    )
    assert req2.top_k == 10
    assert req2.metadata_filters == {"equipment": "S-101"}


def test_rag_response_schema_serialization():
    """Verify complete response construction, serialization, and deserialization."""
    source = SourceDocumentInfo(
        chunk_id="chk-01",
        document_id="DOC-C101-MANUAL",
        document_title="Centrifugal Compressor Manual.pdf",
        page_number=18,
        section_title="4.2 Vibration Thresholds",
        relevance_score=0.965,
        channel="hybrid",
        snippet="Vibration alarm triggers at 4.5 mm/s; trip occurs at 7.1 mm/s."
    )

    citation = CitationInfo(
        citation_id="[1]",
        document_id="DOC-C101-MANUAL",
        document_title="Centrifugal Compressor Manual.pdf",
        page_number=18,
        chunk_id="chk-01",
        text_snippet="Vibration alarm triggers at 4.5 mm/s; trip occurs at 7.1 mm/s.",
        relevance_score=0.965
    )

    claim = ClaimVerificationInfo(
        claim_id="clm-01",
        claim_text="The vibration trip threshold for compressor C-101 is 7.1 mm/s.",
        status="SUPPORTED",
        confidence=0.99,
        supporting_citations=["[1]"]
    )

    grounding = GroundingMetadata(
        grounding_score=1.0,
        is_grounded=True,
        total_claims=1,
        supported_claims=1,
        unsupported_claims=0,
        claims=[claim]
    )

    abstention = AbstentionInfo(
        is_abstained=False
    )

    telemetry = TelemetryMetadata(
        total_latency_ms=145.2,
        retrieval_latency_ms=12.4,
        rerank_latency_ms=45.1,
        generation_latency_ms=72.5,
        verification_latency_ms=15.2,
        prompt_tokens=420,
        completion_tokens=85,
        total_tokens=505,
        tokens_per_second=1172.4,
        model_name="mock-petrogpt-v1"
    )

    response = PetroRAGResponse(
        query_id="req-98765",
        query="What is the vibration trip limit for compressor C-101?",
        intent="TROUBLESHOOTING",
        entities={"equipment": ["C-101"], "parameters": ["vibration"]},
        answer="The vibration trip limit for centrifugal compressor C-101 is 7.1 mm/s per ISO 10816-3 [1].",
        sources=[source],
        citations=[citation],
        grounding=grounding,
        abstention=abstention,
        telemetry=telemetry
    )

    # Validate convenience properties
    assert response.is_grounded is True
    assert response.is_abstained is False
    assert response.confidence == 1.0

    # Serialization roundtrip
    dumped = response.model_dump()
    assert dumped["query_id"] == "req-98765"
    assert len(dumped["sources"]) == 1
    assert dumped["sources"][0]["page_number"] == 18
    assert dumped["grounding"]["supported_claims"] == 1
    assert dumped["telemetry"]["total_tokens"] == 505

    # JSON serialization
    json_str = response.model_dump_json()
    assert "DOC-C101-MANUAL" in json_str

    # Restoration
    restored = PetroRAGResponse.model_validate_json(json_str)
    assert restored.query_id == response.query_id
    assert restored.answer == response.answer
    assert restored.telemetry.total_tokens == 505
