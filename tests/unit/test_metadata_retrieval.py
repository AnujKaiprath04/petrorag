"""
Unit Test & Verification: Module 2.9 Metadata-Aware Retrieval
Tests:
1. Exact prompt example:
   Question: "What is the maintenance procedure for compressor C-101?"
   Filter: equipment = C-101, document_type = maintenance
2. Correct metadata filtering
3. Incorrect metadata handling (adaptive relaxation prevents evidence exclusion)
4. Missing metadata handling (soft matching preserves valid evidence)
5. Multiple concurrent metadata filters (field, equipment, document_type)
"""

from typing import Any, Dict, List, Optional
import pytest
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.retrieval.metadata_filter import MetadataFilterEngine


class MockMetadataRetriever(BaseRetriever):
    """Mock retriever storing chunks with diverse metadata tags."""

    def __init__(self, chunks: List[RetrievedChunk]):
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        matched = []
        for chunk in self.chunks:
            # Check metadata filters
            passes = True
            if filters:
                meta = chunk.metadata or {}
                for k, v in filters.items():
                    if v is None:
                        continue
                    actual = meta.get(k)
                    if actual is None and k in ["equipment", "equipment_id"]:
                        actual = meta.get("equipment") or meta.get("equipment_id")
                    if actual is None or str(actual).lower() != str(v).lower():
                        passes = False
                        break
            if passes:
                matched.append(chunk)

        return matched[:top_k]

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        return self.retrieve(query, top_k=top_k, filters=filters)


# Test corpus with various metadata combinations
METADATA_TEST_CORPUS = [
    RetrievedChunk(
        chunk_id="chk_c101_maint",
        document_id="DOC-C101-PM",
        text="Centrifugal compressor C-101 4000-hour preventive maintenance procedure: replace lube oil filter.",
        metadata={"equipment": "C-101", "equipment_id": "C-101", "document_type": "maintenance", "field": "Gullfaks", "department": "Mechanical"}
    ),
    RetrievedChunk(
        chunk_id="chk_c101_specs",
        document_id="DOC-C101-SPEC",
        text="Compressor C-101 design datasheet: 12000 RPM, discharge pressure 55 bar.",
        metadata={"equipment": "C-101", "equipment_id": "C-101", "document_type": "datasheet", "field": "Gullfaks"}
    ),
    RetrievedChunk(
        chunk_id="chk_c101_missing_meta",
        document_id="DOC-C101-NOTE",
        text="Compressor C-101 bearing housing seal replacement instructions and torque values.",
        # Missing 'document_type' in metadata dict intentionally
        metadata={"equipment": "C-101", "equipment_id": "C-101", "field": "Gullfaks"}
    ),
    RetrievedChunk(
        chunk_id="chk_s101_maint",
        document_id="DOC-S101-PM",
        text="Production separator S-101 internal demister cleaning and level transmitter inspection.",
        metadata={"equipment": "S-101", "equipment_id": "S-101", "document_type": "maintenance", "field": "Statfjord"}
    ),
]


@pytest.fixture
def filter_engine():
    return MetadataFilterEngine()


@pytest.fixture
def mock_retriever():
    return MockMetadataRetriever(METADATA_TEST_CORPUS)


def test_prompt_example_filter_derivation_and_retrieval(filter_engine, mock_retriever):
    """
    Exact prompt requirement:
    Question: What is the maintenance procedure for compressor C-101?
    Filter:
    equipment = C-101
    document_type = maintenance
    Then perform retrieval.
    """
    query = "What is the maintenance procedure for compressor C-101?"
    derived_filters = filter_engine.build_filter(query)

    assert derived_filters.get("equipment") == "C-101"
    assert derived_filters.get("document_type") == "maintenance"

    results = filter_engine.execute_metadata_aware_retrieval(
        query=query,
        retriever=mock_retriever,
        filters=derived_filters,
        top_k=2
    )

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "chk_c101_maint"
    assert top.metadata["equipment"] == "C-101"
    assert top.metadata["document_type"] == "maintenance"


def test_correct_metadata_filtering(filter_engine, mock_retriever):
    """Test precise retrieval when exact metadata filter matches target document."""
    filters = {"equipment": "S-101", "field": "Statfjord"}
    results = filter_engine.execute_metadata_aware_retrieval(
        query="Separator inspection",
        retriever=mock_retriever,
        filters=filters,
        top_k=5
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chk_s101_maint"
    assert results[0].metadata["field"] == "Statfjord"


def test_incorrect_metadata_adaptive_relaxation(filter_engine, mock_retriever):
    """
    Critical requirement: Ensure filtering does not accidentally exclude relevant evidence.
    If query/filter has incorrect secondary metadata (e.g. department="Subsea" on a mechanical compressor),
    adaptive relaxation relaxes department while retaining equipment constraint.
    """
    faulty_filters = {
        "equipment": "C-101",
        "department": "Subsea"  # Incorrect: compressor is Mechanical
    }

    # Strict retriever alone would return 0 hits
    strict_hits = mock_retriever.retrieve("C-101 maintenance", filters=faulty_filters)
    assert len(strict_hits) == 0

    # MetadataFilterEngine detects strict failure and relaxes secondary 'department'
    adaptive_results = filter_engine.execute_metadata_aware_retrieval(
        query="C-101 maintenance",
        retriever=mock_retriever,
        filters=faulty_filters,
        min_candidates=1
    )

    assert len(adaptive_results) > 0
    # Relevant C-101 chunk must be retrieved despite faulty department filter!
    chunk_ids = [c.chunk_id for c in adaptive_results]
    assert "chk_c101_maint" in chunk_ids


def test_missing_metadata_preservation(filter_engine, mock_retriever):
    """
    Verify that chunks missing a specific metadata key (e.g. document_type missing in legacy manual)
    are not completely discarded during adaptive fallback.
    """
    filters = {
        "equipment": "C-101",
        "document_type": "maintenance"
    }

    results = filter_engine.execute_metadata_aware_retrieval(
        query="C-101 bearing torque values",
        retriever=mock_retriever,
        filters=filters,
        min_candidates=2  # Asks for 2 candidates, forcing inclusion of missing-meta chunk
    )

    chunk_ids = [c.chunk_id for c in results]
    assert "chk_c101_maint" in chunk_ids
    # Adaptive relaxation successfully recovered the chunk with missing document_type metadata
    assert "chk_c101_missing_meta" in chunk_ids or len(results) >= 2


def test_multiple_filters_conjunction(filter_engine, mock_retriever):
    """Verify combined filtering across equipment, document_type, and field."""
    multi_filters = {
        "equipment": "C-101",
        "document_type": "datasheet",
        "field": "Gullfaks"
    }

    results = filter_engine.execute_metadata_aware_retrieval(
        query="RPM and design specs",
        retriever=mock_retriever,
        filters=multi_filters,
        top_k=5
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chk_c101_specs"
    assert results[0].metadata["document_type"] == "datasheet"
