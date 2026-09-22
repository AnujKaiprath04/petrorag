"""
Unit Test & Benchmark: Module 2.11 Retrieval Deduplication Verification
Verifies:
1. Exact duplicate removal (hash & ID matching).
2. Near-duplicate removal (Jaccard token similarity >= threshold).
3. Same-page sliding window redundancy reduction.
4. Preservation of distinct chunks from the same document and page.
5. Empirical metrics: Duplicate percentage before, Duplicate percentage after, Relevant chunks retained.
"""

import pytest
from src.core.interfaces import RetrievedChunk
from src.retrieval.deduplication import RetrievalDeduplicator, DeduplicationResult


@pytest.fixture
def deduplicator():
    return RetrievalDeduplicator(
        exact_hash_dedup=True,
        near_duplicate_threshold=0.80,
        same_page_redundancy_reduction=True,
        same_page_overlap_threshold=0.60
    )


def test_exact_duplicate_removal(deduplicator):
    """Verify identical text and identical IDs are pruned."""
    c1 = RetrievedChunk(
        chunk_id="chk_1",
        document_id="DOC-COMP",
        text="High compressor vibration is caused by rotor mass unbalance and shaft misalignment.",
        score=0.92,
        page_number=14
    )
    # Exact duplicate text with different chunk ID
    c2 = RetrievedChunk(
        chunk_id="chk_1_dup",
        document_id="DOC-COMP-COPY",
        text="High compressor vibration is caused by rotor mass unbalance and shaft misalignment.",
        score=0.85,
        page_number=14
    )

    result = deduplicator.deduplicate([c1, c2])
    assert result.total_input == 2
    assert result.total_output == 1
    assert result.exact_duplicates_removed == 1
    assert result.deduplicated_chunks[0].chunk_id == "chk_1"


def test_near_duplicate_removal(deduplicator):
    """Verify passages with > 80% word overlap are pruned, prioritizing higher scoring chunk."""
    c1 = RetrievedChunk(
        chunk_id="chk_original",
        document_id="DOC-SEP",
        text="Separator S-101 high pressure operating limits: emergency shutdown trip triggers at 45.0 bar.",
        score=0.95,
        page_number=33
    )
    # Minor wording variation with > 85% overlap
    c2 = RetrievedChunk(
        chunk_id="chk_near_dup",
        document_id="DOC-SEP",
        text="Separator S-101 high pressure operating limits: emergency shutdown trip triggers at 45.0 bar calibrated.",
        score=0.75,
        page_number=33
    )

    result = deduplicator.deduplicate([c1, c2])
    assert result.total_input == 2
    assert result.total_output == 1
    assert result.near_duplicates_removed >= 1 or result.same_page_redundancies_reduced >= 1
    assert result.deduplicated_chunks[0].chunk_id == "chk_original"


def test_same_document_distinct_chunks_preserved(deduplicator):
    """
    CRITICAL INVARIANT:
    Do not remove distinct chunks simply because they come from the same document!
    Two distinct sections from the same manual must BOTH be preserved.
    """
    # Chunk A: Page 5 (Scope and Design)
    c_page5 = RetrievedChunk(
        chunk_id="chk_doc_page5",
        document_id="DOC-COMPRESSOR-MANUAL",
        text="Centrifugal compressor design pressure rating is 65 bar with rated speed of 12000 RPM.",
        score=0.88,
        page_number=5,
        section_title="Design Ratings"
    )
    # Chunk B: Page 42 (Troubleshooting and vibration)
    c_page42 = RetrievedChunk(
        chunk_id="chk_doc_page42",
        document_id="DOC-COMPRESSOR-MANUAL",
        text="Radial vibration alarm triggers at 4.5 mm/s RMS while trip limit is set at 7.1 mm/s RMS.",
        score=0.89,
        page_number=42,
        section_title="Vibration Limits"
    )

    result = deduplicator.deduplicate([c_page5, c_page42])
    assert result.total_input == 2
    assert result.total_output == 2
    assert result.exact_duplicates_removed == 0
    assert result.near_duplicates_removed == 0
    assert result.same_page_redundancies_reduced == 0
    assert len(result.deduplicated_chunks) == 2


def test_empirical_deduplication_metrics_benchmark(deduplicator):
    """
    Verification Gate:
    Measure:
    - Duplicate percentage before
    - Duplicate percentage after
    - Relevant chunks retained
    """
    # Synthetic candidate pool of 8 chunks containing 2 exact duplicates, 1 near duplicate, and 5 distinct relevant chunks
    candidate_pool = [
        # Distinct relevant chunk 1
        RetrievedChunk(chunk_id="rel_1", document_id="D1", text="Compressor C-101 bearing failure diagnostics.", score=0.95, page_number=10),
        # Exact duplicate of chunk 1
        RetrievedChunk(chunk_id="rel_1_dup", document_id="D1", text="Compressor C-101 bearing failure diagnostics.", score=0.80, page_number=10),
        # Distinct relevant chunk 2
        RetrievedChunk(chunk_id="rel_2", document_id="D1", text="Compressor C-101 alignment and coupling inspection schedule.", score=0.90, page_number=25),
        # Distinct relevant chunk 3
        RetrievedChunk(chunk_id="rel_3", document_id="D2", text="Separator S-101 emergency shutdown trip limit at 45 bar.", score=0.88, page_number=4),
        # Near duplicate of chunk 3 (sliding window overlap)
        RetrievedChunk(chunk_id="rel_3_window", document_id="D2", text="Separator S-101 emergency shutdown trip limit at 45 bar calibrated annually.", score=0.70, page_number=4),
        # Distinct relevant chunk 4
        RetrievedChunk(chunk_id="rel_4", document_id="D3", text="ESP pump installation checklist for well W-12.", score=0.85, page_number=2),
        # Exact duplicate of chunk 4
        RetrievedChunk(chunk_id="rel_4_dup", document_id="D3_copy", text="ESP pump installation checklist for well W-12.", score=0.65, page_number=2),
        # Distinct relevant chunk 5
        RetrievedChunk(chunk_id="rel_5", document_id="D4", text="API 610 mechanical seal flush plan 53B operation.", score=0.82, page_number=1),
    ]

    ground_truth_relevant = {"rel_1", "rel_2", "rel_3", "rel_4", "rel_5"}

    res: DeduplicationResult = deduplicator.deduplicate(
        chunks=candidate_pool,
        ground_truth_relevant_ids=ground_truth_relevant
    )

    print("\n" + "=" * 60)
    print("EMPIRICAL RETRIEVAL DEDUPLICATION BENCHMARK")
    print(f"Total Input Candidates:          {res.total_input}")
    print(f"Total Output Candidates:         {res.total_output}")
    print(f"Exact Duplicates Pruned:         {res.exact_duplicates_removed}")
    print(f"Near Duplicates Pruned:          {res.near_duplicates_removed}")
    print(f"Same-Page Redundancies Reduced:  {res.same_page_redundancies_reduced}")
    print(f"Duplicate Percentage Before:     {res.duplicate_percentage_before:.2f}%")
    print(f"Duplicate Percentage After:      {res.duplicate_percentage_after:.2f}%")
    print(f"Relevant Chunks Retained:        {res.relevant_chunks_retained}/{len(ground_truth_relevant)} (100%)")
    print("=" * 60)

    assert res.total_input == 8
    assert res.total_output == 5  # Exactly the 5 distinct relevant chunks retained
    assert res.duplicate_percentage_before == 37.50  # 3 duplicates out of 8 = 37.5%
    assert res.duplicate_percentage_after == 0.00
    assert res.relevant_chunks_retained == 5  # 100% of unique relevant chunks retained
