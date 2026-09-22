"""
Unit Test & Evaluation: Module 2.13 Context Selection Verification
Evaluates:
1. Context Relevance: Pruning of low-relevance candidates.
2. Context Redundancy: Suppression of repetitive chunks via diversity control.
3. Context Size: Strict compliance with token budget constraints.
4. Relevant Evidence Coverage: Retaining multi-faceted evidence across categories.
5. Lost-in-the-Middle structural ordering.
"""

import pytest
from src.core.interfaces import RerankedChunk, BuiltContext
from src.generation.context_selector import ContextSelector, estimate_tokens


# Test candidate pool representing a spectrum of relevance, redundancy, and token lengths
CANDIDATE_POOL = [
    # High relevance, critical evidence 1 (Causes)
    RerankedChunk(
        chunk_id="chk_causes",
        document_id="DOC-COMP-DIAG",
        document_title="Centrifugal Compressor Vibration Diagnostics.pdf",
        text="Centrifugal compressor high vibration causes include shaft mass unbalance, coupling misalignment, and fluid-film bearing whip.",
        initial_score=0.85,
        reranker_score=0.94,
        rank=1,
        page_number=14,
        section_title="Diagnostic Causes",
        metadata={"document_type": "manual"}
    ),
    # High relevance, critical evidence 2 (Limits)
    RerankedChunk(
        chunk_id="chk_limits",
        document_id="DOC-COMP-LIMITS",
        document_title="Compressor Operating Limits.pdf",
        text="Radial overall vibration trip setpoint is calibrated at 7.1 mm/s RMS while the alarm threshold is 4.5 mm/s RMS.",
        initial_score=0.82,
        reranker_score=0.91,
        rank=2,
        page_number=22,
        section_title="Operating Thresholds",
        metadata={"document_type": "sop"}
    ),
    # Redundant / repetitive variant of evidence 1 (Nearly identical wording)
    RerankedChunk(
        chunk_id="chk_causes_duplicate",
        document_id="DOC-COMP-DIAG-COPY",
        document_title="Compressor Diagnostics Copy.pdf",
        text="Centrifugal compressor high vibration causes include shaft mass unbalance and coupling misalignment and bearing instability.",
        initial_score=0.79,
        reranker_score=0.88,
        rank=3,
        page_number=14,
        section_title="Diagnostic Causes",
        metadata={"document_type": "manual"}
    ),
    # Moderate relevance, complementary evidence 3 (Lubrication SOP)
    RerankedChunk(
        chunk_id="chk_lube_oil",
        document_id="DOC-COMP-LUBE",
        document_title="Lube Oil System Standard Operating Procedure.pdf",
        text="Lube oil pressure must be maintained at 2.8 bar with temperature between 45 deg C and 50 deg C to ensure bearing stability.",
        initial_score=0.75,
        reranker_score=0.78,
        rank=4,
        page_number=5,
        section_title="Lubrication",
        metadata={"document_type": "sop"}
    ),
    # Low relevance / irrelevant candidate (should be pruned by relevance threshold)
    RerankedChunk(
        chunk_id="chk_irrelevant_flare",
        document_id="DOC-FLARE",
        document_title="Offshore Flare System Operations.pdf",
        text="Flare tip purge gas rate must be maintained at 15 scf/h to prevent air ingress and flashback into the knock-out drum.",
        initial_score=0.40,
        reranker_score=0.22,  # Below min_relevance_threshold (0.30)
        rank=5,
        page_number=88,
        section_title="Flare Header",
        metadata={"document_type": "manual"}
    ),
]


@pytest.fixture
def context_selector():
    return ContextSelector(
        min_relevance_threshold=0.30,
        max_redundancy_threshold=0.65,
        lost_in_the_middle_ordering=True
    )


def test_context_relevance_evaluation(context_selector):
    """Verify that chunks with reranker scores below threshold are pruned."""
    context: BuiltContext = context_selector.build_context(CANDIDATE_POOL, token_budget=2000)

    selected_ids = [c.chunk_id for c in context.chunks]
    # The flare system chunk (reranker_score=0.22 < 0.30) must be pruned
    assert "chk_irrelevant_flare" not in selected_ids
    # All selected chunks must have reranker_score >= 0.30
    for chunk in context.chunks:
        assert chunk.reranker_score >= 0.30


def test_context_redundancy_evaluation(context_selector):
    """Verify that redundant, highly overlapping chunks are suppressed."""
    context: BuiltContext = context_selector.build_context(CANDIDATE_POOL, token_budget=2000)

    selected_ids = [c.chunk_id for c in context.chunks]
    # chk_causes (0.94) is kept, but chk_causes_duplicate (overlap > 0.65) must be suppressed!
    assert "chk_causes" in selected_ids
    assert "chk_causes_duplicate" not in selected_ids


def test_context_size_budget_compliance(context_selector):
    """Verify that context size strictly respects small and large token budgets."""
    # Strict small budget (~60 tokens)
    small_context = context_selector.build_context(CANDIDATE_POOL, token_budget=75)
    assert small_context.token_count <= 85
    assert len(small_context.chunks) == 1  # Only fits top candidate

    # Standard budget (2000 tokens)
    standard_context = context_selector.build_context(CANDIDATE_POOL, token_budget=2000)
    assert standard_context.token_count <= 2000


def test_relevant_evidence_coverage_and_parameters(context_selector):
    """
    Verify multi-perspective evidence coverage and parameter preservation:
    - Causes, limits, and lubrication SOP all covered.
    - Physical engineering quantities preserved.
    """
    context = context_selector.build_context(CANDIDATE_POOL, token_budget=2000)

    selected_ids = [c.chunk_id for c in context.chunks]
    assert "chk_causes" in selected_ids
    assert "chk_limits" in selected_ids
    assert "chk_lube_oil" in selected_ids

    # Check preserved parameters
    params = context.preserved_parameters
    assert any("7.1" in p for p in params)
    assert any("4.5" in p for p in params)
    assert any("2.8" in p for p in params)


def test_lost_in_the_middle_ordering(context_selector):
    """
    Verify Lost-in-the-Middle structural placement:
    The two highest scoring chunks (rank 1 and rank 2) must be located
    at the very beginning (index 0) and the very end (index -1) of the context block.
    """
    context = context_selector.build_context(CANDIDATE_POOL, token_budget=2000)

    ordered_chunks = context.chunks
    assert len(ordered_chunks) >= 3

    top_score_chunk = ordered_chunks[0]
    second_score_chunk = ordered_chunks[-1]

    # Highest score (0.94) at front
    assert top_score_chunk.chunk_id == "chk_causes"
    # Second highest score (0.91) at tail
    assert second_score_chunk.chunk_id == "chk_limits"
