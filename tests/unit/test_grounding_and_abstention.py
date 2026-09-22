"""
Unit Test: Modules 2.22 - 2.24 Grounding Verification, Hallucination Detection & Abstention
Verifies:
1. Claim-level factual grounding evaluation against source context.
2. Technical parameter hallucination detection (unsupported numbers/tolerances).
3. Direct operational contradiction detection.
4. MultiBarrierAbstainer: Retrieval quality barrier, Grounding barrier, Hallucination barrier.
5. Empirical Research Benchmark: Abstention Precision, Recall, and F1 on answerable vs unanswerable queries.
"""

import pytest
from src.core.interfaces import BuiltContext, ClaimSupportStatus, GroundingResult, RerankedChunk
from src.verification.abstainer import AbstentionBarrier, AbstentionDecision, MultiBarrierAbstainer
from src.verification.grounding_evaluator import GroundingEvaluator


def _create_sample_context() -> BuiltContext:
    c1 = RerankedChunk(
        chunk_id="chk-c101",
        document_id="DOC-C101-MANUAL",
        document_title="Centrifugal Compressor Manual.pdf",
        text=(
            "Centrifugal compressor C-101 normal discharge pressure is 42.5 bar at 8500 RPM. "
            "High vibration alarm triggers at 4.5 mm/s per ISO 10816-3. "
            "Emergency shutdown (ESD) occurs at 7.1 mm/s."
        ),
        initial_score=0.92,
        reranker_score=0.96,
        rank=1,
        page_number=14
    )
    c2 = RerankedChunk(
        chunk_id="chk-valves",
        document_id="DOC-VALVES-01",
        document_title="Isolation Valve Matrix.pdf",
        text="Valve XV-101 is normally open during routine hydrocarbon production.",
        initial_score=0.88,
        reranker_score=0.91,
        rank=2,
        page_number=5
    )
    return BuiltContext(
        context_text=f"[DOC-C101-MANUAL]\n{c1.text}\n\n---\n\n[DOC-VALVES-01]\n{c2.text}",
        chunks=[c1, c2],
        token_count=65,
        total_candidates_evaluated=2,
        pruned_chunks_count=0
    )


@pytest.mark.asyncio
async def test_grounding_evaluator_fully_supported_answer():
    """Verify that an answer with verified facts and exact parameters receives high grounding score."""
    evaluator = GroundingEvaluator(grounding_threshold=0.70)
    context = _create_sample_context()

    answer = (
        "Centrifugal compressor C-101 normal discharge pressure is 42.5 bar at 8500 RPM. "
        "The high vibration alarm triggers at 4.5 mm/s per ISO 10816-3."
    )

    result = await evaluator.evaluate_grounding(answer, context)

    assert isinstance(result, GroundingResult)
    assert result.is_grounded is True
    assert result.grounding_score >= 0.70
    assert result.hallucination_detected is False
    assert result.abstention_recommended is False
    assert result.supported_claims >= 1


@pytest.mark.asyncio
async def test_grounding_evaluator_hallucinated_parameter_detection():
    """Verify that fabricated parameters not present in context trigger hallucination detection."""
    evaluator = GroundingEvaluator(grounding_threshold=0.70)
    context = _create_sample_context()

    # The context has 42.5 bar and 4.5 mm/s, but the answer fabricates 98.0 bar and 14.5 mm/s
    hallucinated_answer = (
        "Centrifugal compressor C-101 normal discharge pressure is 98.0 bar at 12000 RPM. "
        "Vibration trip limit is calibrated to 14.5 mm/s."
    )

    result = await evaluator.evaluate_grounding(hallucinated_answer, context)

    assert result.is_grounded is False
    assert result.hallucination_detected is True
    assert result.abstention_recommended is True
    assert result.unsupported_claims > 0


@pytest.mark.asyncio
async def test_grounding_evaluator_direct_contradiction_detection():
    """Verify that inverted operational states (normally closed vs normally open) trigger contradiction."""
    evaluator = GroundingEvaluator(grounding_threshold=0.70)
    context = _create_sample_context()

    # Context says XV-101 is normally open
    contradictory_answer = "Valve XV-101 is normally closed during routine production."

    result = await evaluator.evaluate_grounding(contradictory_answer, context)

    assert result.is_grounded is False
    assert result.hallucination_detected is True
    assert any(c.status == ClaimSupportStatus.CONTRADICTED for c in result.claims)


def test_multi_barrier_abstainer_retrieval_barrier():
    """Verify Barrier 2 suppresses responses when no relevant documentation exists."""
    abstainer = MultiBarrierAbstainer(retrieval_threshold=0.30)

    # Sub-case A: Empty candidate pool
    empty_decision = abstainer.evaluate_retrieval_barrier([])
    assert empty_decision.is_abstained is True
    assert empty_decision.barrier == AbstentionBarrier.RETRIEVAL_QUALITY

    # Sub-case B: Candidates below relevance threshold
    poor_chunk = RerankedChunk(
        chunk_id="chk-bad",
        document_id="DOC-UNRELATED",
        text="Unrelated catering menu.",
        initial_score=0.10,
        reranker_score=0.18,  # Below 0.30
        rank=1
    )
    poor_decision = abstainer.evaluate_retrieval_barrier([poor_chunk])
    assert poor_decision.is_abstained is True
    assert poor_decision.barrier == AbstentionBarrier.RETRIEVAL_QUALITY

    # Sub-case C: High relevance chunk passes
    good_chunk = RerankedChunk(
        chunk_id="chk-good",
        document_id="DOC-SOP",
        text="SOP text.",
        initial_score=0.85,
        reranker_score=0.92,
        rank=1
    )
    good_decision = abstainer.evaluate_retrieval_barrier([good_chunk])
    assert good_decision.is_abstained is False


def test_multi_barrier_abstainer_grounding_barrier():
    """Verify Barriers 3 & 4 suppress responses when hallucination or low grounding is detected."""
    abstainer = MultiBarrierAbstainer(grounding_threshold=0.70)

    # Sub-case A: Hallucination detected
    hallucinated_grounding = GroundingResult(
        grounding_score=0.50,
        total_claims=2,
        supported_claims=1,
        unsupported_claims=1,
        is_grounded=False,
        hallucination_detected=True,
        abstention_recommended=True
    )
    h_decision = abstainer.evaluate_grounding_barrier(hallucinated_grounding)
    assert h_decision.is_abstained is True
    assert h_decision.barrier == AbstentionBarrier.HALLUCINATION_DETECTED

    # Sub-case B: Low grounding without explicit hallucination
    weak_grounding = GroundingResult(
        grounding_score=0.40,
        total_claims=2,
        supported_claims=0,
        partially_supported_claims=1,
        unsupported_claims=1,
        is_grounded=False,
        hallucination_detected=False,
        abstention_recommended=True
    )
    w_decision = abstainer.evaluate_grounding_barrier(weak_grounding)
    assert w_decision.is_abstained is True
    assert w_decision.barrier == AbstentionBarrier.GROUNDING_FAILURE

    # Sub-case C: High grounding passes
    clean_grounding = GroundingResult(
        grounding_score=1.0,
        total_claims=2,
        supported_claims=2,
        unsupported_claims=0,
        is_grounded=True,
        hallucination_detected=False,
        abstention_recommended=False
    )
    c_decision = abstainer.evaluate_grounding_barrier(clean_grounding)
    assert c_decision.is_abstained is False


@pytest.mark.asyncio
async def test_empirical_abstention_and_hallucination_benchmark():
    """
    Major Empirical Research Benchmark:
    Evaluates multi-barrier abstention across a representative test suite:
    - 2 fully answerable queries (ground truth: should NOT abstain)
    - 1 out-of-domain query with poor retrieval (ground truth: SHOULD abstain)
    - 1 answer containing fabricated numbers (ground truth: SHOULD abstain)
    - 1 answer with inverted operational state (ground truth: SHOULD abstain)

    Measures:
    - Abstention Precision, Recall, and F1 Score
    - Hallucination Suppression Rate (target: 100%)
    """
    context = _create_sample_context()
    evaluator = GroundingEvaluator(grounding_threshold=0.70)
    abstainer = MultiBarrierAbstainer(retrieval_threshold=0.30, grounding_threshold=0.70)

    test_battery = [
        # Test Case 1: Grounded answer (Answerable)
        {
            "query": "What is the normal operating discharge pressure for C-101?",
            "chunks": context.chunks,
            "answer": "Centrifugal compressor C-101 normal discharge pressure is 42.5 bar at 8500 RPM.",
            "expected_abstain": False
        },
        # Test Case 2: Grounded answer with alarm setpoint (Answerable)
        {
            "query": "What is the alarm setpoint for C-101 vibration?",
            "chunks": context.chunks,
            "answer": "High vibration alarm triggers at 4.5 mm/s per ISO 10816-3.",
            "expected_abstain": False
        },
        # Test Case 3: Out of domain / zero relevant documentation (Unanswerable)
        {
            "query": "What is the maintenance protocol for submarine nuclear reactor coolant pump?",
            "chunks": [
                RerankedChunk(
                    chunk_id="chk-unrelated",
                    document_id="DOC-IRRELEVANT",
                    text="Kitchen inventory checklist for platform mess hall.",
                    initial_score=0.05,
                    reranker_score=0.12,  # < 0.30
                    rank=1
                )
            ],
            "answer": "Maintenance requires coolant flush.",
            "expected_abstain": True
        },
        # Test Case 4: Fabricated numerical parameters (Hallucination)
        {
            "query": "What is the trip setpoint for C-101?",
            "chunks": context.chunks,
            "answer": "Compressor C-101 trip occurs at 88.5 bar and 19.5 mm/s.",
            "expected_abstain": True
        },
        # Test Case 5: Direct contradiction of operational status (Contradiction)
        {
            "query": "What is the normal position of valve XV-101?",
            "chunks": context.chunks,
            "answer": "Valve XV-101 is normally closed during routine hydrocarbon production.",
            "expected_abstain": True
        }
    ]

    tp, fp, tn, fn = 0, 0, 0, 0
    hallucinations_total = 2  # Cases 4 and 5
    hallucinations_blocked = 0

    for idx, tc in enumerate(test_battery):
        chunks = tc["chunks"]
        answer = tc["answer"]
        expected = tc["expected_abstain"]

        # Run retrieval barrier
        retrieval_decision = abstainer.evaluate_retrieval_barrier(chunks)
        if retrieval_decision.is_abstained:
            actual_abstained = True
        else:
            # Run grounding & verification barrier
            grounding_res = await evaluator.evaluate_grounding(answer, BuiltContext(
                context_text="\n".join(c.text for c in chunks),
                chunks=chunks,
                token_count=50,
                total_candidates_evaluated=len(chunks),
                pruned_chunks_count=0
            ))
            final_decision = abstainer.evaluate_all_barriers(chunks, grounding_res)
            actual_abstained = final_decision.is_abstained

        if idx in [3, 4] and actual_abstained:
            hallucinations_blocked += 1

        if actual_abstained and expected:
            tp += 1
        elif actual_abstained and not expected:
            fp += 1
        elif not actual_abstained and not expected:
            tn += 1
        elif not actual_abstained and expected:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    hallucination_suppression_rate = hallucinations_blocked / hallucinations_total

    print("\n" + "=" * 70)
    print("EMPIRICAL RESEARCH BENCHMARK: MULTI-BARRIER ABSTENTION & SAFETY")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Measured Value':<25}")
    print("-" * 70)
    print(f"{'Total Test Battery Cases':<35} | {len(test_battery):<25}")
    print(f"{'True Positives (Properly Abstained)':<35} | {tp:<25}")
    print(f"{'True Negatives (Grounded Released)':<35} | {tn:<25}")
    print(f"{'False Positives (Erroneous Abstention)':<35} | {fp:<25}")
    print(f"{'False Negatives (Hallucination Leaked)':<35} | {fn:<25}")
    print(f"{'Abstention Precision':<35} | {precision:.4f}")
    print(f"{'Abstention Recall':<35} | {recall:.4f}")
    print(f"{'Abstention F1 Score':<35} | {f1:.4f}")
    print(f"{'Hallucination Suppression Rate':<35} | {hallucination_suppression_rate:.4f} ({hallucinations_blocked}/{hallucinations_total})")
    print("=" * 70)

    # Rigorous research assertions
    assert precision == 1.0000
    assert recall == 1.0000
    assert f1 == 1.0000
    assert hallucination_suppression_rate == 1.0000  # 100% of hallucinations blocked!
