"""
Unit Test: Module 2.14 Context Compression Verification
Verifies:
1. Sentence segmentation with technical O&G notations, decimals, and abbreviations.
2. Parameter and safety directive extraction precision.
3. Extractive sentence selection against query with boilerplate pruning.
4. Retention guarantee for critical engineering parameters and safety directives.
5. Comparative empirical experiment: Full Context vs. Extractive Compressed Context.
6. Integration with BuiltContext and RerankedChunk schemas.
"""

import pytest
from src.core.interfaces import (
    BuiltContext,
    CompressedChunk,
    RerankedChunk,
)
from src.generation.context_compressor import (
    ContextCompressor,
    estimate_tokens,
    extract_parameters,
    split_sentences,
)


def test_sentence_segmentation_with_engineering_notation():
    """Verify that decimals, technical abbreviations, and standards do not cause false breaks."""
    text = (
        "Compressor C-101 operates at 42.5 bar and 8500 RPM. "
        "Refer to API 610 Rev. 11 for casing specs e.g. wall thickness. "
        "The alarm triggers at approx. 4.5 mm/s per ISO 10816-3. "
        "Routine logs are archived under Ref. No. 4912."
    )
    sentences = split_sentences(text)

    # Must produce exactly 4 coherent sentences without splitting at 42.5, Rev. 11, e.g., approx., or No. 4912
    assert len(sentences) == 4
    assert "42.5 bar" in sentences[0]
    assert "API 610 Rev. 11" in sentences[1]
    assert "e.g." in sentences[1]
    assert "approx. 4.5 mm/s" in sentences[2]
    assert "Ref. No. 4912." in sentences[3]


def test_parameter_and_tag_extraction():
    """Verify extraction of pressures, vibration, speed, tags, and engineering standards."""
    text = (
        "Separator S-101 high-pressure trip is set to 68.5 bar(g). "
        "Pump P-202A operates at 2950 RPM with vibration under 2.8 mm/s. "
        "Complies with ASME B31.3 and API 610 standards."
    )
    params = extract_parameters(text)

    assert any("68.5" in p for p in params)
    assert any("2950" in p or "RPM" in p for p in params)
    assert any("2.8" in p or "mm/s" in p for p in params)
    assert "S-101" in params
    assert "P-202A" in params
    assert any("API 610" in p for p in params)
    assert any("ASME B31.3" in p for p in params)


def test_chunk_compression_prunes_boilerplate_and_preserves_parameters():
    """Verify that irrelevant administrative sentences are pruned while critical engineering values remain."""
    compressor = ContextCompressor(target_ratio=0.60, min_sentence_score=0.15)

    chunk = RerankedChunk(
        chunk_id="chk-compressor-01",
        document_id="DOC-MAN-C101",
        document_title="Centrifugal Compressor Operating Manual.pdf",
        text=(
            "Centrifugal compressor C-101 is installed on the main process deck. "
            "Normal operating discharge pressure is 42.5 bar at 8500 RPM. "
            "High vibration alarm triggers at 4.5 mm/s, while emergency shutdown (ESD) occurs at 7.1 mm/s per ISO 10816-3. "
            "The equipment was commissioned in August 2021 by ABC Engineering Services Ltd. "
            "Administrative shift handovers must be recorded in the physical desk logbook. "
            "Routine maintenance archives are filed in file cabinet 4."
        ),
        initial_score=0.88,
        reranker_score=0.95,
        rank=1,
        page_number=18,
        section_title="4.2 Vibration Limits"
    )

    query = "What are the vibration alarm and shutdown thresholds for compressor C-101?"
    result = compressor.compress_chunk(chunk, query=query, target_ratio=0.55)

    # Compression verification
    assert isinstance(result, CompressedChunk)
    assert result.compressed_tokens < result.original_tokens
    assert result.compression_ratio < 1.0
    assert result.retained_sentences_count < result.total_sentences_count

    # Critical parameters and entities MUST be preserved
    assert "4.5 mm/s" in result.compressed_text
    assert "7.1 mm/s" in result.compressed_text
    assert "C-101" in result.compressed_text
    assert "ISO 10816-3" in result.compressed_text

    # Boilerplate sentences MUST be pruned away
    assert "ABC Engineering Services Ltd" not in result.compressed_text
    assert "file cabinet 4" not in result.compressed_text
    assert "desk logbook" not in result.compressed_text

    # Chronological sentence order preserved
    pos_c101 = result.compressed_text.find("C-101")
    pos_vib = result.compressed_text.find("High vibration alarm")
    assert pos_c101 < pos_vib


def test_built_context_compression_integration():
    """Verify compress_built_context updates BuiltContext text, token count, and ratio."""
    compressor = ContextCompressor(target_ratio=0.60)

    chunk1 = RerankedChunk(
        chunk_id="chk-01",
        document_id="DOC-01",
        document_title="Separator Manual.pdf",
        text=(
            "Separator S-101 operating pressure is 45.0 bar. "
            "High pressure trip is activated at 52.0 bar by PSV-101. "
            "General paint inspection was completed last quarter."
        ),
        initial_score=0.80,
        reranker_score=0.92,
        rank=1
    )
    chunk2 = RerankedChunk(
        chunk_id="chk-02",
        document_id="DOC-02",
        document_title="Pump Specs.pdf",
        text=(
            "Booster pump P-201A flow rate is 350 m3/h at 1450 RPM. "
            "Suction pressure must be maintained above 3.5 bar. "
            "Operator lunch breaks are scheduled at 12:30 PM daily."
        ),
        initial_score=0.75,
        reranker_score=0.85,
        rank=2
    )

    orig_built = BuiltContext(
        context_text=f"[DOC-01]\n{chunk1.text}\n\n---\n\n[DOC-02]\n{chunk2.text}",
        chunks=[chunk1, chunk2],
        token_count=estimate_tokens(chunk1.text + chunk2.text) + 20,
        total_candidates_evaluated=2,
        pruned_chunks_count=0
    )

    compressed_built = compressor.compress_built_context(
        orig_built,
        query="What is the high pressure trip for separator S-101?",
        target_ratio=0.60
    )

    assert compressed_built.token_count < orig_built.token_count
    assert compressed_built.compression_ratio is not None
    assert compressed_built.compression_ratio < 1.0
    assert "52.0 bar" in compressed_built.context_text
    assert "PSV-101" in compressed_built.context_text
    assert "lunch breaks" not in compressed_built.context_text
    assert "General paint" not in compressed_built.context_text


def test_empirical_compression_experiment_comparison():
    """
    Major Research Experiment:
    Compares Uncompressed Full Context vs. Extractive Compressed Context.
    Evaluates:
    - Original Token Count vs Compressed Token Count
    - Compression Ratio
    - Tokens Saved
    - Critical Parameter Retention Rate (target: 100%)
    """
    chunks = [
        RerankedChunk(
            chunk_id="chk-c101",
            document_id="DOC-OPS-01",
            document_title="Compressor Operations.pdf",
            text=(
                "Compressor C-101 discharge pressure is regulated at 42.0 bar. "
                "Vibration trip limit is strictly calibrated to 7.1 mm/s. "
                "Staff timesheets should be signed on Fridays."
            ),
            initial_score=0.90,
            reranker_score=0.95,
            rank=1
        ),
        RerankedChunk(
            chunk_id="chk-s101",
            document_id="DOC-OPS-02",
            document_title="Separator Operations.pdf",
            text=(
                "Three-phase separator S-101 operates at 45.0 bar and 65 °C. "
                "Liquid carryover occurs if level exceeds 78%. "
                "Coffee machine maintenance is scheduled for Monday morning."
            ),
            initial_score=0.85,
            reranker_score=0.91,
            rank=2
        ),
        RerankedChunk(
            chunk_id="chk-p201",
            document_id="DOC-OPS-03",
            document_title="Pump Operations.pdf",
            text=(
                "Export pump P-201A flow is rated at 520 m3/h at 2950 RPM. "
                "Mechanical seal flush plan 53B pressure is maintained at 6.0 bar. "
                "Visitor badges must be returned to reception."
            ),
            initial_score=0.80,
            reranker_score=0.88,
            rank=3
        ),
    ]

    query = "What are the operating pressure and vibration trip limits for compressor C-101?"
    compressor = ContextCompressor(target_ratio=0.65, min_sentence_score=0.15)

    compressed_results = compressor.compress_chunks(chunks, query=query, target_ratio=0.65)

    total_orig_tokens = sum(c.original_tokens for c in compressed_results)
    total_comp_tokens = sum(c.compressed_tokens for c in compressed_results)
    tokens_saved = total_orig_tokens - total_comp_tokens
    empirical_ratio = total_comp_tokens / total_orig_tokens

    # Ground truth relevant critical parameters for query:
    # C-101, 42.0 bar, 7.1 mm/s
    relevant_target_params = ["C-101", "42.0 bar", "7.1 mm/s"]
    preserved_in_compressed = " ".join(c.compressed_text for c in compressed_results)

    retained_count = sum(1 for p in relevant_target_params if p in preserved_in_compressed)
    param_retention_rate = retained_count / len(relevant_target_params)

    # Print empirical table
    print("\n" + "=" * 70)
    print("EMPIRICAL RESEARCH BENCHMARK: CONTEXT COMPRESSION ABLATION")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Uncompressed':<15} | {'Extractive Compressed':<15}")
    print("-" * 70)
    print(f"{'Total Context Tokens':<35} | {total_orig_tokens:<15} | {total_comp_tokens:<15}")
    print(f"{'Tokens Saved':<35} | {'0':<15} | {tokens_saved:<15}")
    print(f"{'Compression Ratio':<35} | {'1.0000':<15} | {empirical_ratio:.4f}")
    print(f"{'Critical Parameter Retention':<35} | {'1.0000 (3/3)':<15} | {param_retention_rate:.4f} ({retained_count}/{len(relevant_target_params)})")
    print("=" * 70)

    # Assertions
    assert tokens_saved > 0
    assert empirical_ratio < 0.85
    assert param_retention_rate == 1.0  # 100% critical parameters preserved!
    assert "timesheets" not in preserved_in_compressed
    assert "Coffee machine" not in preserved_in_compressed
    assert "Visitor badges" not in preserved_in_compressed
