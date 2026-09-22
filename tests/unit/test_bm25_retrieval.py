"""
Unit Test & Benchmark: Module 2.7 BM25 Keyword Retrieval Verification
Verifies:
1. Exact technical terminology preservation: API 610, ESP, PSV, ESD, MMSCFD, LOTO.
2. Retrieval metrics: Recall@5, Recall@10, Precision@K, MRR across technical queries.
3. Index persistence: save and load verification.
4. Metadata-constrained BM25 retrieval.
"""

from pathlib import Path
import pytest
from src.core.interfaces import RetrievedChunk, RetrievalChannel
from src.retrieval.bm25 import BM25Retriever, tokenize_og_text


# Technical corpus targeting key Oil & Gas acronyms, equipment, and standards
TECHNICAL_OG_CORPUS = [
    RetrievedChunk(
        chunk_id="chunk_api_610",
        document_id="DOC-STD-API610",
        document_title="Centrifugal Pumps for Petroleum Industries.pdf",
        text="API 610 specifies requirements for centrifugal pumps for petroleum, petrochemical and natural gas industries.",
        page_number=1,
        section_title="Scope",
        metadata={"standard": "API 610", "equipment_type": "pump"}
    ),
    RetrievedChunk(
        chunk_id="chunk_esp_pump",
        document_id="DOC-ESP-SOP",
        document_title="ESP Operations Guide.pdf",
        text="ESP electric submersible pump downhole installation and drive head frequency control parameters.",
        page_number=15,
        section_title="Installation",
        metadata={"equipment_type": "esp", "process": "artificial lift"}
    ),
    RetrievedChunk(
        chunk_id="chunk_psv_valve",
        document_id="DOC-PSV-SPECS",
        document_title="Safety Relief Valve Calibration.pdf",
        text="PSV pressure safety valve set point calibration and testing procedure for high-pressure separator protection.",
        page_number=22,
        section_title="Calibration",
        metadata={"equipment_type": "valve", "tag": "PSV-402"}
    ),
    RetrievedChunk(
        chunk_id="chunk_esd_system",
        document_id="DOC-ESD-MANUAL",
        document_title="Emergency Shutdown Protocols.pdf",
        text="ESD emergency shutdown hierarchy Level 1 platform shutdown initiates blowdown and well isolation.",
        page_number=5,
        section_title="Logic Hierarchy",
        metadata={"safety": "ESD"}
    ),
    RetrievedChunk(
        chunk_id="chunk_mmscfd_rate",
        document_id="DOC-PROD-LOG",
        document_title="Daily Production Log 2024.pdf",
        text="Gas export pipeline measured average flow rate of 450 MMSCFD at 72 bar export pressure.",
        page_number=3,
        section_title="Gas Metering",
        metadata={"parameter": "flow_rate", "unit": "MMSCFD"}
    ),
    RetrievedChunk(
        chunk_id="chunk_loto_sop",
        document_id="DOC-SAFE-LOTO",
        document_title="Hazardous Energy Control SOP.pdf",
        text="LOTO lockout tagout procedure requires electrical zero energy verification and mechanical padlock placement.",
        page_number=8,
        section_title="Energy Isolation",
        metadata={"safety_procedure": "LOTO"}
    ),
    RetrievedChunk(
        chunk_id="chunk_sep_pressure",
        document_id="DOC-SEP-MANUAL",
        document_title="Production Separator Operations.pdf",
        text="Production separator high pressure trip is calibrated to 45.0 bar to avoid vessel overpressurization.",
        page_number=33,
        section_title="Operating Limits",
        metadata={"equipment_type": "separator"}
    ),
    RetrievedChunk(
        chunk_id="chunk_comp_vib",
        document_id="DOC-COMP-VIB",
        document_title="Centrifugal Compressor Diagnostics.pdf",
        text="Centrifugal compressor radial vibration probes monitor overall vibration limits up to 4.5 mm/s RMS.",
        page_number=19,
        section_title="Vibration",
        metadata={"equipment_type": "compressor"}
    ),
]

# Benchmark queries targeting exact technical terminology specified in prompt
TECH_EVAL_QUERIES = [
    {
        "query": "API 610 centrifugal pump requirements",
        "target_chunk": "chunk_api_610",
        "term": "API 610"
    },
    {
        "query": "ESP electric submersible pump operation",
        "target_chunk": "chunk_esp_pump",
        "term": "ESP"
    },
    {
        "query": "PSV pressure safety valve set point calibration",
        "target_chunk": "chunk_psv_valve",
        "term": "PSV"
    },
    {
        "query": "ESD emergency shutdown hierarchy",
        "target_chunk": "chunk_esd_system",
        "term": "ESD"
    },
    {
        "query": "450 MMSCFD gas export metering rate",
        "target_chunk": "chunk_mmscfd_rate",
        "term": "MMSCFD"
    },
    {
        "query": "LOTO lockout tagout energy isolation procedure",
        "target_chunk": "chunk_loto_sop",
        "term": "LOTO"
    },
]


@pytest.fixture
def bm25_retriever():
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index_chunks(TECHNICAL_OG_CORPUS)
    return retriever


def test_og_tokenizer():
    """Verify tokenizer correctly preserves technical codes, units, and acronyms."""
    text = "Inspect API 610 pump P-101A with PSV set at 52.4 bar(g) for 450 MMSCFD gas per LOTO SOP."
    tokens = tokenize_og_text(text)

    assert "api" in tokens
    assert "610" in tokens
    assert "p-101a" in tokens
    assert "psv" in tokens
    assert "bar(g)" in tokens
    assert "450" in tokens
    assert "mmscfd" in tokens
    assert "loto" in tokens


def test_exact_technical_term_retrieval(bm25_retriever):
    """
    Exact prompt requirement:
    Test technical terminology queries:
    API 610, ESP, PSV, ESD, MMSCFD, LOTO
    Verify that BM25 retrieves exact technical matches at Rank 1.
    """
    for item in TECH_EVAL_QUERIES:
        query = item["query"]
        target = item["target_chunk"]
        term = item["term"]

        results = bm25_retriever.retrieve(query, top_k=5)
        assert len(results) > 0, f"No results retrieved for '{query}'"

        top_hit = results[0]
        assert top_hit.chunk_id == target, (
            f"Failed exact technical match for '{term}'. "
            f"Expected {target} at rank 1, got {top_hit.chunk_id} with score {top_hit.score}"
        )
        assert top_hit.channel == RetrievalChannel.SPARSE
        assert top_hit.score > 0.0


def test_bm25_empirical_benchmark_metrics(bm25_retriever):
    """
    Empirical research benchmark:
    Measure:
    - Recall@5
    - Recall@10
    - Precision@5
    - MRR (Mean Reciprocal Rank)
    """
    recalls_5 = []
    recalls_10 = []
    precisions_5 = []
    reciprocal_ranks = []

    for item in TECH_EVAL_QUERIES:
        query = item["query"]
        target = item["target_chunk"]

        # Retrieve top 10
        results = bm25_retriever.retrieve(query, top_k=10)
        retrieved_ids = [r.chunk_id for r in results]

        # Recall@5
        hits_5 = 1.0 if target in retrieved_ids[:5] else 0.0
        recalls_5.append(hits_5)

        # Recall@10
        hits_10 = 1.0 if target in retrieved_ids[:10] else 0.0
        recalls_10.append(hits_10)

        # Precision@5 (Target relevant item in top 5)
        prec_5 = 1.0 / 5.0 if target in retrieved_ids[:5] else 0.0
        precisions_5.append(prec_5)

        # MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid == target:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    avg_recall_5 = sum(recalls_5) / len(recalls_5)
    avg_recall_10 = sum(recalls_10) / len(recalls_10)
    avg_prec_5 = sum(precisions_5) / len(precisions_5)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 60)
    print("EMPIRICAL BM25 RETRIEVAL BENCHMARK ON TECHNICAL TERMINOLOGY")
    print(f"Recall@5:     {avg_recall_5:.4f}")
    print(f"Recall@10:    {avg_recall_10:.4f}")
    print(f"Precision@5:  {avg_prec_5:.4f}")
    print(f"MRR:          {mrr:.4f}")
    print("=" * 60)

    assert avg_recall_5 == 1.0
    assert avg_recall_10 == 1.0
    assert mrr == 1.0


def test_bm25_persistence(tmp_path, bm25_retriever):
    """Verify serialization to and loading from JSON file."""
    save_path = tmp_path / "bm25_test_index.json"
    bm25_retriever.save(save_path)
    assert save_path.exists()

    new_retriever = BM25Retriever()
    assert new_retriever.load(save_path) is True
    assert new_retriever.doc_count == len(TECHNICAL_OG_CORPUS)

    # Confirm loaded retriever yields identical results
    results = new_retriever.retrieve("LOTO lockout tagout procedure", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk_loto_sop"


def test_bm25_metadata_filtering(bm25_retriever):
    """Verify that metadata filtering cleanly constrains BM25 candidate selection."""
    query = "inspection procedure"

    # 1. Unfiltered
    unfiltered = bm25_retriever.retrieve(query, top_k=5)
    assert len(unfiltered) > 0

    # 2. Filter by equipment_type: pump
    pump_filtered = bm25_retriever.retrieve(query, top_k=5, filters={"equipment_type": "pump"})
    for hit in pump_filtered:
        assert hit.metadata.get("equipment_type") == "pump"
