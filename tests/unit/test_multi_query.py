"""
Unit Test & Benchmark: Module 2.6 Multi-Query Retrieval Verification
Tests:
1. Exact prompt example multi-query formulation generation.
2. Independent retrieval execution and candidate deduplication.
3. Comparative Benchmark: Single-query retrieval vs. Multi-query retrieval.
"""

from typing import Any, Dict, List, Optional
import pytest
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.query.multi_query import MultiQueryRetriever


class MockDomainRetriever(BaseRetriever):
    """Mock domain retriever with realistic query-dependent technical results."""

    def __init__(self, corpus: Dict[str, Dict[str, Any]]):
        self.corpus = corpus

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        q_terms = set(query.lower().split())
        scored: List[RetrievedChunk] = []

        for cid, doc in self.corpus.items():
            doc_terms = set(doc["text"].lower().split())
            overlap = len(q_terms.intersection(doc_terms))
            if overlap > 0:
                score = round(overlap / (len(q_terms) + 2), 3)
                chunk = RetrievedChunk(
                    chunk_id=cid,
                    document_id=doc["document_id"],
                    document_title=doc.get("document_title"),
                    text=doc["text"],
                    score=score,
                    page_number=doc.get("page_number", 1),
                    section_title=doc.get("section_title"),
                    metadata=doc.get("metadata", {}),
                    channel=RetrievalChannel.DENSE
                )
                scored.append(chunk)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        return self.retrieve(query, top_k=top_k, filters=filters)


# Test corpus covering distinct perspectives of separator pressure issues
TEST_SEPARATOR_CORPUS = {
    "chk_sep_01": {
        "document_id": "DOC-SEP-SOP",
        "text": "Separator high pressure trip operating limits and pressure transmitter calibration.",
        "page_number": 12,
        "section_title": "Limits",
    },
    "chk_sep_02": {
        "document_id": "DOC-SEP-RCA",
        "text": "Separator pressure increase causes include foaming, gas outlet blockage, and mist extractor plugging.",
        "page_number": 45,
        "section_title": "Root Causes",
    },
    "chk_sep_03": {
        "document_id": "DOC-SEP-DIAG",
        "text": "Production separator pressure troubleshooting flowchart: inspect backpressure control valve and bypass lines.",
        "page_number": 68,
        "section_title": "Troubleshooting",
    },
    "chk_sep_04": {
        "document_id": "DOC-SEP-CTRL",
        "text": "Separator pressure control failure diagnosis: pneumatic pilot air pressure loss and diaphragm rupture.",
        "page_number": 89,
        "section_title": "Control Valves",
    },
    "chk_sep_05": {
        "document_id": "DOC-PUMP-GEN",
        "text": "Centrifugal water injection pump minimum flow recirculation line operation.",
        "page_number": 5,
        "section_title": "Pumps",
    },
}


def test_prompt_example_multi_query_generation():
    """
    Exact prompt requirement:
    Original:
    Why is separator pressure high?
    Queries:
    1. separator high pressure
    2. separator pressure increase causes
    3. production separator pressure troubleshooting
    4. separator pressure control failure
    """
    base_retriever = MockDomainRetriever(TEST_SEPARATOR_CORPUS)
    mqr = MultiQueryRetriever(base_retriever=base_retriever, max_queries=4)

    query = "Why is separator pressure high?"
    generated = mqr.generate_queries(query)

    assert "separator high pressure" in generated
    assert "separator pressure increase causes" in generated
    assert "production separator pressure troubleshooting" in generated
    assert "separator pressure control failure" in generated
    assert len(generated) == 4


def test_multi_query_retrieval_and_deduplication():
    """Verify that multi-query executes independently, merges results, and deduplicates chunks."""
    base_retriever = MockDomainRetriever(TEST_SEPARATOR_CORPUS)
    mqr = MultiQueryRetriever(base_retriever=base_retriever, max_queries=4)

    query = "Why is separator pressure high?"
    results = mqr.retrieve(query, top_k=5)

    # 1. Check deduplication: each chunk_id must appear exactly once
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

    # 2. Check that relevant multi-perspective chunks were captured
    assert "chk_sep_01" in chunk_ids
    assert "chk_sep_02" in chunk_ids
    assert "chk_sep_03" in chunk_ids
    assert "chk_sep_04" in chunk_ids

    # 3. Irrelevant chunk (pump) should be ranked below or excluded
    assert results[0].chunk_id != "chk_sep_05"


def test_comparative_retrieval_single_vs_multi():
    """
    Verification Gate:
    Compare Single-query retrieval vs Multi-query retrieval.
    Measure Recall against complete technical evidence set.
    """
    base_retriever = MockDomainRetriever(TEST_SEPARATOR_CORPUS)
    mqr = MultiQueryRetriever(base_retriever=base_retriever, max_queries=4)

    query = "Why is separator pressure high?"
    all_relevant = {"chk_sep_01", "chk_sep_02", "chk_sep_03", "chk_sep_04"}

    # 1. Single query retrieval (using raw user query only)
    single_results = base_retriever.retrieve(query, top_k=4)
    single_ids = set(c.chunk_id for c in single_results)
    single_recall = len(single_ids.intersection(all_relevant)) / len(all_relevant)

    # 2. Multi query retrieval (diverse formulations merged)
    multi_results = mqr.retrieve(query, top_k=4)
    multi_ids = set(c.chunk_id for c in multi_results)
    multi_recall = len(multi_ids.intersection(all_relevant)) / len(all_relevant)

    print("\n" + "=" * 60)
    print("EMPIRICAL COMPARISON: SINGLE-QUERY VS. MULTI-QUERY RETRIEVAL")
    print(f"Single Query Retrieved: {single_ids}")
    print(f"Single Query Recall@4:  {single_recall:.4f}")
    print(f"Multi-Query Retrieved:  {multi_ids}")
    print(f"Multi-Query Recall@4:   {multi_recall:.4f}")
    print("=" * 60)

    # Multi-query should equal or outperform naive single-query on complex O&G queries
    assert multi_recall >= single_recall
    assert multi_recall == 1.0  # Captures all 4 key perspectives (limits, causes, SOP, control failure)
