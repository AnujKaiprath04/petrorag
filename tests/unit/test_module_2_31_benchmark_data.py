"""
Unit Tests for Module 2.31: Benchmark Corpus & Ground Truth Construction
Validates integrity of corpus_documents.json and benchmark_qa.json datasets.
"""

import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus_documents.json"
QA_PATH = ROOT_DIR / "data" / "ground_truth" / "benchmark_qa.json"


@pytest.fixture(scope="module")
def corpus_data():
    assert CORPUS_PATH.exists(), f"Corpus file missing at {CORPUS_PATH}"
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def qa_data():
    assert QA_PATH.exists(), f"QA file missing at {QA_PATH}"
    with open(QA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_corpus_document_structure(corpus_data):
    """Verifies that the corpus contains 12 documents with required metadata."""
    assert len(corpus_data) == 12

    required_doc_fields = {
        "document_id",
        "document_title",
        "asset_id",
        "equipment_type",
        "revision",
        "is_superseded",
        "field",
        "facility",
        "system",
        "chunks"
    }

    all_chunk_ids = set()
    for doc in corpus_data:
        assert required_doc_fields.issubset(doc.keys()), f"Doc {doc.get('document_id')} missing required fields"
        assert len(doc["chunks"]) >= 2, f"Doc {doc['document_id']} should have at least 2 chunks"

        for chunk in doc["chunks"]:
            assert "chunk_id" in chunk
            assert "document_id" in chunk
            assert chunk["document_id"] == doc["document_id"]
            assert "text" in chunk and len(chunk["text"]) > 40
            assert "page_number" in chunk
            assert "section_title" in chunk
            assert "metadata" in chunk

            # Ensure unique chunk IDs
            assert chunk["chunk_id"] not in all_chunk_ids, f"Duplicate chunk ID: {chunk['chunk_id']}"
            all_chunk_ids.add(chunk["chunk_id"])

    assert len(all_chunk_ids) >= 40


def test_corpus_domain_asset_coverage(corpus_data):
    """Verifies representation of all key oilfield equipment and safety standards."""
    asset_ids = {doc["asset_id"] for doc in corpus_data}
    expected_assets = {
        "C-101",
        "V-102",
        "ESDV-201",
        "ESP-304",
        "GDU-401",
        "KO-501",
        "E-205",
        "GTG-01",
        "PL-101",
        "WHCP-01",
        "SOP-SAF-012"
    }
    assert expected_assets.issubset(asset_ids)

    # Verify revision lineage for C-101
    c101_docs = [d for d in corpus_data if d["asset_id"] == "C-101"]
    assert len(c101_docs) == 2
    rev3 = next(d for d in c101_docs if d["revision"] == "REV-03")
    rev4 = next(d for d in c101_docs if d["revision"] == "REV-04")
    assert rev3["is_superseded"] is True
    assert rev3["superseded_by"] == "DOC-C101-REV04"
    assert rev4["is_superseded"] is False


def test_benchmark_qa_distribution_and_schema(qa_data, corpus_data):
    """Verifies the 50-query distribution (40 answerable, 10 unanswerable) and schema."""
    assert len(qa_data) == 50

    answerable = [q for q in qa_data if q["is_answerable"]]
    unanswerable = [q for q in qa_data if not q["is_answerable"]]

    assert len(answerable) == 40
    assert len(unanswerable) == 10

    # Collect all valid doc and chunk IDs from corpus
    valid_doc_ids = {d["document_id"] for d in corpus_data}
    valid_chunk_ids = {c["chunk_id"] for d in corpus_data for c in d["chunks"]}

    # Check answerable queries
    for q in answerable:
        assert q["query_id"].startswith("QA-")
        assert len(q["query_text"]) > 10
        assert len(q["ground_truth_answer"]) > 15
        assert len(q["target_asset_ids"]) >= 1
        assert len(q["supporting_doc_ids"]) >= 1
        assert len(q["supporting_chunk_ids"]) >= 1
        assert len(q["key_factual_units"]) >= 1

        # Check references match actual corpus
        for doc_id in q["supporting_doc_ids"]:
            assert doc_id in valid_doc_ids, f"Query {q['query_id']} references unknown doc {doc_id}"
        for chunk_id in q["supporting_chunk_ids"]:
            assert chunk_id in valid_chunk_ids, f"Query {q['query_id']} references unknown chunk {chunk_id}"

    # Check unanswerable queries
    categories = {q["domain_category"] for q in unanswerable}
    assert "adversarial_security" in categories
    assert "out_of_scope" in categories
    assert "nonexistent_asset" in categories
    assert "ambiguous" in categories

    for q in unanswerable:
        assert q["expected_abstention_barrier"] in {"query_security", "retrieval_barrier"}
        assert len(q["supporting_chunk_ids"]) == 0
