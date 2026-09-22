"""
PetroRAG BM25 Lexical Retrieval Engine (Module 2.7)
Implements BM25Okapi with specialized Oil & Gas tokenization, preserving
exact equipment codes, engineering standards, technical acronyms, and units.
"""

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
from src.core.config import settings
from src.core.interfaces import BaseRetriever, RetrievedChunk, RetrievalChannel
from src.core.logging import logger


class BM25IndexData(BaseModel):
    """Serialized state of the BM25 index."""
    k1: float = 1.5
    b: float = 0.75
    doc_count: int = 0
    avg_doc_len: float = 0.0
    doc_ids: List[str] = Field(default_factory=list)
    doc_lengths: Dict[str, int] = Field(default_factory=dict)
    doc_term_freqs: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    doc_frequencies: Dict[str, int] = Field(default_factory=dict)
    chunk_records: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def tokenize_og_text(text: str) -> List[str]:
    """
    Oil & Gas specialized tokenizer:
    - Preserves hyphenated equipment tags (e.g., S-101, C-101, P-101A)
    - Preserves standard codes (e.g., API 610, ASME B31.3)
    - Preserves technical acronyms (ESP, PSV, ESD, MMSCFD, LOTO, P&ID)
    - Preserves engineering units with symbols (bar(g), m3/d, mm/s)
    """
    if not text:
        return []

    # Replace P&ID with pid for tokenization safety
    text_clean = re.sub(r"\bP&ID\b", "p_and_id", text, flags=re.IGNORECASE)

    # Regex capturing:
    # 1. Words with hyphens/slashes/parentheses: S-101, bar(g), m3/d
    # 2. Standard alphanumeric tokens
    token_pattern = re.compile(
        r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+(?:\([a-zA-Z0-9]+\))?|[A-Za-z0-9]+(?:\([a-zA-Z0-9]+\))?|[A-Za-z0-9]+",
        re.IGNORECASE
    )

    tokens = []
    for match in token_pattern.finditer(text_clean):
        tok = match.group(0).lower()
        if tok == "p_and_id":
            tokens.append("p&id")
        else:
            tokens.append(tok)
    return tokens


class BM25Retriever(BaseRetriever):
    """
    BM25Okapi lexical retriever implementing BaseRetriever interface.
    Features exact technical match preservation, persistence, and metadata filtering.
    """

    def __init__(
        self,
        k1: float = settings.BM25_K1,
        b: float = settings.BM25_B,
        index_path: Optional[Path] = settings.BM25_INDEX_PATH
    ):
        self.k1 = k1
        self.b = b
        self.index_path = Path(index_path) if index_path else None

        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_ids: List[str] = []
        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_frequencies: Dict[str, int] = {}
        self.chunk_records: Dict[str, RetrievedChunk] = {}

    def index_chunks(self, chunks: List[RetrievedChunk]) -> int:
        """
        Build or update the BM25 index from a collection of RetrievedChunk objects.
        """
        for chunk in chunks:
            cid = chunk.chunk_id
            tokens = tokenize_og_text(chunk.text)
            doc_len = len(tokens)

            # Record chunk object
            self.chunk_records[cid] = chunk.model_copy(deep=True)
            self.chunk_records[cid].channel = RetrievalChannel.SPARSE

            # Term frequencies for this document
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            if cid not in self.doc_lengths:
                self.doc_ids.append(cid)
            self.doc_lengths[cid] = doc_len
            self.doc_term_freqs[cid] = tf

        # Recompute corpus statistics
        self.doc_count = len(self.doc_ids)
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = (total_len / self.doc_count) if self.doc_count > 0 else 0.0

        # Recompute document frequencies
        self.doc_frequencies.clear()
        for tf in self.doc_term_freqs.values():
            for term in tf.keys():
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1

        logger.info(f"Indexed {len(chunks)} chunks into BM25 index (total docs: {self.doc_count}, vocab: {len(self.doc_frequencies)})")
        return self.doc_count

    def _idf(self, term: str) -> float:
        """
        Compute standard Robertson-Spärck Jones Okapi IDF with +1 smoothing:
        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
        """
        n_q = self.doc_frequencies.get(term, 0)
        return math.log((self.doc_count - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Execute BM25 lexical retrieval for query, optionally applying metadata filters.
        """
        if self.doc_count == 0:
            return []

        query_tokens = tokenize_og_text(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for doc_id in self.doc_ids:
            # Check metadata filter if present
            chunk = self.chunk_records[doc_id]
            if filters and not self._matches_filter(chunk, filters):
                continue

            doc_len = self.doc_lengths[doc_id]
            tf_dict = self.doc_term_freqs[doc_id]
            doc_score = 0.0

            for q_term in query_tokens:
                if q_term not in tf_dict:
                    continue

                f_q = tf_dict[q_term]
                idf = self._idf(q_term)

                # Standard Okapi BM25 formula
                numerator = f_q * (self.k1 + 1.0)
                denominator = f_q + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (numerator / denominator)
                doc_score += term_score

            if doc_score > 0.0:
                scores[doc_id] = doc_score

        # Sort descending by BM25 score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[RetrievedChunk] = []
        for doc_id, score in sorted_docs:
            chunk = self.chunk_records[doc_id].model_copy(deep=True)
            chunk.score = round(score, 4)
            chunk.channel = RetrievalChannel.SPARSE
            results.append(chunk)

        return results

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Asynchronous BM25 retrieval."""
        return self.retrieve(query, top_k=top_k, filters=filters)

    def _matches_filter(self, chunk: RetrievedChunk, filters: Dict[str, Any]) -> bool:
        """Check if chunk conforms to metadata constraints."""
        meta = chunk.metadata or {}
        for key, expected_val in filters.items():
            if expected_val is None:
                continue
            actual_val = meta.get(key)
            if actual_val is None:
                # Also check direct chunk attributes (document_id, equipment_id)
                if hasattr(chunk, key):
                    actual_val = getattr(chunk, key)
                else:
                    return False
            if str(actual_val).lower() != str(expected_val).lower():
                return False
        return True

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serialize index data to JSON for persistence."""
        dest = filepath or self.index_path
        if not dest:
            raise ValueError("No filepath specified to save BM25 index.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        data = BM25IndexData(
            k1=self.k1,
            b=self.b,
            doc_count=self.doc_count,
            avg_doc_len=self.avg_doc_len,
            doc_ids=self.doc_ids,
            doc_lengths=self.doc_lengths,
            doc_term_freqs=self.doc_term_freqs,
            doc_frequencies=self.doc_frequencies,
            chunk_records={cid: c.model_dump() for cid, c in self.chunk_records.items()}
        )

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, indent=2)

        logger.info(f"Saved BM25 index to {dest}")
        return dest

    def load(self, filepath: Optional[Path] = None) -> bool:
        """Load index from serialized JSON file."""
        src = filepath or self.index_path
        if not src or not src.exists():
            logger.warning(f"BM25 index file {src} not found.")
            return False

        with open(src, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = BM25IndexData.model_validate(raw)
        self.k1 = data.k1
        self.b = data.b
        self.doc_count = data.doc_count
        self.avg_doc_len = data.avg_doc_len
        self.doc_ids = data.doc_ids
        self.doc_lengths = data.doc_lengths
        self.doc_term_freqs = data.doc_term_freqs
        self.doc_frequencies = data.doc_frequencies
        self.chunk_records = {
            cid: RetrievedChunk.model_validate(rec)
            for cid, rec in data.chunk_records.items()
        }

        logger.info(f"Loaded BM25 index from {src} ({self.doc_count} documents)")
        return True
