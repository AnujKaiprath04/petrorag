"""
PetroRAG Retrieval Deduplication Engine (Module 2.11)
Implements multi-tier deduplication:
1. Exact content hash deduplication.
2. Near-duplicate suppression via Jaccard n-gram similarity.
3. Same-page sliding window redundancy reduction without dropping distinct passages.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from src.core.interfaces import RetrievedChunk
from src.core.logging import logger


class DeduplicationResult(BaseModel):
    """Detailed telemetry and audit report for retrieval deduplication."""
    deduplicated_chunks: List[RetrievedChunk] = Field(default_factory=list)
    total_input: int
    total_output: int
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    same_page_redundancies_reduced: int = 0
    duplicate_percentage_before: float = 0.0
    duplicate_percentage_after: float = 0.0
    relevant_chunks_retained: int = 0


def _normalize_for_hash(text: str) -> str:
    """Normalize text for exact hash comparison: lowercase, collapse whitespace, strip punctuation."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _compute_jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Calculate Jaccard similarity coefficient between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def _extract_word_tokens(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric word set."""
    return set(re.findall(r"\b\w+\b", text.lower()))


class RetrievalDeduplicator:
    """
    Deduplicates retrieved chunks from multiple retrieval channels (Dense, Sparse, Multi-Query).
    Preserves distinct passages even from the same document and page, while pruning
    identical content and sliding window chunking redundancies.
    """

    def __init__(
        self,
        exact_hash_dedup: bool = True,
        near_duplicate_threshold: float = 0.80,
        same_page_redundancy_reduction: bool = True,
        same_page_overlap_threshold: float = 0.60
    ):
        self.exact_hash_dedup = exact_hash_dedup
        self.near_duplicate_threshold = near_duplicate_threshold
        self.same_page_redundancy_reduction = same_page_redundancy_reduction
        self.same_page_overlap_threshold = same_page_overlap_threshold

    def deduplicate(
        self,
        chunks: List[RetrievedChunk],
        ground_truth_relevant_ids: Optional[Set[str]] = None
    ) -> DeduplicationResult:
        """
        Execute multi-stage deduplication on a candidate chunk pool.
        """
        if not chunks:
            return DeduplicationResult(total_input=0, total_output=0)

        total_input = len(chunks)

        # Stage 1: Exact chunk_id & content hash deduplication
        seen_ids: Set[str] = set()
        seen_hashes: Dict[str, RetrievedChunk] = {}
        exact_removed = 0
        stage1_chunks: List[RetrievedChunk] = []

        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                exact_removed += 1
                continue
            seen_ids.add(chunk.chunk_id)

            if self.exact_hash_dedup:
                content_hash = hashlib.sha256(_normalize_for_hash(chunk.text).encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    exact_removed += 1
                    # Keep the higher scoring chunk
                    existing = seen_hashes[content_hash]
                    if chunk.score > existing.score:
                        existing.score = chunk.score
                    continue
                seen_hashes[content_hash] = chunk

            stage1_chunks.append(chunk)

        # Sort descending by score before near-duplicate and same-page comparison
        # (guarantees that higher-scoring candidates are prioritized and preserved)
        stage1_chunks.sort(key=lambda c: c.score, reverse=True)

        # Stage 2: Same-page sliding window redundancy reduction
        # Checks chunks originating from same document AND same page
        stage2_chunks: List[RetrievedChunk] = []
        same_page_removed = 0

        for chunk in stage1_chunks:
            is_redundant_same_page = False
            if self.same_page_redundancy_reduction and chunk.page_number is not None:
                c_tokens = _extract_word_tokens(chunk.text)
                for kept in stage2_chunks:
                    if (
                        kept.document_id == chunk.document_id
                        and kept.page_number == chunk.page_number
                    ):
                        kept_tokens = _extract_word_tokens(kept.text)
                        sim = _compute_jaccard_similarity(c_tokens, kept_tokens)
                        if sim >= self.same_page_overlap_threshold:
                            is_redundant_same_page = True
                            same_page_removed += 1
                            break

            if not is_redundant_same_page:
                stage2_chunks.append(chunk)

        # Stage 3: Global Near-Duplicate Suppression (Jaccard token overlap)
        final_chunks: List[RetrievedChunk] = []
        near_dup_removed = 0

        for chunk in stage2_chunks:
            is_near_dup = False
            c_tokens = _extract_word_tokens(chunk.text)

            for kept in final_chunks:
                kept_tokens = _extract_word_tokens(kept.text)
                sim = _compute_jaccard_similarity(c_tokens, kept_tokens)
                if sim >= self.near_duplicate_threshold:
                    is_near_dup = True
                    near_dup_removed += 1
                    break

            if not is_near_dup:
                final_chunks.append(chunk)

        total_output = len(final_chunks)
        total_duplicates_removed = exact_removed + same_page_removed + near_dup_removed

        pct_before = (total_duplicates_removed / total_input * 100.0) if total_input > 0 else 0.0
        pct_after = 0.0  # Output contains zero exact/near duplicates

        # Track relevant chunks retained
        relevant_retained = 0
        if ground_truth_relevant_ids:
            relevant_retained = sum(1 for c in final_chunks if c.chunk_id in ground_truth_relevant_ids)

        return DeduplicationResult(
            deduplicated_chunks=final_chunks,
            total_input=total_input,
            total_output=total_output,
            exact_duplicates_removed=exact_removed,
            near_duplicates_removed=near_dup_removed,
            same_page_redundancies_reduced=same_page_removed,
            duplicate_percentage_before=round(pct_before, 2),
            duplicate_percentage_after=pct_after,
            relevant_chunks_retained=relevant_retained
        )
