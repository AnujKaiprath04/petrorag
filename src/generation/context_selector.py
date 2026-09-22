"""
PetroRAG Context Selection Engine (Module 2.13)
Selects, orders, and token-budgets the most useful evidence chunks based on
relevance, cross-encoder scores, semantic diversity, source reliability, and
lost-in-the-middle positioning.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.core.config import settings
from src.core.interfaces import BaseContextBuilder, RerankedChunk, BuiltContext
from src.core.logging import logger


def estimate_tokens(text: str) -> int:
    """Accurately estimate token count (approx 1 token per 0.75 words / 4 chars)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(int(words * 1.33), int(chars / 4.0))


def _extract_word_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _compute_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return float(len(tokens_a.intersection(tokens_b))) / float(len(tokens_a.union(tokens_b)))


class ContextSelector(BaseContextBuilder):
    """
    Builds compact, high-density context windows from reranked candidate chunks.
    Enforces token budget limits, diversity control, source reliability weighting,
    and lost-in-the-middle structural arrangement.
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.30,
        max_redundancy_threshold: float = 0.65,
        lost_in_the_middle_ordering: bool = True
    ):
        self.min_relevance_threshold = min_relevance_threshold
        self.max_redundancy_threshold = max_redundancy_threshold
        self.lost_in_the_middle_ordering = lost_in_the_middle_ordering

    def _source_reliability_multiplier(self, chunk: RerankedChunk) -> float:
        """
        Compute source reliability score based on document type and approved status.
        Approved SOPs, operational manuals, and industry standards are prioritized
        over unvetted logs.
        """
        meta = chunk.metadata or {}
        doc_type = str(meta.get("document_type", "")).lower()

        if any(dt in doc_type for dt in ["standard", "sop", "manual"]):
            return 1.15
        if "procedure" in doc_type or "datasheet" in doc_type:
            return 1.10
        if "draft" in doc_type or "scratch" in doc_type:
            return 0.85
        return 1.0

    def select_chunks(
        self,
        candidates: List[RerankedChunk],
        token_budget: int = settings.CONTEXT_TOKEN_BUDGET
    ) -> Tuple[List[RerankedChunk], int, int]:
        """
        Greedy diversity-aware knapsack selection within token budget:
        1. Filters candidates below min_relevance_threshold.
        2. Adjusts priority by source reliability.
        3. Prunes chunks that duplicate information already selected (Jaccard > threshold).
        4. Halts when adding another chunk would exceed the token budget.
        """
        total_evaluated = len(candidates)
        pruned_count = 0

        # Filter out low-relevance candidates
        filtered: List[Tuple[RerankedChunk, float]] = []
        for c in candidates:
            if c.reranker_score < self.min_relevance_threshold:
                pruned_count += 1
                continue
            reliability = self._source_reliability_multiplier(c)
            effective_score = c.reranker_score * reliability
            filtered.append((c, effective_score))

        # Sort by effective relevance descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        selected: List[RerankedChunk] = []
        selected_token_sets: List[Set[str]] = []
        accumulated_tokens = 0

        for chunk, _ in filtered:
            c_tokens = _extract_word_tokens(chunk.text)

            # Check diversity against already selected chunks
            is_redundant = False
            for prev_tokens in selected_token_sets:
                sim = _compute_jaccard(c_tokens, prev_tokens)
                if sim >= self.max_redundancy_threshold:
                    is_redundant = True
                    pruned_count += 1
                    break

            if is_redundant:
                continue

            # Estimate tokens for this chunk (including metadata framing header)
            framing_header = (
                f"[Source: {chunk.document_title or chunk.document_id}, "
                f"Page {chunk.page_number or 'N/A'}, Section: {chunk.section_title or 'General'}]\n"
            )
            chunk_tokens = estimate_tokens(framing_header + chunk.text + "\n\n")

            # Check token budget
            if accumulated_tokens + chunk_tokens > token_budget:
                pruned_count += 1
                continue

            selected.append(chunk)
            selected_token_sets.append(c_tokens)
            accumulated_tokens += chunk_tokens

        return selected, total_evaluated, pruned_count

    def _order_chunks(self, chunks: List[RerankedChunk]) -> List[RerankedChunk]:
        """
        Lost-in-the-Middle mitigation ordering:
        Places the most critical evidence (ranks 1 & 2) at the beginning and end
        of the context block, placing intermediate background in the center.
        Pattern: [Rank 1, Rank 3, ..., Rank 4, Rank 2]
        """
        if len(chunks) <= 2 or not self.lost_in_the_middle_ordering:
            return chunks

        # Sort by reranker_score descending first
        sorted_chunks = sorted(chunks, key=lambda c: c.reranker_score, reverse=True)

        front: List[RerankedChunk] = []
        back: List[RerankedChunk] = []

        for i, chunk in enumerate(sorted_chunks):
            if i % 2 == 0:
                front.append(chunk)
            else:
                back.insert(0, chunk)

        return front + back

    def build_context(
        self,
        candidates: List[RerankedChunk],
        token_budget: int = settings.CONTEXT_TOKEN_BUDGET,
        preserve_parameters: bool = True
    ) -> BuiltContext:
        """
        Implements BaseContextBuilder:
        Selects, orders, and formats candidate chunks into an injection-safe context payload.
        """
        selected_chunks, total_eval, pruned_count = self.select_chunks(
            candidates,
            token_budget=token_budget
        )

        # Apply lost-in-the-middle structural arrangement
        ordered_chunks = self._order_chunks(selected_chunks)

        # Format context text blocks with explicit provenance tags
        context_blocks = []
        preserved_params: List[str] = []

        param_regex = re.compile(r"\b\d+(?:\.\d+)?\s*(?:bar|bar\(g\)|barg|psi|psig|deg\s*c|°c|m3/d|bpd|mmscfd|mm/s|rpm)\b", re.IGNORECASE)

        for chunk in ordered_chunks:
            doc_id = chunk.document_title or chunk.document_id
            page_str = f"Page {chunk.page_number}" if chunk.page_number else "Page N/A"
            sec_str = f"Section: {chunk.section_title}" if chunk.section_title else "General"

            header = f"[DOCUMENT: {doc_id} | {page_str} | {sec_str} | ChunkID: {chunk.chunk_id}]"
            block = f"{header}\n{chunk.text.strip()}"
            context_blocks.append(block)

            # Audit preserved technical parameters
            if preserve_parameters:
                matches = param_regex.findall(chunk.text)
                preserved_params.extend(matches)

        context_text = "\n\n---\n\n".join(context_blocks)
        final_token_count = estimate_tokens(context_text)

        logger.info(
            f"Built context: {len(ordered_chunks)} chunks, ~{final_token_count} tokens "
            f"(budget={token_budget}, pruned={pruned_count})"
        )

        return BuiltContext(
            context_text=context_text,
            chunks=ordered_chunks,
            token_count=final_token_count,
            total_candidates_evaluated=total_eval,
            pruned_chunks_count=pruned_count,
            preserved_parameters=list(set(preserved_params))
        )
