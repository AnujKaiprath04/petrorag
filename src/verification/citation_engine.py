"""
PetroRAG Citation Engine (Module 2.19)
Parses, maps, validates, and generates traceable evidence citations linking
answer claims to exact source documents, page numbers, and chunk IDs.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.core.interfaces import (
    BaseCitationEngine,
    BuiltContext,
    Citation,
    RerankedChunk,
)
from src.core.logging import logger
from src.generation.context_compressor import split_sentences


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower()))


def _compute_overlap(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Compute token overlap ratio relative to the smaller set."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    return len(intersection) / min(len(tokens_a), len(tokens_b))


class CitationEngine(BaseCitationEngine):
    """
    Deterministic citation parser, resolver, and validator.
    Binds generated answers to ground-truth document provenance.
    """

    def __init__(self, min_support_overlap: float = 0.20):
        self.min_support_overlap = min_support_overlap

    def _find_bracketed_markers(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract all bracketed citation expressions like [1], [1, 2], [DOC-01],
        [Document: C-101 Manual, Page 14, Chunk: chk-01].
        Returns list of (marker_content, start_pos, end_pos).
        """
        pattern = re.compile(r"\[([^\]\n]+)\]")
        matches = []
        for m in pattern.finditer(text):
            content = m.group(1).strip()
            # Ignore markdown links like [text](url) or empty brackets
            if content:
                matches.append((content, m.start(), m.end()))
        return matches

    def _resolve_chunk_by_marker(
        self,
        marker: str,
        chunks: List[RerankedChunk]
    ) -> List[RerankedChunk]:
        """
        Resolves a citation marker string to one or more candidate chunks.
        Supports:
        1. Numeric indices: '1', '2', '1, 2'
        2. Chunk IDs: 'chk-01', 'Chunk: chk-01'
        3. Document Title / ID: 'DOC-C101', 'Compressor Manual'
        4. Structured: 'DOCUMENT: ..., Page 14, Chunk: chk-01'
        """
        resolved: List[RerankedChunk] = []

        # Case 1: Numeric comma-separated list: '1', '2', '1, 2'
        if re.match(r"^\d+(?:\s*,\s*\d+)*$", marker):
            indices = [int(x.strip()) for x in marker.split(",")]
            for idx in indices:
                # 1-indexed to 0-indexed
                if 1 <= idx <= len(chunks):
                    resolved.append(chunks[idx - 1])
            return resolved

        # Case 2: Chunk ID mention (e.g. 'Chunk: chk-01' or 'chk-01')
        chunk_id_match = re.search(r"\b(chk-[a-zA-Z0-9_\-]+)\b", marker, re.IGNORECASE)
        if chunk_id_match:
            cid = chunk_id_match.group(1).lower()
            for c in chunks:
                if c.chunk_id.lower() == cid:
                    resolved.append(c)
                    return resolved

        # Case 3: Document ID or Title match
        marker_lower = marker.lower()
        for c in chunks:
            doc_id = (c.document_id or "").lower()
            doc_title = (c.document_title or "").lower()
            if (doc_id and doc_id in marker_lower) or (doc_title and doc_title in marker_lower):
                resolved.append(c)

        return resolved

    def _find_best_supporting_snippet(self, statement: str, chunk_text: str) -> str:
        """
        Finds the single sentence within the chunk that best supports the statement.
        """
        chunk_sentences = split_sentences(chunk_text)
        if not chunk_sentences:
            return chunk_text[:200]

        stmt_tokens = _tokenize(statement)
        best_sent = chunk_sentences[0]
        best_overlap = -1.0

        for sent in chunk_sentences:
            sent_tokens = _tokenize(sent)
            overlap = _compute_overlap(stmt_tokens, sent_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = sent

        return best_sent.strip()

    def extract_citations(
        self,
        answer_text: str,
        context: BuiltContext
    ) -> List[Citation]:
        """
        Implements BaseCitationEngine:
        Extracts and resolves all traceable citations from answer text using context chunks.
        If no explicit citations are found, falls back to automatic claim-to-chunk matching.
        """
        chunks = context.chunks
        if not chunks or not answer_text:
            return []

        markers = self._find_bracketed_markers(answer_text)
        citations: List[Citation] = []
        seen_chunk_ids: Set[str] = set()

        # Parse explicit citations
        for marker_str, start_pos, end_pos in markers:
            # Extract the preceding sentence/statement that this citation annotates
            preceding_text = answer_text[:start_pos]
            last_period = max(preceding_text.rfind("."), preceding_text.rfind("\n"))
            statement = preceding_text[last_period + 1:].strip() if last_period != -1 else preceding_text.strip()

            target_chunks = self._resolve_chunk_by_marker(marker_str, chunks)
            for c in target_chunks:
                if c.chunk_id in seen_chunk_ids:
                    continue

                snippet = self._find_best_supporting_snippet(statement or answer_text, c.text)
                citation_num = len(citations) + 1
                citations.append(Citation(
                    citation_id=f"[{citation_num}]",
                    document_id=c.document_id,
                    document_title=c.document_title,
                    page_number=c.page_number,
                    section_title=c.section_title,
                    chunk_id=c.chunk_id,
                    text_snippet=snippet,
                    relevance_score=c.reranker_score
                ))
                seen_chunk_ids.add(c.chunk_id)

        # Fallback: Automatic citation generation if model omitted bracketed markers
        if not citations:
            citations = self.auto_cite(answer_text, context)

        logger.info(f"Extracted {len(citations)} traceable citations from answer.")
        return citations

    def auto_cite(
        self,
        answer_text: str,
        context: BuiltContext
    ) -> List[Citation]:
        """
        Automatic citation synthesis:
        Segments answer into sentences, finds the highest-matching context chunk
        for each sentence exceeding min_support_overlap, and synthesizes citations.
        """
        sentences = split_sentences(answer_text)
        citations: List[Citation] = []
        seen_chunk_ids: Set[str] = set()

        for sent in sentences:
            sent_tokens = _tokenize(sent)
            if len(sent_tokens) < 3:
                continue

            best_chunk: Optional[RerankedChunk] = None
            best_overlap = self.min_support_overlap

            for c in context.chunks:
                chunk_tokens = _tokenize(c.text)
                overlap = _compute_overlap(sent_tokens, chunk_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chunk = c

            if best_chunk and best_chunk.chunk_id not in seen_chunk_ids:
                snippet = self._find_best_supporting_snippet(sent, best_chunk.text)
                citation_num = len(citations) + 1
                citations.append(Citation(
                    citation_id=f"[{citation_num}]",
                    document_id=best_chunk.document_id,
                    document_title=best_chunk.document_title,
                    page_number=best_chunk.page_number,
                    section_title=best_chunk.section_title,
                    chunk_id=best_chunk.chunk_id,
                    text_snippet=snippet,
                    relevance_score=best_chunk.reranker_score
                ))
                seen_chunk_ids.add(best_chunk.chunk_id)

        return citations

    def validate_citations(
        self,
        citations: List[Citation],
        answer_text: str,
        context: BuiltContext
    ) -> Dict[str, Any]:
        """
        Evaluates citation quality metrics:
        - precision: proportion of citations whose text_snippet has meaningful overlap with the answer.
        - valid_count: total verified valid citations.
        - hallucinated_citations: citations pointing to chunks absent from context.
        """
        context_chunk_ids = {c.chunk_id: c for c in context.chunks}
        answer_tokens = _tokenize(answer_text)

        valid = 0
        hallucinated = 0

        for cit in citations:
            if cit.chunk_id not in context_chunk_ids:
                hallucinated += 1
                continue

            snippet_tokens = _tokenize(cit.text_snippet)
            overlap = _compute_overlap(answer_tokens, snippet_tokens)
            if overlap >= self.min_support_overlap:
                valid += 1

        total = len(citations)
        precision = (valid / total) if total > 0 else 1.0

        return {
            "total_citations": total,
            "valid_citations": valid,
            "hallucinated_citations": hallucinated,
            "citation_precision": round(precision, 4)
        }
