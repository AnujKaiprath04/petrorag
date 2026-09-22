"""
PetroRAG Context Compression Engine (Module 2.14)
Extractive sentence selection and relevance filtering that compresses retrieved
context windows without summarizing away critical technical values (e.g.
Pressures, Temperatures, Equipment IDs, Safety Procedures, Standards).
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from src.core.config import settings
from src.core.interfaces import (
    BaseContextCompressor,
    BuiltContext,
    CompressedChunk,
    RerankedChunk,
)
from src.core.logging import logger


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 1 token per 0.75 words / 4 chars)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(int(words * 1.33), int(chars / 4.0))


# Common technical abbreviations to protect from false sentence splits
_ABBREVIATIONS = [
    "e.g.", "i.e.", "approx.", "ref.", "no.", "fig.", "rev.",
    "min.", "max.", "temp.", "press.", "vs.", "vol.", "dept.",
    "dr.", "mr.", "mrs.", "inc.", "ltd.", "corp.", "co.", "spec."
]

# Regex to detect engineering parameters (numerical values + units)
PARAM_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:"
    r"bar|bar\(g\)|barg|bar\(a\)|bara|psi|psig|psia|kpa|mpa|"
    r"deg\s*c|deg\s*f|°c|°f|k|"
    r"m3/d|m3/h|m3/s|bpd|bph|bpm|mmscfd|scfd|scfm|gpm|l/min|kg/s|kg/h|t/h|"
    r"mm/s|in/s|m/s|m/s2|g|"
    r"rpm|hz|khz|"
    r"kw|mw|hp|w|v|kv|a|ka|ma|kwh|mwh|"
    r"mm|cm|m|in|inch|inches|ft|feet|micron|microns|µm|"
    r"kg/m3|g/cm3|api|°api|cp|cst|sg|"
    r"%|ppm|ppb|mol%|vol%|wt%"
    r")\b",
    re.IGNORECASE
)

# Regex to detect equipment tags and codes (e.g., C-101, S-101, P-202A, ESD-04, PSV-501)
TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b")

# Regex to detect industry engineering standards (e.g., API 610, ISO 10816-3, ASME B31.3)
STANDARD_PATTERN = re.compile(
    r"\b(?:API|ISO|ASME|IEC|NFPA|NACE|NORSOK|OSHA|ASTM|BS|DIN|EN)\s*[-_]?[A-Z0-9]+(?:[\.\-_/][0-9A-Z]+)*\b",
    re.IGNORECASE
)

# Regex to detect critical safety terms and trip directives
SAFETY_PATTERN = re.compile(
    r"\b(?:DANGER|WARNING|CAUTION|MANDATORY|TRIP|SHUTDOWN|EMERGENCY|HAZOP|LOTO|ESD|PSV|PERMIT|INTERLOCK|FAILSAFE)\b",
    re.IGNORECASE
)

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def split_sentences(text: str) -> List[str]:
    """
    Splits text into sentences while protecting decimals, technical abbreviations,
    and numbered list items from false breaks.
    """
    if not text or not text.strip():
        return []

    processed = text.strip()

    # Step 1: Mask decimal numbers (e.g. 42.5 -> 42<DEC>5)
    processed = re.sub(r"(\d+)\.(\d+)", r"\1<DEC>\2", processed)

    # Step 2: Mask common abbreviations preserving case
    for abbr in _ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        processed = pattern.sub(lambda m: m.group(0).replace(".", "<DOT>"), processed)

    # Step 3: Mask numbered list items (e.g. "1. " -> "1<LIST> ")
    processed = re.sub(r"(^|\n)(\d+)\.\s+", r"\1\2<LIST> ", processed)

    # Step 4: Split on sentence terminal punctuation followed by space or newline
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'\-])", processed)

    # Step 5: Unmask tokens and clean
    cleaned_sentences: List[str] = []
    for s in raw_sentences:
        # Also split internal newlines if separate paragraphs exist
        parts = re.split(r"\n{2,}", s)
        for part in parts:
            restored = part.replace("<DEC>", ".").replace("<DOT>", ".").replace("<LIST>", ".")
            restored = restored.strip()
            if restored:
                cleaned_sentences.append(restored)

    return cleaned_sentences


def extract_parameters(text: str) -> List[str]:
    """Extract all technical engineering values, equipment tags, and standards."""
    params = []
    params.extend(PARAM_PATTERN.findall(text))
    params.extend(TAG_PATTERN.findall(text))
    params.extend(STANDARD_PATTERN.findall(text))
    return list(set(params))


class ContextCompressor(BaseContextCompressor):
    """
    Extractive context compressor tailored for Oil & Gas technical texts.
    Performs sentence salience evaluation against user queries, preserves critical
    engineering parameters and safety directives, and reconstructs chronological text.
    """

    def __init__(
        self,
        target_ratio: float = 0.65,
        min_sentence_score: float = 0.15,
        preserve_critical_params: bool = True
    ):
        self.target_ratio = target_ratio
        self.min_sentence_score = min_sentence_score
        self.preserve_critical_params = preserve_critical_params

    def _extract_query_tokens(self, query: str) -> Set[str]:
        """Tokenize query and strip common stop words."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", query.lower())
        return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

    def score_sentence(
        self,
        sentence: str,
        query_tokens: Set[str],
        query_tags: Set[str],
        sentence_index: int = 0
    ) -> Tuple[float, List[str]]:
        """
        Calculate sentence relevance score against query.
        Returns (relevance_score, list_of_preserved_parameters).
        """
        sentence_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", sentence.lower()))
        params_found = extract_parameters(sentence)
        safety_found = SAFETY_PATTERN.findall(sentence)

        if not sentence_tokens:
            return 0.0, []

        # 1. Lexical overlap score (Jaccard-like over query terms)
        matching_query_tokens = query_tokens.intersection(sentence_tokens)
        overlap_score = (len(matching_query_tokens) / len(query_tokens)) if query_tokens else 0.0

        # 2. Equipment tag match boost
        tag_match_boost = 0.0
        sentence_upper = sentence.upper()
        for tag in query_tags:
            if tag.upper() in sentence_upper:
                tag_match_boost += 0.35

        # 3. Technical parameter boost
        param_boost = 0.0
        if params_found:
            # Check if query asks for a parameter (pressure, vibration, temp, speed, etc.)
            has_param_query = any(
                p_word in query_tokens
                for p_word in ["pressure", "temperature", "vibration", "speed", "flow", "rate", "level", "limit", "alarm", "trip"]
            )
            param_boost = 0.25 if has_param_query else 0.10

        # 4. Critical safety directive boost
        safety_boost = 0.20 if safety_found else 0.0

        # 5. Position bias (slight lead-in sentence bonus)
        pos_boost = 0.05 if sentence_index == 0 else 0.0

        total_score = overlap_score + tag_match_boost + param_boost + safety_boost + pos_boost
        return total_score, params_found

    def compress_chunk(
        self,
        chunk: RerankedChunk,
        query: str,
        target_ratio: Optional[float] = None
    ) -> CompressedChunk:
        """
        Extractively compress an individual RerankedChunk against the query.
        Guarantees that selected sentences maintain original order and preserved
        technical parameters are audited.
        """
        ratio = target_ratio if target_ratio is not None else self.target_ratio
        orig_text = chunk.text.strip()
        orig_tokens = estimate_tokens(orig_text)

        sentences = split_sentences(orig_text)
        if len(sentences) <= 1:
            # Single sentence or indivisible chunk
            params = extract_parameters(orig_text)
            return CompressedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                original_text=orig_text,
                compressed_text=orig_text,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                compression_ratio=1.0,
                preserved_parameters=params,
                retained_sentences_count=len(sentences),
                total_sentences_count=len(sentences),
                rank=chunk.rank,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                metadata=chunk.metadata
            )

        query_tokens = self._extract_query_tokens(query)
        query_tags = set(TAG_PATTERN.findall(query))

        # Score all sentences
        scored_sentences = []
        all_chunk_params = []
        for idx, s in enumerate(sentences):
            score, params = self.score_sentence(s, query_tokens, query_tags, sentence_index=idx)
            all_chunk_params.extend(params)
            scored_sentences.append({
                "index": idx,
                "text": s,
                "score": score,
                "tokens": estimate_tokens(s),
                "params": params,
                "is_safety": bool(SAFETY_PATTERN.findall(s))
            })

        # Sort sentences by relevance score descending
        ranked_candidates = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)

        token_target = max(15, int(orig_tokens * ratio))
        selected_indices: Set[int] = set()
        accumulated_tokens = 0
        preserved_params: List[str] = []

        # Always include the top-scoring sentence
        top_sentence = ranked_candidates[0]
        selected_indices.add(top_sentence["index"])
        accumulated_tokens += top_sentence["tokens"]
        preserved_params.extend(top_sentence["params"])

        # Greedily include remaining sentences if score meets threshold and within target budget
        for item in ranked_candidates[1:]:
            # If critical parameter preservation is active and sentence contains query tag or safety directive
            has_query_tag = any(t.upper() in item["text"].upper() for t in query_tags)
            is_critical = (has_query_tag or item["is_safety"]) and self.preserve_critical_params

            if item["score"] < self.min_sentence_score and not is_critical:
                continue

            if (accumulated_tokens + item["tokens"] <= token_target) or is_critical:
                selected_indices.add(item["index"])
                accumulated_tokens += item["tokens"]
                preserved_params.extend(item["params"])

        # Reconstruct text in ORIGINAL sentence sequence for natural readability
        retained_sentences = [
            sentences[i] for i in sorted(selected_indices)
        ]
        compressed_text = " ".join(retained_sentences)
        compressed_tokens = estimate_tokens(compressed_text)
        actual_ratio = (compressed_tokens / orig_tokens) if orig_tokens > 0 else 1.0

        return CompressedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            original_text=orig_text,
            compressed_text=compressed_text,
            original_tokens=orig_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=round(actual_ratio, 4),
            preserved_parameters=list(set(preserved_params)),
            retained_sentences_count=len(retained_sentences),
            total_sentences_count=len(sentences),
            rank=chunk.rank,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            metadata=chunk.metadata
        )

    def compress_chunks(
        self,
        chunks: List[RerankedChunk],
        query: str,
        target_ratio: Optional[float] = None
    ) -> List[CompressedChunk]:
        """Compress a pool of candidate chunks against the user query."""
        results: List[CompressedChunk] = []
        for c in chunks:
            compressed = self.compress_chunk(c, query=query, target_ratio=target_ratio)
            results.append(compressed)
        return results

    def to_reranked_chunks(self, compressed_chunks: List[CompressedChunk]) -> List[RerankedChunk]:
        """
        Utility converting CompressedChunks back into RerankedChunks with compressed text,
        enabling drop-in compatibility with downstream modules.
        """
        converted: List[RerankedChunk] = []
        for c in compressed_chunks:
            meta = dict(c.metadata)
            meta["original_tokens"] = c.original_tokens
            meta["compressed_tokens"] = c.compressed_tokens
            meta["compression_ratio"] = c.compression_ratio
            meta["retained_sentences"] = c.retained_sentences_count

            converted.append(RerankedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                text=c.compressed_text,
                initial_score=0.0,
                reranker_score=1.0,  # preserves priority
                rank=c.rank,
                page_number=c.page_number,
                section_title=c.section_title,
                metadata=meta
            ))
        return converted

    def compress_built_context(
        self,
        context: BuiltContext,
        query: str,
        target_ratio: Optional[float] = None
    ) -> BuiltContext:
        """
        Compresses an already constructed BuiltContext payload.
        Updates context_text, token_count, preserved_parameters, and compression_ratio.
        """
        compressed_chunks = self.compress_chunks(
            context.chunks,
            query=query,
            target_ratio=target_ratio
        )
        new_reranked = self.to_reranked_chunks(compressed_chunks)

        # Rebuild formatted context block
        context_blocks = []
        all_preserved_params = set()

        for c in compressed_chunks:
            doc_id = c.document_title or c.document_id
            header = f"[DOCUMENT: {doc_id} | ChunkID: {c.chunk_id}]"
            block = f"{header}\n{c.compressed_text.strip()}"
            context_blocks.append(block)
            all_preserved_params.update(c.preserved_parameters)

        new_context_text = "\n\n---\n\n".join(context_blocks)
        new_token_count = estimate_tokens(new_context_text)
        overall_ratio = (new_token_count / context.token_count) if context.token_count > 0 else 1.0

        return BuiltContext(
            context_text=new_context_text,
            chunks=new_reranked,
            token_count=new_token_count,
            total_candidates_evaluated=context.total_candidates_evaluated,
            pruned_chunks_count=context.pruned_chunks_count,
            preserved_parameters=list(all_preserved_params),
            compression_ratio=round(overall_ratio, 4)
        )
