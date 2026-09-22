"""
PetroRAG Cross-Encoder Reranking Engine (Module 2.12)
Applies second-stage cross-encoder scoring to top hybrid retrieval candidates,
measuring pairwise relevance, computing calibrated reranker scores, and tracking rank shifts.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.core.config import settings
from src.core.interfaces import BaseReranker, RetrievedChunk, RerankedChunk
from src.core.logging import logger


class CrossEncoderReranker(BaseReranker):
    """
    Production Cross-Encoder reranker utilizing sentence-transformers / HuggingFace.
    Computes full cross-attention logits over (query, document) pairs.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = settings.RERANKER_BATCH_SIZE
    ):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device or settings.RERANKER_DEVICE
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model '{self.model_name}' on {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _sigmoid(self, x: float) -> float:
        """Calibrate raw cross-encoder logits into [0, 1] probability range."""
        return float(1.0 / (1.0 + np.exp(-x)))

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        """
        Score (query, chunk.text) pairs and return top_k RerankedChunk items.
        Records initial_score, reranker_score, and final_rank.
        """
        if not candidates:
            return []

        start_time = time.perf_counter()
        pairs = [[query, c.text] for c in candidates]

        # Predict raw logits
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build candidate pool with both initial and reranker scores
        scored_pairs: List[Tuple[RetrievedChunk, float, int]] = []
        for idx, (chunk, raw_score) in enumerate(zip(candidates, raw_scores)):
            calibrated = self._sigmoid(float(raw_score))
            scored_pairs.append((chunk, calibrated, idx + 1))

        # Sort descending by reranker score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_results: List[RerankedChunk] = []
        for final_rank, (chunk, r_score, initial_rank) in enumerate(scored_pairs[:top_k], start=1):
            meta = dict(chunk.metadata or {})
            meta["initial_rank"] = initial_rank
            meta["final_rank"] = final_rank
            meta["rank_delta"] = initial_rank - final_rank
            meta["rerank_latency_ms"] = round(elapsed_ms, 2)

            reranked_item = RerankedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                text=chunk.text,
                initial_score=chunk.score,
                reranker_score=round(r_score, 4),
                rank=final_rank,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                metadata=meta
            )
            reranked_results.append(reranked_item)

        logger.info(
            f"Reranked {len(candidates)} candidates -> top {len(reranked_results)} in {elapsed_ms:.2f}ms"
        )
        return reranked_results

    async def arerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        """Asynchronous reranking."""
        return self.rerank(query, candidates, top_k=top_k)


class MockCrossEncoderReranker(BaseReranker):
    """
    Deterministic pseudo-cross-encoder for fast unit testing and benchmarks.
    Simulates deep cross-attention alignment using lexical-semantic term interaction.
    """

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        if not candidates:
            return []

        start_time = time.perf_counter()
        STOPWORDS = {
            "what", "how", "why", "does", "the", "in", "of", "and", "is", "a", "an",
            "to", "for", "on", "with", "do", "i", "at", "by", "or", "from", "be",
            "this", "that", "it", "are", "under", "your", "my", "all"
        }
        query_clean = query.lower().replace("_", " ")
        raw_words = re.findall(r"\b[a-zA-Z0-9\-\.]+\b", query_clean)
        q_tokens = [w for w in raw_words if w not in STOPWORDS and len(w) > 1]
        if not q_tokens:
            q_tokens = raw_words

        # Check if query specifically targets an equipment ISA/asset tag (e.g. C-101, CH-999, ESDV-201)
        query_equipment_tags = set(re.findall(r"\b[A-Z]{1,4}-\d{2,4}[A-Z]?\b", query))

        scored: List[Tuple[RetrievedChunk, float, int]] = []
        for idx, chunk in enumerate(candidates):
            c_text_clean = chunk.text.lower().replace("_", " ")
            c_tokens = set(re.findall(r"\b[a-zA-Z0-9\-\.]+\b", c_text_clean))

            # Chunk equipment tags from text and metadata
            chunk_tags = set(re.findall(r"\b[A-Z]{1,4}-\d{2,4}[A-Z]?\b", chunk.text))
            if chunk.metadata and chunk.metadata.get("asset_id"):
                chunk_tags.add(chunk.metadata["asset_id"].upper())

            # If query explicitly targets an equipment ID and chunk is explicitly tagged for a different asset:
            # apply strict cross-equipment penalty to prevent cross-asset hallucinations
            if query_equipment_tags and chunk_tags and not query_equipment_tags.intersection(chunk_tags):
                calibrated = 0.08
            else:
                # Simulate cross-encoder: exact matching terms + substring bonus
                overlap = len([t for t in q_tokens if t in c_tokens])
                overlap_ratio = (overlap / len(q_tokens)) if q_tokens else 0.0
                phrase_bonus = 0.5 if any(q in c_text_clean for q in q_tokens if len(q) > 4) else 0.0

                # Suppress score if overlap is negligible (e.g. out of scope or unrelated topic)
                if overlap == 0 or (overlap < 2 and overlap_ratio < 0.20):
                    calibrated = 0.10 + (overlap_ratio * 0.4)
                else:
                    pseudo_logit = (overlap * 1.3) + (overlap_ratio * 2.0) + phrase_bonus + (chunk.score * 0.4)
                    calibrated = float(1.0 / (1.0 + np.exp(-pseudo_logit + 2.2)))

            scored.append((chunk, round(calibrated, 4), idx + 1))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked_results: List[RerankedChunk] = []
        for final_rank, (chunk, r_score, initial_rank) in enumerate(scored[:top_k], start=1):
            meta = dict(chunk.metadata or {})
            meta["initial_rank"] = initial_rank
            meta["final_rank"] = final_rank
            meta["rank_delta"] = initial_rank - final_rank
            meta["rerank_latency_ms"] = round(elapsed_ms, 3)

            reranked_results.append(
                RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    text=chunk.text,
                    initial_score=chunk.score,
                    reranker_score=round(r_score, 4),
                    rank=final_rank,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    metadata=meta
                )
            )

        return reranked_results

    async def arerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RerankedChunk]:
        return self.rerank(query, candidates, top_k=top_k)
