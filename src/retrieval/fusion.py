"""
PetroRAG Reciprocal Rank Fusion (RRF) Engine (Module 2.10)
Fuses heterogeneous ranking lists from dense vector search and sparse BM25 retrieval
into a unified candidate pool using Weighted Reciprocal Rank Fusion.
"""

from typing import Dict, List, Optional
from src.core.config import settings
from src.core.interfaces import RetrievedChunk, RetrievalChannel


def reciprocal_rank_fusion(
    ranked_lists: Dict[RetrievalChannel, List[RetrievedChunk]],
    k: int = settings.RRF_K,
    weights: Optional[Dict[RetrievalChannel, float]] = None,
    top_k: int = 10
) -> List[RetrievedChunk]:
    """
    Execute Weighted Reciprocal Rank Fusion:
    RRF_Score(d) = sum_{m in M} ( w_m / (k + rank_m(d)) )

    Parameters:
    - ranked_lists: Dictionary mapping channel (DENSE, SPARSE) to ordered chunk lists.
    - k: Smoothing constant (default 60).
    - weights: Channel weights (default dense=0.55, sparse=0.45).
    - top_k: Maximum fused candidates to return.
    """
    if not weights:
        weights = {
            RetrievalChannel.DENSE: settings.DENSE_WEIGHT,
            RetrievalChannel.SPARSE: settings.SPARSE_WEIGHT,
        }

    rrf_scores: Dict[str, float] = {}
    channel_ranks: Dict[str, Dict[str, int]] = {}
    best_chunk_instances: Dict[str, RetrievedChunk] = {}

    for channel, chunk_list in ranked_lists.items():
        channel_weight = weights.get(channel, 1.0)
        for rank_zero_idx, chunk in enumerate(chunk_list):
            cid = chunk.chunk_id
            rank_one_idx = rank_zero_idx + 1

            # Compute RRF score component
            term_score = channel_weight / (k + rank_one_idx)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + term_score

            # Track rank per channel for observability
            if cid not in channel_ranks:
                channel_ranks[cid] = {}
            channel_ranks[cid][channel.value] = rank_one_idx

            # Retain best chunk object (preserving rich text and metadata)
            if cid not in best_chunk_instances:
                best_chunk_instances[cid] = chunk.model_copy(deep=True)
            else:
                # Merge metadata fields if any are missing
                existing_meta = best_chunk_instances[cid].metadata or {}
                incoming_meta = chunk.metadata or {}
                for mk, mv in incoming_meta.items():
                    if mk not in existing_meta:
                        existing_meta[mk] = mv
                best_chunk_instances[cid].metadata = existing_meta

    # Sort descending by fused RRF score
    sorted_cids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results: List[RetrievedChunk] = []
    for cid, final_score in sorted_cids:
        chunk = best_chunk_instances[cid]
        chunk.score = round(final_score, 6)
        chunk.channel = RetrievalChannel.HYBRID

        # Attach rank provenance metadata
        chunk.metadata["rrf_score"] = chunk.score
        chunk.metadata["channel_ranks"] = channel_ranks.get(cid, {})
        fused_results.append(chunk)

    return fused_results
